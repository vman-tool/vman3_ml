"""Label quality audit for CCVA training data.

Two-stage pipeline:
  1. Cleanlab  — statistical detection of label noise using out-of-fold
                 cross-validation probabilities from the best trained model.
  2. LLM review — Claude API second-opinion on the top-N flagged records,
                  using narrative text and symptom profile.

Output is a review CSV and JSON summary intended for physician sign-off.
No labels are auto-corrected — this is a review tool only.
"""

import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    _HAS_CLEANLAB = True
except ImportError:
    _HAS_CLEANLAB = False

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from .narrative import NARRATIVE_COLS


_REVIEW_PROMPT = """\
You are a verbal autopsy expert with clinical and epidemiological expertise. \
Review the following case and assess whether the assigned cause of death is \
consistent with the evidence.

ASSIGNED CAUSE OF DEATH (physician label): {true_label}
ML MODEL PREDICTION: {model_pred} (confidence: {confidence:.0%})

NARRATIVE:
{narrative}

POSITIVE SYMPTOMS / SIGNS REPORTED:
{symptoms}

DEMOGRAPHICS: {demographics}

Respond ONLY with this exact JSON (no markdown, no extra text):
{{"plausible": "yes|uncertain|no", "alternative_causes": ["cause1", "cause2"], \
"reasoning": "2-3 sentence clinical justification"}}"""


class LabelAuditor:
    """Detect potential label errors in VA training data and review with LLM.

    Usage::

        auditor = LabelAuditor(cv_folds=5, top_n_llm=50, verbose=True)
        cleanlab_df, llm_df = auditor.run(
            X_scaled=trainer._X_all_scaled,
            y_encoded=trainer._y_all,
            model=trainer.best_model,
            label_encoder=trainer.label_encoder,
            source_df=_source_df,
            audit_source_index=trainer._audit_source_index,
            feature_labels=feature_labels,
            output_dir='reports',
            run_llm=True,
        )
    """

    def __init__(
        self,
        cv_folds:   int = 5,
        top_n_llm:  int = 50,
        llm_model:  str = 'claude-sonnet-4-6',
        api_key:    Optional[str] = None,
        verbose:    bool = False,
    ):
        self.cv_folds  = cv_folds
        self.top_n_llm = top_n_llm
        self.llm_model = llm_model
        self.api_key   = api_key or os.environ.get('ANTHROPIC_API_KEY')
        self.verbose   = verbose

    # ------------------------------------------------------------------ #
    #  Stage 1 — Cleanlab                                                  #
    # ------------------------------------------------------------------ #

    def run_cleanlab(
        self,
        X_scaled:      np.ndarray,
        y_encoded:     np.ndarray,
        model,
        label_encoder,
    ) -> pd.DataFrame:
        """Score label quality using stratified k-fold cross-validation.

        Returns a DataFrame indexed by position in X_scaled, sorted by
        label_quality_score ascending (worst quality first).
        """
        if not _HAS_CLEANLAB:
            raise ImportError("cleanlab is required: pip install cleanlab")

        from sklearn.base import clone
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        import joblib

        n_classes = len(label_encoder.classes_)
        if self.verbose:
            print(
                f"[LabelAudit] Running {self.cv_folds}-fold CV for cleanlab "
                f"({len(y_encoded):,} records, {n_classes} classes)…"
            )

        # Clone and force single-threaded to avoid macOS spawn/OpenMP crash
        model_cv = clone(model)
        for attr in ('nthread', 'n_jobs', 'num_threads'):
            if hasattr(model_cv, attr):
                model_cv.set_params(**{attr: 1})

        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        with joblib.parallel_backend('sequential'):
            pred_probs = cross_val_predict(
                model_cv, X_scaled, y_encoded,
                cv=cv, method='predict_proba', n_jobs=1,
            )

        quality_scores = get_label_quality_scores(labels=y_encoded, pred_probs=pred_probs)

        issue_mask = np.zeros(len(y_encoded), dtype=bool)
        issue_idx  = find_label_issues(
            labels=y_encoded,
            pred_probs=pred_probs,
            return_indices_ranked_by='self_confidence',
            n_jobs=1,   # avoid multiprocessing.Pool spawn crash on macOS
        )
        issue_mask[issue_idx] = True

        classes = label_encoder.classes_
        df = pd.DataFrame(
            {
                'label_quality_score': quality_scores,
                'is_label_issue':      issue_mask,
                'true_label':          classes[y_encoded],
                'model_pred':          classes[np.argmax(pred_probs, axis=1)],
                'model_pred_prob':     pred_probs.max(axis=1).round(4),
            },
            index=pd.RangeIndex(len(y_encoded), name='position'),
        )
        df = df.sort_values('label_quality_score')

        n_issues = int(issue_mask.sum())
        print(
            f"[LabelAudit] Cleanlab flagged {n_issues:,} / {len(y_encoded):,} records "
            f"({100 * n_issues / len(y_encoded):.1f}%) as potential label issues."
        )
        return df

    # ------------------------------------------------------------------ #
    #  Stage 2 — LLM review                                               #
    # ------------------------------------------------------------------ #

    def _get_row(
        self,
        position:          int,
        source_df:         pd.DataFrame,
        audit_source_index: list,
    ) -> pd.Series:
        """Return the source_df row corresponding to a cleanlab position index."""
        try:
            src_key = audit_source_index[position]
            if src_key in source_df.index:
                return source_df.loc[src_key]
            # Fall back to positional lookup if index key is an integer position
            return source_df.iloc[int(src_key)]
        except Exception:
            return pd.Series(dtype=object)

    def _build_context(
        self,
        row:            pd.Series,
        feature_labels: dict,
    ) -> dict:
        """Extract narrative, positive symptoms, and demographics from one row."""
        # Narrative text
        narrative_parts = [
            str(row.get(c, '')).strip()
            for c in NARRATIVE_COLS
            if str(row.get(c, '')).strip().lower() not in ('nan', 'none', 'skipped', '')
        ]
        narrative = ' '.join(narrative_parts) or '(no narrative text)'

        # Positive yes/no symptoms
        yes_items = []
        for col, val in row.items():
            if str(val).strip().lower() == 'yes':
                label = feature_labels.get(str(col).lower(), col)
                yes_items.append(f'  + {label[:60]}')
        symptoms = '\n'.join(yes_items[:25]) or '  (none recorded)'

        # Demographics
        age   = str(row.get('age_group', row.get('agegroup', '?')))
        sx    = str(row.get('id10019', '?')).lower()
        sex   = {'1': 'male', '2': 'female', 'male': 'male', 'female': 'female'}.get(sx, sx)
        demographics = f'age_group={age}, sex={sex}'

        return {'narrative': narrative, 'symptoms': symptoms, 'demographics': demographics}

    def _call_llm(self, prompt: str, client) -> dict:
        """Send prompt to Claude and parse the JSON response. Returns {} on failure."""
        try:
            msg  = client.messages.create(
                model=self.llm_model,
                max_tokens=400,
                messages=[{'role': 'user', 'content': prompt}],
            )
            text = msg.content[0].text.strip()
            # Strip markdown code fences if the model wraps JSON in them
            if text.startswith('```'):
                text = text.split('\n', 1)[-1].rsplit('```', 1)[0].strip()
            return json.loads(text)
        except json.JSONDecodeError:
            return {}
        except Exception as exc:
            if self.verbose:
                print(f"[LabelAudit] API error: {exc}")
            return {}

    def run_llm_review(
        self,
        cleanlab_df:        pd.DataFrame,
        source_df:          pd.DataFrame,
        audit_source_index: list,
        feature_labels:     dict,
        top_n:              Optional[int] = None,
        delay_seconds:      float = 0.3,
    ) -> pd.DataFrame:
        """Run Claude review on the top-N cleanlab-flagged records.

        Returns a DataFrame with one row per reviewed record.
        """
        if not _HAS_ANTHROPIC:
            raise ImportError("anthropic SDK required: pip install anthropic")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY not set. Pass api_key= or export the env var."
            )

        client     = _anthropic.Anthropic(api_key=self.api_key)
        n          = top_n or self.top_n_llm
        candidates = cleanlab_df[cleanlab_df['is_label_issue']].head(n)

        print(f"[LabelAudit] Sending {len(candidates):,} records to {self.llm_model} for review…")

        rows = []
        for rank, (position, clab_row) in enumerate(candidates.iterrows(), 1):
            src_row = self._get_row(int(position), source_df, audit_source_index)
            context = self._build_context(src_row, feature_labels)
            prompt  = _REVIEW_PROMPT.format(
                true_label  = clab_row['true_label'],
                model_pred  = clab_row['model_pred'],
                confidence  = float(clab_row['model_pred_prob']),
                **context,
            )
            result = self._call_llm(prompt, client)

            rows.append({
                'position':            int(position),
                'label_quality_score': round(float(clab_row['label_quality_score']), 4),
                'true_label':          clab_row['true_label'],
                'model_pred':          clab_row['model_pred'],
                'model_pred_prob':     float(clab_row['model_pred_prob']),
                'llm_plausible':       result.get('plausible', ''),
                'llm_alternatives':    '; '.join(result.get('alternative_causes', [])),
                'llm_reasoning':       result.get('reasoning', ''),
                'narrative_snippet':   context['narrative'][:120],
            })

            if rank % 10 == 0 or rank == len(candidates):
                print(f"[LabelAudit]   {rank}/{len(candidates)} reviewed", flush=True)
            time.sleep(delay_seconds)

        df = pd.DataFrame(rows)
        if not df.empty:
            vc    = df['llm_plausible'].value_counts().to_dict()
            n_no  = vc.get('no', 0)
            n_unc = vc.get('uncertain', 0)
            print(
                f"[LabelAudit] LLM assessment: {n_no} clearly wrong, "
                f"{n_unc} uncertain, {len(df) - n_no - n_unc} plausible."
            )
        return df

    # ------------------------------------------------------------------ #
    #  Combined entry point                                                #
    # ------------------------------------------------------------------ #

    def run(
        self,
        X_scaled:           np.ndarray,
        y_encoded:          np.ndarray,
        model,
        label_encoder,
        source_df:          pd.DataFrame,
        audit_source_index: list,
        feature_labels:     dict,
        output_dir:         str = 'reports',
        run_llm:            bool = True,
        top_n_llm:          Optional[int] = None,
    ) -> tuple:
        """Run both stages and write results to output_dir.

        Returns (cleanlab_df, llm_df). llm_df is empty DataFrame if run_llm=False.
        """
        os.makedirs(output_dir, exist_ok=True)

        # ── Stage 1: Cleanlab ─────────────────────────────────────────────
        cleanlab_df   = self.run_cleanlab(X_scaled, y_encoded, model, label_encoder)
        cleanlab_path = os.path.join(output_dir, 'label_audit_cleanlab.csv')

        # Attach narrative snippet and demographics to cleanlab output for context
        snippets, demographics = [], []
        for position in cleanlab_df.index:
            row  = self._get_row(int(position), source_df, audit_source_index)
            narr = ' '.join(
                str(row.get(c, '')).strip()
                for c in NARRATIVE_COLS
                if str(row.get(c, '')).strip().lower() not in ('nan', 'none', 'skipped', '')
            )
            snippets.append(narr[:150] or '')
            age = str(row.get('age_group', row.get('agegroup', '')))
            sx  = str(row.get('id10019', '')).lower()
            sex = {'1': 'male', '2': 'female', 'male': 'male', 'female': 'female'}.get(sx, sx)
            demographics.append(f'{age} / {sex}')

        cleanlab_out = cleanlab_df.copy()
        cleanlab_out['narrative_snippet'] = snippets
        cleanlab_out['demographics']      = demographics
        cleanlab_out.to_csv(cleanlab_path, index=True, encoding='utf-8-sig')
        print(f"Cleanlab audit saved → {cleanlab_path}")

        # ── Stage 2: LLM review ───────────────────────────────────────────
        llm_df = pd.DataFrame()
        if run_llm:
            llm_df    = self.run_llm_review(
                cleanlab_df, source_df, audit_source_index, feature_labels,
                top_n=top_n_llm,
            )
            llm_path  = os.path.join(output_dir, 'label_audit_llm_review.csv')
            llm_df.to_csv(llm_path, index=False, encoding='utf-8-sig')
            print(f"LLM review saved     → {llm_path}")

        # ── Summary JSON ──────────────────────────────────────────────────
        n_flagged = int(cleanlab_df['is_label_issue'].sum())
        summary   = {
            'total_records':         int(len(y_encoded)),
            'flagged_by_cleanlab':   n_flagged,
            'flagged_pct':           round(100 * n_flagged / len(y_encoded), 1),
            'cleanlab_output':       cleanlab_path,
        }
        if not llm_df.empty:
            summary['llm_reviewed']         = len(llm_df)
            summary['llm_plausible_counts'] = llm_df['llm_plausible'].value_counts().to_dict()
            summary['llm_output']           = llm_path

        summary_path = os.path.join(output_dir, 'label_audit_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as fh:
            json.dump(summary, fh, indent=2)
        print(f"Audit summary saved  → {summary_path}")

        return cleanlab_df, llm_df


def build_feature_labels(preprocessor) -> dict:
    """Extract {col_lower → English label} from preprocessor.instrument_dictionary."""
    labels    = {}
    instr     = getattr(preprocessor, 'instrument_dictionary', None)
    if not isinstance(instr, dict):
        return labels
    survey_list = instr.get('survey', []) if 'survey' in instr else [
        q
        for v in instr.values()
        if isinstance(v, dict)
        for q in v.get('survey', [])
    ]
    for q in (survey_list if isinstance(survey_list, list) else []):
        name  = q.get('name') or q.get('normalized_name') or ''
        label = (q.get('label_en') or q.get('label') or '').strip()
        if name and label:
            labels[name.lower()] = label
    return labels
