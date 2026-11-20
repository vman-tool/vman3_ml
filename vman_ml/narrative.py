"""Multilingual narrative embedding for WHO VA free-text fields.

Uses a sentence-transformer model to encode narrative text from id10476,
id10477, id10479, and id10436 into fixed-length numeric feature vectors.
Handles Swahili, Yoruba, Spanish, English and 50+ other languages.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NARRATIVE_COLS = ['id10476', 'id10477', 'id10479', 'id10436']
DEFAULT_EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

try:
    from sentence_transformers import SentenceTransformer as _ST
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _embedding_dim(model) -> int:
    # sentence-transformers renamed this method; prefer the new name to avoid
    # the deprecation warning when it's available.
    getter = getattr(model, 'get_embedding_dimension', None) or model.get_sentence_embedding_dimension
    return getter()


class NarrativeEmbedder:
    """Encodes free-text WHO VA narrative fields into fixed-length float vectors.

    Narrative columns (id10476, id10477, id10479, id10436) are concatenated
    per record and passed through a multilingual sentence-transformer model.

    Records with no narrative text get an all-zero vector plus a trailing
    `narr_emb_<dim>` presence flag (0.0) — not the model's encoding of the
    empty string. A tree model can trivially learn to ignore an exact-zero
    block when the flag says "absent", whereas the old approach (encoding ''
    through the model) gave every narrative-less record the same arbitrary
    *nonzero* point in embedding space - and since narrative columns are
    entirely absent from some source datasets (e.g. all of Eswatini), that
    point was perfectly correlated with "which country this came from",
    risking the model learning "no narrative" as a country shortcut rather
    than disregarding it. Zero + an explicit flag keeps narrative data a pure
    supplement: free to help when present, structurally inert when absent.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        narrative_cols: list[str] | None = None,
        verbose: bool = False,
    ):
        self.model_name = model_name
        self.narrative_cols = narrative_cols or NARRATIVE_COLS
        self.verbose = verbose
        self._model = None

    def __getstate__(self):
        # Drop the loaded transformer when pickling — it reloads lazily from model_name.
        state = self.__dict__.copy()
        state['_model'] = None
        return state

    @property
    def _embedding_model(self):
        if self._model is None:
            if not HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required for narrative embeddings.\n"
                    "Install with: pip install sentence-transformers"
                )
            if self.verbose:
                print(f"[NarrativeEmbedder] Loading '{self.model_name}' …")
            self._model = _ST(self.model_name)
        return self._model

    def _extract_texts(self, df: pd.DataFrame) -> tuple[pd.Series, list[str]]:
        found = [c for c in self.narrative_cols if c in df.columns]
        if not found:
            return pd.Series([''] * len(df), index=df.index), []
        texts = (
            df[found]
            .fillna('')
            .astype(str)
            .apply(lambda row: ' '.join(v.strip() for v in row if v.strip()), axis=1)
        )
        return texts, found

    def embed(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Return a DataFrame of narr_emb_N columns (plus a trailing presence
        flag, see class docstring) aligned to df's index.

        Returns None only when sentence-transformers is not installed, or
        encoding raises.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            if self.verbose:
                print("[NarrativeEmbedder] sentence-transformers not installed — skipping.")
            return None

        texts, found_cols = self._extract_texts(df)
        stripped = texts.str.strip()
        has_text = stripped.ne('')

        try:
            model = self._embedding_model

            # Encode each *unique* non-empty text once. Narrative fields repeat
            # heavily - identical phrasing across records, and every narrative-
            # less record collapses to the same empty string - so deduping
            # before encode() cuts transformer calls substantially (measured:
            # 9,455 combined TZ+ES+NG rows -> 7,145 unique texts, 511 of them
            # the empty string alone - now skipped entirely, not just deduped).
            unique_texts = stripped[has_text].unique().tolist()
            if unique_texts:
                unique_vecs = model.encode(
                    unique_texts, show_progress_bar=False, batch_size=64, convert_to_numpy=True,
                )
                dim    = unique_vecs.shape[1]
                lookup = dict(zip(unique_texts, unique_vecs))
            else:
                dim    = _embedding_dim(model)
                lookup = {}

            embeddings = np.zeros((len(df), dim), dtype=np.float32)
            for row_pos, (text, present) in enumerate(zip(stripped, has_text)):
                if present:
                    embeddings[row_pos] = lookup[text]
            # Rows with no narrative text stay exactly zero (see class docstring).
        except Exception as exc:
            if self.verbose:
                print(f"[NarrativeEmbedder] Encoding failed: {exc} — skipping.")
            return None

        col_names = [f'narr_emb_{i}' for i in range(dim)]
        emb_df = pd.DataFrame(embeddings, index=df.index, columns=col_names)
        # Presence flag, appended as one more narr_emb_* column so every
        # existing narrative_dims_/narr_emb_ prefix check elsewhere (SHAP
        # narrative-contribution grouping, predict.py's zero-fill fallback)
        # picks it up automatically - it just rides along as "one more
        # narrative dimension" that happens to be a clean 0/1 flag.
        emb_df[f'narr_emb_{dim}'] = has_text.astype(np.float32).values

        if self.verbose:
            present = found_cols if found_cols else ['(none found — used empty string)']
            print(
                f"[NarrativeEmbedder] {len(df)} records, {int(has_text.sum())} with text "
                f"({len(unique_texts)} unique non-empty encoded), "
                f"{emb_df.shape[1]} dims (incl. presence flag) | cols used: {present}"
            )

        return emb_df
