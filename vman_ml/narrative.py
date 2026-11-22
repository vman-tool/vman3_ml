"""Multilingual narrative embedding for WHO VA free-text fields.

Uses a sentence-transformer model to encode narrative text from id10476,
id10477, id10479, and id10436 into fixed-length numeric feature vectors.
Handles Swahili, Yoruba, Spanish, English and 50+ other languages.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp

import numpy as np
import pandas as pd

NARRATIVE_COLS = ['id10476', 'id10477', 'id10479', 'id10436']
DEFAULT_EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'

# Availability check only (importlib.util.find_spec has no import side
# effects) - the real `import sentence_transformers` is deferred to
# _embedding_model below. sentence-transformers pulls in torch, which loads
# its own OpenMP runtime as a side effect of the import itself, not of
# actually using it - and that collides with XGBoost's OpenMP thread pool at
# nthread>1 (confirmed: merely `import torch` with nothing else running is
# enough to crash a later multi-threaded XGBoost .fit()). Since this module
# is imported (via processing.py/label_audit.py) in every train.py run
# regardless of whether narrative embedding ever runs, eagerly importing
# sentence_transformers here would poison every process that touches vman_ml
# for any later multi-threaded XGBoost use. See embed_isolated() below for
# how the actual embedding call avoids this for good.
HAS_SENTENCE_TRANSFORMERS = importlib.util.find_spec('sentence_transformers') is not None


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
            # Deliberately deferred to here, not module level - see the
            # HAS_SENTENCE_TRANSFORMERS comment above for why.
            from sentence_transformers import SentenceTransformer as _ST
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

    def embed_isolated(self, df: pd.DataFrame, timeout: float = 300.0) -> pd.DataFrame | None:
        """Same contract as embed(), but runs the actual encoding in a freshly
        spawned child process so sentence-transformers/torch never load into
        this process.

        Why: merely *importing* torch (not even using it) loads its bundled
        OpenMP runtime as an import side effect, and that collides with
        XGBoost's own OpenMP thread pool at nthread>1 - confirmed directly:
        plain `import torch`, nothing else, was enough to crash a later
        multi-threaded XGBoost .fit() with "OMP: Error #179: Function
        pthread_mutex_init failed". Running the embedding entirely inside a
        short-lived child process (spawn, not fork - fork on macOS inherits
        the parent's already-initialised state and is the documented unsafe
        choice here; spawn is Python's own default start method on macOS)
        means this process is never touched by torch at all, so it stays
        free to use XGBoost with multiple threads afterwards.

        Falls back to returning None (same failure contract as embed()) if
        sentence-transformers isn't installed, the child raises, or it
        doesn't finish within `timeout` seconds.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            if self.verbose:
                print("[NarrativeEmbedder] sentence-transformers not installed — skipping.")
            return None

        # Only the columns embed() actually reads cross the process boundary -
        # no need to pickle/ship the full (possibly 500+ column) training frame.
        found = [c for c in self.narrative_cols if c in df.columns]
        narrative_df = df[found].copy() if found else pd.DataFrame(index=df.index)

        ctx   = mp.get_context('spawn')
        queue = ctx.Queue()
        proc  = ctx.Process(
            target=_embed_worker,
            args=(self.model_name, self.narrative_cols, self.verbose, narrative_df, queue),
        )
        proc.start()
        try:
            status, payload = queue.get(timeout=timeout)
        except Exception:
            proc.terminate()
            proc.join()
            if self.verbose:
                print(f"[NarrativeEmbedder] Isolated embedding timed out after {timeout:.0f}s — skipping.")
            return None
        proc.join()

        if status != 'ok' or payload is None:
            if self.verbose:
                print(f"[NarrativeEmbedder] Isolated embedding failed: {payload} — skipping.")
            return None

        return payload


def _embed_worker(model_name, narrative_cols, verbose, narrative_df, queue):
    """Runs inside the child process spawned by embed_isolated() - module-level
    so it's picklable/importable by name under the 'spawn' start method.
    Reconstructs a fresh NarrativeEmbedder here (not passed from the parent)
    so nothing torch-related ever needs to cross the process boundary.
    """
    try:
        embedder = NarrativeEmbedder(model_name=model_name, narrative_cols=narrative_cols, verbose=verbose)
        result = embedder.embed(narrative_df)
        queue.put(('ok', result))
    except Exception as exc:
        queue.put(('error', str(exc)))
