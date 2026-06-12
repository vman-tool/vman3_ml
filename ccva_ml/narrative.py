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


class NarrativeEmbedder:
    """Encodes free-text WHO VA narrative fields into fixed-length float vectors.

    Narrative columns (id10476, id10477, id10479, id10436) are concatenated
    per record and passed through a multilingual sentence-transformer model.
    Records with no text get the embedding of the empty string (consistent
    sentinel), so training and prediction are always aligned.
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
        """Return a DataFrame of narr_emb_N columns aligned to df's index.

        Returns None only when sentence-transformers is not installed.
        Missing or empty narrative text is embedded as the empty-string vector
        so the representation is consistent with training records that also
        had no narratives.
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            if self.verbose:
                print("[NarrativeEmbedder] sentence-transformers not installed — skipping.")
            return None

        texts, found_cols = self._extract_texts(df)

        try:
            model = self._embedding_model
            embeddings = model.encode(
                texts.tolist(),
                show_progress_bar=False,
                batch_size=64,
                convert_to_numpy=True,
            )
        except Exception as exc:
            if self.verbose:
                print(f"[NarrativeEmbedder] Encoding failed: {exc} — skipping.")
            return None

        col_names = [f'narr_emb_{i}' for i in range(embeddings.shape[1])]
        emb_df = pd.DataFrame(embeddings.astype(np.float32), index=df.index, columns=col_names)

        if self.verbose:
            n_nonempty = texts.str.strip().ne('').sum()
            present = found_cols if found_cols else ['(none found — used empty string)']
            print(
                f"[NarrativeEmbedder] {len(df)} records, {n_nonempty} with text, "
                f"{embeddings.shape[1]} dims | cols used: {present}"
            )

        return emb_df
