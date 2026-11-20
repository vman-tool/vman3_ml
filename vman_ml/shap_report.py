# vman3_ml/vman_ml/shap_report.py
"""Global SHAP feature-importance figures for a freshly trained model.

Computed once after training, on a sample of the hold-out set - a separate
concern from prediction.py's CCVAPredictor._get_shap_explainer, which explains
individual predictions at inference time.
"""
from pathlib import Path

import numpy as np


def _shap_3d(explainer, X):
    """Normalise SHAP output to a (N, F, C) array regardless of shap/model version."""
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        return np.stack(raw, axis=-1)
    if raw.ndim == 2:
        return raw[:, :, np.newaxis]
    return raw


def generate_shap_figures(
    model,
    X,
    feature_names,
    output_dir,
    max_display: int = 20,
    sample_size: int = 500,
    random_state: int = 42,
) -> dict:
    """Saves global SHAP figures to output_dir and returns {name: Path}.

    Two figures:
      shap_feature_importance.png   - top `max_display` individual features
      shap_narrative_contribution.png - narrative embedding block vs. everything
                                         else, summed (only when any feature
                                         name starts with 'narr_emb_')

    Returns {} (no figures, nothing written) if shap/matplotlib aren't
    installed or SHAP computation fails - this is a reporting extra, never
    something training should fail over.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_arr = np.asarray(X)
    if sample_size and len(X_arr) > sample_size:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_arr), sample_size, replace=False)
        X_sample = X_arr[idx]
    else:
        X_sample = X_arr

    try:
        explainer = shap.TreeExplainer(model)
        shap_3d   = _shap_3d(explainer, X_sample)  # (N, F, C)
    except Exception:
        return {}

    # Mean absolute SHAP value per feature, averaged over samples and classes -
    # the standard multiclass-safe "global feature importance" summary.
    mean_abs = np.abs(shap_3d).mean(axis=(0, 2))
    feature_names = list(feature_names)

    paths = {}

    order = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    ax.barh(range(len(order)), mean_abs[order][::-1], color='#4C72B0')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel('Mean |SHAP value| (average impact on model output)')
    ax.set_title(f'Top {len(order)} features by global SHAP importance\n'
                 f'(sampled {len(X_sample):,} hold-out records)')
    fig.tight_layout()
    path = output_dir / 'shap_feature_importance.png'
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths['feature_importance'] = path

    is_narrative = np.array([n.startswith('narr_emb_') for n in feature_names])
    if is_narrative.any():
        narrative_total = float(mean_abs[is_narrative].sum())
        other_total     = float(mean_abs[~is_narrative].sum())
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(
            ['Narrative\n(embedding block)', 'All other\nfeatures'],
            [narrative_total, other_total],
            color=['#DD8452', '#4C72B0'],
        )
        ax.set_ylabel('Summed mean |SHAP value|')
        ax.set_title('Narrative embeddings vs. tabular features\n(combined SHAP contribution)')
        for bar in bars:
            ax.annotate(f'{bar.get_height():.3f}', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)
        fig.tight_layout()
        path = output_dir / 'shap_narrative_contribution.png'
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths['narrative_contribution'] = path

    return paths
