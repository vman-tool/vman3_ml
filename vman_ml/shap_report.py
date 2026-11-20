# vman3_ml/vman_ml/shap_report.py
"""Global SHAP feature-importance figures for a freshly trained model.

Computed once after training, on a sample of the hold-out set - a separate
concern from prediction.py's CCVAPredictor._get_shap_explainer, which explains
individual predictions at inference time.
"""
import re
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


def _bar_figure(plt, mean_abs, feature_names, max_display, title, path):
    """Saves one top-`max_display`-features horizontal bar chart to `path`."""
    order = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.35 * len(order))))
    ax.barh(range(len(order)), mean_abs[order][::-1], color='#4C72B0')
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feature_names[i] for i in order][::-1], fontsize=8)
    ax.set_xlabel('Mean |SHAP value| (average impact on model output)')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _slug(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


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

    path = output_dir / 'shap_feature_importance.png'
    _bar_figure(
        plt, mean_abs, feature_names, max_display,
        title=f'Top {max_display} features by global SHAP importance\n'
              f'(sampled {len(X_sample):,} hold-out records)',
        path=path,
    )
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


def generate_stratified_shap_figures(
    model,
    X,
    feature_names,
    strata,
    output_dir,
    max_display: int = 15,
    sample_size: int = 300,
    random_state: int = 42,
    min_group_size: int = 20,
) -> dict:
    """One global-feature-importance figure per (axis, group) in `strata`.

    strata: {axis_name: {group_label: boolean_mask}}, each mask aligned to
    X's rows - e.g. {'age_group': {'Adult': mask, 'Child': mask, ...},
    'sex': {'Male': mask, 'Female': mask},
    'cause_group': {'Infectious and parasitic diseases': mask, ...}}.

    Verbal autopsy feature importance genuinely differs by who died of what:
    a neonate's prediction leans on prematurity/birth-complication fields, an
    adult's on chronic-disease history, an injury death on external-cause
    narrative - averaging all of that into one global chart (generate_shap_
    figures above) hides those differences. This is deliberately NOT one
    figure per individual cause class (31+ classes, many with single-digit
    hold-out support - too sparse for a stable SHAP estimate, same concern as
    the per-class AUROC exclusion in training.py); stratifying by the
    coarser, already-computed groupings above (age flags, sex, WHO major
    cause group) keeps each panel's sample size reasonable.

    A group with fewer than `min_group_size` members is skipped outright (too
    few hold-out records for a stable estimate) rather than plotted misleadingly.
    Returns {(axis_name, group_label): Path} for whichever groups were plotted,
    or {} under the same unavailable-dependency/failure conditions as
    generate_shap_figures.
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
    feature_names = list(feature_names)

    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        return {}

    paths = {}
    for axis_name, groups in strata.items():
        for group_label, mask in groups.items():
            mask = np.asarray(mask, dtype=bool)
            n = int(mask.sum())
            if n < min_group_size:
                continue

            X_group = X_arr[mask]
            if sample_size and len(X_group) > sample_size:
                rng = np.random.RandomState(random_state)
                idx = rng.choice(len(X_group), sample_size, replace=False)
                X_sample = X_group[idx]
            else:
                X_sample = X_group

            try:
                shap_3d = _shap_3d(explainer, X_sample)
            except Exception:
                continue

            mean_abs = np.abs(shap_3d).mean(axis=(0, 2))
            axis_title = axis_name.replace('_', ' ')
            sampled_note = f' (sampled {len(X_sample):,})' if len(X_sample) < n else ''
            path = output_dir / f'shap_{axis_name}_{_slug(group_label)}.png'
            _bar_figure(
                plt, mean_abs, feature_names, max_display,
                title=f'{group_label} — {axis_title}\n'
                      f'top {max_display} features, n={n:,} hold-out records{sampled_note}',
                path=path,
            )
            paths[(axis_name, group_label)] = path

    return paths
