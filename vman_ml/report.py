# vman3_ml/vman_ml/report.py
"""Renders the JSON reports a training run produces (training_audit_report.json,
cv_results.json, holdout_test_results.json) into one human-readable Markdown file.
"""
import json
from pathlib import Path

_SKIP_KEYS = ('accuracy', 'macro avg', 'weighted avg')


def _load(path):
    if not path.exists():
        return None
    with open(path, encoding='utf-8') as fh:
        return json.load(fh)


def _table(rows, headers):
    lines = ['| ' + ' | '.join(headers) + ' |',
              '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join('' if c is None else str(c) for c in row) + ' |')
    return '\n'.join(lines)


def build_notes(audit: dict, model_name: str) -> str:
    """One-line auto-generated training-run summary.

    Shared by train.py (embedded directly in the trained .pkl's artifacts
    ['metrics']['notes'], so the admin panel's upload form can show it
    without anyone hand-typing it - the person uploading a model has no
    basis to describe a run they didn't do) and generate_markdown_summary
    below (same text, same source data, so the two never drift apart).
    """
    tar        = (audit.get('training_audit_report')) or {}
    n_files    = len((audit.get('input_files')) or [])
    versions   = ', '.join((audit.get('instrument_versions')) or []) or 'n/a'
    rows_final = tar.get('rows_final_for_training', 'n/a')
    return (
        f"Combined model trained on {n_files} input file(s) ({versions}); "
        f"{rows_final} rows after quality filtering; best model: {model_name}."
    )


def _classification_table(report_dict, auroc_per_class=None, limit=None):
    auroc_per_class = auroc_per_class or {}
    rows = [
        (cause, f"{m['precision']:.3f}", f"{m['recall']:.3f}", f"{m['f1-score']:.3f}",
         f"{auroc_per_class[cause]:.3f}" if cause in auroc_per_class else 'n/a',
         int(m['support']))
        for cause, m in report_dict.items() if cause not in _SKIP_KEYS
    ]
    rows.sort(key=lambda r: float(r[3]))  # worst F1 first - most actionable
    if limit:
        rows = rows[:limit]
    return _table(rows, ['Cause', 'Precision', 'Recall', 'F1', 'AUROC', 'Support'])


def generate_markdown_summary(report_dir, output_name='training_summary.md'):
    """Reads the JSON reports in report_dir and writes output_name beside them.

    Any of the three JSON files may be missing (e.g. a run without a holdout
    split) - each section is skipped rather than failing.
    """
    report_dir = Path(report_dir)
    audit   = _load(report_dir / 'training_audit_report.json')
    cv      = _load(report_dir / 'cv_results.json')
    holdout = _load(report_dir / 'holdout_test_results.json')

    lines = [f"# Training Summary — {report_dir.name}", '']

    if holdout:
        comparison  = holdout.get('comparison') or {}
        model_name  = holdout.get('model')
        cv_f1_macro = (comparison.get(model_name) or {}).get('cv_f1_macro')
        n_train     = (audit or {}).get('n_training_samples')
        n_test      = (audit or {}).get('n_test_samples', holdout.get('n_records'))

        notes = build_notes(audit or {}, model_name)

        lines += [
            '## Model Upload Info',
            '',
            'For reference - the admin panel reads all of this (including '
            'Notes) directly from the uploaded .pkl\'s own embedded metadata; '
            'nothing here needs to be retyped.',
            '',
            _table([
                ('Version',          report_dir.name),
                ('Notes',            notes),
                ('Accuracy',         holdout.get('accuracy')),
                ('F1 macro',         holdout.get('f1_macro')),
                ('F1 weighted',      holdout.get('f1_weighted')),
                ('CV F1 macro',      cv_f1_macro),
                ('Training samples', n_train),
                ('Test samples',     n_test),
            ], ['Field', 'Value']),
            '',
        ]

    if audit:
        lines += [
            '## Inputs',
            f"- Input files: {', '.join(audit.get('input_files') or [])}",
            f"- Target column: `{audit.get('target_column')}`",
            f"- Instrument version(s): {', '.join(audit.get('instrument_versions') or [])}",
            f"- Feature count: {audit.get('union_feature_count')}",
            '',
        ]

        tar = audit.get('training_audit_report') or {}
        lines += [
            '## Data Pipeline',
            _table([
                ('Input rows',                     tar.get('input_rows')),
                ('After quality filter',           tar.get('rows_after_quality_filter')),
                ('After rare-cause clustering',    tar.get('rows_after_clustering')),
                ('Final rows used for training',   tar.get('rows_final_for_training')),
            ], ['Stage', 'Rows']),
            '',
            '### Rows dropped, by reason',
            _table([
                ('Missing/unknown cause label',            tar.get('missing_or_unknown_labels')),
                ('WHO causelist mapping failure',          tar.get('unmapped_who_cause_labels')),
                ('Demographic-cause implausibility',       tar.get('implausible_demographic_labels')),
                ('vman_dq pregnancy/sex contradiction',    tar.get('vman_dq_pregnancy_male_contradictions')),
                ('vman_dq low ICI (< 70%)',                tar.get('vman_dq_low_ici_rows')),
            ], ['Reason', 'Rows dropped']),
            '',
        ]

        ici = tar.get('vman_dq_ici_summary') or {}
        if ici:
            lines += [
                f"vman_dq ICI: {ici.get('n_scored')} rows scored, mean ICI "
                f"{ici.get('mean_ici', 0):.2f}%, rules applied: "
                f"{', '.join(ici.get('rules_applied') or [])}",
                '',
            ]

        n_clustered = tar.get('clustered_rare_labels')
        if n_clustered:
            n_groups = len({c['cluster'] for c in (tar.get('clustered_causes') or [])})
            lines += [
                f"**Rare-cause clustering:** {n_clustered} causes grouped into "
                f"{n_groups} cluster group(s) (causes below `--min_vc` records).",
                '',
            ]

    if cv and cv.get('comparison'):
        comp_rows = sorted(
            ((name, m['cv_f1_macro'], m['test_accuracy'], m['test_f1_macro'],
              m.get('fit_seconds', 'n/a'))
             for name, m in cv['comparison'].items()),
            key=lambda r: -r[3],
        )
        lines += [
            '## Model Comparison (internal validation split)',
            _table(comp_rows, ['Model', 'CV Macro-F1', 'Val Accuracy', 'Val Macro-F1', 'Fit Time (s)']),
            '',
            f"**Selected model:** {cv.get('model')}",
            '',
        ]

    if audit and (audit.get('training_seconds') is not None or audit.get('total_run_seconds') is not None):
        def _fmt(seconds):
            if seconds is None:
                return 'n/a'
            mins, secs = divmod(int(seconds), 60)
            return f"{mins}m{secs:02d}s"

        lines += [
            '## Timing',
            _table([
                ('Hyperparameter search (model fitting only)', _fmt(audit.get('training_seconds'))),
                ('Total run (data load, preprocessing, embeddings, search, eval)', _fmt(audit.get('total_run_seconds'))),
            ], ['Phase', 'Duration']),
            '',
        ]

    if cv:
        lines += [
            '## Validation Results',
            _table([
                ('Records',      cv.get('n_records')),
                ('Accuracy',     cv.get('accuracy')),
                ('Macro-F1',     cv.get('f1_macro')),
                ('Weighted-F1',  cv.get('f1_weighted')),
                ('Macro-AUROC',  cv.get('auroc_macro')),
            ], ['Metric', 'Value']),
            '',
        ]

    if holdout:
        lines += [
            '## Hold-out Test Results (20%, never seen during training)',
            _table([
                ('Records',      holdout.get('n_records')),
                ('Accuracy',     holdout.get('accuracy')),
                ('Macro-F1',     holdout.get('f1_macro')),
                ('Weighted-F1',  holdout.get('f1_weighted')),
                ('Macro-AUROC',  holdout.get('auroc_macro')),
            ], ['Metric', 'Value']),
            '',
        ]

        cr = holdout.get('classification_report')
        if cr:
            auroc_per_class = holdout.get('auroc_per_class') or {}
            lines += [
                '### Per-cause performance on hold-out (worst 15 by F1)',
                _classification_table(cr, auroc_per_class, limit=15),
                '',
                '<details><summary>All causes</summary>\n',
                _classification_table(cr, auroc_per_class),
                '\n</details>',
                '',
            ]

    shap_figures = [
        ('Global feature importance',           'shap_feature_importance.png'),
        ('Narrative embeddings vs. other features', 'shap_narrative_contribution.png'),
    ]
    present = [(title, name) for title, name in shap_figures if (report_dir / name).exists()]
    if present:
        lines += ['## SHAP Feature Importance', '']
        for title, name in present:
            lines += [f'**{title}**', '', f'![{title}]({name})', '']

    stratified = (audit or {}).get('stratified_shap_figures')
    if stratified:
        lines += [
            '## Stratified SHAP Feature Importance', '',
            'Feature importance split by sub-group — verbal autopsy predictors '
            'genuinely differ by who died of what (e.g. neonates vs. adults, '
            'or infectious vs. external-cause deaths). Collapsed by default; '
            'expand an axis to see its per-group panels.', '',
        ]
        for axis, groups in stratified.items():
            axis_title = axis.replace('_', ' ').title()
            present_groups = [(label, name) for label, name in groups.items()
                               if (report_dir / name).exists()]
            if not present_groups:
                continue
            lines += [f'<details><summary>{axis_title}</summary>', '']
            for label, name in present_groups:
                lines += [f'**{label}**', '', f'![{label} — {axis_title}]({name})', '']
            lines += ['</details>', '']

    output_path = report_dir / output_name
    output_path.write_text('\n'.join(lines), encoding='utf-8')
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Render an existing training run\'s JSON reports (training_audit_report.json, '
                     'cv_results.json, holdout_test_results.json) into training_summary.md, '
                     'without re-running train.py.'
    )
    parser.add_argument('report_dir', help='Directory containing the JSON reports, e.g. reports/v1.2.0')
    args = parser.parse_args()
    path = generate_markdown_summary(args.report_dir)
    print(f"Written: {path}")
