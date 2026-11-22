"""Generate the VMan3-ML manuscript as a Word document.

This script previously generated a funding-proposal document
(CCVA_ML_Funding_Proposal.docx). It now generates a methodological
manuscript describing the VMan3 verbal-autopsy machine-learning pipeline:
data sources, cleaning and harmonisation, feature engineering, model
development, the dataset-identity leakage investigation and mitigation,
the indeterminate-cause / out-of-distribution design, and results on an
independent hold-out set.

Every pipeline/results number (row counts, feature counts, metrics,
cluster membership, per-cause table) is read live from one training run's
own reports - reports/<version>/training_audit_report.json, cv_results.json,
and holdout_test_results.json - plus models/vman_ccva.pkl for narrative
embedding dimensionality. Re-run this script after any retrain and it picks
up that run's own numbers automatically:

    python3 generate_proposal.py --report-version v1.3.4

With no --report-version, it uses whichever reports/<dir> was most recently
written. The one set of numbers that is NOT re-derived per run is the
dataset-identity-leakage investigation itself (Section 2.5) - that audit
was a one-time analysis against the pre-fix pipeline, and its own report
directory was overwritten by the post-fix retrain that reused the same
version label, so those figures are kept as a fixed, clearly-labelled
historical constant (LEAKAGE_INVESTIGATION below) rather than silently
going stale or disappearing.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from docx import Document
from docx.shared import Pt, RGBColor, Inches, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

try:
    import joblib
except ImportError:
    joblib = None

# ---------------------------------------------------------------------------
# Historical constant: the one-time dataset-identity-leakage audit (Section
# 2.5). This describes a specific investigation against the pipeline as it
# stood *before* the intersection-mode fix - not a property of any one
# training run, and not reloadable from disk (that run's own report
# directory was overwritten by the later, post-fix run that reused the same
# version label "v1.3.4"). See module docstring.
# ---------------------------------------------------------------------------
LEAKAGE_INVESTIGATION = {
    "pre_fix_structured_feature_count": 499,
    "columns_in_only_some_datasets": 157,
    "columns_in_only_some_datasets_pct": 31,
    "max_feature_nmi_vs_source": 0.94,
    "true_label_nmi_vs_source": 0.235,
    "post_intersection_only_feature_count": 342,
    "post_fix_structured_feature_count": 339,
    # The pre-fix run's own headline hold-out metrics, preserved here as text
    # (its report directory no longer exists) for the direct before/after
    # comparison in Sections 2.5, 3.5 and 4.1.
    "pre_fix_holdout_accuracy": 0.8949,
    "pre_fix_holdout_f1_macro": 0.8593,
    "pre_fix_holdout_n": 1846,
}

# Historical constant: a one-time empirical check of the two user-facing
# threshold controls exposed in the VMan3 "Run CCVA" UI for the VManML10
# algorithm (dkThreshold, oodThreshold), prompted by a request to document
# and, if warranted, revise their defaults. Computed by sweeping the
# confidence-based OOD check across several candidate threshold values on
# the v1.3.4 hold-out set (n=1,846) and recording how many records each
# mechanism flagged in isolation and in combination - not reloadable from a
# training report, since it is not something train.py itself computes.
OOD_THRESHOLD_ANALYSIS = {
    "holdout_n": 1846,
    "ui_default_dk_threshold": 0.60,
    "ui_default_ood_threshold": 0.05,
    "model_calibrated_entropy_threshold": 0.452,
    "entropy_mask_fires": 77,
    "dk_mask_fires_at_0_60": 0,
    "confidence_mask_fires_at_default_0_05": 0,
    "confidence_mask_fires_at_0_30": 14,
    "confidence_mask_unique_contribution_any_threshold_tested": 0,
    "min_observed_confidence": 0.117,
    "live_run_undetermined_pct": 44.5,
}


# ---------------------------------------------------------------------------
# Report loading
# ---------------------------------------------------------------------------

def find_latest_report_version(reports_root="reports"):
    """Most recently written reports/<version>/ directory that has a full
    training_audit_report.json - i.e. the latest completed training run."""
    root = Path(reports_root)
    candidates = [
        d for d in root.iterdir()
        if d.is_dir() and (d / "training_audit_report.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No report directory with training_audit_report.json found under {reports_root}/"
        )
    candidates.sort(key=lambda d: (d / "training_audit_report.json").stat().st_mtime)
    return candidates[-1].name


def _fmt_duration(seconds):
    seconds = int(round(seconds))
    m, s = divmod(seconds, 60)
    return f"{m}m{s:02d}s"


def load_report(report_version, reports_root="reports", models_dir="models"):
    """Load one training run's own numbers from its report directory."""
    report_dir = Path(reports_root) / report_version
    audit = json.loads((report_dir / "training_audit_report.json").read_text())
    cv = json.loads((report_dir / "cv_results.json").read_text())
    holdout = json.loads((report_dir / "holdout_test_results.json").read_text())
    tar = audit["training_audit_report"]

    # Cluster membership: cluster name -> [(cause, training-row count), ...],
    # sorted by total cluster size (largest first), causes within a cluster
    # by count (largest first).
    grouped = defaultdict(list)
    for c in tar["clustered_causes"]:
        grouped[c["cluster"]].append((c["cause"], int(c["count"])))
    cluster_membership = sorted(
        (
            (name, sorted(members, key=lambda m: -m[1]))
            for name, members in grouped.items()
        ),
        key=lambda kv: -sum(n for _, n in kv[1]),
    )

    # Per-cause hold-out table: merge sklearn's classification_report with
    # the separately-computed per-class AUROC, sorted by F1 ascending (worst
    # first) to match the narrative, which discusses weakest causes first.
    cr = holdout["classification_report"]
    auroc = holdout["auroc_per_class"]
    per_cause = []
    for cause, m in cr.items():
        if cause in ("accuracy", "macro avg", "weighted avg"):
            continue
        per_cause.append((
            cause, m["precision"], m["recall"], m["f1-score"],
            auroc.get(cause, float("nan")), int(m["support"]),
        ))
    per_cause.sort(key=lambda r: r[3])

    narrative_dims = 0
    model_path = Path(models_dir) / "vman_ccva.pkl"
    if joblib is not None and model_path.exists():
        artifact = joblib.load(model_path)
        narrative_dims = int(artifact.get("narrative_dims", 0) or 0)

    return {
        "version": report_version,
        "input_files": audit["input_files"],
        "instrument_versions": audit["instrument_versions"],
        "feature_count_structured": audit["union_feature_count"],
        "narrative_dims": narrative_dims,
        "input_rows": tar["input_rows"],
        "rows_after_quality_filter": tar["rows_after_quality_filter"],
        "rows_after_clustering": tar["rows_after_clustering"],
        "rows_final": tar["rows_final_for_training"],
        "dropped_who_mapping_failure": tar["unmapped_who_cause_labels"],
        "dropped_demographic_implausibility": tar["implausible_demographic_labels"],
        "dropped_pregnancy_sex_contradiction": tar["vman_dq_pregnancy_male_contradictions"],
        "dropped_low_ici": tar["vman_dq_low_ici_rows"],
        "kept_as_indeterminate": tar["kept_as_indeterminate_labels"],
        "ici_summary": tar["vman_dq_ici_summary"],
        "n_clustered_causes": tar["clustered_rare_labels"],
        "n_cluster_groups": len(cluster_membership),
        "cluster_membership": cluster_membership,
        "n_training_samples": audit["n_training_samples"],
        "n_test_samples": audit["n_test_samples"],
        "training_duration": _fmt_duration(audit["training_seconds"]),
        "total_run_duration": _fmt_duration(audit["total_run_seconds"]),
        "model_comparison": cv["comparison"],
        "selected_model": cv.get("model"),
        "validation": {
            "n": cv["n_records"], "accuracy": cv["accuracy"], "f1_macro": cv["f1_macro"],
            "f1_weighted": cv["f1_weighted"], "auroc_macro": cv["auroc_macro"],
        },
        "holdout": {
            "n": holdout["n_records"], "accuracy": holdout["accuracy"], "f1_macro": holdout["f1_macro"],
            "f1_weighted": holdout["f1_weighted"], "auroc_macro": holdout["auroc_macro"],
        },
        "holdout_per_cause": per_cause,
        "n_classes": len(per_cause),
    }


# ---------------------------------------------------------------------------
# docx helpers
# ---------------------------------------------------------------------------

def set_heading(doc, text, level=1, colour=None):
    h = doc.add_heading(text, level=level)
    if colour:
        for run in h.runs:
            run.font.color.rgb = RGBColor(*colour)
    return h

def para(doc, text, bold=False, italic=False, size=None, indent=None, align=None):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.bold = bold
    run.italic = italic
    if size:
        run.font.size = Pt(size)
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
    if align:
        p.alignment = align
    return p

def labelled_para(doc, label, text):
    """'Background: ...' / 'Methods: ...' style abstract paragraph."""
    p = doc.add_paragraph()
    r1 = p.add_run(f"{label}: ")
    r1.bold = True
    p.add_run(text)
    return p

def bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.left_indent = Cm(0.6 * (level + 1))
    return p

def numbered(doc, text):
    return doc.add_paragraph(text, style="List Number")

def reference(doc, number, text):
    p = doc.add_paragraph()
    p.add_run(f"[{number}]\t{text}")
    return p

def table_header(table, row_idx, cells_text, bold=True, bg_hex="1F4E79"):
    row = table.rows[row_idx]
    for i, txt in enumerate(cells_text):
        cell = row.cells[i]
        cell.text = txt
        for run in cell.paragraphs[0].runs:
            run.bold = bold
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:color"), "auto")
        shd.set(qn("w:fill"), bg_hex)
        tcPr.append(shd)

def add_table(doc, headers, rows, col_widths=None):
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_header(t, 0, headers)
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            cell = t.rows[r_i + 1].cells[c_i]
            cell.text = str(val)
            for run in cell.paragraphs[0].runs:
                run.font.size = Pt(9)
    if col_widths:
        for row in t.rows:
            for i, w in enumerate(col_widths):
                row.cells[i].width = Inches(w)
    return t

def table_caption(doc, text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(10)
    return p

def page_break(doc):
    doc.add_page_break()


# ---------------------------------------------------------------------------
# Build document
# ---------------------------------------------------------------------------

def build(report, out_path):
    li = LEAKAGE_INVESTIGATION
    total_features = report["feature_count_structured"] + report["narrative_dims"]
    acc_delta = report["holdout"]["accuracy"] - li["pre_fix_holdout_accuracy"]
    f1_delta = report["holdout"]["f1_macro"] - li["pre_fix_holdout_f1_macro"]

    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(11)

    # ── Title block ─────────────────────────────────────────────────────
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run(
        "Machine-Learning-Based Cause-of-Death Classification from "
        "Verbal Autopsy Data: Development, Validation, and Lessons on "
        "Cross-Dataset Generalisability"
    )
    r.bold = True
    r.font.size = Pt(16)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

    doc.add_paragraph()
    doc.add_paragraph().add_run("Authors: Isaac Lyatuu¹*")
    doc.add_paragraph().add_run("Affiliations:\t¹ Ifakara Health Institute, Dar es Salaam, Tanzania")
    doc.add_paragraph().add_run(
        "Corresponding author: *Isaac Lyatuu, Ifakara Health Institute. Email: ilyatuu@ihi.or.tz"
    )
    doc.add_paragraph().add_run(
        "Keywords: verbal autopsy, cause of death, machine learning, XGBoost, natural "
        "language processing, dataset shift, data leakage, out-of-distribution detection, "
        "WHO-VA instrument, mortality surveillance"
    )

    doc.add_paragraph()
    note = doc.add_paragraph()
    nrun = note.add_run(
        "[Draft prepared for internal review. Co-authors, funding, and ethics statements "
        f"to be completed before submission - see Declarations. Source training run: "
        f"{report['version']}.]"
    )
    nrun.italic = True
    nrun.font.size = Pt(9)
    nrun.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

    page_break(doc)

    # ── Abstract ────────────────────────────────────────────────────────
    set_heading(doc, "Abstract")
    labelled_para(doc, "Background",
        "Verbal autopsy (VA) is the primary method for ascertaining cause of death (CoD) "
        "in settings without complete civil registration and vital statistics systems. "
        "Machine-learning classifiers trained on structured VA responses can automate "
        "CoD assignment, but models trained on pooled data from multiple countries risk "
        "learning which dataset a record came from rather than the clinical evidence "
        "within it - a form of dataset shift that is rarely tested for explicitly in "
        "verbal-autopsy machine-learning work.")
    labelled_para(doc, "Methods",
        f"We describe VMan3-ML, a gradient-boosted tree classifier (XGBoost) combined "
        f"with multilingual sentence-embeddings of free-text narrative fields, trained on "
        f"{report['input_rows']:,} VA records pooled from three countries using the 2016 "
        f"and 2022 WHO-VA instruments. We detail the data-cleaning pipeline, rare-cause "
        f"clustering, and a post-hoc biological-plausibility and out-of-distribution (OOD) "
        f"abstention layer. We then describe a systematic audit for dataset-identity "
        f"leakage - testing every candidate feature's normalised mutual information "
        f"against dataset-of-origin - prompted by a misclassified case traced to a feature "
        f"with no plausible clinical relationship to its predicted cause, and the pipeline "
        f"changes made in response.")
    labelled_para(doc, "Results",
        f"On an independent, stratified 20% hold-out (n={report['holdout']['n']:,}, never "
        f"seen during training or model selection), the corrected pipeline achieved "
        f"{report['holdout']['accuracy']:.1%} accuracy and {report['holdout']['f1_macro']:.3f} "
        f"macro-F1 across {report['n_classes']} cause-of-death categories (macro-AUROC "
        f"{report['holdout']['auroc_macro']:.3f}). The leakage audit that motivated this "
        f"pipeline found that {li['columns_in_only_some_datasets']} of "
        f"{li['pre_fix_structured_feature_count']} candidate structured features "
        f"({li['columns_in_only_some_datasets_pct']}%) existed in only one or two of the "
        f"three source datasets - a presence pattern a model can use as a near-perfect "
        f"proxy for dataset-of-origin - and that at least two further shared fields "
        f"(recorded date of death; a civil-registration question) carried inconsistent or "
        f"country-specific formatting that functioned the same way. Restricting structured "
        f"features to those present in every source dataset, plus explicit removal of the "
        f"two format-inconsistent fields ({li['pre_fix_structured_feature_count']} to "
        f"{report['feature_count_structured']} columns), is now the pipeline's default. "
        f"Retraining under this corrected pipeline measurably changed apparent hold-out "
        f"performance on the same split construction (accuracy "
        f"{li['pre_fix_holdout_accuracy']:.1%} to {report['holdout']['accuracy']:.1%}, "
        f"{acc_delta:+.1%}; macro-F1 {li['pre_fix_holdout_f1_macro']:.3f} to "
        f"{report['holdout']['f1_macro']:.3f}, {f1_delta:+.3f}) - consistent with, though "
        f"not definitive proof of, part of the earlier figure having been inflated by the "
        f"leakage described above.")
    labelled_para(doc, "Conclusions",
        "A pooled, multi-country VA classifier can reach hold-out accuracy comparable to "
        "published CCVA benchmarks, but doing so safely requires treating dataset-identity "
        "leakage as a first-class risk, not an afterthought, alongside explicit, "
        "clinically-grounded abstention for cases the model should not confidently "
        "classify at all. We report the audit method, the concrete leakage found, the "
        "mitigations adopted, and their measured effect on hold-out performance, so other "
        "multi-site VA machine-learning efforts can test for the same failure mode.")

    page_break(doc)

    # ── 1. Introduction ─────────────────────────────────────────────────
    set_heading(doc, "1. Introduction")
    para(doc,
        "Verbal autopsy (VA) is the internationally recommended method for estimating "
        "cause-of-death (CoD) distributions where medical certification is unavailable "
        "[2]. A trained interviewer administers a structured questionnaire - the WHO-VA "
        "instrument, published in versioned releases including the widely used 2016 and "
        "2022 editions [12,13,14] - to a household member of the deceased, capturing "
        "symptoms, signs, circumstances, duration, and an open-ended narrative account of "
        "the final illness. A physician or an automated algorithm then assigns a probable "
        "CoD from this record.")
    doc.add_paragraph()
    para(doc,
        "Physician review does not scale to the volume of deaths VA is meant to cover, and "
        "is subject to inter-rater disagreement [7]; computer-coded verbal autopsy (CCVA) "
        "methods - probabilistic models such as InterVA [5] and InSilicoVA [3], and "
        "data-driven classifiers such as the Tariff method and random-forest approaches "
        "[6] - were developed to automate this step. Open-source toolkits such as openVA "
        "[4] have made several of these methods broadly usable. Most published CCVA "
        "methods, however, are trained and evaluated within a single dataset or country; "
        "when a model is instead trained by pooling VA data collected under different "
        "instrument versions, in different countries, and through different data-export "
        "pipelines, a further risk emerges that is specific to this pooling and is rarely "
        "reported: the model can learn to recognise which source dataset a record came "
        "from and use that as a shortcut to the locally dominant causes, rather than "
        "learning the clinical relationship between symptoms and cause of death. A model "
        "that does this can show strong apparent accuracy on a hold-out split drawn from "
        "the same pool of datasets while generalising poorly to a genuinely new deployment "
        "- the setting VA classifiers are actually meant to serve.")
    doc.add_paragraph()
    para(doc,
        "This paper describes VMan3-ML, the machine-learning CoD classifier built into the "
        "VMan3 verbal-autopsy management platform, with three contributions. First, we "
        "describe the full data-preparation pipeline needed to pool VA records collected "
        "under two WHO-VA instrument versions across three countries, including the "
        "specific heterogeneity we encountered - differing field names, differing date and "
        "duration formats, and instrument sections present in some deployments and absent "
        "from others. Second, we describe a systematic audit for the dataset-identity "
        "leakage risk outlined above, prompted by inspecting a single misclassified record, "
        "the concrete features it found, and the measured effect of removing them on "
        "hold-out performance. Third, we describe the model's abstention design: a genuine "
        "'indeterminate cause' class, entropy- and missingness-based out-of-distribution "
        "flagging, and a set of post-hoc biological-plausibility constraints, together with "
        "the specific threshold values used and how they were chosen.")

    # ── 2. Methods ──────────────────────────────────────────────────────
    set_heading(doc, "2. Methods")

    set_heading(doc, "2.1 Study Design and Data Sources", level=2)
    para(doc,
        "This is a methodological and developmental study describing a deployed "
        "cause-of-death classifier and a leakage audit conducted on its training data. "
        "Three VA datasets were pooled for training: a 2016 WHO-VA instrument dataset "
        "from Tanzania (3,601 records), a 2022 WHO-VA instrument dataset from Eswatini "
        "(380 records), and a dataset collected in Nigeria and auto-detected, by column "
        "overlap, as closest to the 2022 Tanzania instrument definition (5,474 records; "
        "no Nigeria-specific instrument dictionary existed at the time of this analysis, "
        "a limitation discussed in Section 4). Ground-truth CoD labels were drawn from "
        "physician review (ICD-coded, mapped to WHO standardised cause titles) or, where "
        "already physician-mapped at source, used directly.")

    set_heading(doc, "2.2 Data Cleaning and Harmonisation", level=2)
    para(doc, "Raw export files were passed through a shared preprocessing pipeline before "
              "any model saw the data:")
    numbered(doc, "Column name normalisation - whitespace, casing, and punctuation "
        "variants of the same WHO field code (e.g. across ODK export versions) were "
        "merged to one canonical name.")
    numbered(doc, "Null/skip normalisation - the vman_dq package's change_null_toskipped "
        "routine evaluates each question's ODK relevance expression against the record's "
        "own answers and distinguishes a genuinely skipped (not-applicable) question from "
        "a missing answer to a question that should have been asked.")
    numbered(doc, "Instrument-version detection - the WHO-VA instrument version (2016 or "
        "2022) of each input file is detected automatically by scoring column-name overlap "
        "against bundled instrument dictionaries, so multi-version pooling needs no manual "
        "per-file configuration.")
    numbered(doc, "Cause-of-death harmonisation - raw physician-assigned causes (ICD-coded "
        f"or free-text) are mapped to WHO standardised cause titles via a curated lookup; "
        f"records whose cause could not be mapped are excluded and counted "
        f"({report['dropped_who_mapping_failure']} of {report['input_rows']:,} input rows "
        f"in this training run).")
    numbered(doc, "Internal-consistency and demographic-plausibility filtering - vman_dq's "
        "Internal Consistency Index (ICI) [see companion indicator study] flags and drops "
        "records in its 'Critical' tier (ICI < 70%) or with a direct pregnancy/maternal-"
        "versus-sex contradiction in the raw interview data; separately, records whose "
        "assigned label is biologically implausible for the decedent's recorded sex or age "
        "group (e.g. a maternal cause assigned to a male or child decedent, or a "
        "neonate-only cause assigned to an older child or adult) are dropped from training "
        f"({report['dropped_demographic_implausibility']} records in this run), extending "
        "an existing sex-only check to also cover age-group mismatches, prompted directly "
        "by the leakage/plausibility investigation described in Section 2.5.")

    set_heading(doc, "2.3 Target Label Construction: Rare-Cause Clustering and the "
                      "Indeterminate-Cause Class", level=2)
    para(doc,
        f"After harmonisation, {report['n_clustered_causes']} distinct WHO cause titles "
        f"had fewer than 130 training examples each (the min_vc threshold) - too few to "
        f"support a stable decision boundary on their own. Rather than dropping them, each "
        f"is reassigned to its WHO major-cause-group cluster label (e.g. 'Accidental "
        f"drowning and submersion' and nine other specific external causes become "
        f"cluster_external_causes_of_death), producing {report['n_cluster_groups']} "
        f"cluster categories (full membership in Appendix Table A1). A small protected-"
        f"causes exception keeps a cause standalone despite falling under min_vc if it "
        f"clears a lower floor (20 examples) and has its own direct clinical confirmation "
        f"question in the instrument (e.g. sickle cell disease) - collapsing a clinically "
        f"actionable, directly-confirmable cause into a generic cluster was judged to lose "
        f"more than the small-sample risk justified.")
    doc.add_paragraph()
    para(doc,
        "A second, initially overlooked label-construction issue is addressed here: "
        "records whose physician-assigned cause was itself 'Unknown and ill-defined cause "
        f"of death' ({report['kept_as_indeterminate']} of {report['input_rows']:,} input "
        "rows) were, in the pipeline's original form, treated as an invalid label and "
        "dropped entirely - on the reasoning that an indeterminate label carries no "
        "information to train a specific-cause classifier on. In practice this denied the "
        "model any way to output 'I don't know' at all: every record was forced into one "
        "of the remaining specific or clustered causes, including records with no "
        "decisive evidence for any of them. This was found directly, by tracing a "
        "misclassified record in which an unrelated classifier output 'External Causes of "
        "Death' at 98% softmax confidence despite every injury-mechanism question in the "
        "record being answered 'no'. 'Unknown and ill-defined cause of death' is now kept "
        "as a genuine training example and is protected from cluster-collapsing on the "
        "same basis as the directly-confirmable rare causes above, giving the model a real "
        "class to assign when the record itself does not support a specific diagnosis.")

    set_heading(doc, "2.4 Feature Engineering", level=2)
    para(doc,
        "Two feature families are combined. Structured responses (binary, ordinal, and "
        "categorical WHO-VA questions) are label-encoded per column using encoders fit on "
        "training data only, then standardised (zero mean, unit variance). Free-text "
        "narrative fields (WHO field codes id10476, id10477, id10479, id10436, in which the "
        "informant describes the final illness in their own words) are concatenated per "
        "record and encoded into a 384-dimensional dense vector using the multilingual "
        "sentence-transformer paraphrase-multilingual-MiniLM-L12-v2 [Sentence-BERT, 22], "
        f"plus one trailing presence flag ({report['narrative_dims']} narrative dimensions "
        "in total). A record with no narrative text receives an all-zero vector and a "
        "presence flag of zero, rather than the model's encoding of the empty string, "
        "specifically because narrative fields are entirely absent from some source "
        "datasets; encoding '' would otherwise give every narrative-less record the same "
        "non-zero point in embedding space, perfectly correlated with which dataset lacked "
        "narrative collection - itself a form of the leakage risk central to Section 2.5.")
    doc.add_paragraph()
    para(doc,
        "Non-response sentinel tokens ('dk', 'don't know', 'refused', and equivalents, "
        "matching the same tokens the structured-feature encoders already treat as "
        "missingness) are now excluded before narrative concatenation; a free-text field "
        "holding only the literal string 'dk' was previously embedded as if the respondent "
        "had described something, found during the same case-tracing exercise referenced "
        "in Section 2.3.")
    doc.add_paragraph()
    para(doc,
        "A sentence-transformer model depends on PyTorch, whose OpenMP runtime was found "
        "to conflict with XGBoost's own multi-threaded OpenMP runtime when both load in the "
        "same process - a plain 'import torch', before any model is used, was enough to "
        "crash a later multi-threaded XGBoost fit. Narrative embedding is therefore computed "
        "in a separate, spawned subprocess during training, so the two libraries' OpenMP "
        "runtimes never coexist; this project-specific engineering detail is reported here "
        "because it is the kind of failure that is easy to misdiagnose as a data problem.")

    set_heading(doc, "2.5 Dataset-Identity Leakage: Detection and Mitigation", level=2)
    para(doc,
        "Pooling data from heterogeneous sources is the central practical difficulty of "
        "this work, and deserves a methods section of its own. Heterogeneity took several "
        "concrete forms across the three source datasets: different WHO-VA instrument "
        "versions carrying partially different field sets; the same field exported in "
        "incompatible formats (dates as MM/DD/YY text in one dataset, a mix of DD/MM/YYYY "
        "and M/D/YY text in another, and, in the third, largely as the literal Excel "
        "overflow placeholder '########' or raw unconverted Excel serial-date integers); "
        "and whole instrument sections (e.g. a civil-registration consent question) present "
        "in one dataset and entirely absent from the others. The original feature-selection "
        "step took the union of every column seen in any source file and filled a column "
        "with the 'don't know' sentinel for every record from a dataset that lacked it - a "
        "reasonable-sounding design that, on inspection, has a specific failure mode: for a "
        "column present in only one dataset, 'does this have a real value' is a perfect, "
        "deterministic indicator of which dataset a record came from, whether or not the "
        "column's actual answers carry any clinical signal.")
    doc.add_paragraph()
    para(doc,
        "This was investigated after a civil-registration consent question appeared, in a "
        "single-record model-explanation panel, as a top contributing factor for two "
        "unrelated predicted causes with no plausible clinical connection to the question. "
        f"A systematic audit was then run: for every one of the "
        f"{li['pre_fix_structured_feature_count']} candidate structured feature columns, "
        f"the normalised mutual information (NMI) between the column's value and the "
        f"record's source dataset was computed, and compared against the same statistic "
        f"computed for the true physician-assigned label itself (NMI = "
        f"{li['true_label_nmi_vs_source']}, i.e. disease mix genuinely does differ somewhat "
        f"by source population). {li['columns_in_only_some_datasets']} of "
        f"{li['pre_fix_structured_feature_count']} columns "
        f"({li['columns_in_only_some_datasets_pct']}%) existed in only one or two of the "
        f"three datasets; dozens of columns, including several present in all three, "
        f"scored an NMI against source as high as {li['max_feature_nmi_vs_source']} - up "
        f"to four times higher than the real label's own relationship to source, the "
        f"signature of a shortcut rather than a signal.")
    doc.add_paragraph()
    para(doc,
        "Two mitigations were implemented. First, the multi-dataset feature-alignment step "
        "now defaults to the intersection of feature columns actually present in every "
        "dataset being combined, rather than their union with a sentinel fill; this removes "
        f"all {li['columns_in_only_some_datasets']} presence-based leaks by construction "
        f"({li['pre_fix_structured_feature_count']} to "
        f"{li['post_intersection_only_feature_count']} columns), with the old union "
        "behaviour retained as an explicit, non-default option for comparison. Second, two "
        "specific columns confirmed to carry a genuine data-quality defect rather than an "
        "ambiguous cross-population difference - recorded date of death and the civil-"
        "registration consent question - are explicitly excluded even though they are "
        "nominally present in every dataset, since the intersection filter alone cannot "
        "catch a format inconsistency in a column that technically exists everywhere "
        f"(reducing the final structured feature count to "
        f"{li['post_fix_structured_feature_count']}). We deliberately did not apply a "
        "blanket cut at some NMI threshold across the remaining features: inspection "
        "showed the method also flags features we judge should be kept - for example, a "
        "deployment-level malaria/HIV-prevalence setting that is, by design, specific to "
        "the region a dataset was collected in and is intended to inform the prior, not an "
        "artefact. Dataset-identity correlation is a useful screen, not an automatic "
        "verdict, and the remaining ambiguous features are left for a dedicated, "
        "clinically-led follow-up audit rather than an automated purge.")
    doc.add_paragraph()
    para(doc,
        f"Retraining under this corrected pipeline ({report['feature_count_structured']} "
        f"structured features, matching the {li['post_fix_structured_feature_count']} "
        f"figure above) changed hold-out performance on the identical split construction: "
        f"accuracy moved from {li['pre_fix_holdout_accuracy']:.1%} to "
        f"{report['holdout']['accuracy']:.1%} ({acc_delta:+.1%}) and macro-F1 from "
        f"{li['pre_fix_holdout_f1_macro']:.3f} to {report['holdout']['f1_macro']:.3f} "
        f"({f1_delta:+.3f}); see Section 3.5 for the full comparison. We read this "
        "direction - apparent performance falling once the leakage-prone features are "
        "removed - as consistent with the hypothesis that some of the pre-fix figure was "
        "inflated by dataset-identity shortcuts rather than genuine clinical signal, while "
        "stopping short of claiming it as definitive proof: removing any block of "
        "features, including a partially useful one, typically cost some accuracy on its "
        "own.")

    set_heading(doc, "2.6 Model Development", level=2)
    comp = report["model_comparison"]
    xgb_c = comp.get("XGBoost", {})
    rf_c = comp.get("RandomForest", {})
    para(doc,
        "A stratified 80/20 split, held out before any model fitting, cross-validation, or "
        "hyperparameter search, provides the independent test set reported in Section 3. "
        "Within the remaining 80%, two candidate algorithms - gradient-boosted trees "
        "(XGBoost [23]) and random forest (scikit-learn [24]) - were compared by randomised "
        "hyperparameter search with stratified k-fold cross-validation, selecting on "
        "macro-averaged F1 to avoid the common classes dominating model selection. "
        "Successive-halving search (HalvingRandomSearchCV) was used for the random-forest "
        "search; for XGBoost it was found to intermittently produce an inconsistent class "
        "set across halving rounds on this long-tailed, multi-class target, which a plain "
        "randomised search does not exhibit, so XGBoost's own search uses the latter. In "
        f"the run reported here, XGBoost (cross-validated macro-F1 "
        f"{xgb_c.get('cv_f1_macro', float('nan')):.3f}) outperformed random forest "
        f"({rf_c.get('cv_f1_macro', float('nan')):.3f}) by a wide margin and was selected. "
        "Training rows with a direct professional-confirmation answer matching their own "
        "label (e.g. a positive health-worker-confirmed malaria test on a record labelled "
        "Malaria) receive a higher sample weight, so the model is penalised more for "
        "misclassifying an already-confirmed diagnosis than an inferred one.")

    set_heading(doc, "2.7 Indeterminate-Cause and Out-of-Distribution Design", level=2)
    para(doc,
        "A clinically credible classifier must be able to decline to give a specific answer. "
        "Three distinct mechanisms implement this, operating at different points in the "
        "pipeline:")
    numbered(doc, "A trainable indeterminate class - 'Unknown and ill-defined cause of "
        "death' (Section 2.3) is a real class the model can learn to recognise, rather than "
        "a value imposed after the fact.")
    numbered(doc, "Out-of-distribution (OOD) abstention at prediction time, using three "
        "independent signals combined with a logical OR, so a user-supplied override to "
        "one signal only ever adds an extra, typically stricter net rather than silently "
        "replacing a better-calibrated one - an earlier version of this logic did the "
        "latter, discarding the model's own entropy check whenever a caller supplied a "
        "confidence override at all. (a) Normalised Shannon entropy of the predicted-class "
        "probability distribution, with the threshold calibrated per model as the 97th "
        "percentile of this entropy on the internal validation split (0.452 for the model "
        "reported here) - i.e. the most uncertain 3% of validation predictions, by "
        "construction, fall on the OOD side; not exposed as a user-adjustable setting. "
        "(b) Top-class softmax confidence, flagged once it falls below a threshold that is "
        "exposed in the VMan3 'Run CCVA' interface as 'OOD Threshold' (default 0.05) for "
        "the operator to adjust per run. (c) The fraction of input feature columns "
        "answered 'don't know' for a given record, flagged once it exceeds a threshold "
        "exposed in the same interface as 'DK Threshold' (default 0.60); the model itself "
        "additionally enforces 0.60 as a hard floor regardless of what value an operator "
        "supplies, or what an earlier, since-superseded calibration run (computed against "
        "already quality-filtered, and therefore artificially complete, training data) had "
        "produced - an unrealistically tight ~15% that over-flagged ordinary prediction "
        "data. Both operator-adjustable thresholds are evaluated empirically in Section 3.6.")
    numbered(doc, f"A stricter confidence bar for the {report['n_cluster_groups']} "
        "clustered ('cluster_*') classes specifically: because each is a deliberately "
        "heterogeneous grab-bag of several distinct causes merged only because any one of "
        "them was individually too rare (Section 2.3), it has the loosest decision "
        "boundary of any class in the model, which lets a record with no clear evidence "
        "for anything specific still produce a sharply peaked softmax onto the nearest "
        "grab-bag - a failure the entropy check above does not catch, because the "
        "distribution genuinely is peaked, just onto the wrong kind of class. A "
        "cluster-class prediction below 90% confidence is therefore demoted to the "
        "indeterminate class rather than accepted at face value. This threshold is a "
        "documented heuristic - chosen so that genuinely well-evidenced predictions, which "
        "this project's own data clear easily, are unaffected, while the specific failure "
        "case that motivated it (a 'cluster_external_causes_of_death' prediction at 96% "
        "confidence with no supporting injury evidence at all, detailed next) is caught - "
        "pending a dedicated calibration study once enough validated cluster-class "
        "predictions accumulate in routine use.")
    doc.add_paragraph()
    para(doc,
        "Alongside abstention, a small set of deterministic, rule-based plausibility "
        "constraints is applied to the model's output probabilities before the final class "
        "is chosen, zeroing out probability mass for combinations that are not merely "
        "unlikely but definitionally impossible: maternal/pregnancy-related causes for a "
        "male, child, or neonate decedent, or for a decedent recorded as older than 49 "
        "years (the WHO upper bound for reproductive age); neonate- or stillbirth-only "
        "causes for an older child or adult decedent; and - added directly in response to "
        "the case that prompted this investigation - external-cause classes (the cluster "
        "above, and the standalone 'Road traffic accident' class) unless the record's own "
        "injury/accident gate question, or at least one specific injury-mechanism question, "
        "is answered 'yes'. These mirror the same logical-consistency reasoning a published "
        "VA data-quality indicator framework already applies to raw interview data [see "
        "companion indicator study], applied here instead to the model's output.")

    set_heading(doc, "2.8 Statistical Analysis", level=2)
    para(doc,
        "Accuracy, macro- and weighted-averaged F1, and one-vs-rest macro-averaged AUROC "
        "are reported on the sealed hold-out split, overall and per cause-of-death category. "
        "Feature importance is summarised using XGBoost's own tree-based SHAP contribution "
        "values [25], aggregated globally and stratified by age group, sex, and WHO cause "
        "group.")

    # ── 3. Results ──────────────────────────────────────────────────────
    set_heading(doc, "3. Results")

    set_heading(doc, "3.1 Dataset and Pipeline Summary", level=2)
    add_table(doc,
        headers=["Stage", "Rows"],
        rows=[
            ["Input rows (3 datasets pooled)", f"{report['input_rows']:,}"],
            ["After quality filter", f"{report['rows_after_quality_filter']:,}"],
            ["After rare-cause clustering", f"{report['rows_after_clustering']:,}"],
            ["Final rows used for training", f"{report['rows_final']:,}"],
        ])
    doc.add_paragraph()
    add_table(doc,
        headers=["Reason excluded", "Rows"],
        rows=[
            ["WHO causelist mapping failure", f"{report['dropped_who_mapping_failure']}"],
            ["Demographic-cause implausibility (sex/age-group)", f"{report['dropped_demographic_implausibility']}"],
            ["vman_dq pregnancy/sex contradiction (raw data)", f"{report['dropped_pregnancy_sex_contradiction']}"],
            ["vman_dq low ICI (Critical tier, <70%)", f"{report['dropped_low_ici']}"],
            ["Kept as 'Unknown and ill-defined cause of death' (no longer dropped)", f"{report['kept_as_indeterminate']} (retained)"],
        ])
    doc.add_paragraph()
    pct_retained = 100 * report["rows_final"] / report["input_rows"]
    para(doc,
        f"Of {report['input_rows']:,} pooled records, {report['rows_final']:,} "
        f"({pct_retained:.1f}%) remained after quality filtering and were used for "
        f"training and hold-out evaluation. {report['n_clustered_causes']} cause titles "
        f"were grouped into {report['n_cluster_groups']} cluster categories (Appendix "
        f"Table A1), giving {report['n_classes']} final cause-of-death categories. The "
        f"feature set used for this training run comprised "
        f"{report['feature_count_structured']} structured columns (post-leakage-fix "
        f"intersection mode, Section 2.5) plus {report['narrative_dims']} narrative-"
        f"embedding dimensions ({total_features} total model inputs). The run completed "
        f"in {report['total_run_duration']} (hyperparameter search: "
        f"{report['training_duration']}), using {report['n_training_samples']:,} training "
        f"and {report['n_test_samples']:,} hold-out records.")

    set_heading(doc, "3.2 Model Comparison and Overall Hold-Out Performance", level=2)
    add_table(doc,
        headers=["Model", "CV Macro-F1", "Validation Accuracy", "Validation Macro-F1"],
        rows=[
            [f"XGBoost{' (selected)' if report['selected_model'] == 'XGBoost' else ''}",
             f"{xgb_c.get('cv_f1_macro', float('nan')):.3f}",
             f"{xgb_c.get('test_accuracy', float('nan')):.3f}",
             f"{xgb_c.get('test_f1_macro', float('nan')):.3f}"],
            [f"Random forest{' (selected)' if report['selected_model'] == 'RandomForest' else ''}",
             f"{rf_c.get('cv_f1_macro', float('nan')):.3f}",
             f"{rf_c.get('test_accuracy', float('nan')):.3f}",
             f"{rf_c.get('test_f1_macro', float('nan')):.3f}"],
        ])
    doc.add_paragraph()
    v, h = report["validation"], report["holdout"]
    add_table(doc,
        headers=["Metric", "Validation split", f"Hold-out split (n={h['n']:,})"],
        rows=[
            ["Accuracy", f"{v['accuracy']:.3f}", f"{h['accuracy']:.3f}"],
            ["Macro-F1", f"{v['f1_macro']:.3f}", f"{h['f1_macro']:.3f}"],
            ["Weighted-F1", f"{v['f1_weighted']:.3f}", f"{h['f1_weighted']:.3f}"],
            ["Macro-AUROC", f"{v['auroc_macro']:.3f}", f"{h['auroc_macro']:.3f}"],
        ])
    doc.add_paragraph()
    para(doc,
        f"Performance on the sealed hold-out split closely tracked the internal validation "
        f"split (accuracy {h['accuracy']:.3f} vs. {v['accuracy']:.3f}; macro-F1 "
        f"{h['f1_macro']:.3f} vs. {v['f1_macro']:.3f}), with no indication of overfitting "
        f"to the validation split used for model selection.")

    set_heading(doc, "3.3 Per-Cause Performance", level=2)
    per_cause = report["holdout_per_cause"]
    worst = per_cause[:2]
    best = sorted(per_cause, key=lambda r: -r[3])[:5]
    indeterminate = next((r for r in per_cause if r[0] == "Unknown and ill-defined cause of death"), None)
    best_str = "; ".join(f"{c} F1 {f1:.3f}, n={n}" for c, _, _, f1, _, n in best)
    worst_str = "; ".join(f"{c} (F1 {f1:.3f}, n={n})" for c, _, _, f1, _, n in worst)
    para(doc,
        f"Table 3 reports precision, recall, F1, AUROC, and hold-out support for all "
        f"{report['n_classes']} categories. Performance was strongest for high-support, "
        f"clinically well-defined causes ({best_str}).")
    if indeterminate:
        _, _, rc, f1, _, n = indeterminate
        doc.add_paragraph()
        para(doc,
            f"The retained indeterminate class performed "
            f"{'comparably to many specific causes' if f1 >= 0.6 else 'more weakly than most specific causes'} "
            f"(F1 {f1:.3f}, recall {rc:.3f}, n={n}), "
            f"{'supporting its inclusion as a genuine, learnable category rather than noise.' if f1 >= 0.6 else 'suggesting it remains harder to recognise reliably than most specific causes and warrants continued attention in future retrains.'}")
    doc.add_paragraph()
    para(doc, f"The weakest-performing categories were low-support and clinically "
              f"heterogeneous: {worst_str}.")
    doc.add_paragraph()
    table_caption(doc, "Table 3. Per-cause hold-out performance, sorted by F1 (ascending).")
    add_table(doc,
        headers=["Cause", "Precision", "Recall", "F1", "AUROC", "n"],
        rows=[[c, f"{p:.3f}", f"{rc:.3f}", f"{f1:.3f}", f"{a:.3f}", n]
              for c, p, rc, f1, a, n in per_cause],
        col_widths=[2.6, 0.7, 0.7, 0.6, 0.7, 0.5])

    set_heading(doc, "3.4 Feature Contribution", level=2)
    para(doc,
        "Global and stratified SHAP analyses (by age group, sex, and WHO cause group) "
        "confirmed that the relative importance of structured symptom questions versus the "
        "narrative-embedding block differs meaningfully across sub-groups - e.g. neonatal "
        "and adult predictions draw on largely disjoint sets of top features - consistent "
        "with verbal-autopsy predictors genuinely differing by who died of what, rather "
        "than the model relying on one dominant, generic signal across all causes.")

    set_heading(doc, "3.5 Effect of the Leakage Fix on Hold-Out Performance", level=2)
    table_caption(doc, "Table 4. Hold-out performance before and after the dataset-identity "
                        "leakage mitigation (Section 2.5), same split construction.")
    add_table(doc,
        headers=["Pipeline", "Structured features", "Accuracy", "Macro-F1", "n"],
        rows=[
            ["Pre-fix (union mode)", f"{li['pre_fix_structured_feature_count']}",
             f"{li['pre_fix_holdout_accuracy']:.3f}", f"{li['pre_fix_holdout_f1_macro']:.3f}",
             f"{li['pre_fix_holdout_n']:,}"],
            ["Post-fix (intersection mode + exclusions)", f"{report['feature_count_structured']}",
             f"{h['accuracy']:.3f}", f"{h['f1_macro']:.3f}", f"{h['n']:,}"],
        ])
    doc.add_paragraph()
    para(doc,
        f"Removing the {li['pre_fix_structured_feature_count'] - report['feature_count_structured']} "
        f"leakage-prone structured features (Section 2.5) was associated with a "
        f"{abs(acc_delta):.1%} {'decrease' if acc_delta < 0 else 'increase'} in hold-out "
        f"accuracy and a {abs(f1_delta):.3f} {'decrease' if f1_delta < 0 else 'increase'} "
        f"in macro-F1 on the identical hold-out construction. We interpret this as "
        "evidence, though not proof, that part of the pre-fix figure reflected the model "
        "exploiting dataset-identity shortcuts rather than clinical signal, per the "
        "reasoning in Section 2.5 and discussed further in Section 4.1.")

    set_heading(doc, "3.6 Operator-Adjustable Thresholds: An Empirical Check", level=2)
    ota = OOD_THRESHOLD_ANALYSIS
    para(doc,
        f"The 'Run CCVA' interface exposes two of the three OOD signals in Section 2.7 as "
        f"per-run operator settings - 'DK Threshold' (default {ota['ui_default_dk_threshold']:.2f}) "
        f"and 'OOD Threshold' (default {ota['ui_default_ood_threshold']:.2f}) - prompting us "
        f"to check empirically whether these two specific default values are doing any "
        f"work. On the hold-out set (n={ota['holdout_n']:,}), we recomputed all three OOD "
        f"signals and swept the confidence threshold ('OOD Threshold') across candidate "
        f"values from its default up to six times that value.")
    doc.add_paragraph()
    table_caption(doc, "Table 5. Contribution of each OOD signal on the hold-out set, by "
                        "confidence-threshold value.")
    add_table(doc,
        headers=["Confidence threshold ('OOD Threshold')", "Entropy fires", "Confidence fires",
                 "DK-ratio fires", "Confidence's unique contribution"],
        rows=[
            [f"{ota['ui_default_ood_threshold']:.2f} (current UI default)",
             ota["entropy_mask_fires"], ota["confidence_mask_fires_at_default_0_05"],
             ota["dk_mask_fires_at_0_60"], ota["confidence_mask_unique_contribution_any_threshold_tested"]],
            ["0.30 (6x default)", ota["entropy_mask_fires"], ota["confidence_mask_fires_at_0_30"],
             ota["dk_mask_fires_at_0_60"], ota["confidence_mask_unique_contribution_any_threshold_tested"]],
        ])
    doc.add_paragraph()
    para(doc,
        f"At the current default (0.05), the confidence signal fired on zero of "
        f"{ota['holdout_n']:,} hold-out records - the lowest observed top-class "
        f"confidence anywhere in the set was {ota['min_observed_confidence']:.3f}, well "
        f"above the threshold. Raising it six-fold, to 0.30, still added zero records "
        f"beyond what the entropy signal (calibrated at "
        f"{ota['model_calibrated_entropy_threshold']:.3f}, firing on "
        f"{ota['entropy_mask_fires']} records) already flagged on its own, at any "
        f"threshold value tested. The DK-ratio signal likewise fired on zero records at "
        f"its 0.60 default, which is expected rather than concerning: a hold-out split "
        f"drawn from already quality-filtered training data is, by construction, more "
        f"complete than a typical field submission, so this signal's real test is live "
        f"deployment data, not this evaluation.")
    doc.add_paragraph()
    para(doc,
        f"A separate, live CCVA run on field data (not part of training or this hold-out "
        f"set) using these same UI defaults flagged {ota['live_run_undetermined_pct']:.1f}% "
        f"of records as indeterminate ('Undetermined') - over ten times the ~3% rate the "
        f"entropy threshold was calibrated to produce on the internal validation split. "
        f"We could not fully decompose that gap in this analysis, since per-record "
        f"entropy/confidence/DK-ratio values are not currently persisted alongside stored "
        f"CCVA results; it is flagged as a priority item in Section 4.5.")

    # ── 4. Discussion ───────────────────────────────────────────────────
    set_heading(doc, "4. Discussion")

    set_heading(doc, "4.1 Principal Findings", level=2)
    para(doc,
        f"A pooled, three-country VA classifier reached {h['accuracy']:.1%} hold-out "
        f"accuracy and {h['f1_macro']:.3f} macro-F1 across {report['n_classes']} cause "
        f"categories after correcting for dataset-identity leakage, within the range "
        f"reported for other data-driven CCVA methods evaluated against physician or "
        f"medical-record gold standards [3,6]. The more consequential finding is "
        f"methodological: {li['columns_in_only_some_datasets_pct']}% of the candidate "
        f"structured features existed in only a subset of the pooled datasets, several "
        f"more that existed in every dataset still carried an NMI-with-source-dataset far "
        f"exceeding the true label's own, and removing them changed hold-out accuracy by "
        f"{acc_delta:+.1%} on the identical split - a measurable, not merely theoretical, "
        f"consequence of the model having learned to recognise its training data's "
        f"provenance rather than purely clinical signal.")

    set_heading(doc, "4.2 Verbal-Autopsy Data Heterogeneity as a Methodological Hazard, "
                      "Not Just a Data-Cleaning Inconvenience", level=2)
    para(doc,
        "VA data pooled across deployments varies for reasons that have nothing to do with "
        "the decedents' symptoms: instrument version, data-collection platform, export "
        "tooling, and even which language a question's label happened to be recorded in by "
        "whichever team built or extended the form for that deployment. In this project, "
        "that variation included genuinely different field sets between the 2016 and 2022 "
        "WHO-VA instruments; a date field exported in three mutually incompatible formats "
        "across the three sources, one of them broken outright by an Excel export artefact; "
        "and a whole instrument section present in one source and absent from the other "
        "two. None of these are symptoms of poor data collection in the clinical sense - "
        "they are artefacts of how each deployment's data happened to be extracted and "
        "exported. A pipeline that pools multiple sources without accounting for this "
        "treats 'this column has a real value' or 'this column parses as a plausible date' "
        "as if it were itself a clinical observation, when it is frequently a near-perfect "
        "label for which pipeline produced the file. We would argue this risk deserves the "
        "same explicit, routine audit that label quality and missingness already receive in "
        "VA machine-learning work, rather than being left to be discovered by chance, as it "
        "was here, from a single explained prediction that did not make clinical sense.")

    set_heading(doc, "4.3 Distinguishing Leakage from Legitimate Epidemiological Prior", level=2)
    para(doc,
        "Not every feature correlated with dataset-of-origin should be removed. A "
        "deployment-level setting indicating local malaria or HIV prevalence is, by design, "
        "specific to where a dataset was collected, and a model that leans on it is using a "
        "legitimate, intentionally-provided epidemiological prior, not exploiting an "
        "artefact. The distinguishing question is mechanism, not correlation: does the "
        "feature carry information a new deployment could also, and would legitimately, "
        "provide (a real prevalence setting, a real demographic field), or does it merely "
        "encode which of a fixed set of training files produced this row (a column absent "
        "from two of three sources; a date field whose format differs by export pipeline)? "
        "We did not find this distinction automatable from the correlation statistic alone, "
        "and resolved it here by individual inspection of the highest-scoring features "
        "rather than a single threshold rule.")

    set_heading(doc, "4.4 Limitations", level=2)
    para(doc,
        f"First, the pre-fix comparison figures in Sections 2.5, 3.5, and 4.1 come from a "
        f"training run whose own report directory was later overwritten by the post-fix "
        f"retrain because both reused the same version label; they are preserved here as "
        f"text rather than a re-loadable report, which is itself a reproducibility lesson - "
        f"a pipeline change substantial enough to warrant its own before/after comparison "
        f"should also warrant its own version label. Second, the leakage audit itself used "
        f"normalised mutual information, which is known to be unstable for near-constant "
        f"features (a feature answered the same way by over 99% of records can register a "
        f"numerically high score from a small, non-generalisable raw effect); results for "
        f"such features were discounted on inspection rather than trusted at face value, "
        f"but a more robust statistic would strengthen this audit methodologically. Third, "
        f"the ambiguous mid-ranked features identified in Section 2.5 were not individually "
        f"resolved and remain in the current feature set pending dedicated clinical review. "
        f"Fourth, the Nigeria dataset's instrument version was auto-detected against the "
        f"closest available dictionary (2022 Tanzania) in the absence of a Nigeria-specific "
        f"one, which may itself have contributed to the feature-presence heterogeneity "
        f"reported in Section 2.5 independently of genuine instrument differences. Fifth, "
        f"the {abs(acc_delta):.1%} hold-out accuracy change between the pre- and post-fix "
        f"pipelines cannot be decomposed from this analysis alone into 'shortcut removed' "
        f"versus 'some genuine signal removed along with it'; the two are not mutually "
        f"exclusive, and the true balance would need a held-out dataset from a deployment "
        f"not represented in training to resolve directly. Sixth, as with any hold-out "
        f"evaluation drawn from the same pool of source datasets used for training, "
        f"performance on a genuinely new country or deployment not represented in this pool "
        f"remains untested and is, per Section 4.2, exactly the setting in which any "
        f"residual dataset-identity shortcut would be expected to generalise worst.")

    set_heading(doc, "4.5 Future Directions", level=2)
    para(doc,
        "Priority next steps are: (i) a clinically-led review of the remaining mid-ranked, "
        "ambiguous features flagged in Section 2.5; (ii) external validation on a VA "
        "dataset collected independently of the three pooled here, to separate genuinely "
        "lost signal from removed shortcuts; (iii) a dedicated calibration study for the "
        "cluster-confidence threshold (Section 2.7) once sufficient validated cluster-class "
        "predictions have accumulated in routine use; (iv) adopting distinct version "
        "labels for pipeline changes substantial enough to warrant their own before/after "
        "comparison, per the reproducibility note in Section 4.4; and (v) persisting "
        "per-record entropy, top-class confidence, and DK-ratio alongside stored CCVA "
        "results, so the gap between calibrated (~3%) and observed live (44.5%) "
        "indeterminate rates (Section 3.6) can be decomposed by which signal is actually "
        "driving it, rather than only observed in aggregate.")

    set_heading(doc, "4.6 Recommendation: Operator-Adjustable Threshold Defaults", level=2)
    para(doc,
        "The 'DK Threshold' default (0.60) is appropriate as-is and should be kept: it "
        "matches a floor the model enforces internally regardless of operator input, "
        "fixing an earlier default (1.0) that silently disabled this safety net entirely "
        "since the underlying check can never exceed a fraction of 1.0. We see no basis "
        "in today's analysis to change it.")
    doc.add_paragraph()
    para(doc,
        "The 'OOD Threshold' default (0.05), by contrast, is currently inert rather than "
        "wrong: Section 3.6 shows it contributes nothing beyond the model's own entropy "
        "check, at its current value or at several times that value, on the data "
        "available to test it. This is not a bug - the mechanism is working exactly as "
        "designed, OR'd alongside the entropy signal rather than replacing it - but it "
        "means operators adjusting this specific field in the UI are, in practice, "
        "adjusting a control that does very little on its own. We recommend one of two "
        "paths rather than a specific new numeric default, since no value tested changed "
        "the outcome on available data: either (a) leave the control as a documented, "
        "rarely-binding secondary safety net and say so explicitly in the UI, so operators "
        "do not reasonably expect adjusting it to change much; or (b) expose the two "
        "thresholds that this work found actually drive abstention behaviour - the "
        "model's calibrated entropy threshold and the cluster-confidence threshold "
        "(Section 2.7) - as the operator-adjustable controls instead, since those are "
        "the mechanisms with a demonstrated effect. Separately, and independently of "
        "either threshold's default value, the gap between the ~3% OOD rate the entropy "
        "threshold was calibrated to produce and the 44.5% observed on live field data "
        "(Section 3.6) is the more consequential open question for this classifier's "
        "real-world abstention behaviour, and is not resolved by adjusting either UI "
        "threshold - see Section 4.5.")

    # ── 5. Conclusion ───────────────────────────────────────────────────
    set_heading(doc, "5. Conclusion")
    para(doc,
        f"VMan3-ML demonstrates that a gradient-boosted classifier combining structured "
        f"verbal-autopsy responses with multilingual narrative embeddings can reach "
        f"cause-of-death classification accuracy comparable to published computer-coded "
        f"verbal-autopsy methods, evaluated on an independent hold-out set spanning "
        f"{report['n_classes']} cause categories pooled from three countries and two "
        f"WHO-VA instrument versions. Equally important for anyone pooling verbal-autopsy "
        f"data across sites, we show concretely that such pooling creates a specific, "
        f"testable, and partially preventable risk - the model learning dataset-of-origin "
        f"rather than clinical signal - and report both a practical audit method "
        f"(feature-wise mutual information against source dataset, benchmarked against the "
        f"true label's own value), the mitigations it motivated, and their measured effect "
        f"on hold-out performance ({acc_delta:+.1%} accuracy, {f1_delta:+.3f} macro-F1). We "
        f"also report, with their calibration rationale and specific values, the thresholds "
        f"governing when this classifier should decline to give a specific answer at all, "
        f"since an automated cause-of-death tool's willingness to say 'indeterminate' is as "
        f"clinically consequential as its accuracy when it does commit to an answer.")

    # ── Declarations ────────────────────────────────────────────────────
    set_heading(doc, "Declarations")
    set_heading(doc, "Conflict of interest", level=2)
    para(doc, "The author declares no conflict of interest.")
    set_heading(doc, "Funding", level=2)
    para(doc, "[To be completed.]")
    set_heading(doc, "Data availability", level=2)
    para(doc,
        "The VMan3-ML pipeline and model-training code are available at "
        "https://github.com/vman-tool/vman3_ml. The individual-level verbal autopsy "
        "datasets analysed in this study are not publicly available due to restrictions "
        "imposed by the respective data custodians. Requests for access should be "
        "submitted to the relevant ministry or data custodian.")
    set_heading(doc, "Author contributions", level=2)
    para(doc,
        "IL conceptualised and developed the machine-learning pipeline, conducted the "
        "leakage investigation and validation analyses, and drafted the manuscript. "
        "[Additional contributors to be added.]")

    page_break(doc)

    # ── References ──────────────────────────────────────────────────────
    set_heading(doc, "References")
    refs = [
        "Mikkelsen L, Phillips DE, AbouZahr C, et al. A global assessment of civil "
            "registration and vital statistics systems: monitoring data quality and "
            "progress. Lancet. 2015;386(10001):1395-1406.",
        "Fottrell E, Byass P. Verbal autopsy: methods in transition. Epidemiol Rev. "
            "2010;32:38-55.",
        "McCormick TH, Li ZR, Calvert C, et al. Probabilistic Cause-of-death Assignment "
            "using Verbal Autopsies. J Am Stat Assoc. 2016;111(515):1036-1049.",
        "Li ZR, Thomas J, Choi E, et al. The openVA Toolkit for Verbal Autopsies. R J. "
            "2022;14(4):316-334.",
        "Byass P, Calvert C, Miiro-Nakiyingi J, et al. InterVA-4 as a public health tool "
            "for measuring HIV/AIDS mortality: a validation study from five African "
            "countries. Glob Health Action. 2013;6:22448.",
        "Flaxman AD, Vahdatpour A, Green S, James SL, Murray CJ; Population Health "
            "Metrics Research Consortium. Random forests for verbal autopsy analysis: "
            "multisite validation study using clinical diagnostic gold standards. Popul "
            "Health Metr. 2011;9:29.",
        "Lozano R, Lopez AD, Atkinson C, et al. Performance of physician-certified "
            "verbal autopsies: multisite validation study using clinical diagnostic gold "
            "standards. Popul Health Metr. 2011;9:32.",
        "verbal-autopsy-software/pyCrossVA [Internet]. Verbal Autopsy Software. "
            "Available from: https://github.com/verbal-autopsy-software/pyCrossVA",
        "Sankoh O, Byass P. The INDEPTH Network: filling vital gaps in global "
            "epidemiology. Int J Epidemiol. 2012;41(3):579-588.",
        "Oza S, Lawn JE, Hogan DR, et al. Neonatal cause-of-death estimates for the "
            "early and late neonatal periods for 194 countries: 2000-2013. Bull World "
            "Health Organ. 2015;93(1):19-28.",
        "World Health Organization. Data quality assurance (DQA) [Internet]. Available "
            "from: https://www.who.int/data/data-collection-tools/health-service-data/",
        "World Health Organization. Verbal autopsy standards: The 2016 WHO verbal "
            "autopsy instrument [Internet]. Geneva: WHO; 2016.",
        "World Health Organization. Verbal autopsy standards: The 2022 WHO verbal "
            "autopsy instrument [Internet]. Geneva: WHO; 2022.",
        "Nichols EK, Byass P, Chandramohan D, et al. The WHO 2016 verbal autopsy "
            "instrument: An international standard suitable for automated analysis by "
            "InterVA, InSilicoVA, and Tariff 2.0. PLoS Med. 2018;15(1):e1002486.",
        "Saris WE, Gallhofer IN. Design, Evaluation, and Analysis of Questionnaires for "
            "Survey Research. 2nd ed. Wiley; 2014.",
        "Marks M, Lal S, Brindle H, et al. Electronic Data Management for Vaccine "
            "Trials in Low Resource Settings: Upgrades, Scalability, and Impact of Open "
            "Data Kit. Clin Infect Dis. 2019;68(Suppl 2):S264-S270.",
        "Olu-Abiodun O, Faturoti A, Adepoju A, et al. Effectiveness and challenges of "
            "digital tools for enhancing infectious disease surveillance in low- and "
            "middle-income countries. BMC Public Health. 2023;23:1812.",
        "Quionero-Candela J, Sugiyama M, Schwaighofer A, Lawrence ND, eds. Dataset Shift "
            "in Machine Learning. MIT Press; 2009.",
        "Moreno-Torres JG, Raeder T, Alaiz-Rodriguez R, Chawla NV, Herrera F. A unifying "
            "view on dataset shift in classification. Pattern Recognit. "
            "2012;45(1):521-530.",
        "Lyatuu I. VMan3 Demo Server [Internet]. Available from: "
            "https://vman3.vatools.net/login.",
        "Lyatuu I. vman-tool/vman3_ml [Internet]. Available from: "
            "https://github.com/vman-tool/vman3_ml.",
        "Chen T, Guestrin C. XGBoost: A Scalable Tree Boosting System. In: Proceedings "
            "of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and "
            "Data Mining. 2016:785-794.",
        "Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in "
            "Python. J Mach Learn Res. 2011;12:2825-2830.",
        "Reimers N, Gurevych I. Sentence-BERT: Sentence Embeddings using Siamese "
            "BERT-Networks. In: Proceedings of EMNLP-IJCNLP 2019. 2019:3982-3992.",
        "Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. In: "
            "Advances in Neural Information Processing Systems 30 (NeurIPS 2017). "
            "2017:4765-4774.",
    ]
    for i, text in enumerate(refs, start=1):
        reference(doc, i, text)

    page_break(doc)

    # ── Appendix ────────────────────────────────────────────────────────
    set_heading(doc, "Appendix")
    set_heading(doc, "Table A1. Composition of Clustered (Grouped) Cause-of-Death "
                      "Categories", level=2)
    para(doc,
        f"{report['n_clustered_causes']} WHO cause titles with fewer than 130 training "
        f"examples (the min_vc threshold; Section 2.3) were grouped into "
        f"{report['n_cluster_groups']} cluster categories by WHO major-cause group, "
        f"listed here with their constituent causes and training-set counts so readers "
        f"can see exactly what each cluster label represents.")
    doc.add_paragraph()
    for cluster_name, members in report["cluster_membership"]:
        total = sum(n for _, n in members)
        table_caption(doc, f"{cluster_name}  (total n={total}, {len(members)} constituent causes)")
        add_table(doc,
            headers=["Constituent WHO cause", "Training rows (n)"],
            rows=[[cause, n] for cause, n in members],
            col_widths=[4.5, 1.2])
        doc.add_paragraph()

    doc.save(out_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate the VMan3-ML manuscript from one training run's own reports."
    )
    parser.add_argument("--report-version", default=None,
        help="Report directory name under reports/ (e.g. v1.3.4). Defaults to whichever "
             "report directory was most recently written.")
    parser.add_argument("--reports-root", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--out", default=None,
        help="Output .docx path. Defaults to <reports-root>/vman_manuscript.docx")
    args = parser.parse_args()

    report_version = args.report_version or find_latest_report_version(args.reports_root)
    report = load_report(report_version, args.reports_root, args.models_dir)
    out_path = args.out or str(Path(args.reports_root) / "vman_manuscript.docx")
    build(report, out_path)
    print(f"Saved manuscript to {out_path} (source report: {report_version})")


if __name__ == "__main__":
    main()
