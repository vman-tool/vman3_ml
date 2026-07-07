"""
PCVA to WHO VA Cause-List Mapping
===================================
Primary API  : map_causelist(df, ...)         — ICD-code-based mapping (TZ, NG).
               map_ucod_text_to_who(df, ...)  — text-label-based mapping (ES).
Audit API    : export_mapping_excel(df, ...) — two-sheet audit workbook.
CLI          : python -m vman_ml.mapcauselist --who <path> --input <csv> [--output <xlsx>]

WHO columns added by both mapping functions:
  pcva_who_cod    – WHO target cause label  (e.g. "Malaria")
  pcva_who_id     – WHO VAS ID              (e.g. "VAs-01.05")
  pcva_who_major  – WHO Major Cause group   (e.g. "Infectious and parasitic diseases")
  pcva_who_broad  – WHO Broad Group         (e.g. "Communicable")

map_causelist() matching order (ICD codes):
  1. Manual review overrides   (clinically curated look-up table)
  2. Exact match               (full code including sub-code)
  3. Base match                (stem before the dot, e.g. KB21.0 → KB21)
  4. Partial match             (prefix overlap in either direction)
  5. Range match               (code falls within a WHO range, e.g. 1F40–1F4Z)

map_ucod_text_to_who() matching order (text labels, e.g. Eswatini):
  1. Hard overrides            (known abbreviations / alternate spellings)
  2. Exact normalised match    (case-insensitive, whitespace-collapsed)
  3. WHO title starts-with     (handles WHO footnote suffixes, e.g. "Tetanusa")
  4. Label starts-with WHO     (handles appended qualifiers, e.g. "Severe anaemia (iron deficiency)")
"""

from __future__ import annotations

import re
import argparse
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Resource path
# ---------------------------------------------------------------------------

WHO_CAUSELIST_PATH = Path(__file__).parent / 'resources' / 'who_target_list.csv'


# ---------------------------------------------------------------------------
# Manual review overrides
# ---------------------------------------------------------------------------
# Codes that cannot be matched automatically for clinical, coding, or
# data-entry reasons.  Key = base code (before the dot), upper-case.
# Value = (WHO VAS ID, WHO target cause label).
# ---------------------------------------------------------------------------
MANUAL_OVERRIDES: dict[str, tuple[str, str]] = {
    # Congenital malformations
    'LA02': ('VAs-10.06', 'Congenital malformation'),
    'LA04': ('VAs-10.06', 'Congenital malformation'),
    'LA8Z': ('VAs-10.06', 'Congenital malformation'),
    'LB16': ('VAs-10.06', 'Congenital malformation'),
    'LD2Z': ('VAs-10.06', 'Congenital malformation'),
    'LD9Z': ('VAs-10.06', 'Congenital malformation'),
    'DA0E': ('VAs-10.06', 'Congenital malformation'),

    # Non-communicable diseases outside standard NCD ranges
    '8D2Z': ('VAs-98', 'Other and unspecified non-communicable disease'),
    '8D64': ('VAs-98', 'Other and unspecified non-communicable disease'),
    'DB97': ('VAs-98', 'Other and unspecified non-communicable disease'),
    'ME10': ('VAs-98', 'Other and unspecified non-communicable disease'),
    'ME66': ('VAs-98', 'Other and unspecified non-communicable disease'),
    'MG44': ('VAs-98', 'Other and unspecified non-communicable disease'),
    'DB99': ('VAs-06.02', 'Liver cirrhosis'),

    # Renal
    'GC2Z': ('VAs-07.01', 'Renal failure'),

    # Sepsis
    'MA15': ('VAs-01.01', 'Sepsis'),

    # Acute abdomen
    'DB30': ('VAs-06.01', 'Acute abdomen'),
    'DA91': ('VAs-06.01', 'Acute abdomen'),
    'DD53': ('VAs-06.01', 'Acute abdomen'),
    'ME24': ('VAs-06.01', 'Acute abdomen'),

    # Anaemia
    '3A4Z': ('VAs-03.01', 'Severe anaemia'),

    # Diarrhoeal diseases (symptom code used as diagnosis)
    'ME05': ('VAs-01.04', 'Diarrheal diseases'),

    # Respiratory
    'CA2Z': ('VAs-05.01', 'Chronic obstructive pulmonary disease (COPD)'),
    'CB7Z': ('VAs-05.01', 'Chronic obstructive pulmonary disease (COPD)'),
    'AB0Z': ('VAs-01.02', 'Acute respiratory infection, including pneumonia'),

    # External causes
    'PF2Z': ('VAs-12.09', 'Assault'),
    'NA0Z': ('VAs-12.99', 'Other and unspecified external cause of death'),

    # Data-entry error (IF4Z is a typo for 1F4Z = Malaria unspecified)
    'IF4Z': ('VAs-01.05', 'Malaria'),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_base(code: str) -> str:
    """Return the stem before the dot, stripped and upper-cased."""
    return code.split('.')[0].strip().upper()


def _detect_icd_standard(icd_series: pd.Series) -> int:
    """Infer ICD-10 vs ICD-11 from a Series of code strings.

    ICD-10: letter then digit at position 0-1 (e.g. A09, J22)
    ICD-11: digit then letter (1G40) or letter+letter (CA00)
    Returns 10 or 11.
    """
    icd10_pat = re.compile(r'^[A-Z][0-9]', re.IGNORECASE)
    codes = (
        icd_series.dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.split(r'[/&]')
        .str[0]
        .str.strip()
    )
    codes = codes[codes.notna() & (codes != '') & (codes != 'NAN')]
    icd10_votes = codes.apply(lambda c: bool(icd10_pat.match(c))).sum()
    return 10 if icd10_votes / max(len(codes), 1) > 0.5 else 11


def _load_who_causelist(who_path=None) -> pd.DataFrame:
    """Load and normalise the WHO target cause-list CSV."""
    path = who_path or WHO_CAUSELIST_PATH
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        'vas-id':       'vas_id',
        'title  codes': 'who_title',
        'ICD-10 codes': 'icd10_who',
        'ICD-11 codes': 'icd11_who',
    })
    df['who_title'] = df['who_title'].astype(str).str.strip()
    return df


def _parse_who_codes(who_df: pd.DataFrame, standard: int) -> tuple[dict, list]:
    """Build prefix_map and range_map from the WHO cause-list DataFrame.

    prefix_map : { 'CODE_UPPER': (vas_id, title) }
    range_map  : [ (start, end, vas_id, title), ... ]
    """
    prefix_map: dict[str, tuple[str, str]] = {}
    range_map: list[tuple[str, str, str, str]] = []

    use_col = 'icd11_who' if standard == 11 else 'icd10_who'

    for _, row in who_df.iterrows():
        code_str = str(row.get(use_col, '') or '')
        tokens = re.split(r'[;,]', code_str)

        for token in tokens:
            token = token.strip()
            if not token:
                continue

            rm = re.match(
                r'([A-Z0-9]+(?:\.[A-Z0-9]+)?)\s*[-–]\s*([A-Z0-9]+(?:\.[A-Z0-9]+)?)',
                token,
            )
            if rm:
                start = rm.group(1).strip().upper()
                end   = rm.group(2).strip().upper()
                range_map.append((start, end, row['vas_id'], row['who_title']))
                prefix_map[start] = (row['vas_id'], row['who_title'])
            else:
                prefix_map[token.upper()] = (row['vas_id'], row['who_title'])

    return prefix_map, range_map


def _match_single_part(
    part: str,
    prefix_map: dict,
    range_map: list,
) -> tuple[str | None, str | None, str | None]:
    """Try to match one code fragment (no slashes/ampersands) against WHO structures."""
    part = part.strip().upper()
    base = _get_base(part)

    # 1. Exact match
    if part in prefix_map:
        return prefix_map[part][0], prefix_map[part][1], 'Exact'

    # 2. Base match
    if base in prefix_map:
        return prefix_map[base][0], prefix_map[base][1], 'Base match'

    # 3. Partial match
    for key, val in prefix_map.items():
        if part.startswith(key) or key.startswith(base):
            return val[0], val[1], 'Partial match'

    # 4. Range match
    for (start, end, vas_id, title) in range_map:
        if _get_base(start) <= base <= _get_base(end):
            return vas_id, title, 'Range match'

    return None, None, None


def _match_code(
    icd_code: str,
    prefix_map: dict,
    range_map: list,
) -> tuple[str | None, str | None, str]:
    """Full matching pipeline for one ICD code.

    Handles compound codes (joined by '/' or '&') by trying each part.
    Returns (who_vas_id, who_title, match_method).
    """
    if not icd_code or icd_code in ('nan', ''):
        return None, None, 'No code'

    parts = re.split(r'[/&]', icd_code)

    # Manual override (check base of every compound part first)
    for part in parts:
        base = _get_base(part.strip())
        if base in MANUAL_OVERRIDES:
            v, t = MANUAL_OVERRIDES[base]
            return v, t, 'Manual review'

    # Automated matching
    for part in parts:
        vas_id, title, method = _match_single_part(part, prefix_map, range_map)
        if vas_id:
            return vas_id, title, method

    return 'UNMATCHED', None, 'Unmatched'


# ---------------------------------------------------------------------------
# Text-label matching (for datasets without ICD codes, e.g. Eswatini)
# ---------------------------------------------------------------------------

# Hard overrides for ES labels that can't be matched by prefix rules alone.
# Key: normalised source label.  Value: (WHO VAS ID, WHO target cause title).
_TEXT_UCOD_OVERRIDES: dict[str, tuple[str, str]] = {
    'covid-19':                   ('VAs-01.13', 'Coronavirus disease (COVID19)'),
    'chronic obstructive (copd)': ('VAs-05.01', 'Chronic obstructive pulmonary disease (COPD)'),
}


def _normalize_cause_label(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return re.sub(r'\s+', ' ', str(text).strip()).lower()


def map_ucod_text_to_who(
    df: pd.DataFrame,
    ucod_col: str = 'pcva_ucod',
    who_path=None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Match pcva_ucod text labels to WHO VAS target causes by normalised string matching.

    Designed for datasets (e.g. Eswatini) that already use WHO-compatible cause
    labels in pcva_ucod but lack pcva_ucod_icd.

    Matching order per unique label:
      1. Hard override  (_TEXT_UCOD_OVERRIDES — known abbreviations/alternate spellings)
      2. Exact match    (after lowercasing and whitespace normalisation)
      3. WHO-starts-with-label  (handles WHO footnote suffixes: "Tetanusa" → "Tetanus")
      4. Label-starts-with-WHO  (handles extra qualifiers: "Severe anaemia (iron deficiency)")

    Unmatched labels receive None in all four WHO columns and are filtered by
    the training quality filter as unknown causes.
    """
    if ucod_col not in df.columns:
        return df

    who_df = _load_who_causelist(who_path)

    # Build normalised WHO lookup: norm_title → (vas_id, title, major, broad)
    who_lookup: dict[str, tuple[str, str, str, str]] = {}
    for _, row in who_df.iterrows():
        norm = _normalize_cause_label(row['who_title'])
        who_lookup[norm] = (
            row['vas_id'],
            str(row['who_title']).strip(),
            str(row.get('WHO Major Cause', '') or '').strip(),
            str(row.get('Broad Group',     '') or '').strip(),
        )

    # Build override lookup: vas_id → full row tuple
    override_by_id: dict[str, tuple[str, str, str, str]] = {
        vas_id: next(
            (v for v in who_lookup.values() if v[0] == vas_id),
            (vas_id, title, '', ''),
        )
        for _, (vas_id, title) in _TEXT_UCOD_OVERRIDES.items()
    }

    def _match(label: str) -> tuple[str | None, str | None, str | None, str | None]:
        if pd.isna(label):
            return None, None, None, None
        norm = _normalize_cause_label(label)

        # 1. Hard override
        if norm in _TEXT_UCOD_OVERRIDES:
            vas_id, _ = _TEXT_UCOD_OVERRIDES[norm]
            return override_by_id.get(vas_id, (vas_id, None, None, None))

        # 2. Exact
        if norm in who_lookup:
            return who_lookup[norm]

        # 3. WHO title starts with label (e.g. "Acute cardiac disease" ~ "Acute cardiac diseasec")
        for wn, row_tuple in who_lookup.items():
            if wn.startswith(norm) and len(wn) - len(norm) <= 5:
                return row_tuple

        # 4. Label starts with WHO title (e.g. "Severe anaemia (iron deficiency)" ~ "Severe anaemia")
        for wn, row_tuple in who_lookup.items():
            if len(wn) >= 6 and norm.startswith(wn):
                return row_tuple

        return None, None, None, None

    rows = []
    for label in df[ucod_col].unique():
        vid, wt, wmaj, wbro = _match(label)
        rows.append({
            ucod_col:         label,
            'pcva_who_cod':   wt,
            'pcva_who_id':    vid,
            'pcva_who_major': wmaj,
            'pcva_who_broad': wbro,
        })

    lookup_df = pd.DataFrame(rows)
    who_cols  = ['pcva_who_cod', 'pcva_who_id', 'pcva_who_major', 'pcva_who_broad']
    df = df.drop(columns=[c for c in who_cols if c in df.columns]).copy()
    df = df.merge(lookup_df, on=ucod_col, how='left')

    if verbose:
        mapped   = df['pcva_who_cod'].notna().sum()
        total    = len(df)
        unmatched = lookup_df[lookup_df['pcva_who_cod'].isna()][ucod_col].tolist()
        print(f"WHO text mapping: {mapped}/{total} rows mapped ({mapped/total:.1%})")
        if unmatched:
            print(f"  Unmatched labels ({len(unmatched)}): {unmatched}")

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_causelist(
    df: pd.DataFrame,
    icd_col: str = 'pcva_ucod_icd',
    who_path=None,
    standard: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Enrich a DataFrame with four WHO standardised cause columns.

    Reads the ICD code from `icd_col`, matches it against the WHO VA target
    cause-list, and adds:
      pcva_who_cod    – WHO target cause label
      pcva_who_id     – WHO VAS ID
      pcva_who_major  – WHO Major Cause group
      pcva_who_broad  – WHO Broad Group

    Records whose ICD code cannot be matched receive None in all four columns
    and are treated as unknown causes during training (filtered by quality filter).

    Args:
        df:        Input DataFrame containing at least `icd_col`.
        icd_col:   Column with ICD-10 or ICD-11 codes. Default 'pcva_ucod_icd'.
        who_path:  Path to the WHO target cause-list CSV. Defaults to the
                   bundled resources/who_target_list.csv.
        standard:  Force ICD standard (10 or 11). Auto-detected when None.
        verbose:   Print mapping statistics.

    Returns:
        DataFrame with the four new columns appended.
    """
    if icd_col not in df.columns:
        if verbose:
            print(f"Warning: ICD column '{icd_col}' not found — skipping WHO causelist mapping.")
        return df

    who_df = _load_who_causelist(who_path)

    icd_series = df[icd_col].astype(str).str.strip()
    if standard is None:
        standard = _detect_icd_standard(icd_series)
        if verbose:
            print(f"WHO causelist: auto-detected ICD-{standard}")

    prefix_map, range_map = _parse_who_codes(who_df, standard)

    vas_meta: dict[str, tuple[str, str]] = {
        row['vas_id']: (
            str(row.get('WHO Major Cause', '') or '').strip(),
            str(row.get('Broad Group',     '') or '').strip(),
        )
        for _, row in who_df.iterrows()
    }

    # Build per-unique-code lookup table
    rows = []
    for code in icd_series.unique():
        vas_id, who_title, _ = _match_code(code, prefix_map, range_map)
        if vas_id and vas_id != 'UNMATCHED':
            major, broad = vas_meta.get(vas_id, ('', ''))
        else:
            who_title = None
            vas_id    = None
            major     = None
            broad     = None
        rows.append({
            icd_col:          code,
            'pcva_who_cod':   who_title,
            'pcva_who_id':    vas_id,
            'pcva_who_major': major,
            'pcva_who_broad': broad,
        })

    lookup_df = pd.DataFrame(rows)

    # Drop any pre-existing WHO columns before merging
    who_cols = ['pcva_who_cod', 'pcva_who_id', 'pcva_who_major', 'pcva_who_broad']
    df = df.drop(columns=[c for c in who_cols if c in df.columns]).copy()
    df = df.merge(lookup_df, on=icd_col, how='left')

    if verbose:
        mapped = df['pcva_who_cod'].notna().sum()
        total  = len(df)
        unmatched_codes = lookup_df[lookup_df['pcva_who_cod'].isna()][icd_col].tolist()
        print(f"WHO causelist: {mapped}/{total} rows mapped ({mapped/total:.1%})")
        if unmatched_codes:
            print(f"  Unmatched codes ({len(unmatched_codes)}): {unmatched_codes[:10]}")

    return df


def build_mapping_report(
    df: pd.DataFrame,
    icd_col: str = 'pcva_ucod_icd',
    ucod_col: str = 'pcva_ucod',
    who_path=None,
    standard: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build mapping and summary DataFrames for audit/export.

    Returns (detail_df, summary_df).
    """
    if icd_col not in df.columns:
        raise ValueError(f"ICD column '{icd_col}' not found in DataFrame.")

    who_df = _load_who_causelist(who_path)
    icd_series = df[icd_col].astype(str).str.strip()
    if standard is None:
        standard = _detect_icd_standard(icd_series)

    prefix_map, range_map = _parse_who_codes(who_df, standard)
    vas_meta = {
        row['vas_id']: (
            str(row.get('WHO Major Cause', '') or '').strip(),
            str(row.get('Broad Group',     '') or '').strip(),
        )
        for _, row in who_df.iterrows()
    }

    counts = df.groupby(icd_col).size().rename('Case_Count')
    unique_df = df.drop_duplicates(subset=[icd_col])[[icd_col] + ([ucod_col] if ucod_col in df.columns else [])].copy()
    unique_df[icd_col] = unique_df[icd_col].astype(str).str.strip()

    detail_rows = []
    for _, row in unique_df.iterrows():
        code = row[icd_col]
        vas_id, who_title, method = _match_code(code, prefix_map, range_map)
        vid = vas_id or 'UNMATCHED'
        major, broad = vas_meta.get(vid, ('', ''))
        detail_rows.append({
            'PCVA_UCOD':        row.get(ucod_col, ''),
            'PCVA_ICD_Code':    code,
            'WHO_VAS_ID':       vid,
            'WHO_Target_Cause': who_title or '—',
            'WHO_Major_Cause':  major,
            'Broad_Group':      broad,
            'Match_Method':     method or 'No code',
        })

    detail_df = pd.DataFrame(detail_rows)
    detail_df = detail_df.merge(
        counts.reset_index().rename(columns={icd_col: 'PCVA_ICD_Code'}),
        on='PCVA_ICD_Code', how='left',
    ).sort_values(['WHO_VAS_ID', 'PCVA_ICD_Code']).reset_index(drop=True)

    summary_df = (
        detail_df
        .groupby(['WHO_VAS_ID', 'WHO_Target_Cause', 'WHO_Major_Cause', 'Broad_Group'], as_index=False)
        .agg(
            Unique_Codes=('PCVA_ICD_Code', 'count'),
            Total_Cases=('Case_Count', 'sum'),
        )
        .sort_values('WHO_VAS_ID')
    )

    return detail_df, summary_df


def export_mapping_excel(
    df: pd.DataFrame,
    output_path: str = 'who_mapping.xlsx',
    icd_col: str = 'pcva_ucod_icd',
    ucod_col: str = 'pcva_ucod',
    who_path=None,
    standard: int | None = None,
) -> None:
    """Write a two-sheet audit workbook (Sheet 1: full mapping, Sheet 2: summary)."""
    detail_df, summary_df = build_mapping_report(
        df, icd_col=icd_col, ucod_col=ucod_col, who_path=who_path, standard=standard,
    )

    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        detail_df.to_excel(writer, sheet_name='PCVA to WHO Mapping', index=False)
        summary_df.to_excel(writer, sheet_name='Summary by WHO Target', index=False)

    print(f"Mapping audit saved: {output_path}")
    matched   = (detail_df['WHO_VAS_ID'] != 'UNMATCHED').sum()
    unmatched = (detail_df['WHO_VAS_ID'] == 'UNMATCHED').sum()
    print(f"  Unique codes matched  : {matched}")
    print(f"  Unique codes unmatched: {unmatched}")
    if unmatched:
        print("  Unmatched codes:")
        print(detail_df[detail_df['WHO_VAS_ID'] == 'UNMATCHED'][['PCVA_UCOD', 'PCVA_ICD_Code']].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli():
    parser = argparse.ArgumentParser(
        description='Map PCVA ICD codes to WHO VA target cause-list.'
    )
    parser.add_argument('--who',      default=str(WHO_CAUSELIST_PATH),
                        help='Path to who_target_list.csv (default: bundled resource)')
    parser.add_argument('--input',    required=True,
                        help='Input CSV with at least a pcva_ucod_icd column')
    parser.add_argument('--icd-col',  default='pcva_ucod_icd', dest='icd_col',
                        help='Name of the ICD code column (default: pcva_ucod_icd)')
    parser.add_argument('--ucod-col', default='pcva_ucod',     dest='ucod_col',
                        help='Name of the cause-of-death label column (default: pcva_ucod)')
    parser.add_argument('--output',   default='who_mapping.xlsx',
                        help='Output Excel file (default: who_mapping.xlsx)')
    parser.add_argument('--standard', type=int, choices=[10, 11], default=None,
                        help='Force ICD standard (10 or 11); auto-detected when omitted')
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()

    import chardet
    with open(args.input, 'rb') as fh:
        enc = chardet.detect(fh.read(100_000))['encoding']
    df = pd.read_csv(args.input, encoding=enc, low_memory=False)

    export_mapping_excel(
        df,
        output_path=args.output,
        icd_col=args.icd_col,
        ucod_col=args.ucod_col,
        who_path=args.who,
        standard=args.standard,
    )


if __name__ == '__main__':
    _cli()
