# Four Practical Indicators for Real-Time Verbal Autopsy Data Quality Assessment: A Cross-National Validation Study

**Authors:** Isaac Lyatuu¹

**Affiliation:** ¹ Ifakara Health Institute, Dar es Salaam, Tanzania

**Corresponding author:** Isaac Lyatuu, Ifakara Health Institute. Email: ilyatuu@ihi.or.tz

**Running title:** Real-time VA data quality indicators

**Keywords:** verbal autopsy, data quality, completeness, reliability, internal consistency, interview duration, WHO-VA instrument, mortality surveillance

---

## Abstract

**Background:** Verbal autopsy (VA) data quality is conventionally assessed after or during cause-of-death (CoD) assignment, a step that requires aggregated data and introduces delays in identifying and correcting data problems. No standardised, instrument-agnostic set of quality indicators has been published that can be applied to individual VA records in real time. We propose and validate four complementary indicators — the Informative Completeness Score (ICS), Respondent Reliability Score (RRS), Internal Consistency Index (ICI), and Average Interview Duration (AID) — that can be computed for a single VA record, enabling immediate feedback to interviewers and supervisors.

**Methods:** The four indicators were developed using field names and response codes defined in the 2016 and 2022 World Health Organization Verbal Autopsy (WHO-VA) instruments. We validated the indicators on three publicly accessible datasets: va_2022_es (Eswatini, n=380, 2022 WHO-VA), va_2016_tz (Tanzania, n=3,601, 2016 WHO-VA), and va_2022_ng (Nigeria, n=5,468, 2022 WHO-VA). Descriptive statistics, tier distributions, and per-rule violation frequencies were computed for each indicator.

**Results:** ICS was high across all datasets (means: Eswatini 96.4%, Tanzania 98.7%, Nigeria 98.7%), with occasional low-outlier records indicating interviewers who relied heavily on "don't know" or "refused" responses. RRS was computable for Eswatini (n=349, mean=82.4, 63.9% High tier) and Tanzania (n=3,546, mean=85.2, 70.1% High tier); the Nigeria dataset lacked the required relationship and date fields. ICI revealed that cough duration exceeding total illness duration (C4) was the most frequent logical error in both Eswatini (n=23, 6.1%) and Tanzania (n=28, 0.8%); Nigeria had the highest absolute number of C3 violations (fever duration > illness duration; n=115, 2.1%). AID was computable for Eswatini (median=27.3 min, n=281) and Tanzania (median=32.0 min, n=676); Nigeria's dataset did not include interview time fields.

**Conclusions:** The four proposed indicators are practical, transparent, and computable at the record level using standard WHO-VA fields. They enable real-time quality feedback before CoD assignment, do not require large samples, and are compatible with both the 2016 and 2022 WHO-VA instruments. Integration into data management platforms can substantially shorten the feedback loop between data collection and quality improvement.

---

## 1. Introduction

Verbal autopsy is the primary method for ascertaining causes of death in settings where vital registration systems are absent or incomplete — conditions that characterise much of sub-Saharan Africa, South Asia, and parts of Latin America [1,2]. A VA interview involves a structured questionnaire administered to a close relative or caregiver of a recently deceased person; the resulting data are used to assign a probable CoD through algorithmic methods such as InSilicoVA [3], InterVA [4], SmartVA [5], or physician-coded review.

The quality of CoD estimates depends directly on the quality of the underlying VA data. Yet formal quality assessment has typically been embedded in the CoD assignment workflow: data analysts or automated algorithms detect inconsistencies or implausible values when processing aggregated datasets, usually weeks or months after the interviews are conducted [6,7]. This approach has two key limitations. First, problems identified late are costly to rectify; the interviewer who recorded contradictory responses may no longer be accessible or may have continued the same error across dozens of additional interviews. Second, algorithmic CoD methods require a sufficient number of records to produce stable estimates; a single VA cannot be evaluated for "cause-of-death plausibility" in isolation. Consequently, individual-record quality problems may go undetected until they affect population-level estimates.

To our knowledge, no published framework provides a standardised set of quality indicators that can be applied to a single VA record, in real time, independently of CoD assignment, and in a format compatible with both major WHO-VA instrument versions. Existing quality checks tend to be embedded within specific software packages, are not consistently defined or published, and address only subsets of the data quality problem (for example, completeness alone, or logical range checks only).

We address this gap by proposing four complementary indicators — ICS, RRS, ICI, and AID — each targeting a distinct dimension of VA data quality. These indicators can be computed automatically as soon as an interview is uploaded, providing supervisors with actionable, record-level feedback. We demonstrate their application on three national datasets spanning two WHO-VA instrument versions.

---

## 2. Methods

### 2.1 Study Design

This is a methodological paper describing the development and cross-national application of four VA data quality indicators. The indicators are derived from the field structure of the WHO-VA instrument and applied to three existing datasets. No primary data collection was undertaken; accordingly, no ethics approval was required beyond that already governing the source datasets.

### 2.2 WHO-VA Instrument Compatibility

The WHO publishes the VA instrument in versioned releases; the two most widely used are the 2016 and 2022 versions [8,9]. Both share a common question-naming convention (e.g., `id10011` for interview start time, `id10019` for deceased sex, `id10023` for date of death). The indicators proposed here are constructed from field names present in both versions, making them applicable to data collected with either instrument. Where a field is absent or differently structured in a particular dataset version, computation of the relevant indicator is not possible; this is treated as a data limitation, not an indicator failure.

### 2.3 Indicator Definitions

#### 2.3.1 Informative Completeness Score (ICS)

The ICS quantifies the proportion of binary (yes/no) questions answered informatively — that is, with a definitive "yes" or "no" rather than "don't know" (dk) or "refused" (ref). Let $B_i$ be the set of binary-response fields for record $i$ that received any response (excluding system-missing), and let $I_i \subseteq B_i$ be the subset answered "yes" or "no". The ICS is:

$$\text{ICS}_i = \frac{|I_i|}{|B_i|} \times 100$$

ICS ranges from 0% (all responses are "don't know" or "refused") to 100% (all binary responses are informative). A record with ICS=100% does not imply high accuracy — a respondent may confidently give wrong answers — but a systematically low ICS signals that the respondent could not or did not engage meaningfully with the interview. ICS is computed over all question fields whose names begin with the standard WHO-VA prefix `id`, restricting to rows with at least one binary response.

#### 2.3.2 Respondent Reliability Score (RRS)

The RRS is a weighted composite of four factors that are empirically associated with the reliability of proxy-reported information: relationship to the deceased, presence at death, recall period, and respondent literacy. Each factor is scored independently and summed to a 0–100 scale.

| Component | Field(s) | Maximum Points | Scoring |
|-----------|----------|---------------|---------|
| Relationship (*W*_rel) | id10008 | 40 | Spouse/parent/child: 40; family member: 20; other: 10 |
| Presence at death (*W*_prox) | id10009 | 30 | Yes: 30; No: 15 |
| Recall period (*W*_rec) | id10023, id10012 | 20 | <90 days: 20; 90–179 days: 15; 180–364 days: 10; ≥365 days: 0 |
| Literacy/education (*W*_edu) | id10064, id10063 | 10 | Literate or secondary+ education: 10; No formal education or illiterate: 5 |

**Total RRS = *W*_rel + *W*_prox + *W*_rec + *W*_edu**

Tier classification: High (RRS ≥ 80), Moderate (50–79), Low (< 50). Records for which the death date or interview date cannot be parsed are excluded from RRS computation, as the recall period component cannot be determined. Like ICS, RRS can be computed for a single record.

#### 2.3.3 Internal Consistency Index (ICI)

The ICI identifies logical contradictions within a VA record — cross-field combinations that are biologically impossible or implausible. Six consistency rules are evaluated per record:

| Rule | Fields | Description |
|------|--------|-------------|
| C1 | id10019, id10305 | Pregnancy reported for a male decedent |
| C2 | id10153, id10157 | Blood reported in stool/urine without cough |
| C3 | id10120, id10148 | Fever duration (days) exceeds total illness duration (days) |
| C4 | id10120, id10154 | Cough duration exceeds total illness duration |
| C5 | id10120, id10182 | Diarrhoea duration exceeds total illness duration |
| C6 | id10120, id10161 | Breathlessness duration exceeds total illness duration |

Each violated rule increments the error count for that record. The ICI at the dataset or interviewer level is defined as:

$$\text{ICI} = \frac{\text{records with zero violations}}{\text{total records}} \times 100$$

Tier classification: Excellent (ICI ≥ 90%), Good (70–89%), Critical (< 70%). At the individual record level, each violation is flagged directly. The ICI is most informative when stratified by interviewer, as systematic violations point to training gaps with a specific interviewer rather than random errors distributed across a team.

#### 2.3.4 Average Interview Duration (AID)

The AID is the elapsed time in minutes between interview start (id10011) and interview end (id10481). Duration is computed as:

$$\text{AID}_i = \frac{(\text{id10481}_i - \text{id10011}_i)}{60} \text{ minutes}$$

Records with non-positive or implausible durations (≥ 480 minutes, i.e., 8 hours) are excluded from analysis. The 480-minute threshold was chosen to filter multi-day data entry artefacts while retaining genuinely long interviews. Median AID is the preferred summary statistic due to skewness caused by outliers and data entry errors; the mean and percentile distribution are reported to characterise the full range.

An important limitation applies to the 2016 WHO-VA instrument: the fields `id10011` and `id10481` store only the time of day (HH:MM:SS) without a date component. For such datasets, duration computation is possible only when start and end times fall on the same calendar day; a midnight crossing cannot be detected and would yield an erroneous negative duration (handled by modular addition of 1,440 minutes), and the 480-minute cap prevents inclusion of cross-day artefacts. In the 2022 instrument, these fields store full ISO 8601 datetimes, allowing duration computation across date boundaries.

### 2.4 Datasets

Three national VA datasets were used to illustrate indicator performance:

**Eswatini (va_2022_es):** A dataset of 380 VA records collected using the 2022 WHO-VA instrument. All four indicators were computable; interview start and end times were stored as full ISO 8601 datetimes with timezone offsets.

**Tanzania (va_2016_tz):** A dataset of 3,601 VA records collected using the 2016 WHO-VA instrument. ICS, RRS, and ICI were fully computable. AID was computed with the time-only limitation described above; records with cross-midnight or multi-day patterns are excluded by the 480-minute cap. The dataset contained duplicate column headers for a subset of fields; the first occurrence was retained.

**Nigeria (va_2022_ng):** A dataset of 5,468 VA records collected using the 2022 WHO-VA instrument. ICS and ICI were fully computable. The fields required for RRS (id10008, relationship to deceased; id10012, interview date) and AID (id10011, interview start time; id10481, interview end time) were absent from this dataset, precluding computation of those two indicators.

### 2.5 Statistical Analysis

Descriptive statistics (mean, median, standard deviation, minimum, maximum, 5th and 95th percentiles) were computed for ICS and AID. For RRS, tier distributions (High/Moderate/Low) were reported in addition to continuous summary statistics. For ICI, the proportion of records passing all consistency checks and the absolute count of violations per rule were reported. All analyses were performed in Python 3.11 using pandas and NumPy. All indicator computations are designed to be reproducible from the raw CSV exports of WHO-VA instruments.

---

## 3. Results

### 3.1 Dataset Overview

Table 1 summarises the three datasets.

**Table 1. Dataset characteristics**

| Dataset | Country | WHO-VA version | N | AID computable | RRS computable |
|---------|---------|---------------|---|---------------|---------------|
| va_2022_es | Eswatini | 2022 | 380 | Yes (full datetime) | Yes |
| va_2016_tz | Tanzania | 2016 | 3,601 | Partial (time-only) | Yes |
| va_2022_ng | Nigeria | 2022 | 5,468 | No (fields absent) | No |

### 3.2 Informative Completeness Score (ICS)

ICS was computed for all records in all three datasets (Table 2). All three datasets showed median ICS of ≥ 98.8%, indicating that the vast majority of binary questions received definitive responses.

**Table 2. ICS results by dataset**

| Dataset | N | Mean (%) | Median (%) | SD | Min (%) | 5th pct (%) | 95th pct (%) |
|---------|---|---------|-----------|-----|--------|------------|-------------|
| Eswatini (2022) | 380 | 96.4 | 98.8 | 6.6 | 25.0 | 86.4 | 100.0 |
| Tanzania (2016) | 3,601 | 98.7 | 100.0 | 3.1 | 48.7 | 94.2 | 100.0 |
| Nigeria (2022) | 5,468 | 98.7 | 100.0 | 3.0 | 8.4 | 94.7 | 100.0 |

Eswatini showed slightly greater variability (SD=6.6%) and a lower minimum (25.0%) than the African datasets, consistent with the smaller sample size and the presence of records where respondents answered a substantial proportion of binary questions with "don't know". Tanzania and Nigeria, despite differing instrument versions, produced nearly identical ICS distributions (mean=98.7%, median=100%), suggesting high respondent engagement with binary questions in both programmes.

The low-end tail (ICS < 86%) in Eswatini represents 5% of records and may indicate cases where the respondent had limited direct knowledge of the circumstances of death, or where language or comprehension barriers reduced informative response rates. These records warrant targeted interviewer or supervisor follow-up.

### 3.3 Respondent Reliability Score (RRS)

RRS was computed for Eswatini (n=349 of 380; 31 records excluded due to missing or unparseable death date) and Tanzania (n=3,546 of 3,601; 55 records excluded) (Table 3, Table 4). Across both datasets, the mean RRS exceeded 80, placing the average respondent in the High reliability tier.

**Table 3. RRS continuous summary by dataset**

| Dataset | N (eligible) | Mean | Median | SD | Min | Max |
|---------|-------------|------|--------|-----|-----|-----|
| Eswatini (2022) | 349 | 82.4 | 87 | 12.6 | 35 | 100 |
| Tanzania (2016) | 3,546 | 85.2 | 90 | 12.5 | 40 | 100 |

**Table 4. RRS tier distribution by dataset**

| Dataset | High (≥ 80) | Moderate (50–79) | Low (< 50) |
|---------|------------|----------------|-----------|
| Eswatini (2022) | 63.9% | 35.0% | 1.1% |
| Tanzania (2016) | 70.1% | 29.8% | 0.2% |

Tanzania demonstrated a higher proportion of High-tier respondents (70.1% vs 63.9%) and a higher mean and median RRS. The Low-tier proportion was small in both datasets (<1.1%), suggesting that respondents with very low reliability scores are uncommon in well-managed programmes.

The Nigeria dataset was excluded from RRS computation because the relationship-to-deceased field (id10008) and interview date field (id10012) were absent from the exported data. This illustrates that RRS requires specific fields that may not be available in all dataset exports and highlights the importance of ensuring data export completeness.

### 3.4 Internal Consistency Index (ICI)

The ICI and per-rule violation counts are presented in Table 5.

**Table 5. ICI results and rule-level violation counts**

| Rule | Description | Eswatini (n=380) | Tanzania (n=3,601) | Nigeria (n=5,468) |
|------|-------------|--------------|-------------------|------------------|
| C1 | Pregnancy in male | 0 (0.0%) | 0 (0.0%) | 1 (0.02%) |
| C2 | Blood without cough | 0 (0.0%) | 0 (0.0%) | 0 (0.0%) |
| C3 | Fever dur > illness | 4 (1.1%) | 28 (0.8%) | 115 (2.1%) |
| C4 | Cough dur > illness | 23 (6.1%) | 28 (0.8%) | 10 (0.2%) |
| C5 | Diarrhoea dur > illness | 7 (1.8%) | 4 (0.1%) | 11 (0.2%) |
| C6 | Breathlessness dur > illness | 8 (2.1%) | 15 (0.4%) | 12 (0.2%) |
| **Records passing all rules** | | **91.8%** | **98.5%** | **97.5%** |
| Mean errors per record | | 0.111 | 0.021 | 0.027 |

Eswatini achieved an ICI of 91.8% (Excellent tier), Tanzania 98.5% (Excellent), and Nigeria 97.5% (Excellent). No dataset fell below the Good threshold (70%). The most striking finding is the pattern of C4 violations in Eswatini: 23 records (6.1%) had a cough duration exceeding the reported total illness duration, compared to 28 records in Tanzania's much larger dataset (0.8%). This discrepancy may reflect a data entry convention in the Eswatini dataset, a structural difference in how interviewers were trained to elicit duration responses, or ambiguity in how respondents distinguished disease-specific symptom duration from total illness episode duration.

The C1 rule (pregnancy in a male decedent) was virtually absent across all datasets (only 1 case in Nigeria), consistent with this being a clear biological impossibility that interviewers and data entry systems readily avoid. C2 (blood reported without cough) was also absent across all datasets, suggesting that the rule as currently defined may reflect a low-frequency physiological pairing rather than a common data entry error, or that interviewers naturally avoid this combination.

The C3 violation (fever duration > illness duration) was most prevalent in Nigeria (n=115, 2.1%), pointing to a potential systematic data entry pattern worth investigating at the interviewer level. When stratified by interviewer, the ICI would likely reveal whether these violations are concentrated in specific enumerators.

### 3.5 Average Interview Duration (AID)

AID results are presented in Table 6. The AID for Eswatini was computed from full ISO 8601 datetime fields; for Tanzania, from time-of-day fields (see Methods section 2.3.4 for the associated limitation).

**Table 6. AID results by dataset**

| Dataset | N (valid) | Mean (min) | Median (min) | SD | 5th pct (min) | 95th pct (min) |
|---------|----------|-----------|-------------|-----|--------------|---------------|
| Eswatini (2022) | 281 | 56.9 | 27.3 | 79.1 | 11.7 | 235.1 |
| Tanzania (2016)* | 676 | 36.1 | 32.0 | 27.5 | 15.0 | 62.0 |

\* Time-of-day fields only; cross-midnight interviews are excluded by the 480-minute cap and cannot be detected.

The median interview duration in Eswatini (27.3 minutes) is consistent with expected VA interview lengths for a standard instrument administered by trained interviewers. The high mean (56.9 minutes) and wide standard deviation (79.1 minutes) reflect a right-skewed distribution with a substantial proportion of records reporting very long durations (95th percentile = 235.1 minutes). In datasets storing full datetimes, multi-day intervals (e.g., where the interview was paused and resumed the following day) can produce apparent durations of many hours; these are captured before the 480-minute exclusion and inflate the mean.

Tanzania's AID distribution is more concentrated (median=32.0 min, SD=27.5 min), which is partly an artefact of the time-only field limitation: cross-midnight interviews are excluded, and very long within-day interviews are capped. The 95th percentile of 62.0 minutes is a plausible upper bound for a within-day VA interview.

Nigeria's dataset did not include the interview start and end time fields (id10011, id10481), preventing AID computation. Programmes using the 2022 WHO-VA instrument that wish to monitor AID should verify that these fields are included in data exports.

Collectively, an AID below approximately 10–12 minutes (5th percentile across both datasets) should trigger supervisor review, as it may indicate that sections of the questionnaire were skipped. Similarly, records with AID above the 95th percentile may represent multi-session interviews or data entry artefacts.

---

## 4. Discussion

### 4.1 Principal Findings

We have proposed and applied four complementary VA data quality indicators that can be computed from a single interview record, without requiring CoD assignment or aggregated data. The indicators were applicable to data collected with both the 2016 and 2022 WHO-VA instruments and performed consistently across datasets from three countries. Across all datasets where indicators were computable, the majority of records achieved satisfactory performance (ICS > 94%, RRS High tier, ICI pass), confirming that these datasets represent well-conducted field programmes. At the same time, each indicator identified a non-trivial subset of records or patterns worthy of follow-up, demonstrating that the indicators add value even in high-quality programmes.

### 4.2 Advantages Over Post-Hoc Quality Assessment

The central argument for this indicator framework is temporal: quality problems identified before CoD assignment can still be corrected. A supervisor alerted in real time that a specific interviewer is generating a disproportionate number of C4 violations (cough duration > illness duration) can provide targeted retraining. A data manager notified that 15% of today's uploads have ICS below 80% can investigate the cause — interviewer fatigue, a difficult community, or a software display problem — before the situation worsens.

In contrast, quality issues identified during CoD coding may require the original respondent to be re-contacted, which is often logistically impossible months after the interview. Some errors, such as contradictory duration fields, may be structurally unresolvable and require the record to be flagged or excluded.

Moreover, unlike CoD assignment algorithms which require a population distribution to produce meaningful estimates (e.g., InSilicoVA requires at least 100 cases for stable output [3]), the indicators proposed here are fully interpretable at the level of a single record. A programme with 10 records can assess ICS, RRS, and ICI immediately; it need not wait for a sufficient sample to run a CoD algorithm.

### 4.3 Complementarity of the Four Indicators

The four indicators measure distinct aspects of quality that are not reducible to one another. A record can score perfectly on ICS (all binary questions answered definitively) yet still contain internal contradictions (low ICI) because the respondent confidently gave inconsistent answers. An interview can be internally consistent yet reflect an unreliable respondent (low RRS) — for example, a distant acquaintance interviewed years after the death. An interview can be conducted by a highly reliable respondent and contain no logical errors, yet be completed in three minutes, suggesting that key sections were skipped (low AID). The simultaneous availability of all four indicators gives supervisors a multidimensional view of record quality.

### 4.4 Dataset-Specific Observations

The absent fields in Nigeria's dataset (id10008, id10011, id10012, id10481) underscore that the indicator framework's utility depends on data export completeness. Field absence is itself a data quality signal: if a programme is not capturing interview timestamps, the AID indicator cannot serve its monitoring function. Programmes should ensure that all instrument fields, including administrative and timing fields, are exported alongside the interview responses.

The anomalously high C4 violation rate in Eswatini compared to Tanzania and Nigeria (6.1% vs 0.8% and 0.2%, respectively) illustrates that ICI performance is context-dependent. Possible explanations include: (1) interviewer training that did not sufficiently emphasise the distinction between symptom onset and total illness duration; (2) cultural or linguistic differences in how respondents conceptualise "illness duration" versus "symptom duration"; or (3) data entry conventions specific to the Eswatini programme. Investigating this pattern at the interviewer level (by stratifying ICI by interviewer ID) would be the appropriate next step.

### 4.5 Instrument Compatibility

A strength of this framework is compatibility with both the 2016 and 2022 WHO-VA instruments. The field names used for each indicator are defined consistently across versions. The primary instrument-version caveat is the AID computation: the 2016 instrument stores interview times as time-only values, while the 2022 instrument stores full datetimes. Programmes using the 2016 instrument should interpret AID results with the understanding that cross-midnight interviews are excluded, and median rather than mean duration should be reported.

Researchers considering application of these indicators to earlier instrument versions (pre-2016) should verify field name correspondence, as the question numbering convention evolved substantially before the 2016 release.

### 4.6 Limitations

Several limitations should be acknowledged. First, the thresholds defining ICS, RRS, and ICI tiers (e.g., ICI Excellent ≥ 90%, RRS High ≥ 80) are expert-derived and have not yet been empirically validated against CoD concordance data. Future work should assess whether records in different ICS or RRS tiers show systematically different CoD assignment outcomes in comparison with gold-standard CoD data. Second, the ICI rules (C1–C6) represent a subset of possible logical inconsistencies in the WHO-VA instrument; additional rules could be added to cover paediatric and neonatal sections, pregnancy-related death sections, and age-sex combinations for other conditions. Third, RRS is limited to datasets that capture respondent characteristics; programmes that do not record respondent relationship or death date will find this indicator inapplicable. Fourth, the datasets used for validation are not random samples and may not represent the full range of data quality conditions in their respective countries.

### 4.7 Future Directions

The indicator framework is intended to be extensible. Future work should: (1) validate tier thresholds against linked CoD data; (2) expand the ICI rule set to cover additional sections of the WHO-VA instrument (neonatal, under-5, and maternal modules); (3) investigate machine-learning extensions that learn anomalous response patterns from historical data and flag records that deviate from expected distributions; (4) develop composite data quality scores that aggregate across the four indicators; and (5) integrate the indicators into field data collection platforms to provide immediate feedback to enumerators.

---

## 5. Conclusion

We propose four complementary indicators — ICS, RRS, ICI, and AID — that collectively address completeness, respondent reliability, logical consistency, and interview process quality in verbal autopsy data. Unlike existing approaches that assess quality as a byproduct of CoD assignment, these indicators are designed for real-time application to individual records, enabling supervisors to identify and act on quality problems during data collection rather than after it. The indicators are compatible with both the 2016 and 2022 WHO-VA instruments, require no additional data beyond the standard instrument, and can be computed automatically by data management systems. Cross-national application to datasets from Eswatini, Tanzania, and Nigeria demonstrated consistent performance and identified context-specific patterns worthy of programmatic attention. We encourage their adoption as a standard component of VA data management workflows.

---

## Declarations

**Conflict of interest:** The author declares no conflict of interest.

**Funding:** No specific funding was received for this methodological study.

**Data availability:** The indicator computation algorithms are implemented in the vman3 data management platform (Ifakara Health Institute). The three validation datasets used in this study are described by their respective data custodians; access conditions are determined by those custodians.

**Author contributions:** IL conceptualised the indicator framework, implemented the computational algorithms, performed the validation analysis, and wrote the manuscript.

---

## References

1. Mikkelsen L, Phillips DE, AbouZahr C, et al. A global assessment of civil registration and vital statistics systems: monitoring data quality and progress. *Lancet*. 2015;386(10001):1395–1406.

2. Fottrell E, Byass P. Verbal autopsy: methods in transition. *Epidemiologic Reviews*. 2010;32(1):38–55.

3. McCormick TH, Li ZR, Calvert C, et al. Probabilistic cause-of-death assignment using verbal autopsies. *Journal of the American Statistical Association*. 2016;111(515):1073–1083.

4. Byass P, Calvert C, Miiro-Nakiyingi J, et al. InterVA-4 as a public health tool for measuring HIV/AIDS mortality: a validation study from five African countries. *Global Health Action*. 2013;6:22448.

5. Serina P, Riley I, Stewart A, et al. A shortened verbal autopsy instrument for use in routine mortality surveillance systems. *PLOS ONE*. 2015;10(6):e0129683.

6. Sankoh O, Byass P. The INDEPTH Network: new developments and future directions. *Global Health Action*. 2012;5:1–5.

7. Oza S, Lawn JE, Hogan DR, Mathers C, Cousens SN. Neonatal cause-of-death estimates for the early and late neonatal periods for 194 countries: 2000–2013. *Bulletin of the World Health Organization*. 2015;93(1):19–28.

8. World Health Organization. *Verbal Autopsy Standards: The 2016 WHO Verbal Autopsy Instrument*. Geneva: WHO; 2016.

9. World Health Organization. *WHO Verbal Autopsy Instrument 2022*. Geneva: WHO; 2022.

10. Salter ML, Maybury TS, Nicol E, Byass P. The ALPHA Network study on causes of death in sub-Saharan Africa: the InterVA method revisited. *Global Health Action*. 2020;13(1):1833677.

11. Nichols EK, Byass P, Chandramohan D, et al. The WHO 2016 verbal autopsy instrument: an international standard suitable for automated analysis by InterVA, InSilicoVA, and TARIFF 2.0. *PLOS Medicine*. 2018;15(1):e1002486.

12. Iyaniwura CA, Yussuf Q. Facilitators of and constraints to health facility delivery in rural Ogun State, Nigeria. *African Journal of Reproductive Health*. 2009;13(3):51–62.

13. Garrib A, Jaffar S, Knight S, Bradshaw D, Bennish ML. Rates and causes of child mortality in an area of high HIV prevalence in rural South Africa. *Tropical Medicine & International Health*. 2006;11(12):1841–1848.

14. AbouZahr C, de Savigny D, Mikkelsen L, et al. Civil registration and vital statistics: progress in the data revolution for counting and accountability. *Lancet*. 2015;386(10001):1373–1385.

15. Clark SJ, McCormick TH. openVA: An open-source implementation of verbal autopsy algorithms. *PLOS Computational Biology*. 2019;15(9):e1007333.

---

## Appendix: Indicator Summary Reference

**Table A1. Indicator field requirements by WHO-VA version**

| Indicator | Required fields | 2016 compatible | 2022 compatible | Notes |
|-----------|----------------|-----------------|-----------------|-------|
| ICS | All id* binary fields | Yes | Yes | Requires ≥1 binary response |
| RRS | id10008, id10009, id10023, id10012 or id10011, id10064, id10063 | Yes | Yes | Date fields must be parseable |
| ICI | id10019, id10305, id10153, id10157, id10120, id10148, id10154, id10182, id10161 | Yes | Yes | Duration rules require numeric fields |
| AID | id10011, id10481 | Partial† | Yes | †2016 stores time-only; cross-midnight excluded |

**Table A2. RRS scoring rubric**

| Component | Field | Value | Points |
|-----------|-------|-------|--------|
| Relationship (*W*_rel) | id10008 | Spouse, parent, or child | 40 |
| | | Family member | 20 |
| | | Friend, health worker, official, or other | 10 |
| Presence (*W*_prox) | id10009 | Yes | 30 |
| | | No | 15 |
| Recall (*W*_rec) | id10023 – id10012 | < 90 days | 20 |
| | | 90–179 days | 15 |
| | | 180–364 days | 10 |
| | | ≥ 365 days | 0 |
| Literacy (*W*_edu) | id10064 / id10063 | Literate (yes) | 10 |
| | | Primary / secondary / higher education | 10 |
| | | Illiterate (no) | 5 |
| | | No formal education | 5 |

**Table A3. ICI consistency rules**

| Rule ID | Field(s) | Condition flagged as violation |
|---------|---------|-------------------------------|
| C1 | id10019, id10305 | id10019 = "male" AND id10305 = "yes" |
| C2 | id10153, id10157 | id10153 = "no" (no cough) AND id10157 = "yes" (blood) |
| C3 | id10120, id10148 | id10148 (fever days) > id10120 (illness days) |
| C4 | id10120, id10154 | id10154 (cough days) > id10120 (illness days) |
| C5 | id10120, id10182 | id10182 (diarrhoea days) > id10120 (illness days) |
| C6 | id10120, id10161 | id10161 (breathlessness days) > id10120 (illness days) |
