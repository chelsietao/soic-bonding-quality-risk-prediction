# SoIC Bonding Quality Risk Prediction

[繁體中文主管講解版](README.zh-TW.md) · [Presentation](presentations/SoIC_Process_Engineer_Briefing_ZH.pptx) · [Open in Colab](https://colab.research.google.com/github/chelsietao/soic-bonding-quality-risk-prediction/blob/main/notebooks/SoIC_Bonding_01_EDA.ipynb)

## 1. Project overview

This project evaluates inspection prioritization using synthetic bonding records, with a workflow covering data quality checks, time-based evaluation, predictive modeling and model interpretation.

The engineering question is: **when an inspection budget covers 20% of records, how many defects fall within the selected records?** The prediction point is after the recorded processing step and before quality inspection, assuming the seven numerical process features are available at that point.

On a 15,000-record chronological test set, the gradient boosting model selected 3,000 records containing **1,927 of 2,657 defects**. This corresponds to **72.5% recall and 64.2% precision**, with 730 defects remaining outside the selected subset.

**Scope:** the data are synthetic, the study demonstrates an analytical method rather than production qualification. Physical units, the exact bonding step and defect inspection criteria remain undocumented, so parameter values must not serve as actual SoIC recipe recommendations.

## 2. Relevance to the Process Engineer role

The role context is the user-supplied posting for **2027預聘_SoIC AP7P2(嘉義) & SoIC AZ(美國) 3DAS Process Engineer**, including process development and sustaining, equipment troubleshooting, collaboration, SPC/DOE knowledge, Python/AI capability and an AZ assignment following Chiayi training.

| Role responsibility | Evidence in this project | Further evidence needed |
|---|---|---|
| Process development and improvement | Nonlinear risk analysis and explicit evaluation of inspection coverage | Approved experiments and real process measurements |
| Daily sustaining and equipment troubleshooting | Tool/time stratification and a proposed investigation sequence | Equipment logs, chamber history, recipe revisions and maintenance records |
| Cross-functional communication | Traceable metrics, explanations and an engineering handoff plan | Actual collaboration outcomes and signed ownership records |
| SPC and DOE | A documented extension plan linking monitoring and hypothesis testing | Implemented control charts and executed experiments |
| AI and Python | Colab analysis, scikit-learn models, saved artifacts and reproducible input data | Production integration and monitoring |
| Technician mentoring and fab presence | Proposed standard investigation checklist | On-site practice and technician training evidence |
| Chiayi training and AZ assignment | Bilingual documentation can support technical handover | Personal availability, willingness and assignment readiness |

The project supports a discussion of transferable analytical skills, while manufacturing experience and relocation commitments require separate evidence.

## 3. Data contract

The fixed CSV contains **100,000 rows, 12 columns, 4,000 lots and three tools**, spanning January through March 2026 in UTC+08:00. There are **10,046 defect labels**, giving an overall defect rate of **10.046%**, with no missing cells or fully duplicated rows in the checked input.

| Field | Interpretation | Use | Unit status |
|---|---|---|---|
| `wafer_id` | Synthetic wafer identifier | Traceability | Identifier |
| `timestamp` | Processing timestamp | Chronological split and trends | UTC+08:00 |
| `lot_id` | Synthetic lot identifier | Lot separation | Identifier |
| `tool_id` | Synthetic equipment identifier | Exploratory stratification | Identifier |
| `temperature_mean` | Mean recorded process temperature | Model feature | Physical unit undocumented |
| `temperature_std` | Within-record temperature variation | Model feature | Same temperature scale, undocumented |
| `pressure_mean` | Mean recorded process pressure | Model feature | Physical unit undocumented |
| `process_time` | Recorded processing duration | Model feature | Duration unit undocumented |
| `alignment_error` | Pre-process alignment error | Model feature | Magnitude/direction and physical unit undocumented |
| `surface_particle_count` | Pre-process particle count | Model feature | Count, sampled area and detection threshold undocumented |
| `runs_since_maintenance` | Runs accumulated after maintenance | Model feature | Run count, maintenance reset definition undocumented |
| `defect_flag` | 0: no defect, 1: defect | Target | Binary, defect taxonomy undocumented |

The data-generation source code is unavailable, so the checked-in CSV and its SHA-256 checksum define the reproducible input. A mean temperature near 250 does not establish either Celsius or Fahrenheit, nor does it identify a real bonding or annealing recipe.

## 4. Exploratory analysis

### Parameter bins and defect rates

`pd.qcut(feature, q=10, duplicates="drop")` creates up to ten groups with similar record counts, followed by grouping and aggregation. Each plotted point uses the **within-bin feature median** on the X axis and **defect count / record count** on the Y axis, so one point summarizes a group rather than one wafer.

For the lowest temperature bin, 1,530 of 10,000 records have defects and the median is about 245.07. The point therefore represents **(245.07, 15.30%)**, with the temperature unit unresolved. Repeated integer values can reduce the number of bins, and the particle-count feature yields fewer groups.

### Correlation, equipment and time

Spearman correlation evaluates consistency of rank direction, and a near-zero coefficient can coexist with a U-shaped relationship. Alignment error has a positive rank relationship with the label, while temperature, pressure and duration require examination of their full range.

Tool-level observed defect rates are **13.46% for TOOL_C, 8.50% for TOOL_A and 8.18% for TOOL_B**, making TOOL_C a candidate for stratified investigation. Product mix, time period and recipe conditions may contribute to the difference, so the result does not isolate an equipment fault.

The late-period defect rate increases, which motivates chronological evaluation and future drift monitoring. The daily and seven-day trend plots are descriptive, with no SPC control limits or formal out-of-control rules implemented.

## 5. Split and modeling method

Lots are ordered by their earliest processing timestamps, then assigned in a 70%/15%/15% split. The notebook keeps each lot within one split and asserts strictly non-overlapping time ranges, which holds for this dataset.

| Dataset | Records | Lots | Defect rate | Purpose |
|---|---:|---:|---:|---|
| Train | 70,000 | 2,800 | 8.19% | Learn preprocessing and model parameters |
| Validation | 15,000 | 600 | 11.02% | Compare models and inspection policies |
| Test | 15,000 | 600 | 17.71% | Evaluate the fixed model and 20% policy |

The baseline is `SimpleImputer(strategy="median")`, followed by `StandardScaler()` and `LogisticRegression(max_iter=1000)`. Training data determine the imputation and scaling rules, with validation data only transformed afterward.

The second model is `HistGradientBoostingClassifier`, with `learning_rate=0.05`, `max_iter=200`, `max_leaf_nodes=15`, `min_samples_leaf=50`, `l2_regularization=1.0`, `early_stopping=False` and `random_state=42`. It uses the same seven numerical features, allowing threshold-based splits to express nonlinear patterns and interactions.

The project uses one selected tree configuration, with no exhaustive hyperparameter search. `wafer_id`, `lot_id`, `tool_id` and `timestamp` do not enter the model, and any future extension must recheck feature timing and leakage.

## 6. Metrics and inspection policy

| Metric | Definition | Engineering interpretation |
|---|---|---|
| Precision | TP / (TP + FP) | Fraction of selected records that contain defects |
| Recall | TP / (TP + FN) | Fraction of all labeled defects covered by selection |
| Alert rate | Selected records / all records | Fraction of records sent for inspection |
| AP | Average precision across ranking thresholds | Defect concentration toward the high-score end |
| ROC-AUC | Pairwise ranking discrimination, ties count half | Separation of defect and non-defect scores |
| Lift at 20% | Precision / defect prevalence | Enrichment relative to random selection |

The final policy selects the **highest-scoring 20% of a complete batch**, breaking score ties by original row order. This is an inspection fraction, distinct from a probability cutoff of 0.20, with no true label used during selection.

The earlier Logistic Regression threshold experiment demonstrates policy tradeoffs, with threshold 0.50 producing 250 alerts and 10.7% recall, while threshold 0.10 produces 5,209 alerts and 73.1% recall. These validation threshold results belong to the baseline model, so they must not be attributed to the final gradient boosting model.

## 7. Model comparison and test results

| Validation model | AP | ROC-AUC | Inspected | Defects covered | Defects outside selection | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.421 | 0.794 | 3,000 | 958 | 695 | 31.9% | 58.0% |
| Gradient Boosting | 0.593 | 0.887 | 3,000 | 1,226 | 427 | 40.9% | 74.2% |

At the same validation inspection budget, gradient boosting covers **268 additional defects**, reducing defects outside selection by **38.6%** relative to the baseline. This is a selection improvement, with no demonstrated reduction in physical defect generation or manufacturing cost.

![Validation model comparison](images/01_model_comparison.png)

| Final gradient boosting evaluation | Validation | Test |
|---|---:|---:|
| Records | 15,000 | 15,000 |
| Defects | 1,653 | 2,657 |
| Selected records | 3,000 | 3,000 |
| Defects within selection | 1,226 | 1,927 |
| Defects outside selection | 427 | 730 |
| Non-defects within selection | 1,774 | 1,073 |
| Precision | 40.9% | 64.2% |
| Recall | 74.2% | 72.5% |
| AP | 0.593 | 0.756 |
| ROC-AUC | 0.887 | 0.912 |

Random selection of 3,000 test records would cover **531.4 defects in expectation**, compared with 1,927 under the model policy, giving lift of **3.63**. The 12,000 unselected records still contain 730 defects, a **6.08% residual defect rate** within that group.

Test recall is 1.7 percentage points below validation recall, while higher test prevalence contributes to the interpretation of precision and AP. This evaluation reports one period and one policy, with no confidence intervals or evidence of production stability.

![Final evaluation](images/02_final_evaluation.png)

## 8. Model explanation

Permutation importance uses validation AP, with five random shuffles per feature while holding the trained model fixed.

| Feature | Mean AP decrease | Shuffle standard deviation |
|---|---:|---:|
| Alignment error | 0.3237 | 0.0065 |
| Mean temperature | 0.2090 | 0.0092 |
| Process time | 0.1459 | 0.0038 |
| Mean pressure | 0.0765 | 0.0030 |
| Surface particle count | 0.0369 | 0.0027 |
| Temperature variation | 0.0046 | 0.0017 |
| Runs since maintenance | 0.0036 | 0.0012 |

These values measure dependence of this model on each feature, conditional on the other included features. Correlated or substitutable inputs can share information, and importances do not sum to a causal allocation of defects. Shuffle standard deviations describe repeat variation, rather than uncertainty over new fab datasets.

![Feature importance](images/03_feature_importance.png)

Partial dependence replaces one feature across validation records and averages predicted probabilities, over its 5th to 95th percentile range. Alignment error shows rising model risk, while mean temperature and process time show U-shaped patterns, consistent with the exploratory analysis.

The hypothetical replacements can create rare combinations, and the outputs do not establish physical interventions or qualified process windows. Probability calibration was not evaluated, so probability-shaped outputs serve primarily as risk scores in this study.

![Partial dependence](images/04_partial_dependence.png)

## 9. Proposed sustaining and equipment investigation workflow

This section is an extension proposal, with no implemented fab connection or automatic action.

1. Validate data and labels, checking sensor units, recipe versions, wafer/lot mapping, metrology timing and inspection coverage.
2. Define affected scope, stratifying by tool/chamber, product, recipe, time, maintenance event and upstream history.
3. Join risk patterns with physical evidence, reviewing alignment calibration, tool alarms, sensor traces, particle metrology and maintenance records as available.
4. Assign owners and preserve records, documenting the hypothesis, supporting evidence, containment decision and required follow-up under the site's approved procedures.
5. Verify recovery, using independent metrology and subsequent production windows before concluding an action worked.

Equipment and process engineers would review tool and recipe evidence, integration/yield teams would examine downstream and product effects, metrology owners would confirm measurement validity, and technicians would follow the approved checklist and escalation path. These are proposed interfaces, with actual responsibilities determined by the organization.

## 10. SPC and DOE extension plan

### SPC, proposed

Existing plots show trends, while formal SPC requires a defined measurement system, rational subgrouping and an appropriate stable reference period. An extension could evaluate a p-chart for wafer-level defective fractions when assumptions support it, or suitable continuous-variable charts for qualified measurements, with stratification for relevant product and equipment differences.

Control limits describe the reference process variation, while specification limits come from engineering requirements. The current CSV has no approved specification limits, so no capability claim such as Cpk is supported. Temporal dependence, batch clustering and overdispersion also require assessment, before applying textbook limits. See [NIST control chart guidance](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm).

### DOE, proposed

Model patterns can prioritize hypotheses, then an approved experiment can test selected controllable factors within qualified ranges. A conceptual plan could study temperature and duration settings, including interaction and curvature, with nuisance variation handled through appropriate blocks and randomization. Alignment error is an observed response or covariate unless a safe, controllable intervention is defined, so shuffling it does not provide a DOE treatment.

The plan requires an explicit response definition, measurement-system checks, independent replication, power or precision assessment, execution order and acceptance criteria. Where settings are difficult to change, the randomization structure and analysis must reflect that constraint. No experimental runs or physical conclusions exist in this project, see [NIST design selection](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm) and [randomized blocks](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm).

## 11. Qualification gaps and deployment conditions

- Full-period EDA preceded splitting, making this an educational holdout evaluation rather than a strictly untouched confirmation study.
- Units, bonding step, defect taxonomy and generation assumptions require documentation, before physical interpretation.
- Real data should include representative recipe, tool/chamber, product and maintenance context, with point-in-time feature availability verified.
- Future evaluation should include additional time windows, lot-aware uncertainty estimates, calibration and subgroup performance, with separate data for decisions made after seeing test results.
- Shadow-mode validation should preserve existing inspection and release rules, with owners, escalation criteria and versioned logs agreed before operational changes.
- Quality requirements may require inspection beyond the ranked subset, with no demonstrated yield gain, cost saving or release qualification claimed here.
- Cross-site transfer between Chiayi and AZ requires harmonized units, labels and equipment/recipe context, followed by local validation rather than assuming portability.

## 12. Reproduction and repository contents

```text
README.md                 English technical overview
README.zh-TW.md           Traditional Chinese explanation and interview guide
notebooks/                Colab notebook
data/                     Fixed synthetic CSV and SHA-256 checksum
models/                   Saved tree model and feature/policy metadata
results/                  Model comparison, evaluation and importance tables
images/                   Project figures
presentations/            Chinese briefing deck and speaking notes
```

Run the notebook top to bottom in Colab, after downloading and uploading the checked-in CSV. The training and interpretation cells precede the export cells, which download local artifacts when run. Saved notebook outputs and account-specific execution metadata were removed, while the committed CSV tables and figures preserve recorded results.

The saved model uses scikit-learn **1.6.1**, NumPy **2.1.3**, pandas **2.2.3** and joblib **1.6.0**, with feature order and inspection policy recorded in `models/model_metadata.json`. `requirements.txt` records these core versions and plotting dependencies, with full environment reproduction still subject to Colab/runtime compatibility.

```python
import json
import joblib
import numpy as np

with open("models/model_metadata.json", encoding="utf-8") as file:
    metadata = json.load(file)

model = joblib.load("models/bonding_risk_model.joblib")
scores = model.predict_proba(records[metadata["features"]])[:, 1]
k = int(np.ceil(len(scores) * metadata["inspection_policy"]["fraction"]))
selected_rows = np.argsort(-scores, kind="stable")[:k]
```

In this example `records` is a batch DataFrame with validated numerical features, and selected indices are positional indices. Only load trusted joblib files, since deserialization can execute code.

## 13. Evidence and references

- Numerical results, `results/model_comparison.csv`, `results/final_evaluation.csv`, `results/permutation_importance.csv` and `results/threshold_comparison_logistic.csv`
- Implementation, `notebooks/SoIC_Bonding_01_EDA.ipynb`
- Role requirements, the posting text supplied by the project author for discussion, with current vacancy status not independently verified
- [NIST, control charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc31.htm)
- [NIST, experimental design selection](https://www.itl.nist.gov/div898/handbook/pri/section3/pri3.htm)
- [NIST, randomized block designs](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)
