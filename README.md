# SoIC Bonding Quality Risk Prediction

A machine learning learning project that ranks synthetic bonding records by defect risk and evaluates how many defects can be covered by a limited inspection budget.

## Main result

On the chronological test set, inspecting the highest-scoring 20% of records covers **72.5% of defects**, with **64.2% precision**. The model selects 3,000 of 15,000 records, including 1,927 of 2,657 defects. The remaining 12,000 records contain 730 defects.

This is a synthetic-data study. The dataset does not represent TSMC production data or establish actual SoIC operating conditions.

## Run in Google Colab

[Open the notebook in Colab](https://colab.research.google.com/github/chelsietao/soic-bonding-quality-risk-prediction/blob/main/notebooks/SoIC_Bonding_01_EDA.ipynb)

1. Download [synthetic_bonding_data.csv](data/synthetic_bonding_data.csv) using GitHub's download button.
2. Open the notebook in Colab and run the first cell, uploading the CSV when prompted.
3. Run the analysis cells from top to bottom. Later cells depend on variables created in earlier cells.
4. The final export cells download the trained model, tables and figures. They are optional when reproducing the analysis.

Saved notebook outputs and account-specific execution metadata are removed from the published notebook. The committed tables and images preserve the reported results. `requirements.txt` records the saved core package versions, with plotting dependencies included. The notebook uses Colab-specific upload/download functions.

## Data and prediction task

The CSV contains **100,000 records and 12 columns**, spanning January through March 2026, with 4,000 lots and three tools. Its overall defect rate is 10.046%. Each row represents a synthetic wafer bonding record. Prediction is framed after processing and before quality inspection, so the process summaries are assumed available at prediction time.

| Column | Meaning | Role |
|---|---|---|
| `wafer_id` | Wafer identifier | Traceability |
| `timestamp` | Processing timestamp | Time split |
| `lot_id` | Lot identifier | Group separation |
| `tool_id` | Tool identifier | Exploratory analysis |
| `temperature_mean` | Mean process temperature | Model input |
| `temperature_std` | Within-record temperature standard deviation | Model input |
| `pressure_mean` | Mean process pressure | Model input |
| `process_time` | Processing duration | Model input |
| `alignment_error` | Pre-process alignment error | Model input |
| `surface_particle_count` | Pre-process particle count | Model input |
| `runs_since_maintenance` | Cumulative runs since maintenance | Model input |
| `defect_flag` | 0: no defect, 1: defect | Target |

Physical units, recipe specifications, defect inspection criteria and the exact bonding step were not documented during data generation. Numerical ranges should not be treated as recommended process settings. The data-generation source code is not included, so the checked-in CSV is the fixed input for reproduction. Its SHA-256 checksum is recorded in `data/SHA256SUMS`.

## Method

- Exploratory analysis of parameter ranges, defect rates, tool differences and time trends.
- Chronological split by lot start time: 70,000 training records, 15,000 validation records and 15,000 test records. Whole lots remain together, with assertions checking non-overlapping time ranges.
- Logistic Regression baseline with training-fitted median imputation and standardization.
- HistGradientBoostingClassifier using the same seven numeric features.
- Validation comparison using AP, ROC-AUC and the highest-risk 20% inspection policy.
- Test evaluation with the trained tree model and inspection fraction fixed.
- Permutation importance and partial dependence on validation data for explanation.

The inspection policy ranks a complete evaluation batch and selects its highest-scoring 20%, breaking score ties by original row order. It is distinct from a fixed probability threshold and requires batch-level ranking. Selection uses predictions, then quality labels are used for evaluation. Inspection outcomes are assumed perfectly observable in this simulation.

## Model comparison on validation data

| Model | AP | ROC-AUC | Inspected | Defects covered | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.421 | 0.794 | 3,000 | 958 | 31.9% | 58.0% |
| Gradient Boosting | 0.593 | 0.887 | 3,000 | 1,226 | 40.9% | 74.2% |

![Validation model comparison](images/01_model_comparison.png)

## Final evaluation

| Dataset | Defect rate | AP | ROC-AUC | Defects covered | Defects outside selection | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation | 11.02% | 0.593 | 0.887 | 1,226 | 427 | 40.9% | 74.2% |
| Test | 17.71% | 0.756 | 0.912 | 1,927 | 730 | 64.2% | 72.5% |

Both sets contain 15,000 records, with 3,000 selected for inspection. Test precision is about 3.6 times the test prevalence, corresponding to enrichment over random selection. The rising defect prevalence affects precision and AP, so higher values across periods do not alone establish better generalization.

![Final evaluation](images/02_final_evaluation.png)

## Model interpretation

Permutation importance measures the validation AP decrease after shuffling one feature, repeated five times. The largest decreases occur for alignment error (0.3237), mean temperature (0.2090) and process time (0.1459). These are conditional model dependencies, not shares of causal defect contribution. Error bars show shuffle-repeat standard deviations, not confidence intervals over new production data.

![Permutation importance](images/03_feature_importance.png)

Partial dependence shows increasing model risk with alignment error and U-shaped relationships for mean temperature and process time. Each point averages predictions after replacing one feature across validation records. These hypothetical combinations may be rare in the original data, and the plots do not identify safe operating limits or intervention effects.

![Partial dependence](images/04_partial_dependence.png)

## Limitations

- Full-period EDA preceded the split, so this is an educational holdout evaluation rather than a strictly untouched confirmation study.
- Synthetic relationships and metrics require validation on real process data before operational use.
- Probability calibration, inspection costs, confidence intervals and repeated time-window evaluation were not assessed.
- The model uses seven numeric process features, excluding tool identifiers and timestamps. Differences between periods may reflect changes not captured by those inputs.
- Results support inspection prioritization, with defects remaining outside the selected subset.

## Repository contents

```text
notebooks/   Colab analysis notebook
data/        Fixed synthetic CSV and checksum
models/      Saved tree model and feature/policy metadata
results/     Model comparison, evaluation and importance tables
images/      Four project figures
```

The model artifact was saved with scikit-learn 1.6.1, NumPy 2.1.3, pandas 2.2.3 and joblib 1.6.0. Only load trusted joblib files, and provide features in the order recorded in `models/model_metadata.json`.
