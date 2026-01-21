# PHM RUL Modeling (NASA CMAPSS)

An engineering-oriented PHM (Prognostics and Health Management) pipeline for equipment degradation analysis and Remaining Useful Life (RUL) prediction using the NASA CMAPSS dataset.

This repo focuses on:

* reproducible data-mining workflow
* controlled experiments (unit-wise split)
* interpretable degradation / health curve visualization

---

## 1. Project Overview

* Task: Degradation modeling and RUL prediction
* Dataset: NASA CMAPSS (FD001)
* Data type: Multi-sensor run-to-failure time series
* Goals:

  * build temporal statistical features from sensor streams
  * train explainable regressors for RUL
  * compare RF vs XGBoost under identical feature settings
  * visualize health degradation curves

---

## 2. Repository Structure

phm_anomaly/

* src/

  * step1_load_check.py
  * step2_make_rul_and_baseline.py
  * step3_window_features.py
  * step4_plot_health_curve.py
  * step5_eval_on_test.py
* artifacts/

  * train_fd001_w30_features.csv
  * model_compare_trainval.csv
  * model_compare_test.csv
  * health_curve_unit_compare_with_truth.png
* README.md
* .gitignore

Note: Raw CMAPSS data is stored locally and NOT included in this repository.

Example local data path:

* /home/didu/projects/datasets/cmapss/

  * train_FD001.txt
  * test_FD001.txt
  * RUL_FD001.txt

---

## 3. Methodology

### 3.1 RUL Construction

For training data, each unit runs until failure.

* failure_cycle(unit) = max(cycle) within that unit
* true_RUL(t) = failure_cycle(unit) − cycle(t)

This provides a physically meaningful regression target.

### 3.2 Sliding-Window Feature Engineering

Raw sensor readings are noisy and may not directly reflect degradation.
To encode temporal structure explicitly, sliding-window statistics are computed per unit:

* rolling mean
* rolling standard deviation
* deviation from rolling mean (current − rolling_mean)

These features capture low-frequency degradation patterns and improve model stability.

### 3.3 Model Comparison

Two tree-based regressors are trained and compared under identical feature settings:

* Random Forest (RF)
* XGBoost (XGB)

Evaluation uses unit-wise splitting to avoid leakage (units in validation are not seen during training).

Metrics:

* MAE (mean absolute error)
* RMSE (root mean squared error)

---

## 4. Results

### 4.1 Validation (unit-wise split)

From artifacts/model_compare_trainval.csv:

* XGB: MAE ≈ 22.44, RMSE ≈ 32.73
* RF : MAE ≈ 22.44, RMSE ≈ 33.07

Observation:

* RF and XGB perform very similarly, suggesting that feature representation is the dominant factor.

### 4.2 Test Set (official CMAPSS labels)

From artifacts/model_compare_test.csv:

* XGB: MAE ≈ 19.31, RMSE ≈ 27.13
* RF : MAE ≈ 19.91, RMSE ≈ 26.02

Observation:

* No significant gap between RF and XGB on test data.

---

## 5. Health Degradation Curve

A normalized health score (0 to 1) is derived from RUL:

* health(t) = (RUL(t) − min_RUL) / (max_RUL − min_RUL + eps)

The figure artifacts/health_curve_unit_compare_with_truth.png contains three curves:
<img width="1280" height="960" alt="health_curve_unit_compare_with_truth" src="https://github.com/user-attachments/assets/dd1847e6-9b08-4a6d-9377-99be76778410" />
* True health curve (from true RUL)
* RF-predicted health curve
* XGB-predicted health curve

Engineering interpretation:

* overall degradation trend is physically consistent
* mid-life fluctuations can occur due to operating condition changes and sensor noise
* agreement near end-of-life is the most critical for PHM decisions

---

## 6. How to Run

Install dependencies:

* pandas, numpy, scikit-learn, matplotlib
* optional: xgboost

Run scripts in order (from project root):

1. Data loading & sanity check
   python src/step1_load_check.py

2. Build RUL labels + baseline
   python src/step2_make_rul_and_baseline.py

3. Sliding-window feature engineering
   python src/step3_window_features.py

4. Model comparison + plot health curve
   python src/step4_plot_health_curve.py

5. Evaluate on test set
   python src/step5_eval_on_test.py

---

## 7. Key Takeaways

* Sliding-window statistical features effectively encode degradation information.
* Once temporal structure is represented well, RF and XGB converge to similar performance.
* In this PHM setup, feature quality matters more than using a more complex model.

---

## 8. Notes

* FD001 is used as the main dataset.
* FD002–FD004 can be integrated similarly for extended experiments.
* The pipeline is intentionally designed to be reproducible and interview-ready.
