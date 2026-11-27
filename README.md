 # Addis Ababa Road Traffic Accident (RTA) Severity Prediction (`T7Vr2-B5`)

Predicting the severity of road traffic accidents (`Accident_severity`) in Addis Ababa using pre-accident environmental, road, vehicle, and driver features.

---

## Repository Structure

| File | Description |
| :--- | :--- |
| `raw_Addis_Ababa_city_RTA.csv` | Original raw dataset (13,063 rows, 32 columns). |
| `T7Vr2-B5_dataset_clean.csv` | Cleaned dataset (25 columns: 24 leak-free features + 1 target). |
| `T7Vr2-B5_notebook.ipynb` | Main Google Colab / Jupyter notebook covering Steps 0 through 5 with full code and executed outputs. |
| `README.md` | Project overview, pipeline summary, and current progress. |

---

## Pipeline Progress (Steps 0 – 5)

- [x] **Step 0 — Problem Formulation:** Multi-class classification of `Accident_severity` (`Slight Injury`, `Serious Injury`, `Fatal Injury`) evaluated primarily via **Macro-F1** and **Fatal Recall** due to severe class imbalance.
- [x] **Step 1 — Exploratory Data Analysis (EDA):** Inspected 13,063 crash records, identified missing values, categorical distributions, and class imbalance.
- [x] **Step 2 — Data Cleaning & Leakage Prevention:** Imputed missing values and dropped post-accident target-leakage columns (`Casualty_class`, `Casualty_gender`, `Age_band_of_casualty`, `Work_of_casualty`, `Fitness_of_casualty`, `Pedestrian_movement`, and `Number_of_casualties`), producing `T7Vr2-B5_dataset_clean.csv` (25 columns).
- [x] **Step 3 — Stratified Train/Test Split:** Split the cleaned dataset into **80% Training** and **20% Test** sets while preserving class proportions.
- [x] **Step 4 — Baseline Model:** Evaluated a `DummyClassifier(strategy="most_frequent")` baseline to establish a minimum performance floor (**Macro-F1: `0.3013`**).
- [x] **Step 5 — Decision Tree Hyperparameter Tuning:** Ran 5-fold stratified `GridSearchCV` across **20 configurations** (`max_depth` $\in \{2, 3, 5, 10, \text{None}\}$, `min_samples_leaf` $\in \{1, 2, 5, 10\}$) with `class_weight="balanced"`, improving **Macro-F1 to `0.4818`** (`max_depth=None`, `min_samples_leaf=1`) and generating validation curves/heatmaps.
- [ ] **Step 6 — Next Stage:** Advanced model comparison / final test-set evaluation.

---

## Key Results Summary

| Model | Configuration | CV Macro-F1 | Notes |
| :--- | :--- | :---: | :--- |
| **Baseline (`DummyClassifier`)** | `strategy="most_frequent"` | `0.3013` | Predicts majority class only; 0% recall on minority classes. |
| **Decision Tree (Tuned)** | `max_depth=None`, `min_samples_leaf=1`, `class_weight="balanced"` | **`0.4818`** | Best across 20 grid-search combinations; significantly improves minority/fatal class detection. |

---

## How to Run

1. Open `T7Vr2-B5_notebook.ipynb` directly in **Google Colab** using the badge/link at the top of the notebook file on GitHub.
2. Upload `raw_Addis_Ababa_city_RTA.csv` (or `T7Vr2-B5_dataset_clean.csv` if starting from Step 3) to the Colab session files.
3. Run cells sequentially from top to bottom.
