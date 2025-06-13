# Outcome Prediction After Chikungunya Hospitalization

## MC853 - Artificial Intelligence in Healthcare - Unicamp

**Authors:**
- Pietro Grazzioli Golfeto – 223694
- Leandro Henrique Silva Resende – 213437
- Yvens Ian Prado Porto – 184031

---

## 1. Overview

This project aims to develop a machine learning model to predict the clinical outcome (recovery or death) for patients hospitalized due to Chikungunya fever. The analysis uses public health data from Brazil's Notifiable Diseases Information System (SINAN).

The project is divided into three main parts, each contained in a separate Jupyter Notebook:
1.  **Preprocessing and Exploratory Data Analysis (EDA):** Cleaning and preparing the raw data, engineering relevant features, and visualizing key aspects of the dataset.
2.  **Model Training and Evaluation:** Training various classification models, handling class imbalance, and evaluating performance to predict patient outcomes.
3.  **Fairness Analysis:** Assessing the trained models for biases, particularly concerning demographic groups like race and gender.

---

## 2. Dataset

-   **Source:** The data is from the **DataSUS SINAN** system, containing reported cases of Chikungunya in Brazil.
-   **Years:** The dataset spans from **2018 to 2025**.
-   **Initial Population:** All reported cases of Chikungunya.
-   **Study Population:** The analysis is filtered to focus only on **hospitalized patients**, as the primary goal is to predict outcomes in a hospital setting.

---

## 3. Setup and Installation

### Prerequisites
-   Python `3.10.12` or higher.
-   Jupyter Notebook or JupyterLab.

### Installation
Clone the repository and install the required packages using the `requirements.txt` file:

```bash
git clone <your-repository-url>
cd <your-repository-name>
pip install -r requirements.txt
```

### `requirements.txt`
```
pandas
seaborn
matplotlib
numpy
scikit-learn
imbalanced-learn
ipykernel
import-ipynb
```

### Project Structure
Before running the notebooks, you need to set up the data directories. The notebooks expect the following structure:

```
.
├── source/
│   └── csv/
│       ├── CHIKBR18.csv
│       ├── CHIKBR19.csv
│       └── ... (and so on for all years)
├── datasets/
│   └── (This directory is used to store cleaned and processed data)
├── part_1_preprocessing.ipynb
├── part_2_training.ipynb
├── part_3_fairness.ipynb
├── plots.ipynb
└── README.md
```

### Configuration
At the beginning of each notebook part, you must add a path dictionary for your own system. Find the cell with the pietro_path and leandro_path dictionaries and add your own, like this:
```python
# Add your own path dictionary here
your_name_path = {
    'sinan_path': '/path/to/your/source/csv/',
    'cleaned_path': '/path/to/your/datasets/',
    'train_path': '/path/to/your/datasets/train.csv',
}

# Then, add your path to the `if/elif` check
if os.path.isdir(pietro_path['sinan_path']):
    path = pietro_path
elif os.path.isdir(leandro_path['sinan_path']):
    path = leandro_path
elif os.path.isdir(your_name_path['sinan_path']): # Add this line
    path = your_name_path
else:
    raise Exception('Path not found. Please check the paths in the script.')
```

## 4. Part 1: Preprocessing and Exploratory Data Analysis

This notebook (`part_1_preprocessing.ipynb`) handles the initial data preparation.

### Key Preprocessing Steps:
1.  **Data Loading:** Loads multiple yearly CSV files (`CHIKBR18.csv` to `CHIKBR25.csv`).
2.  **Column Consolidation:** Identifies and retains only the columns common to all yearly files.
3.  **Filtering:** Selects only records of **hospitalized patients**.
4.  **Feature Cleaning:** Drops constant columns and features with more than **1% missing values**.
5.  **Feature Engineering:** The target variable `EVOLUCAO` is mapped to **0 (Cure)** and **1 (Death)**. Categorical features like region, gender, and race are one-hot encoded.
6.  **Train-Test Split:** A **temporal split** is performed. The training set contains data from **2018-2023**, and the test set contains data from **2024-2025**.
7.  **Data Export:** The final processed `train.csv` and `test.csv` files are saved to the `datasets/` directory.

### Exploratory Data Analysis (EDA) Highlights
The notebook includes visualizations showing that patient age, geographical region, and certain comorbidities are strongly correlated with the clinical outcome.

---

## 5. Part 2: Model Training and Evaluation

This notebook (`part_2_training.ipynb`) focuses on building, tuning, and evaluating predictive models.

### Further Preprocessing:
-   **Imputation:** Missing values remaining after Part 1 are imputed using `KNNImputer`.
-   **Outlier Removal:** Rows where the time between symptoms and notification (`TIME_DIFF_DAYS`) exceeds 45 days are removed.
-   **Scaling:** Continuous features (`AGE`, `TIME_DIFF_DAYS`, `TIME`) are scaled using `RobustScaler`, which is resilient to outliers.

### Class Imbalance Handling
The training data is highly imbalanced. To address this, a resampling pipeline combining **SMOTE** (oversampling) and **RandomUnderSampler** (undersampling) is applied within the cross-validation loop to create balanced training sets.

### Model Evaluation Strategy
A robust **nested cross-validation** approach is used to tune and evaluate three models: **K-Nearest Neighbors**, **Logistic Regression**, and **Random Forest**. Models are optimized for **Recall** on the positive class (`recall_1`) to prioritize the correct identification of patients at risk of death (minimizing false negatives).

### Results
After tuning, the final models were evaluated on the held-out test set. The **Logistic Regression** model achieved the best performance on the primary metric, but the **Random Forest** model achieved best performance overall across all metrics.

-   **Best Model (Test Set `recall_1`):** **Random Forest** with a score of **0.820**.
-   **Other Models:** Logistic Regression (`recall_1`: 0.917) and KNN (`recall_1`: 0.544).

---

## 6. Part 3: Fairness Analysis

This notebook (`part_3_fairness.ipynb`) investigates whether the machine learning model exhibits bias towards a specific demographic group.

### Methodology
The analysis focuses on the **Random Forest** model and uses **Gender** as the sensitive attribute to assess fairness.

-   **Fairness Metric:** The primary metric is **Equal Opportunity**, which measures whether the model's ability to correctly identify a positive outcome (death) is the same across different gender groups. It is calculated by comparing the **True Positive Rate (TPR)** between male and female patients. A smaller difference in TPRs indicates a fairer model.

-   **Analysis Scenarios:** Two distinct models are trained and compared:
    1.  **"Aware" Model:** Trained *with* the `GENDER` column included as a feature.
    2.  **"Unaware" Model:** Trained *without* the `GENDER` column to see if removing direct knowledge of the attribute mitigates bias. This approach is known as *Fairness through Unawareness*.

-   **Data Strategy:** A careful data separation strategy is employed to ensure valid results:
    -   **Model Fitting:** Models are trained on the **resampled** training data to benefit from a balanced class distribution.
    -   **Fairness Evaluation:** Fairness metrics are calculated on the **original, non-resampled** training and test sets to assess bias on the true data distribution.

### Results and Conclusion
The performance and fairness of both the "aware" (with gender) and "unaware" (without gender) models were evaluated on the test set. The analysis answers the key questions about the fairness-performance trade-off.

1. Does the model "aware" of gender show a significant fairness gap?

- Yes. The model trained with the gender feature shows a notable bias. On the test set, the True Positive Rate (TPR) for males was 0.870, while for females it was 0.759.
- This results in a fairness gap (Equal Opportunity difference) of 0.112, indicating the model is better at correctly identifying at-risk males.
2. Does removing the gender feature reduce this gap?

- Yes, dramatically. The model trained without the gender feature is significantly fairer. The TPR for males became 0.816 and for females 0.808, shrinking the fairness gap to just 0.008.
3. Is there a significant trade-off in performance?

- No. The improvement in fairness comes at a minimal cost to overall performance. The balanced accuracy only dropped slightly from 0.6492 to 0.6364 (~1.3%), and the crucial recall for the positive class (death) remained high, decreasing from 0.8203 to 0.8125.
- Given the substantial gain in fairness and the negligible drop in predictive power, the model trained without the gender feature is the superior choice. It achieves a much better balance between accuracy and equity, making it more suitable for ethical deployment in a real-world healthcare setting.