# Integrated Retail Analytics for Store Optimization and Demand Forecasting

An end-to-end data science project that applies machine learning to a 45-store, 81-department Walmart weekly sales dataset to solve four connected business problems: demand forecasting, anomaly detection, store segmentation, and quantifying the impact of external economic factors.

## Project Objective

Utilize machine learning and data analysis techniques to optimize store performance, forecast weekly sales, and enhance customer experience through segmentation and personalized marketing strategies.

## Highlights

- **Target metric (WMAE)**: Weighted Mean Absolute Error with holiday weeks penalized 5× — the same metric used in the original Walmart Kaggle competition.
- **Best model**: Tuned XGBoost reaching R² ≈ 0.97 on a held-out chronological test set, a ~10× reduction in WMAE versus a linear baseline.
- **Four store clusters** (Flagship A, Steady A/B, Small-format C, Holiday-driven B) validated with four independent clustering quality metrics (Elbow, Silhouette, Davies-Bouldin, Calinski-Harabasz).
- **Time-series decomposition** reveals a flat long-run trend — a strategic insight that no single chart could have produced.

## Datasets

| File | Rows | Description |
|---|---|---|
| `sales data-set.csv` | 421,570 | Weekly sales at Store × Dept × Date grain. **Target: `Weekly_Sales`.** |
| `Features data set.csv` | 8,190 | Weekly Store-level weather, fuel price, five MarkDown streams, CPI, unemployment, holiday flag. |
| `stores data-set.csv` | 45 | Store metadata — format Type (A / B / C) and Size (sqft). |

## Notebook Structure

The notebook (`Integrated_Retail_Analytics.ipynb`) follows a standard data-science template and contains 321 cells (68 code, 253 markdown).

1. **Know Your Data** — schema, nulls, duplicates.
2. **Understanding Your Variables** — variable dictionary and uniqueness counts.
3. **Data Wrangling** — merge, imputation, calendar feature engineering.
4. **Data Visualization (15 UBM charts)** — univariate, bivariate, and multivariate storytelling, each with Why / Insight / Business Impact commentary.
5. **Hypothesis Testing** — Welch t-test (holiday effect), ANOVA (store type), Pearson (markdown–sales correlation).
6. **Feature Engineering & Preprocessing** — IsolationForest for outliers, ordinal encoding, lag & rolling-mean features, chronological 85/15 split.
7. **ML Model Implementation** — Linear / Ridge / Random Forest / XGBoost with RandomizedSearchCV tuning, K-Means segmentation, Apriori market-basket analysis.
   - **7A. Time-Based Anomaly Detection** — seasonal decomposition (trend / seasonal / residual).
   - **7B. Segmentation Quality Evaluation** — four metrics triangulated.
   - **7C. Impact of External Factors** — correlation + standardized OLS + XGBoost importance.
   - **7D. Short-term vs Long-term Forecasting** — horizon trade-off analysis.
   - **7E. Personalization Strategies** — per-cluster marketing & inventory playbook.
   - **7F. Real-World Application & Challenges** — implementation plan and KPI framework.
8. **Future Work** — model serialized with `joblib`, reloaded for sanity prediction.

## Results Summary

| Model | R² | WMAE | Notes |
|---|---|---|---|
| Linear Regression | ~0.09 | highest | Baseline — cannot capture interactions |
| Ridge (tuned) | ~0.09 | high | Confirms linear family is inadequate |
| Random Forest | ~0.95 | much lower | Ensemble lift is large |
| Random Forest (tuned) | ~0.96 | lower | RandomSearchCV over depth / estimators |
| XGBoost | ~0.96 | lower | Gradient boosting out-of-the-box |
| **XGBoost (tuned)** | **~0.97** | **best** | **Final production model** |

## Key Findings

- Store size is the single strongest external driver of weekly sales.
- Markdowns have a modest positive effect on sales — meaningfully smaller than the discount itself, so ROI should always be assessed on incremental margin, not top-line.
- CPI and Unemployment are small negative drivers; Temperature and Fuel Price effects are non-linear and captured only by tree ensembles.
- Chain-wide trend is flat across 2010–2012 → growth must come from share-of-wallet, not organic volume.

## Tech Stack

- Python 3.10+
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LinearRegression, Ridge, RandomForest, KMeans, IsolationForest, PCA, RandomizedSearchCV)
- scipy (hypothesis tests)
- statsmodels (seasonal decomposition)
- xgboost (final forecasting model)
- mlxtend (Apriori association rules)
- joblib (model serialization)

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/vipul-solanki/Data-Science-Projects.git
   cd "Data-Science-Projects/Integrated Retail Analytics for Store Optimization"
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter:
   ```bash
   jupyter notebook Integrated_Retail_Analytics.ipynb
   ```
4. In Jupyter, select **Kernel → Restart & Run All**. End-to-end runtime is approximately 3–5 minutes on a modern laptop.

## Author

**Vipul Solanki**
GitHub: [@vipul-solanki](https://github.com/vipul-solanki)
