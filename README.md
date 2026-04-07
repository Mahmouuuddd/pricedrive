# 🚗 PriceDrive — Data in the Driver's Seat

> Exploring 550,000+ vehicle sales transactions to uncover pricing patterns, market trends, and predict selling prices using machine learning.

---

## 📌 Project Overview

PriceDrive is an end-to-end data exploration, preparation, and visualization project built on the **Vehicle Sales and Market Trends Dataset**. The project covers the full data science pipeline — from raw data cleaning to an interactive dashboard with an integrated machine learning model that predicts vehicle selling prices in real time.

---

## 🔗 Links

| Resource | URL |
|---|---|
| 🌐 Live Dashboard | [PriceDrive on Hugging Face Spaces](https://huggingface.co/spaces/MahmoudSaqr22/pricedrive) |
| 📓 Jupyter Notebook | [project.ipynb](./project.ipynb) |
| 📊 Dataset | [Vehicle Sales Data — Kaggle](https://www.kaggle.com/datasets/syedanwarafridi/vehicle-sales-data) |

---

## 📂 Project Structure

```
pricedrive/
├── app.py                  # Dash dashboard application
├── requirements.txt        # Python dependencies
├── project.ipynb           # Full EDA, cleaning & ML notebook
├── Dockerfile              # For Hugging Face Spaces deployment
├── dataset/
│   ├── car_prices.csv      # Raw dataset
│   └── cleaned_dataset.csv # Cleaned & engineered dataset
└── models/
    ├── xgb_pricedrive.pkl  # Trained XGBoost model
    ├── label_encoders.pkl  # Label encoders for categorical features
    └── feature_columns.pkl # Feature column order used in training
```

---

## 📊 Dataset

The dataset contains **558,837** vehicle sales transactions with the following key features:

| Feature | Description |
|---|---|
| `year` | Manufacturing year of the vehicle |
| `make` | Brand / manufacturer |
| `model` | Specific vehicle model |
| `trim` | Trim level / variant |
| `body` | Body type (Sedan, SUV, Truck, etc.) |
| `transmission` | Transmission type |
| `condition` | Condition rating (1–50) |
| `odometer` | Mileage at time of sale |
| `sellingprice` | Final sale price ($) |
| `vehicle_age` | Age of vehicle at sale (engineered) |

---

## 🧹 Data Preparation

The following cleaning steps were applied:

- **Brand normalization** — unified inconsistent make names (e.g. `bmw`, `BMW`, `bmw` → `bmw`)
- **Invalid state values** — dropped rows where VIN strings appeared in the `state` column
- **Saledate parsing** — converted raw timezone strings to datetime and extracted `sale_year`, `sale_month`, `sale_dayofweek`
- **Missing value handling**:
  - `transmission` → filled with `"unknown"` (11% missing, not inferrable)
  - `condition` → filled with median per make
  - `body` → filled with mode per make+model combination
  - `color` / `interior` → filled with `"unknown"`; replaced `"—"` placeholder
  - `make`, `model`, `trim`, `body` → KNN imputation using label-encoded features
- **Outlier removal**:
  - Odometer capped between 10 and 300,000 miles
  - Selling price bounded between $500 and $150,000
- **Feature engineering**:
  - `vehicle_age` = sale year − manufacturing year
  - `price_ratio` = selling price / MMR market value
  - `price_vs_mmr` = selling price − MMR

---

## ❓ Business Questions & Insights

### Q1 — How have vehicle sales volume and average price changed over time?
Sales activity peaks in early 2015 driven by post-holiday auction cycles. Average selling prices remain stable between $12,000–$14,000, with a slight upward trend during low-volume months — suggesting an inverse relationship between supply and price.

### Q2 — Which car makes dominate the market and do high-volume brands sell at lower prices?
Ford, Nissan, and Chevrolet lead in volume and cluster in the $10,000–$15,000 range. Luxury brands (Porsche, BMW, Mercedes-Benz) sit at lower volumes but command average prices above $30,000 — a clear inverse relationship between volume and price.

### Q3 — How do vehicle condition and mileage affect selling price?
Condition has a strong positive effect — vehicles rated 41–50 sell for nearly double those rated 1–10. Odometer shows a steep negative relationship with price up to 100,000 miles, flattening thereafter. Vehicle age amplifies both effects.

### Q4 — Are vehicles being sold above or below their MMR market value?
50.8% of vehicles sold below MMR and 47.2% above, making the market slightly buyer-favorable. The price ratio distribution peaks sharply at 1.0, confirming MMR is an excellent market predictor.

### Q5 — Which body types are most popular and which command the highest prices?
Sedans dominate in volume. However, trucks and convertibles rank highest by average price — often 40–60% more expensive than sedans — showing that popularity does not directly reflect value.

### Q6 — How does vehicle age drive price depreciation and at what age does value stabilize?
Depreciation is steepest in the first 5 years, dropping from ~$24,000 to ~$10,000. A notable cliff appears at year 5. After year 10, prices flatten between $1,000–$3,000. The persistent gap between mean and median across all ages reflects luxury vehicles pulling the average upward.

---

## 🤖 Machine Learning Model

### Model
**XGBoost Regressor** — chosen for its native support of integer-encoded categorical features, speed on large datasets, and strong performance on tabular regression tasks.

### Features Used
```
year, make, model, trim, body, transmission, condition, odometer, vehicle_age
```

### Hyperparameter Tuning
`RandomizedSearchCV` with 5-fold cross-validation (`KFold`) was used to explore the hyperparameter space efficiently. The best parameters found:

```python
XGBRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=10,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    gamma=0.3,
    tree_method="hist"
)
```

### Evaluation Metrics

| Metric | Value |
|---|---|
| MAE | *(run notebook to see)* |
| RMSE | *(run notebook to see)* |
| R² | *(run notebook to see)* |

### Integration
The model is integrated into the **Price Predictor** tab of the Dash dashboard. Users input vehicle details and receive an instant predicted selling price along with a market comparison from similar vehicles in the dataset and a feature importance chart.

---

## 📈 Dashboard

The dashboard is built with **Dash + Plotly** and deployed on Hugging Face Spaces. It has 4 tabs:

| Tab | Contents |
|---|---|
| 📊 Overview | 6 KPI cards, condition distribution, price distribution, top makes bar chart, odometer vs price scatter |
| 📈 Sales Trends | Interactive filters (make, body, transmission) updating 3 charts: depreciation curve, condition vs price, sales by year |
| 🏷️ Market Analysis | Body type donut, bubble chart (volume vs price), side-by-side body bars, condition box plot |
| 🤖 Price Predictor | Cascading dropdowns, input validation, predicted price, market context, feature importance chart |

### Run Locally

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/MahmoudSaqr22/pricedrive
cd pricedrive

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python app.py
```

Then open [http://127.0.0.1:8050](http://127.0.0.1:8050) in your browser.

---

## 🚀 Deployment

The dashboard is deployed on **Hugging Face Spaces** using Docker.

The `Dockerfile` runs the app with `gunicorn` on port `7860` as required by Hugging Face:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["gunicorn", "app:server", "--bind", "0.0.0.0:7860"]
```

---

## 🛠 Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | KNN imputation, model evaluation, hyperparameter tuning |
| XGBoost | Price prediction model |
| Plotly / Dash | Interactive dashboard |
| Dash Bootstrap Components | Dashboard styling |
| Joblib | Model serialization |
| Matplotlib / Seaborn | EDA static charts |
| Hugging Face Spaces | Deployment |

---

## 📄 License

This project is for educational purposes as part of a Data Exploration, Preparation & Visualization course.