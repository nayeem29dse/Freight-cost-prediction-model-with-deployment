# Freight Cost Prediction API

A machine learning project that predicts freight cost for vendor orders and flags high-risk shipments before invoice arrival. Built with a real inventory database, deployed as a live REST API.

🔗 **Live API:** https://freight-cost-prediction-model-with.onrender.com/

---

## Problem Statement

In supply chain management, freight costs are often unknown until the invoice arrives — making it difficult to forecast true procurement costs in advance. This project builds two models to solve that:

1. **Predict the exact freight cost** of an incoming vendor order
2. **Flag orders with unusually high freight rates** (>1% of invoice value) before they ship

---

## Project Workflow

### 1. Data Source
- Data imported from a local **PostgreSQL** database using **SQLAlchemy**
- Source table: `vendor_invoice` from an inventory management database
- 5,543 records covering vendor purchase orders with freight charges

### 2. Exploratory Data Analysis
- Found that `Freight` and `Dollars` have **0.985 correlation** — freight is essentially a fixed ~0.5% of invoice value
- Identified that high freight outliers (>1% rate) are small orders with disproportionately high shipping costs
- Discovered all 84 outlier orders concentrated in December — flagged as a data anomaly

### 3. Feature Engineering
- Converted date columns to datetime and extracted:
  - `lead_time` — days from PO to invoice
  - `payment_gap` — days from invoice to payment
  - `po_month` — month of purchase order
- Label encoded `VendorName` for model input

### 4. Models Built

#### Model 1: Freight Cost Regression (Linear Regression)
- **Features:** Quantity, Dollars, po_month, vendor_encoded
- **R²: 0.9939** — explains 99.4% of variance in freight cost
- **RMSE: 65.58**
- Insight: Freight is almost entirely determined by invoice value at ~0.5% rate

#### Model 2: High Freight Risk Classifier (Random Forest)
- **Target:** Binary — high freight rate (>1% of invoice value) vs normal
- Handled severe class imbalance (1.5% positive class) using **SMOTE**
- Tuned classification threshold to **0.1** for maximum recall
- **Recall: 0.94** at threshold 0.1 — catches 94% of high-risk orders
- Caught and investigated a **data leakage issue** via feature importance analysis (`po_month` was a December anomaly proxy, not a real signal)

### 5. Deployment
- Built a **Flask REST API** with two endpoints
- Deployed on **Render** (free tier, cloud hosted)
- Models serialized with `pickle`

---

## API Endpoints

### `GET /`
Returns API info and available endpoints.

**Response:**
```json
{
  "message": "Freight Cost Prediction API",
  "endpoints": {
    "/predict": "Predict freight cost in USD",
    "/classify": "Flag high freight risk orders"
  }
}
```

---

### `POST /predict`
Predicts the freight cost for a given order.

**Request:**
```json
{
  "Quantity": 100,
  "Dollars": 5000,
  "po_month": 3,
  "VendorName": "BACARDI USA INC            "
}
```

**Response:**
```json
{
  "predicted_freight": 24.33,
  "currency": "USD"
}
```

---

### `POST /classify`
Flags whether an order is at risk of having an unusually high freight rate.

**Request:**
```json
{
  "Quantity": 100,
  "Dollars": 5000,
  "VendorName": "BACARDI USA INC            "
}
```

**Response:**
```json
{
  "high_freight_risk": false,
  "probability": 0.06,
  "threshold": 0.1
}
```

---

## Tech Stack

| Layer | Tool |
|---|---|
| Database | PostgreSQL (local) via pgAdmin 4 |
| Data migration | SQLite → PostgreSQL via Python/SQLAlchemy |
| Data manipulation | Pandas, NumPy |
| Modeling | Scikit-learn (LinearRegression, RandomForestClassifier) |
| Imbalance handling | imbalanced-learn (SMOTE) |
| API | Flask |
| Deployment | Render |
| Version control | GitHub |

---

## Key Findings

- Freight cost is **almost entirely determined by invoice value** at a near-constant rate of ~0.5% — a simple linear rule `Freight ≈ 0.005 × Dollars` explains 99% of variation
- High freight outliers are **small orders** (avg $17,655) with disproportionately high shipping costs compared to normal orders (avg $58,695)
- All 84 outliers were concentrated in **December** — suggesting a seasonal pricing event or vendor contract anomaly rather than a systematic pattern
- Feature importance analysis revealed `po_month` was acting as a **data leakage proxy** — caught and corrected during model evaluation

---

## Repository Structure

```
├── Freight cost prediction.ipynb   # Full analysis notebook
├── app.py                          # Flask API
├── freight_model.pkl               # Trained regression model
├── freight_classifier.pkl          # Trained classification model
├── label_encoder.pkl               # Vendor name encoder
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Author

**Mehedee Hasan Nayeem**  
Data Analyst | Data Science Student  
University of Dhaka — BSc Statistics  
[GitHub](https://github.com/nayeem29dse)
