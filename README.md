# Algorithmic Pricing Optimization  

A data-science project that uses **Regression** and **XGBoost** models to predict optimal **unit prices** for retail products based on multiple business features such as cost, freight price, and competitor prices.  
Developed as part of a **Machine Learning & Predictive Analytics** portfolio to demonstrate the use of regression modeling in **pricing strategy optimization**.  

---

## Project Overview  

Pricing optimization is a crucial part of retail strategy — companies must set the right price to balance **profitability** and **competitiveness**.  
This project builds an **end-to-end predictive model** to estimate the ideal unit price of a product using data such as product weight, freight cost, customer behavior, and competitor pricing.  

The system applies:
- **Linear Regression** – for interpretable baseline prediction.  
- **XGBoost Regression** – for advanced, non-linear performance improvement.  

---

## Objectives  

- Build regression-based predictive models for **unit price estimation**.  
- Use **feature engineering** and **data cleaning** to improve model accuracy.  
- Compare model performance between **Linear Regression** and **XGBoost**.  
- Visualize **actual vs. predicted** prices to assess accuracy.  
- Save trained models (`.pkl` files) for reuse and deployment.  

---

## Dataset Description  

**Dataset Source:** [Retail Product Pricing Dataset](https://www.kaggle.com/datasets)  
*(Dataset originally distributed as `archive.zip`, extracted as `retail_price.csv`.)*

This dataset contains **676 rows** and **27 columns**, covering key attributes such as:  

| Feature | Description |
|:---------|:-------------|
| `qty` | Quantity sold per order |
| `total_price` | Total sale amount |
| `freight_price` | Shipping cost |
| `unit_price` | Target price variable |
| `product_weight_g` | Weight of product in grams |
| `product_score` | Rating given by customer |
| `customers` | Customer segment or group |
| `comp_1`, `comp_2`, `comp_3` | Competitor pricing indicators |
| `ps1`, `ps2`, `fp1`, `fp2` | Promotional and freight-related parameters |

---

## Workflow  

###  **Data Preparation**
- Load and inspect dataset (`.csv`) using `pandas`.  
- Check for missing values, data types, and descriptive statistics.  

### **Feature Selection**
- Use all numerical columns as predictors.  
- Target variable: `unit_price`.  
- Split data into **training (80%)** and **testing (20%)** sets.  

### **Modeling**
- Train two models:  
  - **Linear Regression** — interpretable baseline.  
  - **XGBoost Regressor** — high-performance boosting model.  

### **Evaluation**
- Metrics used:  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - R² Score  
- Results:  

| Model | MAE | MSE | R² |
|:------|:----:|:----:|:--:|
| **Linear Regression** | 4.36 | 65.58 | 0.988 |
| **XGBoost Regressor** | 3.36 | 43.03 | 0.992 |

**XGBoost achieved the best predictive performance.**

### **Visualization**
A scatter plot comparing **Actual vs. Predicted Prices** shows strong alignment, indicating excellent model fit.

---

## Repository Structure  

```
│
├── README.md
├── requirements.txt
├── retail_price.csv
├── Algorithmic_Pricing_Optimization.ipynb
│
├── models/
│ ├── linear_regression_model.pkl
│ └── xgboost_model.pkl
│
└── visuals/
└── actual_vs_predicted.png
```

---

## Model Persistence  

Both trained models are saved as `.pkl` files using **joblib** for easy reloading:

```python
joblib.dump(lr, "models/linear_regression_model.pkl")
joblib.dump(xgb, "models/xgboost_model.pkl")
```


