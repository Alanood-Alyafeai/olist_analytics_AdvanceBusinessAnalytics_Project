# Olist Delivery Intelligence — Late Delivery Prediction

**Course:** DSAI 4103 — Advanced Business Analytics (Winter 2026)
**Author:** Alanood Alyafeai

An end-to-end business analytics project on the Brazilian Olist e-commerce dataset. The goal is to understand the drivers of late deliveries, predict the probability that an order will arrive late, and deliver actionable recommendations through an interactive Tableau dashboard and a production-ready scoring script.

---

## Project Structure

```
Olist_analytics_AdvanceBusinessAnalytics_Project/
├── data/                                              # Raw Olist datasets (CSV)
├── model/                                             # Trained model + scoring script
│   ├── late_delivery_model.pkl
│   ├── model_features.json
│   └── score.py
├── plots/                                             # Exported figures (EDA, model diagnostics, SHAP, fairness)
├── tableau/                                           # Tableau workbook + aggregated CSVs used by the dashboard
│   └── Olist_Delivery_Intelligence_Hub.twb
├── olist_analytics_project_AdvanceBusinessAnalytics.ipynb   # Main analysis notebook
├── Olist_Delivery_Intelligence_Presentation.pptx      # Final presentation deck
├── DSAI4103_Project_Description_Winter_2026.pdf       # Course project brief
└── README.md
```

---

## Objectives

1. **Descriptive analytics** — characterise order, customer, seller, and delivery patterns across Brazilian states and product categories.
2. **Diagnostic analytics** — identify which factors (geography, category, freight, payment, seller history, timing) drive late deliveries and low review scores.
3. **Predictive modelling** — train and evaluate classification models that predict the probability of a late delivery for a new order.
4. **Explainability & fairness** — use permutation importance and SHAP to explain model decisions, and audit fairness across Brazilian regions.
5. **Deployment & communication** — package the model as a reusable scoring script and present findings through a Tableau dashboard and executive slide deck.

---

## Dataset

The analysis uses the public **Brazilian E-Commerce Public Dataset by Olist** (2016–2018), covering ~100k orders across customers, order items, payments, reviews, products, sellers, and the product category translation table. All raw CSV files live in [data/](data/).

---

## Methodology

- **Data preparation:** joining the nine raw tables, handling missing values, engineering delivery-time, freight-ratio, seller-history, and temporal features.
- **Exploratory analysis:** univariate and bivariate analysis of delivery days, late rate by state and category, monthly volume vs. late rate, payment behaviour, and review-score impact (figures 1–8 in [plots/](plots/)).
- **Regression benchmark:** predicting delivery days, with actual-vs-predicted, feature importance, and residual diagnostics (figures 9–11).
- **Classification:** predicting `is_late` using logistic regression and gradient boosting, compared via ROC curves, confusion matrices, and feature importance (figures 12–15).
- **Explainability:** permutation importance, individual-order explanations, and SHAP global, beeswarm, dependence, and waterfall plots (figures 16–21).
- **Fairness audit:** late-rate and model-performance breakdown by Brazilian region (figure 22).
- **Deployment:** `LateDeliveryScorer` class in [model/score.py](model/score.py) with CLI, Python API, and Flask serving modes, plus risk-tier bucketing and recommended actions.

---

## Final Model

- **Algorithm:** Gradient Boosting Classifier (scikit-learn).
- **Features (17):** approval delay, purchase hour / day-of-week / month, weekend flag, item counts, total price & freight, freight ratio, seller counts, average item price / weight / volume, payment installments and value, same-state flag, and seller historical late rate. See [model/model_features.json](model/model_features.json).
- **Decision threshold:** 0.30, tuned to maximise F1 on the held-out test set.
- **Risk tiers:** `low` (<0.10), `medium` (0.10–0.25), `high` (0.25–0.40), `critical` (≥0.40), each mapped to a recommended operational action.

---

## How to Run

### 1. Reproduce the notebook

```bash
pip install -r requirements.txt   # pandas, numpy, scikit-learn, matplotlib, seaborn, shap, joblib, flask
jupyter notebook olist_analytics_project_AdvanceBusinessAnalytics.ipynb
```

### 2. Score new orders with the saved model

```bash
cd model
python score.py --input new_orders.csv --output predictions.csv
```

Or from Python:

```python
from model.score import LateDeliveryScorer
import pandas as pd

scorer = LateDeliveryScorer()
preds = scorer.predict(pd.read_csv("new_orders.csv"))
print(preds[["late_probability", "risk_tier", "recommended_action"]])
```

Or as a Flask service:

```bash
-------------------
```

### 3. Open the Tableau dashboard

Open [tableau/Olist_Delivery_Intelligence_Hub.twb](tableau/Olist_Delivery_Intelligence_Hub.twb) in Tableau Desktop. Supporting aggregated extracts are in the same folder.

---

## Deliverables

- **Notebook:** [olist_analytics_project_AdvanceBusinessAnalytics.ipynb](olist_analytics_project_AdvanceBusinessAnalytics.ipynb)
- **Presentation:** [Olist_Delivery_Intelligence_Presentation.pptx](Olist_Delivery_Intelligence_Presentation.pptx)
- **Tableau dashboard:** [tableau/Olist_Delivery_Intelligence_Hub.twb](tableau/Olist_Delivery_Intelligence_Hub.twb)
- **Scoring package:** [model/](model/)
- **Project brief:** [DSAI4103_Project_Description_Winter_2026.pdf](DSAI4103_Project_Description_Winter_2026.pdf)
