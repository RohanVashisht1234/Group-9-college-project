# ✈️ Airline Ticket Price Forecasting & Market Competition Dynamics

> **Group 9** | AI & Economics College Project

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1SjFBKjVB4VhKEnatz0Vy76dGqHitB-mj)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://group-9-college-project.streamlit.app/)
[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/datasets/dilwong/flightprices)

---

## 📌 Business Problem Statement

Airline ticket prices fluctuate dramatically based on demand, competition, timing, and route characteristics. This volatility creates challenges for both travelers trying to find the best deals and airlines seeking to maximize revenue.

This project addresses two core problems:

1. **For Travelers** — When is the optimal window to book a flight to minimize cost?
2. **For Airlines** — How do competitive dynamics, seat scarcity, and route characteristics drive pricing strategy?

To answer these questions, we built a machine learning price prediction model, performed market segmentation via clustering, and conducted a formal demand elasticity analysis on 200,000 real flight itineraries.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| **Source** | [Itinerary Flight Prices – Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices) |
| **File** | `itineraries.csv` |
| **Sample Used** | 200,000 rows |
| **Total Columns** | 27 |
| **Key Fields** | `baseFare`, `totalFare`, `seatsRemaining`, `travelDuration`, `startingAirport`, `destinationAirport`, `segmentsAirlineName`, `isBasicEconomy`, `isRefundable`, `segmentsCabinCode` |

The dataset contains real flight search results scraped from Expedia across multiple U.S. routes in 2022, making it highly representative of real-world airline pricing behavior.

---

## 💡 Economic Concepts Applied

### 1. Price Elasticity of Demand (PED)
$$\text{PED} = \frac{\% \Delta \text{ Quantity Demanded}}{\% \Delta \text{ Price}}$$

- **|PED| > 1** → Elastic demand (leisure travelers, price-sensitive)
- **|PED| < 1** → Inelastic demand (business travelers, time-sensitive)

Last-minute flights exhibit inelastic demand — travelers *must* fly, so airlines capture this with premium pricing.

### 2. Yield Management & Revenue Optimization
Airlines practice dynamic pricing by adjusting fares in real time based on load factor (seats remaining), booking lead time, and competitive pressure. Our analysis confirms fares are lowest in the **14–60 day** booking window before departure.

### 3. Supply-Demand Dynamics & Scarcity Pricing
As `seatsRemaining` decreases, prices rise — a direct application of the scarcity principle. This is modeled explicitly in our regression features.

### 4. Market Competition Analysis
Routes with more competing airlines show lower average fares, consistent with standard competitive market theory. Monopoly/duopoly routes command significant price premiums.

### 5. Price Discrimination
Airlines charge different prices for the same route based on cabin class (coach vs. premium), refundability, and fare basis codes — a textbook example of third-degree price discrimination.

---

## 🤖 AI Techniques Used

### 1. K-Means Clustering — Market Segmentation
**Goal:** Segment flight itineraries into distinct market tiers (budget, mid-range, premium).

- **Features used:** `totalFare`, `segmentsDistance`, `seatsRemaining`, number of competing airlines, lead time
- **Output:** Cluster labels that reveal distinct pricing tiers and the competitive/operational characteristics driving each tier
- **Business Value:** Helps airlines identify which market segment a route belongs to and price accordingly

### 2. Linear Regression — Price Prediction Model
**Goal:** Predict `totalFare` from flight characteristics.

- **Features used:** Travel duration (minutes), distance, seats remaining, days until departure, number of stops, airline encoded, route encoded, cabin class, `isBasicEconomy`, `isRefundable`
- **Evaluation Metrics:** MAE, RMSE, R²
- **Business Value:** Enables real-time fare recommendations and "fair price" benchmarking for travelers

### 3. Demand Elasticity Analysis
**Goal:** Quantify how sensitive demand (proxied by seats remaining) is to price changes across route segments.

- Routes and fare buckets are grouped, and PED coefficients are computed per segment
- Results distinguish elastic (leisure) from inelastic (business/last-minute) travel demand

---

## 🗂️ Project Structure

```
├── Group_9_1.ipynb          # Main analysis notebook (all sections below)
│   ├── 0. Setup & Data Loading
│   ├── 1. Data Cleaning & Preprocessing
│   ├── 2. Exploratory Data Analysis (EDA)
│   ├── 3. K-Means Clustering — Market Segmentation
│   ├── 4. Linear Regression — Price Prediction
│   ├── 5. Demand Elasticity Analysis
│   ├── 6. Business Interpretation & Strategic Insights
│   └── 7. Conclusion
├── app.py                   # Streamlit deployment app
└── README.md
```

---

## 📊 Key Findings & Strategic Insights

| Finding | Economic Concept | Business Implication |
|---------|-----------------|----------------------|
| Last-minute fares are highest | Inelastic demand | Revenue opportunity; risk to brand perception |
| Fares lowest 14–60 days out | Yield management | Travelers should book in the 3–8 week window |
| Fewer seats → higher price | Scarcity / Supply constraint | Dynamic pricing maximizes revenue per flight |
| Non-stop flights cost ~20–30% more | Willingness-to-pay premium | Differentiated product pricing |
| More competitors → lower fares | Competitive market theory | Monopoly routes yield significantly higher margins |

---

## 🚀 Links

| Resource | Link |
|----------|------|
| 📓 Google Colab Notebook | [Open in Colab](https://colab.research.google.com/drive/1SjFBKjVB4VhKEnatz0Vy76dGqHitB-mj) |
| 📊 Dataset (Kaggle) | [kaggle.com/datasets/dilwong/flightprices](https://www.kaggle.com/datasets/dilwong/flightprices) |
| 🌐 Live Streamlit App | [group-9-college-project.streamlit.app](https://group-9-college-project.streamlit.app/) |

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/your-group-repo/airline-price-forecasting.git
cd airline-price-forecasting

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

# OR open the notebook
jupyter notebook Group_9_1.ipynb
```

**Core Dependencies:**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
kagglehub
streamlit
```

---

## 👥 Group 9

> Submitted as part of the AI & Economics course project.

---

## 📄 License

This project is for academic purposes. Dataset credit: [Dilwong on Kaggle](https://www.kaggle.com/datasets/dilwong/flightprices).
