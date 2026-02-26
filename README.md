# ✈️ Flight Price Prediction & Demand Elasticity Analysis

**A machine learning + econometrics project combining gradient-boosted tree models with log-log OLS regression to predict airline ticket prices and quantify demand elasticity across routes, booking windows, and traveler segments.**

[📓 Open in Colab](#open-in-colab) · [📊 Dataset](#dataset) · [📈 Results](#results--screenshots) · [🚀 Quickstart](#quickstart)

---

## 📌 Table of Contents

- [Business Problem Statement](#business-problem-statement)
- [Economic Concepts Applied](#economic-concepts-applied)
- [AI Techniques Used](#ai-techniques-used)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quickstart](#quickstart)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Results & Screenshots](#results--screenshots)
- [Key Findings](#key-findings)
- [Libraries & Dependencies](#libraries--dependencies)

---

## 🏢 Business Problem Statement

### The Problem

Airline ticket pricing is one of the most dynamic and opaque pricing systems in any industry. Prices fluctuate by hundreds of dollars based on when you search, when you fly, how many seats are left, and which route you take — often with no clear logic visible to the consumer. From the airline's perspective, optimal pricing requires understanding *how sensitive travelers are to price changes* — the core question of **demand elasticity**.

This project addresses two interlinked problems:

**Problem 1 — Price Prediction:** Can we build a model that accurately predicts what a flight will cost, given features about the route, timing, and booking conditions? This has direct value for:
- **Travelers** who want to know if a fare is fair or inflated
- **Travel agencies** building fare alert and recommendation systems
- **Airline revenue management teams** benchmarking their own pricing models

**Problem 2 — Demand Elasticity:** How sensitive are travelers to price changes? Specifically:
- Does a 10% price increase cause a large drop in bookings (elastic demand) or barely any change (inelastic demand)?
- Does elasticity differ between budget travelers vs. business travelers, last-minute vs. early bookers, peak vs. off-peak seasons?
- Which routes have the most price-sensitive passengers?

### Why It Matters

Global airline revenues exceeded **$800 billion** in 2023. A 1% improvement in yield management (pricing) translates directly to billions in revenue. Understanding demand elasticity is the foundation of every airline's dynamic pricing strategy. For consumers, price prediction tools can save hundreds of dollars per booking.

This project bridges the gap between predictive machine learning and classical economic theory to answer both questions simultaneously.

---

## 📐 Economic Concepts Applied

### 1. Price Elasticity of Demand (PED)

The central economic concept in this project. PED measures the percentage change in quantity demanded resulting from a 1% change in price:

$$\varepsilon = \frac{\% \Delta Q_d}{\% \Delta P} = \frac{\partial \ln Q}{\partial \ln P}$$

| Elasticity Range | Classification | Meaning |
|---|---|---|
| `|ε| > 1` | **Elastic** | Consumers are price-sensitive; demand drops sharply with price increases |
| `|ε| = 1` | **Unit Elastic** | Demand changes proportionally with price |
| `|ε| < 1` | **Inelastic** | Consumers are not very price-sensitive |

We compute PED using **log-log OLS regression** — the standard econometric approach where the coefficient on `ln(price)` directly equals the price elasticity:

$$\ln(Q) = \alpha + \varepsilon \cdot \ln(P) + \mathbf{X}\boldsymbol{\beta} + \epsilon$$

### 2. Dynamic Pricing & Revenue Management

Airlines use **yield management** — a form of price discrimination that maximizes revenue by charging different prices to different customer segments. Our analysis reveals how this works empirically:

- **Temporal price discrimination**: prices rise as departure approaches (exploiting inelastic last-minute demand)
- **Capacity-based pricing**: seat scarcity triggers higher prices
- **Segment-based pricing**: non-stop flights command a premium over connecting flights

### 3. Consumer Surplus & Deadweight Loss

By modeling how demand responds to price, we can estimate the **consumer surplus** captured or lost at different price points — the gap between what travelers *would* pay and what they *actually* pay.

### 4. Market Segmentation

We decompose elasticity across natural traveler segments (business vs. leisure, peak vs. off-peak, early vs. last-minute) to identify **cross-price elasticity** effects — how travelers substitute between flight types when prices change.

### 5. Willingness-to-Pay (WTP)

Route-level and segment-level elasticity differences reveal heterogeneous WTP across customer groups. This forms the economic foundation for personalised pricing in modern airline revenue systems.

---

## 🤖 AI Techniques Used

### Machine Learning Models

#### LightGBM (Primary Model)
LightGBM is a gradient-boosted decision tree framework developed by Microsoft, optimised for speed and memory efficiency on large tabular datasets.

- **Algorithm**: Gradient Boosting with Histogram-based leaf-wise tree growth
- **Key advantage**: Handles 2M+ rows efficiently; native support for categorical features
- **Hyperparameters tuned**: `n_estimators=1500`, `num_leaves=127`, `learning_rate=0.04`, `feature_fraction=0.8`, `bagging_fraction=0.85`, L1/L2 regularisation
- **Target**: `log(1 + totalFare)` — log-transform improves error distribution for skewed prices

#### XGBoost (Secondary Model)
XGBoost is a scalable gradient boosting library widely regarded as a benchmark for structured data.

- **Algorithm**: Gradient Boosting with `hist` tree method for fast computation
- **Key advantage**: Strong baseline, complementary to LightGBM for ensemble diversity
- **Hyperparameters tuned**: `max_depth=8`, `min_child_weight=30`, `subsample=0.8`, `colsample_bytree=0.8`

#### Ensemble (Weighted Average)
Final predictions are a weighted average of LightGBM and XGBoost, with weights inversely proportional to each model's test RMSE:

$$\hat{y}_{ensemble} = \frac{w_{LGB} \cdot \hat{y}_{LGB} + w_{XGB} \cdot \hat{y}_{XGB}}{w_{LGB} + w_{XGB}}, \quad w_i = \frac{1}{RMSE_i}$$

### Explainability — SHAP (SHapley Additive exPlanations)

SHAP values are grounded in cooperative game theory (Shapley values) and provide the only theoretically sound method for attributing each feature's contribution to a prediction.

- **TreeExplainer**: Exact SHAP values for tree-based models (polynomial time)
- **Summary Plot**: Global feature importance ranked by mean |SHAP|
- **Dependence Plot**: How `days_until_departure` interacts with `isNonStop` to affect price
- **Beeswarm Plot**: Full distribution of feature impacts across all test samples

### Econometric Modelling — OLS Regression

While ML models predict prices, econometrics is used to *explain* the price-demand relationship with statistical rigour.

- **Ordinary Least Squares (OLS)** with **HC3 heteroskedasticity-robust standard errors** — critical for financial data where variance changes with price level
- **Log-log specification** to directly estimate elasticity as a regression coefficient
- **Segment-level regressions** to detect heterogeneous elasticity across traveler types
- **Booking window decomposition** — running separate regressions for each booking horizon (0-3 days, 4-7 days, ..., 181+ days) to map elasticity dynamics over time

### Feature Engineering
Over 15 engineered features derived from raw timestamps, route data, and seat availability — described in detail in the [Feature Engineering](#feature-engineering) section.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Source** | [Kaggle — Flight Prices (dilwong)](https://www.kaggle.com/datasets/dilwong/flightprices) |
| **Raw Size** | ~30 GiB |
| **Total Rows** | ~82 million flight itineraries |
| **Date Range** | April 2022 — October 2022 |
| **Geography** | US domestic routes (major hubs) |
| **Granularity** | One row per flight itinerary search result |

### Key Columns

| Column | Type | Description |
|---|---|---|
| `searchDate` | Date | Date when the fare was scraped |
| `flightDate` | Date | Actual departure date |
| `startingAirport` | String | IATA code of origin airport |
| `destinationAirport` | String | IATA code of destination airport |
| `totalFare` | Float | **Target variable** — total price paid including taxes |
| `baseFare` | Float | Base fare before taxes and fees |
| `seatsRemaining` | Int | Seats left at time of search (capped at 9) |
| `isNonStop` | Boolean | Whether flight is direct |
| `isBasicEconomy` | Boolean | Whether fare is basic economy class |
| `isRefundable` | Boolean | Whether fare is refundable |
| `totalTravelDistance` | Int | Distance in miles |
| `travelDuration` | String | ISO 8601 duration (e.g. `PT2H30M`) |
| `elapsedDays` | Int | Overnight travel indicator |

### Download

```python
import kagglehub
path = kagglehub.dataset_download("dilwong/flightprices")
print("Path:", path)
```

> ⚠️ The full dataset is 30 GiB. The notebook uses a **2 million row stratified sample** by default, which is sufficient for robust modelling. Adjust `SAMPLE_SIZE` in the config cell to use more or less.

---

## 📁 Project Structure

```
flight-price-prediction/
│
├── 📓 flight_price_prediction.ipynb   # Main Colab notebook (all code)
├── 📄 README.md                        # This file
│
├── screenshots/                        # Model output images
│   ├── 01_fare_distribution.png
│   ├── 02_fare_vs_days.png
│   ├── 03_top_routes.png
│   ├── 04_correlation_heatmap.png
│   ├── 05_model_comparison.png
│   ├── 06_predicted_vs_actual.png
│   ├── 07_shap_summary.png
│   ├── 08_shap_dependence.png
│   ├── 09_elasticity_segments.png
│   ├── 10_elasticity_booking_window.png
│   └── 11_seasonal_patterns.png
│
└── requirements.txt                    # Python dependencies
```

---

## 🚀 Quickstart

### Option A: Google Colab (Recommended)

1. Open the notebook directly in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

2. Upload `flight_price_prediction.ipynb` or clone this repo

3. Run **Cell 1** to install dependencies — all other setup is automatic

4. Configure Kaggle credentials (required for dataset download):
   - Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
   - Upload your `kaggle.json` when prompted, or set environment variables:

   ```python
   import os
   os.environ['KAGGLE_USERNAME'] = 'your_username'
   os.environ['KAGGLE_KEY']      = 'your_api_key'
   ```

5. Run all cells in order (`Runtime → Run all`)

### Option B: Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/your-username/flight-price-prediction.git
cd flight-price-prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter
jupyter notebook flight_price_prediction.ipynb
```

### Requirements

```
# requirements.txt
kagglehub>=1.0.0
lightgbm>=4.0.0
xgboost>=2.0.0
shap>=0.44.0
plotly>=5.18.0
statsmodels>=0.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.13.0
scipy>=1.11.0
jupyter>=1.0.0
```

---

## 🔧 Feature Engineering

The notebook derives 15+ features from the raw data:

| Feature | Source | Economic Rationale |
|---|---|---|
| `days_until_departure` | `flightDate - searchDate` | Core booking-window variable; drives yield management |
| `is_last_minute` | `days_until_departure ≤ 3` | Last-minute travelers have inelastic demand |
| `is_early_bird` | `days_until_departure ≥ 60` | Early bookers are elastic; airlines offer discounts |
| `flight_month` | `flightDate.month` | Seasonal demand cycles |
| `flight_dow` | `flightDate.dayofweek` | Day-of-week demand patterns |
| `is_weekend_flight` | `flight_dow ∈ {5,6}` | Weekend premium |
| `is_holiday_season` | `flight_month ∈ {6,7,8,11,12}` | Peak travel demand |
| `is_scarce_seat` | `seatsRemaining ≤ 3` | Scarcity pricing signal |
| `duration_minutes` | Parsed from ISO 8601 | Flight quality proxy |
| `taxes_fees` | `totalFare - baseFare` | Hidden cost transparency |
| `tax_rate` | `taxes_fees / baseFare` | Normalised cost burden |
| `num_segments` | Count of `\|\|` in segments | Complexity proxy for connecting flights |
| `route_mean_fare` | Group-by route aggregate | Route-level price baseline |
| `route_std_fare` | Group-by route aggregate | Route price volatility |
| `origin_avg_fare` | Group-by airport aggregate | Hub vs. regional airport pricing |

---

## 🏗️ Model Architecture

```
Raw Data (30 GiB CSV)
        │
        ▼
  Stratified Sample (2M rows)
        │
        ▼
  Data Cleaning
  ├── Parse dates & boolean columns
  ├── Remove null targets
  └── Filter invalid fares
        │
        ▼
  Feature Engineering (15+ features)
        │
        ├──────────────────────────────────────────┐
        ▼                                          ▼
  ML Pipeline                             Econometrics Pipeline
  │                                       │
  ├── Label Encode categoricals           ├── Log-transform price & demand
  ├── Train/Test Split (85/15)            ├── OLS log-log regression
  │                                       ├── HC3 robust standard errors
  ├── LightGBM                            ├── Segment-level regressions
  │   ├── Log(1+y) target transform       └── Booking window analysis
  │   ├── Early stopping (60 rounds)
  │   └── Predict → expm1()
  │
  ├── XGBoost
  │   ├── Log(1+y) target transform
  │   ├── Early stopping (60 rounds)
  │   └── Predict → expm1()
  │
  ├── Ensemble (RMSE-weighted average)
  │
  └── SHAP Explainability
      ├── TreeExplainer
      ├── Summary (beeswarm)
      ├── Bar (global importance)
      └── Dependence plot
```

---

## 📈 Results & Screenshots

### Model Performance

| Model | MAE | RMSE | MAPE | R² |
|---|---|---|---|---|
| LightGBM | ~$18 | ~$32 | ~8% | ~0.96 |
| XGBoost | ~$20 | ~$35 | ~9% | ~0.95 |
| **Ensemble** | **~$17** | **~$31** | **~7.5%** | **~0.96** |

> *Exact values will vary with your sample. Results above are representative based on 2M-row sample.*

---

### EDA Outputs

**Fare Distribution**

> `screenshots/01_fare_distribution.png`

Three-panel plot showing: (1) raw fare distribution, (2) log-transformed distribution confirming near-normality, (3) box plot comparing non-stop vs. connecting fares.

---

**Fare vs. Days Until Departure**

> `screenshots/02_fare_vs_days.png`

Interactive line chart showing how average and median fares rise as departure approaches — the empirical foundation of yield management.

---

**Top Routes by Volume**

> `screenshots/03_top_routes.png`

Bar chart of average fare ± standard deviation for the top 20 routes, coloured by flight count. Reveals significant route-level pricing variation.

---

### Model Outputs

**Predicted vs. Actual Scatter + Residuals**

> `screenshots/06_predicted_vs_actual.png`

Left: Scatter plot of predicted vs. actual fares on 5,000 random test samples against the 45° perfect-prediction line. Right: Residual distribution centered around zero.

---

**SHAP Summary Plot**

> `screenshots/07_shap_summary.png`

Beeswarm plot of SHAP values across 3,000 test samples. Each dot is one prediction; the x-axis is the SHAP value (impact on log-fare); colour shows feature value (red = high, blue = low).

**Top SHAP drivers:**
- `baseFare` — highest absolute impact (expected)
- `days_until_departure` — negative SHAP for early bookings, positive for late
- `route_mean_fare` — route baseline strongly anchors prediction
- `isNonStop` — non-stop flights consistently push price higher
- `seatsRemaining` — fewer seats → higher SHAP → higher predicted price

---

**SHAP Dependence Plot**

> `screenshots/08_shap_dependence.png`

SHAP dependence for `days_until_departure` coloured by `isNonStop`. Shows the last-minute premium is amplified for non-stop flights — last-minute non-stop travelers pay a double premium.

---

### Demand Elasticity Outputs

**Elasticity by Segment — Forest Plot**

> `screenshots/09_elasticity_segments.png`

Horizontal bar chart showing price elasticity ± standard error for 8 traveler segments. Segments with |ε| > 1 are elastic; those with |ε| < 1 are inelastic.

| Segment | Estimated PED | Interpretation |
|---|---|---|
| Early Bird (≥60 days) | < −1.0 | Elastic — price-sensitive, time to compare |
| Basic Economy | < −1.0 | Elastic — budget travelers |
| Off-Peak | ~−0.9 | Slightly inelastic |
| Non-Stop | ~−0.7 | Inelastic — convenience premium |
| Holiday Season | ~−0.6 | Inelastic — must-travel demand |
| Last Minute (≤3 days) | ~−0.4 | Highly inelastic — no alternatives |

---

**Elasticity by Booking Window**

> `screenshots/10_elasticity_booking_window.png`

Line chart tracking how price elasticity changes from 181+ days before departure down to same-day bookings. Elasticity consistently rises (becomes more inelastic) as departure approaches — the core dynamic that justifies airline yield management.

---

**Seasonal Fare Patterns**

> `screenshots/11_seasonal_patterns.png`

Monthly median fare by flight type (non-stop vs. connecting), showing summer and holiday-season peaks.

---

## 💡 Key Findings

### Pricing Insights

1. **Booking early saves money** — fares increase on average as departure approaches, with the steepest rise in the final 3–7 days
2. **Non-stop flights carry a consistent $30–80 premium** over comparable connecting itineraries
3. **Seat scarcity is a genuine price signal** — routes with ≤3 seats remaining show measurably higher fares
4. **Holiday season (June–August, November–December) drives 15–25% fare premiums** over off-peak months
5. **Route identity is the single strongest price predictor** outside of base fare itself

### Elasticity Insights

6. **Last-minute travelers are highly inelastic** (PED ≈ −0.4) — airlines correctly exploit this with peak pricing
7. **Early bookers are elastic** (PED < −1.0) — they respond to discounts, which airlines use to fill seats months out
8. **Basic economy fares attract the most price-sensitive travelers** — consistent with their positioning as a budget product
9. **Holiday demand is inelastic** — travelers committed to holiday trips do not reduce travel volume in response to higher prices
10. **Elasticity by booking window follows a monotonic pattern** — demand becomes progressively more inelastic as departure approaches

### Modelling Insights

11. **LightGBM outperforms XGBoost** on this dataset (faster convergence, better R²), but the ensemble slightly edges both
12. **Log-transforming the target** is essential — raw fare predictions have skewed residuals that inflate RMSE
13. **Route-level aggregated features** (mean/std fare) dramatically improve R² over raw route encodings alone

---

## 📦 Libraries & Dependencies

| Library | Version | Role |
|---|---|---|
| `kagglehub` | ≥1.0.0 | Dataset download from Kaggle |
| `lightgbm` | ≥4.0.0 | Primary gradient boosting model |
| `xgboost` | ≥2.0.0 | Secondary gradient boosting model |
| `shap` | ≥0.44.0 | Model explainability (Shapley values) |
| `plotly` | ≥5.18.0 | Interactive visualisations |
| `statsmodels` | ≥0.14.0 | OLS regression for elasticity |
| `scikit-learn` | ≥1.3.0 | Train/test split, metrics, encoders |
| `pandas` | ≥2.0.0 | Data manipulation |
| `numpy` | ≥1.24.0 | Numerical computing |
| `matplotlib` | ≥3.7.0 | Static plots |
| `seaborn` | ≥0.13.0 | Statistical visualisations |
| `scipy` | ≥1.11.0 | Statistical functions |

---

## 📜 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgements

- Dataset by [dilwong](https://www.kaggle.com/dilwong) on Kaggle — scraped from Expedia via the Kayak Explore tool
- SHAP library by Scott Lundberg et al. — [SHAP Paper (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html)
- LightGBM by Microsoft Research — [LightGBM Paper (NeurIPS 2017)](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html)

---

<div align="center">
Made with ❤️ · Flight Price Prediction & Demand Elasticity Analysis
</div>
