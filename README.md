# 📈 Stock Market Upcoming Trend Prediction (XGBoost + Streamlit)

A beginner-friendly ML project that predicts a stock's **upcoming trend**
(Up / Sideways / Down over the next 5 trading days) from OHLC price data
and technical indicators, using **XGBoost**, with a **Streamlit** UI to
enter inputs and see the predicted trend and its probability.

---

## 1. Project structure

```
stock_trend_predictor/
├── data/
│   └── sample_stock_data.csv     # ready-to-use sample dataset (see note below)
├── generate_sample_dataset.py    # (re)creates the sample dataset
├── data_fetcher.py               # optional: download REAL data via yfinance
├── feature_engineering.py        # technical indicators + target label logic
├── train_model.py                # trains & saves the XGBoost model
├── app.py                        # Streamlit UI
├── models/                       # created after training: model, scaler, metadata
├── database/                     # optional MySQL integration (schema, helpers, loader)
├── .env.example                  # template for DB credentials
├── requirements.txt
└── README.md
```

## 2. About the dataset

`data/sample_stock_data.csv` included here is **synthetic** (randomly
generated with realistic price-movement statistics) so you can run the
whole pipeline immediately with no downloads or signups. It is NOT real
market data — don't use a model trained on it to make real trading
decisions, and say so if you're submitting this as coursework.

To train on **real** data instead, you have two easy options:

**Option A — auto-download with the included script (recommended):**
```bash
pip install yfinance
python data_fetcher.py --ticker RELIANCE.NS --years 6
```
This overwrites `data/sample_stock_data.csv` with real daily OHLCV data
for that ticker (works for NSE `.NS`, BSE `.BO`, US tickers, and indices
like `^NSEI`).

**Option B — download a ready-made dataset:**
- Kaggle: *"Huge Stock Market Dataset"* (user: borismarjanovic) — OHLCV
  CSVs for thousands of US stocks/ETFs.
- Kaggle: search *"NSE India stock data"* for Indian exchange data.
- Any CSV works, as long as it has columns: `Date, Open, High, Low, Close, Volume`.

## 3. Setup

```bash
cd stock_trend_predictor
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Train the model

```bash
python train_model.py
```
This will:
1. Load `data/sample_stock_data.csv` (or pass `--csv your_file.csv`)
2. Compute 18 features (raw OHLCV + technical indicators)
3. Build the 3-class trend label (Down / Sideways / Up), looking 5 days ahead
4. Split the data **by time** (train on older data, test on newer data —
   this matters for time series, a random shuffle-split would leak future
   information and give falsely high accuracy)
5. Scale features and train an `XGBClassifier`
6. Print accuracy, a classification report, confusion matrix and top
   feature importances
7. Save `models/xgb_trend_model.pkl`, `models/scaler.pkl`, `models/metadata.json`

## 5. Run the app

```bash
streamlit run app.py
```
Opens in your browser. Two ways to get a prediction:
- **From CSV** — upload ≥35 rows of recent OHLCV data; the app computes
  the technical indicators for you and predicts the trend for the most
  recent day.
- **Manual input** — type in the OHLC values and indicator values yourself.

The output shows the predicted trend (Up/Sideways/Down) plus the model's
probability for each of the three classes.

## 6. Work-history logging via MySQL

The training data and model stay as local files (`data/*.csv`,
`models/*.pkl`) — only the app's **work history** (every prediction it
makes) is logged to a MySQL database, in a single table:

```
database/
├── schema.sql       # CREATE DATABASE stock_trend_db + one table: work_history
├── db_config.py      # reads connection settings from a .env file / environment vars
└── db_utils.py       # log_prediction() and fetch_history() helper functions
```

**`work_history` table** — one row per prediction: timestamp, input source
(manual/CSV), the OHLCV values, the full feature vector (as JSON), the
predicted trend, and the three class probabilities.

**Setup:**
```bash
# 1. Install MySQL locally (or use XAMPP), or a free cloud instance (Aiven, Railway, etc.)
# 2. Create the schema
mysql -u root -p < database/schema.sql

# 3. Set your credentials
cp .env.example .env      # then edit .env with your real DB_USER / DB_PASSWORD
pip install mysql-connector-python python-dotenv
```

**What happens automatically:** once MySQL is reachable, `app.py` detects it
(sidebar shows "MySQL connected") and logs every prediction — inputs,
predicted trend, and probabilities — into `work_history`. The app also has
a "Work history (from MySQL)" panel that shows the last 50 logged
predictions. If MySQL isn't set up, the app just skips logging and works
exactly as before.

You can also query the log directly, e.g.:
```sql
SELECT predicted_trend, COUNT(*) FROM work_history GROUP BY predicted_trend;
```

**Where the data physically lives:** on whatever machine runs the MySQL
*server* — for local installs, inside MySQL's own data directory (e.g.
`/var/lib/mysql/` on Linux, `C:\ProgramData\MySQL\MySQL Server8.0\Data\` on
Windows), managed entirely by MySQL as InnoDB files. You never read/write
those files directly; you always go through SQL via `db_utils.py`.

## 7. How it works (for your report / viva)

**Inputs (features):**
| Category | Features |
|---|---|
| Raw price/volume | Open, High, Low, Close, Volume |
| Trend indicators | SMA_10, SMA_20, EMA_12, EMA_26, MACD, MACD_Signal |
| Momentum indicators | RSI_14, Stochastic %K, Stochastic %D, Momentum_10 |
| Volatility indicators | Bollinger Band Width, ATR_14, 10-day return std-dev |

**Target label:** for each day, look `FORWARD_DAYS` (default 5) trading
days ahead and compute the percentage price change. If it's ≥ +1.5% →
**Up**; if ≤ −1.5% → **Down**; otherwise → **Sideways**. These thresholds
are set in `feature_engineering.py` and easy to tune.

**Model:** `XGBClassifier` (gradient-boosted decision trees) with
`objective="multi:softprob"` for 3-class probability outputs — this is
what lets the app show "70% Up / 20% Sideways / 10% Down" instead of
just a single label.

**Evaluation:** accuracy, precision/recall/F1 per class, and a confusion
matrix, using a chronological (not random) train/test split — the honest
way to evaluate any time-series model.

## 8. Ideas to extend this project

- Add more indicators (OBV, ADX, Fibonacci retracement levels)
- Try predicting further ahead (10/20 days) or a different up/down threshold
- Add hyperparameter tuning (`GridSearchCV` / `Optuna`)
- Add SHAP values to explain individual predictions in the UI
- Train one model per stock, or one multi-stock model with a "ticker" feature
- Backtest a simple trading strategy based on the predicted trend

## 9. Disclaimer

This project is for **educational purposes only**. Stock markets are
influenced by countless factors a model like this cannot capture (news,
macroeconomics, sentiment, company fundamentals). Nothing here is
financial advice.
