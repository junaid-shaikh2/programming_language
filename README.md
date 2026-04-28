# Forex Direction Prediction Project

This project predicts the next-day direction of a forex pair using a machine learning classifier trained on historical exchange-rate data. It includes:

- A `Streamlit` dashboard for interactive training and prediction
- Basic backtesting and model evaluation metrics
- External `API` integration for forex prices and optional market-news sentiment
- A Vercel-ready Python API route for serverless prediction requests

## Tech Stack

- `Streamlit` for the user interface
- `scikit-learn` for the ML classifier
- `Alpha Vantage API` for FX historical price data
- `NewsAPI` for optional headline-based sentiment context
- `FastAPI` for a deployable API endpoint

## Project Structure

```text
.
|-- api/
|   `-- predict.py
|-- src/
|   |-- config.py
|   |-- data_sources.py
|   |-- features.py
|   |-- model.py
|   `-- service.py
|-- artifacts/
|-- .env.example
|-- requirements.txt
|-- streamlit_app.py
|-- README.md
`-- vercel.json
```

## Setup

1. Create a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create a `.env` file from `.env.example` and add your API keys:

```env
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
NEWSAPI_KEY=your_newsapi_key
```

## Run Streamlit Locally

```powershell
streamlit run streamlit_app.py
```

If `streamlit` is not recognized, use:

```powershell
python -m streamlit run streamlit_app.py
```

## Run the API Locally

```powershell
uvicorn api.predict:app --reload
```

Then open:

- `http://127.0.0.1:8000/api/predict?base=EUR&quote=USD`

## ML Workflow

The model:

- Fetches historical daily forex candles
- Builds technical features like returns, volatility, moving-average gaps, and momentum
- Labels each row as `1` if the next candle closes higher, otherwise `0`
- Trains a `RandomForestClassifier`
- Predicts whether the next day is more likely to move `UP` or `DOWN`
- Shows test accuracy, precision, recall, confusion matrix, and a simple directional backtest

## Vercel Deployment Notes

This repository includes a Vercel-ready API route via `api/predict.py`.

### Deploy the API to Vercel

1. Push this project to GitHub.
2. Import the repository into Vercel.
3. Set these environment variables in Vercel:
   - `ALPHA_VANTAGE_API_KEY`
   - `NEWSAPI_KEY`
4. Deploy.

Your endpoint will be similar to:

```text
https://your-project.vercel.app/api/predict?base=EUR&quote=USD
```

### About Streamlit on Vercel

Vercel is best suited here for the API route, not the Streamlit app itself. For the Streamlit UI, the easiest hosting options are:

- Streamlit Community Cloud
- Render
- Railway

If you want, you can still keep this repo as a single codebase and deploy:

- the API on Vercel
- the Streamlit app on Streamlit Community Cloud

## Future Improvements

- Add live intraday prediction
- Store training history in a database
- Add multiple model choices such as XGBoost or LSTM
- Integrate economic-calendar features
