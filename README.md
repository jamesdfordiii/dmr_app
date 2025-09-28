# Daily Market Report (DMR) Streamlit App

## Quickstart

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate # Mac/Linux
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Run Streamlit app:
   ```bash
   streamlit run dmr_app.py
   ```

## Features
- Enter catalysts, econ calendar, upgrades/downgrades.
- Auto snapshot of selected indices/commodities/crypto.
- Options flow (yfinance option chain).
- Markdown summary + export.
