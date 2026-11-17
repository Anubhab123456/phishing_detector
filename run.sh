#!/usr/bin/env bash
# One-command setup + launch (Linux / macOS)
set -e
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
[ -f models/model.joblib ] || python -m src.train
streamlit run app.py
