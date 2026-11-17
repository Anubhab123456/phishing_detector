@echo off
REM One-command setup + launch (Windows)
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
if not exist models\model.joblib python -m src.train
streamlit run app.py
