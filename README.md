# 🛡️ Phishing Website Detection using Machine Learning

Enter a URL → extract 37 lexical features → ML model predicts **legitimate** or **phishing**
→ dashboard shows **confidence** and **feature importance** (global + per-URL explanation).

The analysis is lexical only: the URL is **never visited**, so it is safe to test malicious-looking links.

## Project structure
```
phishing_detector/
├── app.py               # Streamlit dashboard (analyze URL / live stream / performance)
├── cli.py               # Command-line classifier
├── requirements.txt
├── run.sh / run.bat     # One-command setup + launch
├── src/
│   ├── features.py      # URL feature extraction (37 features)
│   ├── dataset.py       # Synthetic dataset generator (offline demo)
│   ├── train.py         # Trains RF / GradientBoosting / LogReg, saves the best
│   └── predict.py       # Inference + per-URL explanation
├── tests/test_features.py
├── data/                # urls.csv is generated here
└── models/              # model.joblib + metrics.json are generated here
```

## Step-by-step setup

**Prerequisite:** Python 3.9+ (`python --version`).

### Option A - one command
- Linux/macOS: `chmod +x run.sh && ./run.sh`
- Windows: double-click `run.bat` (or run it in Command Prompt)

### Option B - manual
1. **Unzip** the project and open a terminal inside the `phishing_detector` folder.
2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Train the model** (~15 seconds; creates `models/model.joblib` and `models/metrics.json`)
   ```bash
   python -m src.train
   ```
5. **Launch the dashboard**
   ```bash
   streamlit run app.py
   ```
   Your browser opens at http://localhost:8501.

### Optional extras
- Classify from the terminal: `python cli.py "http://paypal.com.secure-login.xyz/verify"`
- Run tests: `pip install pytest && python -m pytest tests`

## Running the simulation
1. **🔍 Analyze URL** - pick an example or paste a URL, click **Classify**. You get the verdict, confidence,
   a probability gauge, a bar chart explaining which features pushed the decision, and the full feature table.
2. **📡 Live Stream Simulation** - set the number of URLs and delay, click **▶ Start simulation**.
   Held-out test URLs are classified one by one, with a live counter, probability chart, and results table
   (including ground truth and running accuracy).
3. **📊 Model Performance** - model comparison table, confusion matrix, and global feature importance.

## Using a real dataset (recommended for real accuracy)
The bundled dataset is **synthetic** so the project runs offline; its scores will not match real-world performance.
To train on real data, get a labelled CSV and train with it:
```bash
python -m src.train --csv path/to/your_data.csv    # needs columns: url, label (1 = phishing, 0 = legitimate)
```
Public sources: PhishTank / OpenPhish (phishing URLs), Tranco list (legitimate domains),
Kaggle "Phishing Site URLs" or the UCI "PhiUSIIL Phishing URL" dataset (rename columns to `url`,`label`).

## How it works
- **Features:** length stats, special-character counts, IP-as-host, HTTPS, subdomain depth, abused TLDs,
  URL shorteners, suspicious keywords (login/verify/...), brand names in the wrong place, entropy, punycode,
  `@` trick, and more (see `src/features.py`).
- **Models:** Random Forest, Gradient Boosting, and Logistic Regression are compared; the best F1 is saved.
- **Confidence:** the model's predicted class probability.
- **Global importance:** Random Forest feature importances.
- **Per-URL explanation:** each feature is replaced with the average legitimate-URL value; the resulting change
  in phishing probability is that feature's contribution (red = toward phishing, green = toward legitimate).

## Limitations
URL-only features miss attacks on compromised legitimate domains with clean-looking URLs. For production, combine
with page-content, WHOIS/domain-age, and reputation signals, and retrain regularly.

## Troubleshooting
- `Model not found` → run `python -m src.train`.
- `streamlit: command not found` → activate the virtualenv, or use `python -m streamlit run app.py`.
- Port busy → `streamlit run app.py --server.port 8502`.
