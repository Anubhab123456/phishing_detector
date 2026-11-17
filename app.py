"""Live Phishing Website Classification Dashboard (Streamlit).
Run:  streamlit run app.py
"""
import random
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.features import FEATURE_DESCRIPTIONS
from src.predict import PhishingDetector, load_metrics

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="wide")


@st.cache_resource
def get_detector():
    return PhishingDetector()


@st.cache_data
def get_metrics():
    return load_metrics()


try:
    det, metrics = get_detector(), get_metrics()
except FileNotFoundError:
    st.error("Model not found. Run `python -m src.train` first, then reload this page.")
    st.stop()

EXAMPLES = {
    "Legit - GitHub": "https://github.com/anthropics",
    "Legit - Wikipedia": "https://en.wikipedia.org/wiki/Machine_learning",
    "Phish - brand subdomain": "http://paypal.com.secure-login-x7k2.xyz/verify/account.php",
    "Phish - raw IP": "http://185.23.44.9/paypal/signin/login.php",
    "Phish - @ trick": "http://google.com@a8d7f6s5d.tk/login",
}


def gauge(prob: float) -> go.Figure:
    color = "#d62728" if prob >= 0.5 else "#2ca02c"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%"},
        title={"text": "Phishing probability"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": color},
               "steps": [{"range": [0, 30], "color": "#e6f4ea"},
                         {"range": [30, 70], "color": "#fff4d6"},
                         {"range": [70, 100], "color": "#fde7e7"}],
               "threshold": {"line": {"color": "black", "width": 3}, "value": 50}}))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def contrib_chart(contribs: dict, top: int = 10) -> go.Figure:
    items = list(contribs.items())[:top][::-1]
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h",
                           marker_color=["#d62728" if v > 0 else "#2ca02c" for v in vals]))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=30, b=10),
                      title="Why this prediction? (red = pushes toward phishing)",
                      xaxis_title="Change in phishing probability")
    return fig


# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("🛡️ Phishing Detector")
    st.caption("ML-based URL classification")
    st.metric("Active model", metrics["best_model"])
    best = metrics["results"][metrics["best_model"]]
    st.metric("Test accuracy", f"{best['accuracy']:.1%}")
    st.metric("F1 score", f"{best['f1']:.3f}")
    st.caption(f"Trained on {metrics['n_train']:,} URLs, tested on {metrics['n_test']:,}.")
    st.info("Analysis is lexical only: the URL is never visited or fetched.")

tab1, tab2, tab3 = st.tabs(["🔍 Analyze URL", "📡 Live Stream Simulation", "📊 Model Performance"])

# ---------------- Tab 1 ----------------
with tab1:
    st.subheader("Check a website URL")
    ex = st.selectbox("Try an example (or type your own below)", ["-"] + list(EXAMPLES))
    default = EXAMPLES.get(ex, "")
    url = st.text_input("URL", value=default, placeholder="https://example.com/login")
    if st.button("Classify", type="primary") and url.strip():
        r = det.predict(url.strip())
        c1, c2 = st.columns([1, 1.4])
        with c1:
            if r["label"] == "phishing":
                st.error(f"⚠️ PHISHING  ·  confidence {r['confidence']:.1%}")
            else:
                st.success(f"✅ LEGITIMATE  ·  confidence {r['confidence']:.1%}")
            st.plotly_chart(gauge(r["phishing_probability"]), width="stretch")
        with c2:
            st.plotly_chart(contrib_chart(r["contributions"]), width="stretch")

        st.markdown("#### Extracted features")
        df = pd.DataFrame({
            "feature": list(r["features"]),
            "value": [round(v, 3) for v in r["features"].values()],
            "contribution": [round(r["contributions"][k], 4) for k in r["features"]],
            "description": [FEATURE_DESCRIPTIONS[k] for k in r["features"]],
        }).sort_values("contribution", key=abs, ascending=False)
        st.dataframe(df, width="stretch", hide_index=True, height=300)

# ---------------- Tab 2 ----------------
with tab2:
    st.subheader("Simulated real-time URL feed")
    st.caption("Replays held-out test URLs through the model as if they were arriving from a live traffic feed.")
    c1, c2, c3 = st.columns(3)
    n_urls = c1.slider("URLs to stream", 10, 200, 40)
    delay = c2.slider("Delay per URL (seconds)", 0.0, 1.5, 0.3, 0.1)
    run = c3.button("▶ Start simulation", type="primary")
    ph_metrics, ph_chart, ph_table = st.empty(), st.empty(), st.empty()

    if run:
        pool = list(metrics["stream_sample"])
        random.shuffle(pool)
        rows, correct, flagged, history = [], 0, 0, []
        bar = st.progress(0.0)
        for i, item in enumerate(pool[:n_urls], 1):
            r = det.predict(item["url"])
            pred = int(r["label"] == "phishing")
            correct += int(pred == item["label"])
            flagged += pred
            history.append(r["phishing_probability"] * 100)
            rows.insert(0, {"#": i, "URL": item["url"][:90], "Prediction": r["label"].upper(),
                            "Confidence": f"{r['confidence']:.1%}",
                            "Ground truth": "PHISHING" if item["label"] else "LEGITIMATE",
                            "Correct": "✅" if pred == item["label"] else "❌"})
            with ph_metrics.container():
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Processed", i)
                m2.metric("Flagged as phishing", flagged)
                m3.metric("Marked safe", i - flagged)
                m4.metric("Running accuracy", f"{correct / i:.1%}")
            ph_chart.line_chart(pd.DataFrame({"phishing probability (%)": history}), height=180)
            ph_table.dataframe(pd.DataFrame(rows).head(15), width="stretch", hide_index=True)
            bar.progress(i / min(n_urls, len(pool)))
            time.sleep(delay)
        st.success("Simulation finished.")

# ---------------- Tab 3 ----------------
with tab3:
    st.subheader("Model comparison (held-out test set)")
    res = pd.DataFrame({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
                        for k, v in metrics["results"].items()}).T.round(4)
    st.dataframe(res, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Confusion matrix - {metrics['best_model']}**")
        cm = best["confusion_matrix"]
        fig = go.Figure(go.Heatmap(z=cm, x=["Pred legit", "Pred phishing"], y=["Actual legit", "Actual phishing"],
                                   text=cm, texttemplate="%{text}", colorscale="Blues", showscale=False))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("**Global feature importance (Random Forest)**")
        imp = list(metrics["feature_importance"].items())[:15][::-1]
        fig = go.Figure(go.Bar(x=[v for _, v in imp], y=[k for k, _ in imp], orientation="h",
                               marker_color="#1f77b4"))
        fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig, width="stretch")
