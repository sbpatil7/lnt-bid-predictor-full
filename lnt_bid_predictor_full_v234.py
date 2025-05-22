
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from io import BytesIO
import datetime

st.set_page_config(page_title="L&T Bid Predictor - Full Version", layout="wide")

@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    cat_cols = [
        'Product Type', 'Project Region', 'Project Geography/ Location', 'Licensor',
        'Shell (MOC)', 'Weld Overlay/ Clad Applicable (Yes or No)', 'Sourcing Restrictions (Yes or No)'
    ]
    num_cols = ['ID (mm)', 'Weight (MT)', 'Price($ / Kg)']
    if 'Bid Date' in df.columns:
        df['Bid Month'] = pd.to_datetime(df['Bid Date']).dt.month

    df = df.dropna(subset=['Result(w/L)'])
    df.fillna(method='ffill', inplace=True)

    le_dict, inv_le_dict = {}, {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        inv_le_dict[col] = dict(zip(le.classes_, le.transform(le.classes_)))

    all_cols = cat_cols + num_cols
    if 'Bid Month' in df.columns:
        all_cols += ['Bid Month']
    X = df[all_cols]
    y = LabelEncoder().fit_transform(df['Result(w/L)'])
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])
    return X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, df

X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, full_data = load_data("data.xlsx")

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

st.title("🏗️ L&T Bid Predictor – Full Dashboard (Phase 2–4)")

# Sidebar metrics
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
st.sidebar.text("Classification Report")
st.sidebar.text(classification_report(y_test, model.predict(X_test)))

st.subheader("📌 Confusion Matrix")
fig1, ax1 = plt.subplots()
sns.heatmap(confusion_matrix(y_test, model.predict(X_test)), annot=True, fmt="d", cmap="Blues", ax=ax1)
st.pyplot(fig1)

st.subheader("📊 Feature Importance")
importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=True)
fig2, ax2 = plt.subplots()
importance.plot(kind='barh', color='skyblue', ax=ax2)
st.pyplot(fig2)

# Phase 2: Visualizations
if 'Project Geography/ Location' in full_data.columns:
    st.subheader("🌍 Geo Impact Analysis")
    geo_df = full_data.copy()
    geo_df['Result Encoded'] = y
    geo_fig = px.histogram(geo_df, x='Project Geography/ Location', color='Result Encoded', barmode='group')
    st.plotly_chart(geo_fig)

if 'Product Type' in full_data.columns:
    st.subheader("📉 Historical Win Rate by Product Type")
    hist_fig = px.histogram(full_data, x='Product Type', color='Result(w/L)', barmode='group')
    st.plotly_chart(hist_fig)

if 'Bid Month' in X.columns:
    st.subheader("📅 Win Trends by Bid Month")
    month_fig = px.histogram(full_data, x='Bid Month', color='Result(w/L)', barmode='group')
    st.plotly_chart(month_fig)

# Phase 3: What-if Sliders
st.subheader("🔍 Predict Individual Bid (With What-if Sliders)")
user_input = {}
col1, col2 = st.columns(2)
with col1:
    for col in cat_cols:
        user_input[col] = st.selectbox(col, list(inv_le_dict[col].keys()))
with col2:
    for col in num_cols:
        user_input[col] = st.slider(col, float(full_data[col].min()), float(full_data[col].max()), float(full_data[col].median()))

bid_month = st.slider("Bid Month", 1, 12, 6)

if st.button("🚩 Predict Now"):
    input_df = pd.DataFrame([user_input])
    for col in cat_cols:
        input_df[col] = le_dict[col].transform([input_df[col][0]])
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    input_df["Bid Month"] = bid_month

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    result = "✅ WIN" if pred == 1 else "❌ LOSE"
    st.markdown(f"### 🎯 Prediction: {result} ({proba:.2%} confidence)")

    st.subheader("🧠 Why this outcome?")
    top_feats = importance.sort_values(ascending=False).head(3).index.tolist()
    for feat in top_feats:
        val = input_df[feat].values[0]
        avg = X[feat].mean()
        direction = "higher" if val > avg else "lower"
        st.write(f"- **{feat}** is {direction} than average ({val:.2f} vs {avg:.2f})")

    st.subheader("🛠 Suggestions to Improve")
    for feat in top_feats:
        tip = "Try reducing" if input_df[feat].values[0] > X[feat].mean() else "Try optimizing"
        st.write(f"- {tip} **{feat}**")

    report = f"Prediction: {result} ({proba:.2%})\n\nTop Factors:\n"
    for feat in top_feats:
        report += f"- {feat}: input={input_df[feat].values[0]:.2f}, avg={X[feat].mean():.2f}\n"
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
    st.download_button("📥 Download Summary Report", buffer, file_name="prediction_summary.txt")

# Phase 3: Batch Upload Preview
st.subheader("📦 Upload CSV for Batch Predictions (Coming Soon)")
batch_file = st.file_uploader("Upload .csv file", type=["csv"])
if batch_file:
    df_batch = pd.read_csv(batch_file)
    st.dataframe(df_batch.head())
    st.warning("Batch prediction is under construction.")

# Phase 4 Placeholder: Email and Admin Controls
st.subheader("📧 Email Result (Coming Soon)")
st.text_input("Enter your email")
st.button("Send Report (disabled)")
