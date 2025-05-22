import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="L&T Bid Predictor – EBM & Price Logic", layout="wide")

# Load data with price logic features
def load_data():
    df = pd.read_excel("data.xlsx")

    cat_cols = [
        'Product Type', 'Project Region', 'Project Geography/ Location', 'Licensor',
        'Shell (MOC)', 'Weld Overlay/ Clad Applicable (Yes or No)', 'Sourcing Restrictions (Yes or No)'
    ]
    num_cols = ['ID (mm)', 'Weight (MT)', 'Price($ / Kg)', 'Total Cost($)', 'Total price($)', 'Cost ($ / Kg)']

    df = df.dropna(subset=['Result(w/L)', 'Weight (MT)', 'Price($ / Kg)', 'Total price($)'])
    df.fillna(method='ffill', inplace=True)

    # Business logic feature: Bid Value
    df['Total Bid Value'] = df['Weight (MT)'] * df['Price($ / Kg)']
    num_cols.append('Total Bid Value')

    le_dict, inv_le_dict = {}, {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le
        inv_le_dict[col] = dict(zip(le.transform(le.classes_), le.classes_))

    X = df[cat_cols + num_cols].copy()
    y = LabelEncoder().fit_transform(df['Result(w/L)'])
    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, df

X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, full_data = load_data()

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = ExplainableBoostingClassifier(interactions=5)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

st.title("🏗️ L&T Bid Predictor – EBM with Total Price Logic")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
st.sidebar.text("Classification Report")
st.sidebar.text(classification_report(y_test, model.predict(X_test)))

st.subheader("🔍 Predict Your Bid")
user_input = {}
col1, col2 = st.columns(2)
with col1:
    for col in cat_cols:
        user_input[col] = st.selectbox(col, list(inv_le_dict[col].values()))
with col2:
    for col in num_cols:
        min_val = float(full_data[col].min())
        max_val = float(full_data[col].max())
        default = float(full_data[col].median())
        user_input[col] = st.slider(col, min_val, max_val, default)

if st.button("🚩 Predict Bid Result"):
    input_df = pd.DataFrame([user_input])
    for col in cat_cols:
        input_df[col] = le_dict[col].transform([input_df[col][0]])
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    result = "✅ WIN" if pred == 1 else "❌ LOSE"
    st.markdown(f"### 🎯 Prediction: {result} ({proba:.2%} confidence)")

    st.subheader("🧠 Feature Effects (Top 3)")
    ebm_feats = pd.DataFrame({
        'feature': model.feature_names,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False).head(3)

    for _, row in ebm_feats.iterrows():
        st.write(f"- **{row['feature']}** had high impact on outcome (Importance: {row['importance']:.4f})")

    st.subheader("📊 Key Bid Metric")
    raw_price = float(user_input['Price($ / Kg)'])
    weight = float(user_input['Weight (MT)'])
    bid_value = raw_price * weight
    st.markdown(f"**💰 Total Bid Value (Price × Weight)** = ${bid_value:,.2f}")

    buffer = BytesIO()
    report = f"Prediction: {result} ({proba:.2%} confidence)\n\nTop Features:\n"
    for _, row in ebm_feats.iterrows():
        report += f"- {row['feature']}: Importance = {row['importance']:.4f}\n"
    buffer.write(report.encode())
    buffer.seek(0)
    st.download_button("📥 Download Summary Report", buffer, file_name="prediction_summary.txt")
