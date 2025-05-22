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

st.set_page_config(page_title="L&T Bid Predictor - Smart Adjust", layout="wide")

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

st.title("🏗️ L&T Bid Predictor – Phase 5: Smart Adjust")

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

st.subheader("🔍 Predict Individual Bid")
user_input = {}
col1, col2 = st.columns(2)
with col1:
    for col in cat_cols:
        user_input[col] = st.selectbox(col, list(inv_le_dict[col].keys()))
with col2:
    for col in num_cols:
        user_input[col] = st.slider(col, float(full_data[col].min()), float(full_data[col].max()), float(full_data[col].median()))

if 'Bid Month' in X.columns:
    bid_month = st.slider("Bid Month", 1, 12, 6)
else:
    bid_month = None

if st.button("🚩 Predict Now"):
    input_df = pd.DataFrame([user_input])
    for col in cat_cols:
        input_df[col] = le_dict[col].transform([input_df[col][0]])
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    if bid_month:
        input_df["Bid Month"] = bid_month

    input_df = input_df.reindex(columns=X.columns, fill_value=0)

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

    # 🔁 Smart Auto Adjust
    if result == "❌ LOSE":
        st.subheader("🔁 Make this Bid WIN (Auto Adjust)")
        adjusted_input_df = input_df.copy()

        for feat in top_feats:
            if feat in num_cols:
                adjusted_input_df[feat] = X[feat].mean() * 0.9
            elif feat in cat_cols:
                most_common = X[feat].value_counts().idxmax()
                adjusted_input_df[feat] = most_common

        adjusted_input_df = adjusted_input_df.reindex(columns=X.columns, fill_value=0)
        new_pred = model.predict(adjusted_input_df)[0]
        new_proba = model.predict_proba(adjusted_input_df)[0][new_pred]
        new_result = "✅ WIN" if new_pred == 1 else "❌ Still LOSE"

        if new_result == "✅ WIN":
            st.success("🎯 Success: Auto-adjusted bid turns into WIN!")
            st.markdown(f"**Updated Confidence:** {new_proba:.2%}")
            st.markdown("#### 🔧 Adjusted Values:")
            for feat in top_feats:
                old = input_df[feat].values[0]
                new = adjusted_input_df[feat].values[0]
                if old != new:
                    st.write(f"- {feat}: {old:.2f} ➝ {new:.2f}")
        else:
            st.warning("⚠️ Tried all adjustments but still predicted as LOSS")

    # 📥 Download Report
    report = f"Prediction: {result} ({proba:.2%})\\n\\nTop Factors:\\n"
    for feat in top_feats:
        report += f"- {feat}: input={input_df[feat].values[0]:.2f}, avg={X[feat].mean():.2f}\\n"
    buffer = BytesIO()
    buffer.write(report.encode())
    buffer.seek(0)
    st.download_button("📥 Download Summary Report", buffer, file_name="prediction_summary.txt")
