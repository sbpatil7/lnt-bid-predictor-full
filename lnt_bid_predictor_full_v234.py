import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="L&T Bid Predictor – LightGBM Model", layout="wide")

def load_data():
    df = pd.read_excel("data.xlsx")

    # Safe column check
    required_cols = ['Result(w/L)', 'Weight (MT)', 'Price($ / Kg)', 'Total price($)']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Preprocessing
    cat_cols = [
        'Product Type', 'Project Region', 'Project Geography/ Location', 'Licensor',
        'Shell (MOC)', 'Weld Overlay/ Clad Applicable (Yes or No)', 'Sourcing Restrictions (Yes or No)'
    ]
    num_cols = ['ID (mm)', 'Weight (MT)', 'Price($ / Kg)', 'Total Cost($)', 'Total price($)', 'Cost ($ / Kg)']

    df = df.dropna(subset=['Result(w/L)'])
    df.fillna(method='ffill', inplace=True)

    # Business logic feature: Total Bid Value
    df['Total Bid Value'] = df['Weight (MT)'] * df['Price($ / Kg)']
    num_cols.append('Total Bid Value')

    le_dict, inv_le_dict = {}, {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            inv_le_dict[col] = dict(zip(le.transform(le.classes_), le.classes_))
    cat_cols = [col for col in cat_cols if col in df.columns]

    all_cols = cat_cols + num_cols
    X = df[all_cols].copy()
    y = LabelEncoder().fit_transform(df['Result(w/L)'])

    scaler = MinMaxScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, df

# Load and prepare data
X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, full_data = load_data()

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.1, max_depth=6)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

# Streamlit UI
st.title("🏗️ L&T Bid Predictor – Price Aware Model")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
st.sidebar.text("Classification Report")
st.sidebar.text(classification_report(y_test, model.predict(X_test)))

st.subheader("🔍 Enter Bid Parameters")
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
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    result = "✅ WIN" if pred == 1 else "❌ LOSE"
    st.markdown(f"### 🎯 Prediction: {result} ({proba:.2%} confidence)")

    st.subheader("🧠 Top Influencing Features")
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    top_feats = feature_importance.sort_values(ascending=False).head(3)

    for feat in top_feats.index:
        st.write(f\"- **{feat}**: Importance Score = {top_feats[feat]:.2f}\")

    st.subheader(\"📊 Total Bid Insight\")
    bid_val = user_input['Price($ / Kg)'] * user_input['Weight (MT)']
    st.markdown(f\"**💰 Total Bid Value:** ${bid_val:,.2f}\")

    # Summary report
    buffer = BytesIO()
    report = f\"Prediction: {result} ({proba:.2%})\\nTop Features:\\n\"
    for feat in top_feats.index:
        report += f\"- {feat}: importance = {top_feats[feat]:.2f}\\n\"
    buffer.write(report.encode())
    buffer.seek(0)
    st.download_button(\"📥 Download Summary Report\", buffer, file_name=\"bid_prediction_summary.txt\")
