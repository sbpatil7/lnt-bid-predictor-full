import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

st.set_page_config(page_title="L&T Bid Predictor – Smart Adjust", layout="wide")

# Load static Excel file (no upload feature)
def load_data():
    df = pd.read_excel("data.xlsx")

    cat_cols = [
        'Product Type', 'Project Region', 'Project Geography/ Location', 'Licensor',
        'Shell (MOC)', 'Weld Overlay/ Clad Applicable (Yes or No)', 'Sourcing Restrictions (Yes or No)'
    ]
    num_cols_all = [
        'ID (mm)', 'Weight (MT)', 'Price($ / Kg)', 'Unit Cost($)', 'Total Cost($)',
        'Off top (%)', 'Unit Price($)', 'Total price($)', 'Cost ($ / Kg)'
    ]
    num_cols = [col for col in num_cols_all if col in df.columns]

    if 'Bid Date' in df.columns:
        df['Bid Month'] = pd.to_datetime(df['Bid Date']).dt.month

    df = df.dropna(subset=['Result(w/L)'])
    df.fillna(method='ffill', inplace=True)

    le_dict, inv_le_dict = {}, {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
            inv_le_dict[col] = dict(zip(le.transform(le.classes_), le.classes_))
    cat_cols = [col for col in cat_cols if col in df.columns]

    all_cols = cat_cols + num_cols
    if 'Bid Month' in df.columns:
        all_cols.append('Bid Month')

    X = df[all_cols].copy()
    y = LabelEncoder().fit_transform(df['Result(w/L)'])
    scaler = StandardScaler()
    if num_cols:
        X[num_cols] = scaler.fit_transform(X[num_cols])

    return X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, df

X, y, scaler, le_dict, inv_le_dict, cat_cols, num_cols, full_data = load_data()

@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = XGBClassifier(n_estimators=250, learning_rate=0.1, max_depth=4, eval_metric='logloss', use_label_encoder=False)
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(X, y)

st.title("🏗️ L&T Bid Predictor – Smart Adjust")

st.sidebar.header("📊 Model Performance")
st.sidebar.metric("Accuracy", f"{accuracy_score(y_test, model.predict(X_test)):.2%}")
st.sidebar.text("Classification Report")
st.sidebar.text(classification_report(y_test, model.predict(X_test)))

st.subheader("🔍 Predict Individual Bid")
user_input = {}
col1, col2 = st.columns(2)
with col1:
    for col in cat_cols:
        user_input[col] = st.selectbox(col, list(inv_le_dict[col].values()))
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
    if num_cols:
        input_df[num_cols] = scaler.transform(input_df[num_cols])
    if bid_month:
        input_df["Bid Month"] = bid_month

    input_df = input_df.reindex(columns=X.columns, fill_value=0)
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][pred]
    result = "✅ WIN" if pred == 1 else "❌ LOSE"
    st.markdown(f"### 🎯 Prediction: {result} ({proba:.2%} confidence)")

    st.subheader("🧠 Why this outcome?")
    top_feats = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(3).index.tolist()
    for feat in top_feats:
        val = input_df[feat].values[0]
        avg = X[feat].mean()
        if feat in cat_cols:
            val_name = list(inv_le_dict[feat].values())[list(inv_le_dict[feat].keys()).index(int(val))]
            st.write(f"- **{feat}** is `{val_name}`, most common is `{inv_le_dict[feat][X[feat].value_counts().idxmax()]}`")
        else:
            direction = "higher" if val > avg else "lower"
            st.write(f"- **{feat}** is {direction} than average ({val:.2f} vs {avg:.2f})")

    if result == "❌ LOSE":
        st.subheader("🛠 Suggested Changes to Convert to WIN")
        suggested_changes = {}
        for feat in top_feats:
            if feat in num_cols:
                new_val = round(float(X[feat].mean()), 2)
                suggested_changes[feat] = (float(input_df[feat].values[0]), new_val)
                st.write(f"- **{feat}**: {input_df[feat].values[0]:.2f} ➝ {new_val:.2f}")
            elif feat in cat_cols:
                current_code = int(input_df[feat].values[0])
                common_code = int(X[feat].value_counts().idxmax())
                if current_code != common_code:
                    old_label = inv_le_dict[feat][current_code]
                    new_label = inv_le_dict[feat][common_code]
                    suggested_changes[feat] = (old_label, new_label)
                    st.write(f"- **{feat}**: '{old_label}' ➝ '{new_label}'")

        if st.button("✅ Apply Suggested Changes and Predict Again"):
            for feat, (old, new) in suggested_changes.items():
                if feat in cat_cols:
                    input_df[feat] = le_dict[feat].transform([new])[0]
                else:
                    input_df[feat] = new
            input_df = input_df.reindex(columns=X.columns, fill_value=0)
            new_pred = model.predict(input_df)[0]
            new_proba = model.predict_proba(input_df)[0][new_pred]
            new_result = "✅ WIN" if new_pred == 1 else "❌ Still LOSE"
            st.markdown(f"### 🔁 New Prediction After Changes: {new_result} ({new_proba:.2%} confidence)")
            if new_result == "✅ WIN":
                st.success("🎯 Prediction improved to WIN!")
            else:
                st.warning("⚠️ Still predicted as LOSS after suggested changes.")

    buffer = BytesIO()
    report = f"Prediction: {result} ({proba:.2%} confidence)\n\nTop Features:\n"
    for feat in top_feats:
        report += f"- {feat}: {input_df[feat].values[0]}\n"
    buffer.write(report.encode())
    buffer.seek(0)
    st.download_button("📥 Download Summary Report", buffer, file_name="prediction_summary.txt")
