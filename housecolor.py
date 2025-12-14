import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# ------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------
st.set_page_config(page_title="🏡 House Price Prediction AI App",
                   layout="wide",
                   page_icon="🏠")

# ------------------------------------------
# HEADER SECTION (MODERN STYLE)
# ------------------------------------------
st.markdown("""
    <h1 style='text-align:center;color:#4CAF50;'>🏡 AI-Based House Price Prediction</h1>
    <p style='text-align:center;font-size:18px;'>
        A complete <b>Machine Learning + Data Engineering</b> pipeline with advanced UI.
    </p>
""", unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv")

df = load_data()

# ------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------
st.sidebar.header("⚙️ App Controls")
st.sidebar.write("Navigate through the app")

page = st.sidebar.radio("Go to", ["📊 Dataset", "🤖 Model Training", "🏠 Prediction", "⬇ Download Results"])

# ------------------------------------------
# PAGE 1: DATASET
# ------------------------------------------
if page == "📊 Dataset":
    st.header("📌 Dataset Overview")
    st.dataframe(df.head())

    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("🔍 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    corr = df.corr()
    heatmap = ax.imshow(corr, cmap="coolwarm")
    plt.colorbar(heatmap)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90)
    ax.set_yticklabels(corr.columns)
    st.pyplot(fig)

# ------------------------------------------
# PAGE 2: MODEL TRAINING
# ------------------------------------------
elif page == "🤖 Model Training":
    st.header("🤖 Model Training & Evaluation")

    X = df.drop("medv", axis=1)
    y = df["medv"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2:.3f}")
    col2.metric("Mean Absolute Error", f"{mae:.3f}")

    st.subheader("📈 Feature Importance")
    fig, ax = plt.subplots()
    importances = model.feature_importances_
    indices = np.argsort(importances)

    ax.barh(range(len(indices)), importances[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(X.columns[indices])
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance")

    st.pyplot(fig)

# ------------------------------------------
# PAGE 3: PREDICTION
# ------------------------------------------
elif page == "🏠 Prediction":

    st.header("🏠 Predict House Price")

    X = df.drop("medv", axis=1)
    feature_order = list(X.columns)

    user_values = {}
    for feature in feature_order:
        user_values[feature] = st.number_input(
            feature,
            min_value=float(df[feature].min()),
            max_value=float(df[feature].max()),
            value=float(df[feature].mean())
        )

    user_df = pd.DataFrame([[user_values[f] for f in feature_order]], columns=feature_order)

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X, df["medv"])

    if st.button("🔮 Predict Price"):
        price = model.predict(user_df)[0]
        st.success(f"### Predicted Price: **${price * 1000:,.2f} USD**")

# ------------------------------------------
# PAGE 4: DOWNLOAD PREDICTIONS
# ------------------------------------------
elif page == "⬇ Download Results":
    st.header("⬇ Download Sample Predictions")

    X = df.drop("medv", axis=1)
    y = df["medv"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
    st.dataframe(pred_df.head())

    csv = pred_df.to_csv(index=False)
    st.download_button("Download Predictions CSV", csv, "predictions.csv", "text/csv")

st.markdown("---")
st.info("💡 This UI is optimized for interviews. Talk about pipelines, feature importance, and ML workflow.")
