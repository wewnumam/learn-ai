import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------
# App Config
# -----------------------
st.set_page_config(page_title="Iris Prediction App", layout="wide")
st.title("🌸 Iris Flower Prediction App (Ensemble Model)")

# -----------------------
# Load Dataset
# -----------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["Species"] = iris.target
df["Species"] = df["Species"].map(dict(enumerate(iris.target_names)))

# -----------------------
# Sidebar
# -----------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["EDA (Training Data)", "Model Evaluation", "Make Prediction"]
)

# -----------------------
# Train/Test Split
# -----------------------
X = df[iris.feature_names]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------
# Ensemble Model
# -----------------------
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(random_state=42)

ensemble_model = VotingClassifier(
    estimators=[("rf", rf), ("gb", gb)],
    voting="soft"
)

ensemble_model.fit(X_train, y_train)

# -----------------------
# Predictions
# -----------------------
y_pred = ensemble_model.predict(X_test)

# =====================================================
# PAGE 1: EDA
# =====================================================
if page == "EDA (Training Data)":
    st.header("📊 Exploratory Data Analysis (EDA)")

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Class Distribution")
    fig, ax = plt.subplots()
    sns.countplot(x="Species", data=df, ax=ax)
    st.pyplot(fig)

    st.subheader("Feature Distributions")
    fig = sns.pairplot(df, hue="Species")
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.iloc[:, :-1].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# =====================================================
# PAGE 2: MODEL EVALUATION
# =====================================================
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation (Test Data)")

    accuracy = accuracy_score(y_test, y_pred)
    st.metric("Accuracy", f"{accuracy:.2f}")

    st.subheader("Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=iris.target_names)

    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =====================================================
# PAGE 3: PREDICTION
# =====================================================
else:
    st.header("🔮 Make a Prediction")

    st.write("Enter flower measurements:")

    col1, col2 = st.columns(2)

    with col1:
        sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
        sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)

    with col2:
        petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
        petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    if st.button("Predict"):
        prediction = ensemble_model.predict(input_data)[0]
        probabilities = ensemble_model.predict_proba(input_data)[0]

        st.success(f"🌼 Predicted Species: **{prediction}**")

        st.subheader("Prediction Probabilities")
        prob_df = pd.DataFrame(
            {
                "Species": ensemble_model.classes_,
                "Probability": probabilities
            }
        )
        st.bar_chart(prob_df.set_index("Species"))
