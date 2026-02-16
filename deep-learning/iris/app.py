import os
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import joblib


# =========================
# PATH CONFIG (FIXED)
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.pkl")

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================
# STREAMLIT CONFIG
# =========================

st.set_page_config(
    page_title="Iris Deep Learning App",
    layout="wide"
)

st.title("🌸 Iris Prediction App (TensorFlow Deep Learning)")


# =========================
# LOAD DATA
# =========================

@st.cache_data
def load_data():

    iris = load_iris()

    df = pd.DataFrame(
        iris.data,
        columns=iris.feature_names
    )

    df["Species"] = iris.target
    df["Species"] = df["Species"].map(
        dict(enumerate(iris.target_names))
    )

    return df, iris


df, iris = load_data()


# =========================
# TRAIN MODEL FUNCTION
# =========================

def train_and_save_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    X = df[iris.feature_names]
    y = df["Species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=SEED
    )

    # Label encoding
    encoder = LabelEncoder()
    y_train_enc = encoder.fit_transform(y_train)
    y_test_enc = encoder.transform(y_test)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # One-hot encoding
    y_train_cat = tf.keras.utils.to_categorical(y_train_enc)

    # Model architecture
    model = tf.keras.Sequential([

        tf.keras.layers.Input(shape=(4,)),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            16,
            activation="relu"
        ),

        tf.keras.layers.Dense(
            3,
            activation="softmax"
        )

    ])

    model.compile(

        optimizer="adam",

        loss="categorical_crossentropy",

        metrics=["accuracy"]

    )

    model.fit(

        X_train_scaled,
        y_train_cat,

        epochs=100,
        batch_size=8,

        verbose=0

    )

    # Save artifacts
    model.save(MODEL_PATH, save_format="keras")

    joblib.dump(scaler, SCALER_PATH)

    joblib.dump(encoder, ENCODER_PATH)


# =========================
# LOAD MODEL FUNCTION
# =========================

@st.cache_resource
def load_model():

    # Always ensure directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Check if files exist and are not empty
    model_exists = os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0
    scaler_exists = os.path.exists(SCALER_PATH) and os.path.getsize(SCALER_PATH) > 0
    encoder_exists = os.path.exists(ENCODER_PATH) and os.path.getsize(ENCODER_PATH) > 0

    if not (model_exists and scaler_exists and encoder_exists):

        st.info("Training model for first time...")

        train_and_save_model()

        st.success("Model trained and saved.")

    model = tf.keras.models.load_model(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    encoder = joblib.load(ENCODER_PATH)

    return model, scaler, encoder


model, scaler, encoder = load_model()


# =========================
# PREPARE TEST SET
# =========================

X = df[iris.feature_names]
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=SEED
)

X_test_scaled = scaler.transform(X_test)

y_pred_probs = model.predict(X_test_scaled, verbose=0)

y_pred = encoder.inverse_transform(
    np.argmax(y_pred_probs, axis=1)
)


# =========================
# SIDEBAR
# =========================

st.sidebar.header("Navigation")

page = st.sidebar.radio(

    "Go to",

    [
        "EDA (Training Data)",
        "Model Evaluation",
        "Make Prediction"
    ]

)


# =========================
# PAGE 1: EDA
# =========================

if page == "EDA (Training Data)":

    st.header("Exploratory Data Analysis")

    st.dataframe(df.head())

    fig, ax = plt.subplots()

    sns.countplot(
        x="Species",
        data=df,
        ax=ax
    )

    st.pyplot(fig)

    fig = sns.pairplot(
        df,
        hue="Species"
    )

    st.pyplot(fig)

    fig, ax = plt.subplots()

    sns.heatmap(

        df.iloc[:, :-1].corr(),

        annot=True,

        cmap="coolwarm",

        ax=ax

    )

    st.pyplot(fig)


# =========================
# PAGE 2: EVALUATION
# =========================

elif page == "Model Evaluation":

    st.header("Model Evaluation")

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    st.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    st.dataframe(
        pd.DataFrame(report).transpose()
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=encoder.classes_
    )

    fig, ax = plt.subplots()

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=encoder.classes_,

        yticklabels=encoder.classes_,

        ax=ax

    )

    st.pyplot(fig)


# =========================
# PAGE 3: PREDICTION
# =========================

else:

    st.header("Make Prediction")

    col1, col2 = st.columns(2)

    with col1:

        sepal_length = st.slider(
            "Sepal Length",
            4.0, 8.0, 5.1
        )

        sepal_width = st.slider(
            "Sepal Width",
            2.0, 4.5, 3.5
        )

    with col2:

        petal_length = st.slider(
            "Petal Length",
            1.0, 7.0, 1.4
        )

        petal_width = st.slider(
            "Petal Width",
            0.1, 2.5, 0.2
        )

    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    if st.button("Predict"):

        input_scaled = scaler.transform(
            input_data
        )

        probs = model.predict(
            input_scaled,
            verbose=0
        )[0]

        pred_index = np.argmax(probs)

        prediction = encoder.inverse_transform(
            [pred_index]
        )[0]

        st.success(
            f"Predicted Species: {prediction}"
        )

        prob_df = pd.DataFrame({

            "Species": encoder.classes_,

            "Probability": probs

        })

        st.bar_chart(
            prob_df.set_index("Species")
        )
