import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp

# Helper function to plot loss and accuracy
def plot_history(history, title):
    plt.figure(figsize=(10, 6))
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss', color='red')
    plt.title(f'{title} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Plot Accuracy (if available)
    if 'accuracy' in history.history:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Accuracy', color='blue')
        plt.title(f'{title} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.show()

# 1. LINEAR REGRESSION (Titanic)
def run_linear_regression():
    print("--- Starting Linear Regression (Titanic Dataset) ---")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = os.path.join(current_dir, "train.csv")
    eval_file = os.path.join(current_dir, "eval.csv")

    if not os.path.exists(train_file):
        print(f"Error: {train_file} not found.")
        return

    dftrain = pd.read_csv(train_file)
    dfeval = pd.read_csv(eval_file)
    y_train = dftrain.pop('survived')
    y_eval = dfeval.pop('survived')

    CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck', 'embark_town']
    NUMERIC_COLUMNS = ['age', 'fare']
    
    USED_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

    # --- DATA CLEANING ---
    for feature in NUMERIC_COLUMNS:
        dftrain[feature] = pd.to_numeric(dftrain[feature], errors='coerce').fillna(0.0).astype('float32')
        dfeval[feature] = pd.to_numeric(dfeval[feature], errors='coerce').fillna(0.0).astype('float32')

    for feature in CATEGORICAL_COLUMNS:
        dftrain[feature] = dftrain[feature].astype(str).replace('nan', 'Unknown')
        dfeval[feature] = dfeval[feature].astype(str).replace('nan', 'Unknown')
        
    # Create Inputs
    inputs = {}
    for name in USED_COLUMNS:
        if name in CATEGORICAL_COLUMNS:
            dtype = tf.string
        else:
            dtype = tf.float32
        inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)

    # Process Inputs
    concat_features = []
    
    for name in NUMERIC_COLUMNS:
        concat_features.append(inputs[name])

    for name in CATEGORICAL_COLUMNS:
        vocab = dftrain[name].unique()
        lookup = tf.keras.layers.StringLookup(vocabulary=vocab, output_mode='one_hot')
        concat_features.append(lookup(inputs[name]))

    x = tf.keras.layers.Concatenate()(concat_features)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    def df_to_dataset(dataframe, labels, shuffle=True, batch_size=32):
        ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        if shuffle:
            ds = ds.shuffle(1000)
        ds = ds.batch(batch_size)
        return ds

    train_ds = df_to_dataset(dftrain, y_train)
    eval_ds = df_to_dataset(dfeval, y_eval, shuffle=False)

    # Train (and capture history)
    print("Training Linear Model...")
    # Increased epochs to 20 to make the graph more interesting
    history = model.fit(train_ds, epochs=20, verbose=1)
    
    # VISUALIZE
    print("Displaying Linear Regression Plot (Close window to continue)...")
    plot_history(history, "Titanic Linear Regression")

    # Evaluate
    loss, accuracy = model.evaluate(eval_ds, verbose=0)
    print(f"\nModel Accuracy: {accuracy:.2f}")
    print("-" * 30 + "\n")


# 2. CLASSIFICATION (Iris)
def run_classification():
    print("--- Starting Classification (Iris Dataset) ---")

    CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(current_dir, "iris_training.csv")
    test_path = os.path.join(current_dir, "iris_test.csv")

    if not os.path.exists(train_path):
        print("Error: Iris dataset not found.")
        return

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)

    train_y = train.pop('Species')
    test_y = test.pop('Species')

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(30, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Training Classification Model...")
    history = model.fit(train.values, train_y, epochs=50, batch_size=256, verbose=0)

    # VISUALIZE
    print("Displaying Classification Plot (Close window to continue)...")
    plot_history(history, "Iris Classification")

    loss, accuracy = model.evaluate(test.values, test_y, verbose=0)
    print(f"\nTest set accuracy: {accuracy:0.3f}\n")
    print("-" * 30 + "\n")


# 3. HIDDEN MARKOV MODEL (Weather)
def run_hidden_markov_model():
    print("--- Starting Hidden Markov Model (Weather Prediction) ---")
    print("(Note: This HMM uses pre-defined probabilities, so there is no training loss to plot.)")
    
    tfd = tfp.distributions

    initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) 
    transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
    observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])

    model = tfd.HiddenMarkovModel(
        initial_distribution=initial_distribution,
        transition_distribution=transition_distribution,
        observation_distribution=observation_distribution,
        num_steps=7)

    try:
        mean_temperatures = model.mean()
        print("Predicted average temperatures for the next 7 days:")
        print(mean_temperatures.numpy())
    except Exception as e:
        print(f"HMM Error: {e}")
    print("-" * 30 + "\n")


def main():
    run_linear_regression()
    run_classification()
    run_hidden_markov_model()

if __name__ == "__main__":
    main()