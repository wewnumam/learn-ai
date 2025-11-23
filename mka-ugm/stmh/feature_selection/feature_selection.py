import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random

# --------------------------------------------
# Genetic Algorithm Functions
# --------------------------------------------

def init_population(pop_size, n_features):
    return [np.random.randint(0, 2, n_features) for _ in range(pop_size)]

def fitness_func(model_func, X_train, X_test, y_train, y_test, chromosome):
    idx = np.where(chromosome == 1)[0]
    if len(idx) == 0:
        return 0
    model = model_func()
    model.fit(X_train[:, idx], y_train)
    preds = model.predict(X_test[:, idx])
    return accuracy_score(y_test, preds)

def selection(population, fitnesses):
    probs = fitnesses / np.sum(fitnesses)
    return population[np.random.choice(len(population), p=probs)]

def crossover(parent1, parent2, rate):
    if random.random() < rate:
        point = random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2
    return parent1.copy(), parent2.copy()

def mutate(chromosome, rate):
    for i in range(len(chromosome)):
        if random.random() < rate:
            chromosome[i] = 1 - chromosome[i]
    return chromosome

# --------------------------------------------
# Streamlit UI
# --------------------------------------------

st.title("Genetic Algorithm Feature Selection for Stress Level Prediction")

# Load data
file_path = "StressLevelDataset.csv"
df = pd.read_csv(file_path)

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Sidebar Parameters
st.sidebar.header("GA & Model Parameters")
test_size = st.sidebar.slider("Train/Test Split", 0.1, 0.5, 0.2)
pop_size = st.sidebar.slider("Population Size", 10, 200, 50)
generations = st.sidebar.slider("Max Generations", 10, 200, 50)
crossover_rate = st.sidebar.slider("Crossover Rate", 0.1, 1.0, 0.7)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.01, 0.5, 0.1)
runs = st.sidebar.slider("Jumlah Run Pengujian", 1, 20, 5)

st.sidebar.subheader("Model Parameters")
rf_n = st.sidebar.slider("Random Forest n_estimators", 10, 500, 100)
knn_k = st.sidebar.slider("KNN k", 1, 30, 5)
svm_c = st.sidebar.slider("SVM C", 0.1, 10.0, 1.0)

run_btn = st.sidebar.button("RUN")

# Model factory functions
rf_model = lambda: RandomForestClassifier(n_estimators=rf_n)
knn_model = lambda: KNeighborsClassifier(n_neighbors=knn_k)
svm_model = lambda: SVC(C=svm_c)

# --------------------------------------------
# Baseline Metrics
# --------------------------------------------

st.header("1. Distribusi Stress Level")
st.bar_chart(df["stress_level"].value_counts())

X = df.drop("stress_level", axis=1).values
y = df["stress_level"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

models = {"Random Forest": rf_model, "KNN": knn_model, "SVM": svm_model}
baseline_results = []

for name, m in models.items():
    model = m()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    baseline_results.append([
        name,
        accuracy_score(y_test, preds),
        precision_score(y_test, preds, average="macro"),
        recall_score(y_test, preds, average="macro")
    ])

st.header("2. Baseline Metrics (Semua Fitur)")
st.table(pd.DataFrame(baseline_results, columns=["Model", "Accuracy", "Precision", "Recall"]))

# --------------------------------------------
# Run GA
# --------------------------------------------

if run_btn:
    n_features = X.shape[1]

    st.header("3. Pelatihan GA")
    tab1, tab2, tab3 = st.tabs(["Random Forest", "KNN", "SVM"])

    model_tabs = [
        (tab1, rf_model, "Random Forest"),
        (tab2, knn_model, "KNN"),
        (tab3, svm_model, "SVM")
    ]

    top_features_all = {}

    for tab, model_func, model_name in model_tabs:
        with tab:
            st.subheader(model_name)

            best_runs_acc = []
            best_features = None

            progress = st.progress(0)

            for r in range(runs):
                progress.progress((r+1)/runs)

                population = init_population(pop_size, n_features)
                best_fit_progress = []

                for g in range(generations):
                    fitnesses = np.array([
                        fitness_func(model_func, X_train, X_test, y_train, y_test, chrom)
                        for chrom in population
                    ])
                    best_fit_progress.append(np.max(fitnesses))

                    new_pop = []
                    while len(new_pop) < pop_size:
                        p1 = selection(population, fitnesses)
                        p2 = selection(population, fitnesses)
                        c1, c2 = crossover(p1, p2, crossover_rate)
                        new_pop.append(mutate(c1, mutation_rate))
                        if len(new_pop) < pop_size:
                            new_pop.append(mutate(c2, mutation_rate))
                    population = new_pop

                final_fitness = np.array([
                    fitness_func(model_func, X_train, X_test, y_train, y_test, chrom)
                    for chrom in population
                ])
                best_idx = np.argmax(final_fitness)
                best_acc = final_fitness[best_idx]
                best_runs_acc.append(best_acc)

                if best_features is None or best_acc > max(best_runs_acc):
                    best_features = population[best_idx]
                    top_features_all[model_name] = best_features

            st.subheader("Perkembangan Fitness Terbaik Per Generasi")
            st.line_chart(best_fit_progress)

            st.subheader("Distribusi Akurasi Setiap Run")
            st.line_chart(best_runs_acc)

            st.write("Fitur Terpilih: ", list(np.where(best_features == 1)[0]))

    # --------------------------------------------
    # Intersection Results
    # --------------------------------------------
    st.header("4. Irisan Top Fitur GA (RF ∩ KNN ∩ SVM)")

    inter = (top_features_all["Random Forest"] *
             top_features_all["KNN"] *
             top_features_all["SVM"])

    idx = np.where(inter == 1)[0]
    if len(idx) == 0:
        st.write("Tidak ada irisan fitur.")
    else:
        st.write("Fitur Irisan: ", list(idx))

        # Hitung metrik dengan fitur irisan
        model_inter_results = []
        for name, m in models.items():
            if len(idx) == 0:
                model_inter_results.append([name, 0, 0, 0])
            else:
                model = m()
                model.fit(X_train[:, idx], y_train)
                preds = model.predict(X_test[:, idx])
                model_inter_results.append([
                    name,
                    accuracy_score(y_test, preds),
                    precision_score(y_test, preds, average="macro"),
                    recall_score(y_test, preds, average="macro")
                ])

        st.table(pd.DataFrame(model_inter_results, columns=["Model", "Accuracy", "Precision", "Recall"]))

    # --------------------------------------------
    # 5. Tabel Perbandingan Akhir
    # --------------------------------------------
    st.header("5. Perbandingan Akhir Semua Metode")

    comparison_rows = []

    # Baseline
    for r in baseline_results:
        comparison_rows.append([r[0] + " (Baseline)", r[1], r[2], r[3]])

    # GA Top (RF, KNN, SVM)
    for model_name, feats in top_features_all.items():
        feat_idx = np.where(feats == 1)[0]
        if len(feat_idx) == 0:
            comparison_rows.append([model_name + " (GA)", 0, 0, 0])
        else:
            model = models[model_name]()
            model.fit(X_train[:, feat_idx], y_train)
            preds = model.predict(X_test[:, feat_idx])
            comparison_rows.append([
                model_name + " (GA)",
                accuracy_score(y_test, preds),
                precision_score(y_test, preds, average="macro"),
                recall_score(y_test, preds, average="macro")
            ])

    # Intersection
    inter = (top_features_all["Random Forest"] * top_features_all["KNN"] * top_features_all["SVM"])
    idx = np.where(inter == 1)[0]
    if len(idx) == 0:
        comparison_rows.append(["Intersection (RF ∩ KNN ∩ SVM)", 0, 0, 0])
    else:
        # Gunakan model default: Random Forest untuk evaluasi intersection
        model = rf_model()
        model.fit(X_train[:, idx], y_train)
        preds = model.predict(X_test[:, idx])
        comparison_rows.append([
            "Intersection (RF ∩ KNN ∩ SVM)",
            accuracy_score(y_test, preds),
            precision_score(y_test, preds, average="macro"),
            recall_score(y_test, preds, average="macro")
        ])

    st.table(pd.DataFrame(comparison_rows, columns=["Metode", "Accuracy", "Precision", "Recall"]))

st.success("Selesai! Siap dijalankan 🚀")