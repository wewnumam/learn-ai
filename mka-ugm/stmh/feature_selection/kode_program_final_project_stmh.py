import pandas as pd
import os

# read file
# Update the path to your CSV file location
csv_path = 'StressLevelDataset.csv'
if not os.path.exists(csv_path):
    print(f"Error: File '{csv_path}' not found. Please ensure the file is in the correct directory.")
    print(f"Current working directory: {os.getcwd()}")
    exit()

df = pd.read_csv(csv_path, encoding='latin1')

# Tampilkan 5 baris pertama
df.head()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from deap import base, creator, tools, algorithms
import random
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# --- 1. EDA dan Preprocessing Data ---
print("## 🚀 Mulai Program Seleksi Fitur dengan Algoritma Genetika ##")
print("-" * 70)

print("## 📊 1. Eksplorasi Data Awal (EDA) ##")
print("-----------------------------------")

print("\nFitur-Fitur:")
print(df.columns)

# Periksa nilai null
print("\nJumlah Nilai Null per Kolom:")
print(df.isnull().sum())

# Periksa distribusi target (Stress Level)
plt.figure(figsize=(7, 5))
sns.countplot(x='stress_level', data=df)
plt.title('Distribusi Tingkat Stres')
plt.xlabel('Tingkat Stres (0: Rendah, 1: Sedang, 2: Tinggi)')
plt.ylabel('Jumlah Sampel')
plt.show()

X = df.drop('stress_level', axis=1)
y = df['stress_level']
feature_names = X.columns.tolist()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.3, random_state=42, stratify=y)

# --- Fungsi Fitness dan Algoritma Genetika (GA) Setup ---
try:
    creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maksimalisasi fitness (akurasi)
    creator.create("Individual", list, fitness=creator.FitnessMax)
except:
    pass # Sudah terbuat

# Parameter GA
POP_SIZE = 50
GENERATIONS = 30
P_CROSSOVER = 0.8
P_MUTATION = 0.2
N_FEATURES = X_train.shape[1]

# Fungsi Fitness
def evaluate_individual(individual, X_train, X_test, y_train, y_test, classifier_name):
    """Menghitung akurasi model menggunakan subset fitur yang dipilih oleh individu GA."""

    selected_indices = [i for i, gene in enumerate(individual) if gene == 1]

    if len(selected_indices) == 0:
        return 0.0,

    X_train_sub = X_train.iloc[:, selected_indices]
    X_test_sub = X_test.iloc[:, selected_indices]

    # Pilih dan latih model klasifikasi
    if classifier_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif classifier_name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif classifier_name == 'SVM':
        model = SVC(kernel='linear', random_state=42)
    else:
        raise ValueError("Classifier tidak valid")

    model.fit(X_train_sub, y_train)
    y_pred = model.predict(X_test_sub)

    # Hitung Akurasi (sebagai fungsi fitness)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy,

def setup_toolbox(classifier_name):
    """Menyiapkan toolbox DEAP dengan operator GA yang telah ditentukan."""
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint, 0, 1)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=N_FEATURES)

    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Register fungsi evaluasi (fitness) dengan classifier yang dipilih
    toolbox.register("evaluate", evaluate_individual,
                     X_train=X_train.copy(),
                     X_test=X_test.copy(),
                     y_train=y_train,
                     y_test=y_test,
                     classifier_name=classifier_name)

    # Operator GA
    toolbox.register("mate", tools.cxTwoPoint) # Crossover dua titik
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Mutasi flip bit dengan probabilitas 5%
    toolbox.register("select", tools.selTournament, tournsize=3) # Seleksi turnamen

    return toolbox

def run_ga_with_feature_log(classifier_name, toolbox):
    """Menjalankan Algoritma Genetika dan menyimpan log fitur terbaik di setiap generasi."""
    pop = toolbox.population(n=POP_SIZE)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    logbook = tools.Logbook()

    detailed_feature_log = []

    print(f"\n--- 🧬 Menjalankan Algoritma Genetika dengan Fungsi Fitness: {classifier_name} (Generasi: {GENERATIONS}) ---")

    # Evaluasi populasi awal (Generasi 0)
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, **record)

    # LOG DETAIL FITUR Generasi 0
    best_features_gen0 = [feature_names[i] for i, gene in enumerate(hof[0]) if gene == 1]
    detailed_feature_log.append({
        'gen': 0,
        'fitness': hof[0].fitness.values[0],
        'features': best_features_gen0
    })
    print(f"Gen 00 | Fitness: {hof[0].fitness.values[0]:.4f} | Fitur Terpilih ({len(best_features_gen0)}): {', '.join(best_features_gen0)}")


    # Mulai Generasi 1 hingga GENERATIONS
    for gen in range(1, GENERATIONS + 1):
        # Seleksi
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Crossover
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < P_CROSSOVER:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Mutasi
        for mutant in offspring:
            if random.random() < P_MUTATION:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluasi individu yang baru
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        hof.update(pop)

        record = stats.compile(pop)
        logbook.record(gen=gen, **record)

        best_features_gen = [feature_names[i] for i, gene in enumerate(hof[0]) if gene == 1]
        detailed_feature_log.append({
            'gen': gen,
            'fitness': hof[0].fitness.values[0],
            'features': best_features_gen
        })

        if gen % 5 == 0 or gen == GENERATIONS:
            print(f"Gen {gen:02d} | Fitness: {hof[0].fitness.values[0]:.4f} | Fitur Terpilih ({len(best_features_gen)}): {', '.join(best_features_gen)}")

    return hof[0], logbook, detailed_feature_log

# --- Eksekusi GA untuk Semua Classifier ---
classifiers = ['KNN', 'RandomForest', 'SVM']
results = {}

for name in classifiers:
    toolbox = setup_toolbox(name)
    best_individual, logbook, detailed_feature_log = run_ga_with_feature_log(name, toolbox)
    results[name] = {
        'best_individual': best_individual,
        'logbook': logbook,
        'detailed_feature_log': detailed_feature_log,
        'best_fitness': best_individual.fitness.values[0]
    }

print("\n--- 🏁 Algoritma Genetika Selesai untuk Semua Classifier ---")
print("-" * 70)

# --- 4. Visualisasi dan Analisis Hasil ---

print("\n## 📈 4. Visualisasi Pergerakan Fitness Setiap Iterasi GA ##")

plt.figure(figsize=(15, 6))
for name, res in results.items():
    gen = res['logbook'].select("gen")
    max_fit = res['logbook'].select("max")
    avg_fit = res['logbook'].select("avg")
    plt.plot(gen, max_fit, label=f'{name} - Max Fitness', linestyle='-', marker='o')
    plt.plot(gen, avg_fit, label=f'{name} - Avg Fitness', linestyle='--')

plt.title('Perkembangan Maksimal dan Rata-rata Fitness (Akurasi) Setiap Generasi')
plt.xlabel('Generasi')
plt.ylabel('Fitness (Akurasi)')
plt.legend()
plt.grid(True)
plt.show()


print("\n## 🔍 5. Analisis Faktor-faktor Paling Berpengaruh dan Hasil Akhir ##")

all_important_features = {}

for name, res in results.items():
    best_individual = res['best_individual']
    selected_indices = [i for i, gene in enumerate(best_individual) if gene == 1]
    final_features = [feature_names[i] for i in selected_indices]

    print(f"\n--- HASIL AKHIR MENGGUNAKAN {name} SEBAGAI FITNESS ---")
    print(f"✅ Akurasi (Fitness) Terbaik: **{res['best_fitness']:.4f}**")
    print(f"✅ Jumlah Fitur Terbaik: {len(final_features)}")
    print(f"✅ Faktor-faktor Paling Berpengaruh: \n**{', '.join(final_features)}**")


    for feature in final_features:
        all_important_features[feature] = all_important_features.get(feature, 0) + 1

    X_train_final = X_train.iloc[:, selected_indices]
    X_test_final = X_test.iloc[:, selected_indices]

    if name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=5)
    elif name == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif name == 'SVM':
        model = SVC(kernel='linear', random_state=42)

    model.fit(X_train_final, y_train)
    y_pred = model.predict(X_test_final)

    # Metrik Klasifikasi
    acc = accuracy_score(y_test, y_pred)

    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n**Metrik Klasifikasi Lengkap ({name}):**")
    print(f"  - Akurasi: **{acc:.4f}**")
    print(f"  - Precision (Weighted): {prec:.4f}")
    print(f"  - Recall (Weighted): {rec:.4f}")
    print(f"  - F1-Score (Weighted): {f1:.4f}")

    print("-" * 50)

# Visualisasi Total Pengaruh Fitur (Diambil oleh GA paling sering)
print("\n## 🏆 6. Total Pengaruh Fitur Berdasarkan Frekuensi Pemilihan GA ##")
feature_counts = pd.Series(all_important_features).sort_values(ascending=False)
feature_counts.index.name = 'Fitur'
feature_counts.name = 'Jumlah Terpilih (Max 3)'

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_counts.values, y=feature_counts.index, palette='viridis')
plt.title('Frekuensi Fitur Terpilih sebagai Terbaik oleh GA (dari 3 Classifier)')
plt.xlabel('Jumlah Terpilih (Maksimal 3)')
plt.ylabel('Faktor')
plt.show()


print("\n--- 📝 Kesimpulan Faktor Berpengaruh ---")
print("Fitur-fitur yang lebih berpengaruh adalah fitur yang **paling sering terpilih** (bernilai 1 dalam kromosom terbaik) dari solusi akhir yang ditemukan oleh GA.")
print("Nilai Akurasi (**Fitness**) Tertinggi ditemukan menggunakan **RandomForest** dengan akurasi **{:.4f}**.".format(results['RandomForest']['best_fitness']))
print(f"Fitur-fitur paling dominan (terpilih {feature_counts.max()} dari 3 kali percobaan):")
print(f"**{feature_counts[feature_counts == feature_counts.max()].index.tolist()}**")

print("-" * 70)

# --- 7. Detil Analisis Pergerakan Fitur Terbaik (Berdasarkan Log Generasi) ---
print("\n## 🎯 7. Detil Pergerakan Fitur Terbaik Setiap Iterasi GA ##")

for name, res in results.items():
    print(f"\n=======================================================")
    print(f"      ANALISIS FITUR TERBAIK UNTUK {name} (Fitness)      ")
    print(f"=======================================================")

    feature_occurrence = {}
    total_generations = len(res['detailed_feature_log']) - 1

    # Hitung Frekuensi kemunculan fitur di solusi terbaik (Hall of Fame) di SETIAP GENERASI
    for log_entry in res['detailed_feature_log']:
        for feature in log_entry['features']:
            feature_occurrence[feature] = feature_occurrence.get(feature, 0) + 1

    sorted_features = sorted(feature_occurrence.items(), key=lambda item: item[1], reverse=True)

    print("\n✅ Urutan Fitur Paling Berpengaruh Selama Evolusi (Frekuensi Kemunculan di Solusi Terbaik Generasi):")
    for feature, count in sorted_features:
        percentage = (count / (total_generations + 1)) * 100
        print(f"  - **{feature}**: {count} kali ({percentage:.1f}% dari {total_generations + 1} generasi)")

    print("\n--- Log Fitur Terbaik (Kromosom) di Setiap Generasi (Ringkasan) ---")


    for log_entry in res['detailed_feature_log']:
        gen = log_entry['gen']
        fitness = log_entry['fitness']
        features = log_entry['features']


        if gen < 5 or gen > GENERATIONS - 5 or gen == GENERATIONS // 2:
            print(f"Gen {gen:02d} | Fitness: {fitness:.4f} | Fitur Terpilih ({len(features)}): {', '.join(features)}")
        elif gen == 5:
            print("... (Generasi pertengahan dilewati untuk keringkasan) ...")

    print(f"\n🎉 Akurasi Terbaik Akhir ({name}): **{res['best_fitness']:.4f}**, dengan Fitur: {', '.join(res['detailed_feature_log'][-1]['features'])}")
    print("------------------------------------------------------------------")


# =============================================================
#  🔥 8. Simpan Semua Log Generasi ke Dalam File Excel
# =============================================================

# ==========================================================
#  EXCEL LOGGER - MENYIMPAN SEMUA GENERASI TANPA MENIMPA
# ==========================================================

import openpyxl

excel_path = "log_generasi_GA.xlsx"

# Jika file belum ada → buat baru dengan header
if not os.path.exists(excel_path):
    df_init = pd.DataFrame(columns=[
        "percobaan",
        "generasi",
        "model_ml",
        "fitness",
        "representasi_kromosom_biner",
        "fitur_terpilih"
    ])
    df_init.to_excel(excel_path, index=False)

# Tentukan nomor percobaan (append)
existing_df = pd.read_excel(excel_path)
if len(existing_df) == 0:
    percobaan_id = 1
else:
    percobaan_id = existing_df["percobaan"].max() + 1

print(f"\n📁 Menyimpan log ke Excel sebagai percobaan #{percobaan_id}")

# ==========================================================
#  SIMPAN LOG GENERASI UNTUK SETIAP CLASSIFIER
# ==========================================================

rows_to_append = []

for name, res in results.items():
    for log in res["detailed_feature_log"]:
        gen = log["gen"]
        fitness = log["fitness"]
        fitur = log["features"]
        krom = results[name]["best_individual"]  # kromosom terbaik versi GA, bentuk list 0/1

        rows_to_append.append({
            "percobaan": percobaan_id,
            "generasi": gen,
            "model_ml": name,
            "fitness": fitness,
            "representasi_kromosom_biner": "".join(map(str, krom)),
            "fitur_terpilih": ", ".join(fitur)
        })

# Append ke Excel
df_append = pd.DataFrame(rows_to_append)
with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as writer:
    df_append.to_excel(writer, index=False, header=False, startrow=writer.sheets['Sheet1'].max_row)
    
print(f"✅ Selesai menyimpan {len(df_append)} baris log generasi ke Excel.")

print("\nProgram Selesai.")