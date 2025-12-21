# ============================================================
# TITANIC FULL NOTEBOOK – MACHINE LEARNING + ENSEMBLE
# ============================================================

# ===================
# 1. SETUP PATH
# ===================
DATA_PATH = "./machine-learning/kaggle-competition/titanic/"


# ===================
# 2. IMPORT LIBRARIES
# ===================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# ===================
# 3. LOAD DATA
# ===================
train = pd.read_csv(DATA_PATH + "train.csv")
test  = pd.read_csv(DATA_PATH + "test.csv")


# ===================
# 4. PREPROCESSING
# ===================

# Gabungkan train + test untuk preprocessing konsisten
full = pd.concat([train.drop("Survived", axis=1), test], ignore_index=True)

# Isi missing values
full["Age"].fillna(full["Age"].median(), inplace=True)
full["Fare"].fillna(full["Fare"].median(), inplace=True)
full["Embarked"].fillna("S", inplace=True)

# Label encoding kolom kategori
for col in ["Sex", "Embarked"]:
    le = LabelEncoder()
    full[col] = le.fit_transform(full[col])

# Buat fitur baru (opsional, biasanya meningkatkan skor)
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1

# Pilihan fitur
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"]

X_full = full[features]
X_train = X_full[:len(train)]
X_test = X_full[len(train):]
y_train = train["Survived"]


# ===================
# 5. TRAIN / VALIDATION SPLIT
# ===================
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)


# ===================
# 6. FUNGSI EVALUASI MODEL
# ===================
results = []

def evaluate_model(name, model, X_val, y_val):
    pred = model.predict(X_val)
    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_val, pred),
        "Precision": precision_score(y_val, pred),
        "Recall": recall_score(y_val, pred),
        "F1-Score": f1_score(y_val, pred)
    })


# ===================
# 7. MODEL: DECISION TREE
# ===================
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_tr, y_tr)
evaluate_model("Decision Tree", dt, X_val, y_val)


# ===================
# 8. MODEL: SVM
# ===================
from sklearn.svm import SVC

svm = SVC(kernel='rbf', probability=True)
svm.fit(X_tr, y_tr)
evaluate_model("SVM (RBF)", svm, X_val, y_val)


# ===================
# 9. MODEL: KNN
# ===================
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_tr, y_tr)
evaluate_model("KNN", knn, X_val, y_val)


# ===================
# 10. MODEL: RANDOM FOREST
# ===================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    random_state=42
)
rf.fit(X_tr, y_tr)
evaluate_model("Random Forest", rf, X_val, y_val)


# ===================
# 11. MODEL: XGBOOST
# ===================
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_tr, y_tr)
evaluate_model("XGBoost", xgb, X_val, y_val)


# ===================
# 12. ENSEMBLE: VOTING CLASSIFIER
# ===================
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('dt', dt),
        ('svm', svm),
        ('knn', knn),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft'
)
voting_clf.fit(X_tr, y_tr)
evaluate_model("Voting Ensemble", voting_clf, X_val, y_val)


# ===================
# 13. ENSEMBLE: STACKING CLASSIFIER
# ===================
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

stacking_clf = StackingClassifier(
    estimators=[
        ('dt', dt),
        ('svm', svm),
        ('knn', knn),
        ('rf', rf),
        ('xgb', xgb)
    ],
    final_estimator=LogisticRegression(max_iter=200),
    passthrough=True
)

stacking_clf.fit(X_tr, y_tr)
evaluate_model("Stacking Ensemble", stacking_clf, X_val, y_val)


# ===================
# 14. TABEL PERBANDINGAN MODEL
# ===================
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by="F1-Score", ascending=False)
print(df_results_sorted)


# ===================
# 15. SUBMISSION FINAL (PAKAI MODEL TERBAIK)
# ===================
BEST_MODEL = stacking_clf   # ganti jika Voting lebih bagus

final_pred = BEST_MODEL.predict(X_test)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": final_pred
})

submission.to_csv(DATA_PATH + "submission_final_ensemble.csv", index=False)
print("Saved:", DATA_PATH + "submission_final_ensemble.csv")
