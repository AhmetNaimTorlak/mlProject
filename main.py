import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# =========================================================
# LABEL OTOMATİK BULMA
# =========================================================
def detect_label_column(df):
    possible = [
        "Label", "label", "LABEL",
        "Attack", "attack",
        "Class", "class",
        "Classification", "classification",
        "Attack Category", "AttackCategory",
        " Label", "Label ", " Label "
    ]
    for p in possible:
        if p in df.columns:
            return p

    print("\n❌ LABEL COLUMN NOT FOUND!")
    print(df.columns)
    return None


# =========================================================
# VERİ TEMİZLİĞİ (UYARISIZ SÜRÜM)
# =========================================================
def clean_dataset(df, label_col):

    df[label_col] = df[label_col].astype(str)

    # Non-numeric kolonları tespit et
    numeric_df = df.apply(pd.to_numeric, errors='ignore')

    # INF → NaN
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)

    # NaN doldurma
    for col in numeric_df.columns:
        if col != label_col:
            if pd.api.types.is_numeric_dtype(numeric_df[col]):
                median = numeric_df[col].median()
                numeric_df[col] = numeric_df[col].fillna(median)

    return numeric_df


# =========================================================
# IDS MODEL
# =========================================================
def run_ids(df, filename, label_col):

    print("\n===============================================")
    print(f"DATASET: {filename}")
    print(f"Label Column: {label_col}")
    print("===============================================")

    # Binary label oluştur
    df["binary"] = df[label_col].apply(lambda x: 0 if x == "BENIGN" else 1)

    # Eğer tüm değerler 0 veya tüm değerler 1 ise SVM yapılamaz
    if df["binary"].nunique() < 2:
        print("⚠️  Stage-1 SVM Skipped → Only ONE class in dataset.")
        return

    # -----------------------------
    # Stage 1 — SVM
    # -----------------------------
    X = df.drop(columns=[label_col, "binary"])
    y = df["binary"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    svm = SVC(kernel="rbf", C=10, gamma=0.01)
    svm.fit(X_train_s, y_train)

    acc = svm.score(X_test_s, y_test)
    print(f"Stage 1 (SVM) Accuracy: {acc:.4f}")

    # -----------------------------
    # Stage 2 — Decision Tree
    # -----------------------------
    df_attack = df[df[label_col] != "BENIGN"]

    # Eğer tüm saldırıların türü aynıysa DT yapılamaz
    if df_attack[label_col].nunique() < 2:
        print("⚠️ Stage-2 DT Skipped → Only one attack type.")
        return

    X2 = df_attack.drop(columns=[label_col, "binary"])
    y2 = df_attack[label_col]

    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.2, random_state=42, stratify=y2
    )

    dt = DecisionTreeClassifier(max_depth=12, min_samples_leaf=20)
    dt.fit(X2_train, y2_train)

    dt_acc = dt.score(X2_test, y2_test)
    print(f"Stage 2 (Decision Tree) Accuracy: {dt_acc:.4f}")



# =========================================================
# ANA PROGRAM — TÜM CSV DOSYALARI
# =========================================================
DATA_FOLDER = "data"
csv_files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".csv")]

for file in csv_files:

    print(f"\nLoading: {file}")
    df = pd.read_csv(os.path.join(DATA_FOLDER, file))

    label_col = detect_label_column(df)
    if label_col is None:
        continue

    df = clean_dataset(df, label_col)

    run_ids(df, file, label_col)
