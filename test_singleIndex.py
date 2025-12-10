import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ================== AYARLAR ==================
DATA_CSV = Path("combined_with_svm.csv")   # Aynı dosyayı kullanıyoruz
ROW_INDEX = 14789  # Test etmek istediğin satırın index'i (0, 1, 12345 vs.)
# ============================================

# ---- Modelleri ve feature kolonlarını yükle ----
svm_clf = joblib.load("svm_model.joblib")
dt_clf = joblib.load("dt_model.joblib")
feature_cols = joblib.load("feature_columns.joblib")

print("Modeller ve feature kolonları yüklendi.")
print("Feature sayısı:", len(feature_cols))

# ---- Veriyi oku ----
df = pd.read_csv(DATA_CSV, header=0, low_memory=False)
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

print(f"Toplam satır sayısı: {len(df)}")

# ---- Etiket kolonunu tekrar bulalım (rapor için) ----
def find_label_column(columns) -> str | None:
    candidates_norm = {
        "label",
        "target",
        "class",
        "attackcategory",
        "attack_cat",
        "attacktype",
    }
    normalized_map = {}
    for col in columns:
        norm = (
            str(col)
            .strip()
            .lower()
            .replace(" ", "")
            .replace("_", "")
        )
        normalized_map[col] = norm

    if "Label" in columns:
        return " Label" if " Label" in columns else "Label"
    if "Target" in columns:
        return "Target"

    for original, norm in normalized_map.items():
        if norm in candidates_norm:
            return original

    return None

label_col = find_label_column(df.columns)
print("Orijinal etiket kolonu:", label_col)

# ---- Tek satır seç ----
row = df.iloc[ROW_INDEX]
print(f"\nSeçilen satır index: {ROW_INDEX}")
print("Gerçek etiket      :", row[label_col])
print("svm_label          :", row["svm_label"])

# ---- Ortak prediction fonksiyonu ----
def predict_flow_from_series(sample_row: pd.Series) -> str:
    """
    Tek bir satır için:
      1) SVM ile benign / attack / other_attack kararı
      2) attack ise Decision Tree ile saldırı tipini belirler.
    """
    # Sadece feature kolonlarını sırayla al
    x = sample_row[feature_cols].values.astype(float).reshape(1, -1)

    svm_pred = svm_clf.predict(x)[0]

    if svm_pred == "benign":
        return "BENIGN"
    elif svm_pred == "other_attack":
        return "OTHER_ATTACK"
    else:
        dt_pred = dt_clf.predict(x)[0]
        return dt_pred

# ---- Tahmin yap ----
final_pred = predict_flow_from_series(row)
print("\n=== TEK KAYIT TAHMİNİ ===")
print("Gerçek Label :", row[label_col])
print("SVM Label    :", row["svm_label"])
print("Final prediction (SVM+DT):", final_pred)
