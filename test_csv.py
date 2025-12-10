import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# ================== AYARLAR ==================
# Eğitimde kullandığın model ve feature dosyaları
SVM_MODEL_PATH = "svm_model.joblib"
DT_MODEL_PATH = "dt_model.joblib"
FEATURE_COLS_PATH = "feature_columns.joblib"
# ============================================


def normalize_label_str(s):
    """
    Label stringlerini normalize eder:
      - Farklı tire karakterlerini '-' yapar
      - Bozuk '�' karakterini '-' ile değiştirir
      - Baştaki/sondaki boşlukları temizler
    """
    if pd.isna(s):
        return s
    s = str(s)
    # Bozuk karakterleri ve farklı tireleri normalize et
    s = s.replace("�", "-")
    s = s.replace("–", "-")
    s = s.strip()
    return s


# ---- Modelleri ve feature kolonlarını yükle ----
svm_clf = joblib.load(SVM_MODEL_PATH)
dt_clf = joblib.load(DT_MODEL_PATH)
feature_cols = joblib.load(FEATURE_COLS_PATH)

print("✅ Modeller ve feature kolonları yüklendi.")
print("Feature sayısı:", len(feature_cols))

# ---- Test CSV yolunu kullanıcıdan iste ----
test_path_str = input("\nTest için kullanacağın CSV dosyasının yolunu gir (örn: data/benign_only.csv): ").strip()
TEST_CSV = Path(test_path_str)

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Dosya bulunamadı: {TEST_CSV.resolve()}")

print(f"\n📂 Test verisi okunuyor: {TEST_CSV.resolve()}")
df_test = pd.read_csv(TEST_CSV, header=0, low_memory=False)
df_test = df_test.loc[:, ~df_test.columns.astype(str).str.startswith("Unnamed")]

print("\nTest dosyasındaki sütunlar:")
for c in df_test.columns:
    print(f"- '{c}'")

print(f"\nToplam test satırı: {len(df_test)}")


# ---------- Etiket kolonunu akıllı tespit et ----------
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

    # Birebir Label / Target var mı?
    if "Label" in columns:
        return " Label" if " Label" in columns else "Label"
    if "Target" in columns:
        return "Target"

    # Normalize edip adaylara bak
    for original, norm in normalized_map.items():
        if norm in candidates_norm:
            return original

    return None


label_col = find_label_column(df_test.columns)
if label_col is not None:
    print(f"\n🔎 Bulunan etiket kolonu: '{label_col}'")
    # Label stringlerini normalize et
    df_test[label_col] = df_test[label_col].apply(normalize_label_str)
else:
    print("\n⚠ Bu test CSV'sinde etiket kolonu bulunamadı (Label/Target yok). "
          "Yine de tahmin yapacağız ama metrik hesaplayamayacağız.")

# ================== SINIF GRUPLARI ==================
# Eğitimde kullandığın gruplarla uyumlu tutuyoruz
major_attacks = [
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "PortScan",
    "SSH-Patator",
]

# Web attack'ler ve diğer nadirler (normalize edilmiş halleri dahil)
rare_attacks = [
    "Bot",
    "Heartbleed",
    "Infiltration",
    # Web Attack'ler - normalize edilmiş tire ile
    "Web Attack - Brute Force",
    "Web Attack - Sql Injection",
    "Web Attack - XSS",
    # Olası eski varyantlar (güvenlik için ekliyorum)
    "Web Attack – Brute Force",
    "Web Attack – Sql Injection",
    "Web Attack – XSS",
]
# ====================================================


def make_svm_label_from_original(lbl) -> str:
    """Test setindeki orijinal label'dan benign/attack/other_attack üretir."""
    if pd.isna(lbl):
        return "other_attack"
    s = normalize_label_str(lbl)
    if s == "BENIGN":
        return "benign"
    elif s in rare_attacks:
        return "other_attack"
    else:
        return "attack"


# ------------ FEATURE MATRİSİNİ HAZIRLA ------------

# Yalnızca eğitimde kullandığımız kolonları al
missing_cols = [c for c in feature_cols if c not in df_test.columns]
if missing_cols:
    print("\n⚠ Uyarı: Test CSV içinde aşağıdaki feature kolonları yok:")
    for c in missing_cols:
        print("  -", c)
    print("Bu kolonlar 0 ile doldurulacak (eğer çoksa schema farkı var demektir).")

# Eksik feature kolonlarını oluşturup 0 ile doldur
for c in missing_cols:
    df_test[c] = 0.0

# Sadece feature_cols sırasıyla alınır
feature_df_test = df_test[feature_cols].copy()

# Numerik dönüşüm, inf temizliği, NaN doldurma
for col in feature_df_test.columns:
    if not np.issubdtype(feature_df_test[col].dtype, np.number):
        feature_df_test[col] = pd.to_numeric(feature_df_test[col], errors="coerce")

feature_df_test.replace([np.inf, -np.inf], np.nan, inplace=True)
feature_df_test = feature_df_test.fillna(0.0)

X_test = feature_df_test.values

print(f"\nTest için kullanılan feature sayısı: {len(feature_cols)}")
print("Örnek feature kolonları:", feature_cols[:10])

# ====================================================
#           1) SVM TAHMİNİ (benign/attack/other_attack)
# ====================================================

svm_pred = svm_clf.predict(X_test)

print("\n🔹 SVM tahmin dağılımı:")
unique, counts = np.unique(svm_pred, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c}")

# ====================================================
#      2) SVM+DT PIPELINE (final prediction) TAHMİNİ
# ====================================================

final_pred = []

# DT'yi toplu çağırmak için önce attack index'lerini bulalım
attack_indices = np.where(svm_pred == "attack")[0]
if len(attack_indices) > 0:
    X_attack = X_test[attack_indices]
    dt_pred_attack = dt_clf.predict(X_attack)
else:
    dt_pred_attack = np.array([])

attack_idx_to_dt = {idx: dt_pred_attack[i] for i, idx in enumerate(attack_indices)}

for i in range(len(df_test)):
    sp = svm_pred[i]
    if sp == "benign":
        final_pred.append("BENIGN")
    elif sp == "other_attack":
        final_pred.append("OTHER_ATTACK")
    else:  # "attack"
        final_pred.append(attack_idx_to_dt[i])

final_pred = np.array(final_pred)

print("\n🔹 Final prediction (SVM+DT) dağılımı:")
unique_f, counts_f = np.unique(final_pred, return_counts=True)
for u, c in zip(unique_f, counts_f):
    print(f"  {u}: {c}")

# ====================================================
#          3) ETİKET VARSA METRİKLERİ HESAPLA
# ====================================================

if label_col is not None:
    y_true_orig = df_test[label_col].values

    # SVM için: orijinal label'dan benign/attack/other_attack map et
    y_true_svm = np.array([make_svm_label_from_original(v) for v in y_true_orig])

    print("\n=== SVM (3 sınıf) - Test CSV üzerinde performans ===")
    print(classification_report(y_true_svm, svm_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_true_svm, svm_pred, labels=["benign", "attack", "other_attack"]))

    print("\n=== SVM+DT Final Prediction - Test CSV üzerinde ===")
    print("Not: Burada orijinal label ile final_prediction karşılaştırılıyor.")
    print(classification_report(y_true_orig, final_pred, zero_division=0))
    print("Confusion Matrix (rows=true, cols=pred):")
    labels_final = sorted(list(set(y_true_orig) | set(final_pred)), key=lambda x: str(x))
    print("Kullanılan label sırası:", labels_final)
    print(confusion_matrix(y_true_orig, final_pred, labels=labels_final))
else:
    print("\n⚠ Etiket kolonu bulunamadığı için metrik hesaplanmadı. "
          "Yukarıdaki dağılımları (SVM ve final prediction) inceleyebilirsin.")
