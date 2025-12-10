import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import SGDClassifier  # ✅ Linear SVM için
from sklearn.utils import resample             # ✅ Oversample için

# İstersen joblib ile modelleri kaydedebiliriz
try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# ================== AYARLAR ==================
DATA_CSV = Path("combined_with_svm.csv")  # svm_label eklenmiş dosya
TEST_SIZE_SVM = 0.2
TEST_SIZE_DT = 0.2
RANDOM_STATE = 42

USE_SUBSAMPLE = False      # Büyük veri için istersen True yap
SUBSAMPLE_FRAC = 0.2       # frac=0.2 => verinin %20'si ile eğitim (şu an kullanılmıyor)
# ============================================

print(f"Veri okunuyor: {DATA_CSV.resolve()}")
df = pd.read_csv(DATA_CSV, header=0, low_memory=False)

# Gereksiz Unnamed kolonlarını temizle
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

print("\nBulunan sütunlar:")
for c in df.columns:
    print(f"- '{c}'")

# ------------------ OPSİYONEL: SUBSAMPLE ------------------
if USE_SUBSAMPLE:
    df = df.sample(frac=SUBSAMPLE_FRAC, random_state=RANDOM_STATE)
    print(f"\nSubsample uygulandı. Yeni satır sayısı: {len(df)}")
# ----------------------------------------------------------


# ---------- Etiket kolonunu akıllı tespit et ----------
def find_label_column(columns) -> str | None:
    """
    Sütun adlarını normalize ederek olası label/target kolonunu bulmaya çalışır.
    Örn:
      ' Label '           -> 'label'
      'Attack category'   -> 'attackcategory'
    """
    candidates_norm = {
        "label",
        "target",
        "class",
        "attackcategory",
        "attack_cat",
        "attacktype",
    }

    # orijinal -> normalize eşlemesi
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

    # Önce birebir Label / Target var mı diye bak
    if "Label" in columns:
        # başında boşluklu hali varsa onu kullan
        return " Label" if " Label" in columns else "Label"
    if "Target" in columns:
        return "Target"

    # Sonra normalize edip adaylarla eşleştir
    for original, norm in normalized_map.items():
        if norm in candidates_norm:
            print(f"\nOlası etiket kolonu bulundu: '{original}' (normalize: '{norm}')")
            return original

    # Bulunamazsa None dön
    return None


label_col = find_label_column(df.columns)

if label_col is None:
    raise ValueError(
        "\nNe 'Label' ne de 'Target' ne de bilinen isimlerde bir etiket kolonu bulunamadı.\n"
        "Yukarıdaki sütun isimlerine bakıp etiket kolonunun adını netleştirip\n"
        "find_label_column içindeki candidates_norm set'ine uygun bir anahtar eklemen gerekiyor."
    )

print(f"\nKullanılacak orijinal etiket kolonu: '{label_col}'")

if "svm_label" not in df.columns:
    raise ValueError("'svm_label' kolonu bulunamadı. Önce svm_label script'ini çalıştırmalısın.")

print(f"SVM etiketi kolonu                     : 'svm_label'")


# ================== SINIF GRUPLARI ==================
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

rare_attacks = [
    "Bot",
    "Heartbleed",
    "Infiltration",
    "Web Attack – Brute Force",
    "Web Attack - Brute Force",
    "Web Attack – Sql Injection",
    "Web Attack - Sql Injection",
    "Web Attack – XSS",
    "Web Attack - XSS",
]
# ====================================================


# ------------ FEATURE MATRİSİNİ HAZIRLA ------------

# Label ve svm_label dışındaki kolonlar özellik adayı
feature_df = df.drop(columns=[label_col, "svm_label"], errors="ignore").copy()

# Tüm kolonları numeriğe çevirmeyi dene (non-numeric -> NaN)
for col in feature_df.columns:
    if not np.issubdtype(feature_df[col].dtype, np.number):
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce")

# Sonsuz değerleri (inf, -inf) NaN yap
feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Tamamen NaN olan kolonları at
feature_df = feature_df.dropna(axis=1, how="all")

# Kalan NaN'leri 0 ile doldur
feature_df = feature_df.fillna(0.0)

feature_cols = feature_df.columns.tolist()
X_all = feature_df.values

print(f"\nKullanılan özellik sayısı: {len(feature_cols)}")
print("Örnek feature kolonları:", feature_cols[:10])


# ================== OVERSAMPLE FONKSİYONU ==================

def oversample_other_attack(X_train, y_train, random_state=42, factor=20):
    """
    Sadece train set üzerinde 'other_attack' sınıfını oversample eder.
    factor: other_attack'i yaklaşık kaç katına çıkarmak istediğin.
    Ama hedef sayı attack ve benign'den büyük olmayacak.
    """
    X_df = pd.DataFrame(X_train)
    y_sr = pd.Series(y_train)

    print("\n[Önce] SVM train sınıf dağılımı:")
    print(y_sr.value_counts())

    mask_other = (y_sr == "other_attack")
    mask_attack = (y_sr == "attack")
    mask_benign = (y_sr == "benign")

    n_other = mask_other.sum()
    n_attack = mask_attack.sum()
    n_benign = mask_benign.sum()

    if n_other == 0:
        print("other_attack sınıfı train set'te yok, oversample yapılmadı.")
        return X_train, y_train

    # Hedef: other_attack'i factor * n_other seviyesine çıkar,
    # ama yine de attack ve benign'den büyük olmasın
    raw_target = n_other * factor
    target_other = min(raw_target, n_attack, n_benign)

    if target_other <= n_other:
        print("other_attack zaten hedef veya üzerinde, oversample yapılmadı.")
        return X_train, y_train

    n_new = target_other - n_other
    print(
        f"\nother_attack için hedef örnek sayısı: {target_other} "
        f"(şu an {n_other}), eklenecek yeni örnek: {n_new}"
    )

    X_other = X_df[mask_other]
    y_other = y_sr[mask_other]

    X_other_up, y_other_up = resample(
        X_other,
        y_other,
        replace=True,
        n_samples=n_new,
        random_state=random_state
    )

    X_rest = X_df[~mask_other]
    y_rest = y_sr[~mask_other]

    X_bal = pd.concat([X_rest, X_other, X_other_up], ignore_index=True)
    y_bal = pd.concat([y_rest, y_other, y_other_up], ignore_index=True)

    print("\n[Sonra] SVM train sınıf dağılımı (oversample sonrası):")
    print(y_bal.value_counts())

    return X_bal.values, y_bal.values


# ====================================================
#                   1) SVM EĞİTİMİ
# ====================================================

y_svm = df["svm_label"].values

X_train_svm, X_test_svm, y_train_svm, y_test_svm = train_test_split(
    X_all,
    y_svm,
    test_size=TEST_SIZE_SVM,
    random_state=RANDOM_STATE,
    stratify=y_svm
)

# 🔹 Train set üzerinde other_attack oversample (factor ile ayarlı)
X_train_svm, y_train_svm = oversample_other_attack(
    X_train_svm,
    y_train_svm,
    random_state=RANDOM_STATE,
    factor=20  # gerekirse 10 / 30 vs. deneyebilirsin
)

# ✅ Linear SVM (SGDClassifier ile)
svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SGDClassifier(
        loss="hinge",           # linear SVM
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        random_state=RANDOM_STATE
    ))
])

print("\n=== SVM (linear, 3 sınıf: benign / attack / other_attack) eğitiliyor ===")
svm_clf.fit(X_train_svm, y_train_svm)

y_pred_svm = svm_clf.predict(X_test_svm)

print("\n--- SVM Classification Report ---")
print(classification_report(y_test_svm, y_pred_svm))

print("--- SVM Confusion Matrix ---")
print(confusion_matrix(y_test_svm, y_pred_svm, labels=["benign", "attack", "other_attack"]))


# ====================================================
#            2) DECISION TREE (ATTACK SUBSET)
# ====================================================

# Sadece major attack sınıflarını içeren satırlar
df_dt = df[df[label_col].isin(major_attacks)].copy()

if df_dt.empty:
    raise ValueError("Decision Tree için major_attacks sınıflarına ait satır bulunamadı.")

# Aynı feature kolonlarını, aynı sırayla kullan
X_dt_all = feature_df.loc[df_dt.index].values
y_dt_all = df_dt[label_col].values

X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
    X_dt_all,
    y_dt_all,
    test_size=TEST_SIZE_DT,
    random_state=RANDOM_STATE,
    stratify=y_dt_all
)

dt_clf = DecisionTreeClassifier(
    max_depth=None,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

print("\n=== Decision Tree (major attack tipleri) eğitiliyor ===")
dt_clf.fit(X_train_dt, y_train_dt)

y_pred_dt = dt_clf.predict(X_test_dt)

print("\n--- Decision Tree Classification Report ---")
print(classification_report(y_test_dt, y_pred_dt))

print("--- Decision Tree Confusion Matrix ---")
print(confusion_matrix(y_test_dt, y_pred_dt, labels=major_attacks))


# ====================================================
#         3) ORTAK PREDICTION AKIŞI ÖRNEK FONKSİYON
# ====================================================

def predict_flow(sample_row: pd.Series):
    """
    Tek bir satır için:
      1) SVM ile benign / attack / other_attack kararı
      2) attack ise Decision Tree ile saldırı tipini belirler.

    Dönen:
      - "BENIGN"
      - "OTHER_ATTACK"
      - veya major_attacks listesinden biri
    """
    x = sample_row[feature_cols].values.astype(float).reshape(1, -1)

    svm_pred = svm_clf.predict(x)[0]

    if svm_pred == "benign":
        return "BENIGN"
    elif svm_pred == "other_attack":
        return "OTHER_ATTACK"
    else:
        dt_pred = dt_clf.predict(x)[0]
        return dt_pred


print("\nÖrnek ortak prediction (ilk 5 satır):")
for idx in df.index[:5]:
    row = df.loc[idx]
    final_label = predict_flow(row)
    print(f"Index {idx}: gerçek={row[label_col]} | svm_label={row['svm_label']} | final={final_label}")


# ====================================================
#                     4) MODELLERİ KAYDET
# ====================================================

if HAS_JOBLIB:
    joblib.dump(svm_clf, "svm_model.joblib")
    joblib.dump(dt_clf, "dt_model.joblib")
    joblib.dump(feature_cols, "feature_columns.joblib")
    print("\nModeller ve feature kolon listesi kaydedildi: "
          "svm_model.joblib, dt_model.joblib, feature_columns.joblib")
else:
    print("\njoblib bulunamadı, modeller kaydedilmedi. "
          "İstersen 'pip install joblib' ile kurup tekrar çalıştırabilirsin.")
