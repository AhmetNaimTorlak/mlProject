import pandas as pd
from pathlib import Path

# ================== AYARLAR ==================
INPUT_CSV = Path("combined.csv")
OUTPUT_CSV = Path("combined_with_svm.csv")
# Aynı dosyanın üzerine yazmak istersen:
# OUTPUT_CSV = INPUT_CSV
# ============================================


print(f"Dosya okunuyor: {INPUT_CSV.resolve()}")
df = pd.read_csv(INPUT_CSV, header=0, low_memory=False)

# Gereksiz Unnamed index kolonlarını temizle
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

print("\nBulunan sütunlar:")
for c in df.columns:
    print(f"- '{c}'")

# ---------- Etiket kolonunu akıllı tespit et ----------

def find_label_column(columns) -> str | None:
    """
    Sütun adlarını normalize ederek olası label/target kolonunu bulmaya çalışır.
    Örn:
      ' Label ' -> 'label'
      'Attack category' -> 'attackcategory'
    """
    candidates_norm = {
        "label",
        "target",
        "class",
        "attackcategory",
        "attack_cat",
        "attackcategory",
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
        return "Label"
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
        "script içinde find_label_column fonksiyonuna eklemen gerekiyor."
    )

print(f"\nKullanılacak etiket kolonu: '{label_col}'")

# ================== SINIF GRUPLARI ==================

# >= 5000 kayıtlı ana saldırılar
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

# < 5000 kayıtlı nadir saldırılar (OTHER_ATTACK)
rare_attacks = [
    "Bot",
    "Heartbleed",
    "Infiltration",
    "Web Attack – Brute Force",
    "Web Attack - Brute Force",   # farklı tire ihtimalleri
    "Web Attack – Sql Injection",
    "Web Attack - Sql Injection",
    "Web Attack – XSS",
    "Web Attack - XSS",
]

# ====================================================

def make_svm_label(lbl) -> str:
    """
    SVM için 3 sınıflı etiket üretir:
    - 'benign'
    - 'attack'
    - 'other_attack'
    """
    if pd.isna(lbl):
        return "other_attack"   # eksik label olursa güvenli taraf

    s = str(lbl).strip()

    if s == "BENIGN":
        return "benign"
    elif s in rare_attacks:
        return "other_attack"
    else:
        # major_attacks + varsa diğer saldırı tipleri
        return "attack"


# 3) Yeni kolon: svm_label
df["svm_label"] = df[label_col].apply(make_svm_label)

# 4) Kontrol için dağılım yazdır
print("\nSVM etiket dağılımı (svm_label):")
print(df["svm_label"].value_counts())

print("\nOrijinal etiket dağılımı (ilk 20):")
print(df[label_col].value_counts().head(20))

# 5) Kaydet
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Güncellenmiş dosya kaydedildi: {OUTPUT_CSV.resolve()}")
