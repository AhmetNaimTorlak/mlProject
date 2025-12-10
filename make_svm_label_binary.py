from pathlib import Path
import pandas as pd

CSV_PATH = Path("combined_with_svm.csv")

print(f"Dosya okunuyor: {CSV_PATH.resolve()}")
df = pd.read_csv(CSV_PATH, low_memory=False)

if "svm_label" not in df.columns:
    raise ValueError("'svm_label' kolonu bulunamadı. Bu dosya eski pipeline'dan mı geldi, kontrol et.")

print("\n[Önce] svm_label dağılımı:")
print(df["svm_label"].value_counts())

# other_attack -> attack
df["svm_label"] = df["svm_label"].replace("other_attack", "attack")

print("\n[Sonra] svm_label dağılımı:")
print(df["svm_label"].value_counts())

# Yedek al
backup_path = CSV_PATH.with_name("combined_with_svm_backup_before_binary.csv")
print(f"\nYedek kaydediliyor: {backup_path.resolve()}")
df.to_csv(backup_path, index=False)

# Asıl dosyanın üzerine yaz
print(f"Asıl dosya üzerine yazılıyor: {CSV_PATH.resolve()}")
df.to_csv(CSV_PATH, index=False)

print("\n✅ Tamamlandı. Artık svm_label sadece 'benign' ve 'attack' içeriyor.")
