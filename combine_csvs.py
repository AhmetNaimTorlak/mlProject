import pandas as pd
from pathlib import Path

# ================== AYARLAR ==================
DATA_DIR = Path("data")          # CSV'lerin olduğu klasör
OUTPUT_CSV = Path("combined.csv")  # Çıkacak dosya adı
# ============================================

def main():
    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"{DATA_DIR} klasöründe .csv dosyası bulunamadı.")

    print("Birleştirilecek dosyalar:")
    for f in csv_files:
        print(" -", f.name)

    combined_dfs = []
    reference_columns = None

    for i, csv_path in enumerate(csv_files):
        print(f"\n[{i+1}/{len(csv_files)}] Okunuyor: {csv_path.name}")

        # ÖNEMLİ: header=0 (ilk satır sütun adları)
        df = pd.read_csv(csv_path, header=0, low_memory=False)

        # Gereksiz Unnamed index kolonlarını temizle (varsaysa)
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]

        if reference_columns is None:
            reference_columns = list(df.columns)
            print("Referans kolonlar:")
            print(reference_columns)
        else:
            current_columns = list(df.columns)
            if current_columns != reference_columns:
                raise ValueError(
                    f"{csv_path.name} içindeki sütun adları diğer dosyalarla uyuşmuyor.\n"
                    f"Beklenen: {reference_columns}\n"
                    f"Gelen   : {current_columns}"
                )

        combined_dfs.append(df)

    # Tüm dataframeleri alt alta birleştir
    combined_df = pd.concat(combined_dfs, ignore_index=True)

    # Tek bir combined.csv olarak kaydet
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Tüm CSV'ler birleştirildi: {OUTPUT_CSV.resolve()}")

if __name__ == "__main__":
    main()
