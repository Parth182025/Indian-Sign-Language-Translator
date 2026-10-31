import os
import pandas as pd

DATASET_PATH = "dataset"

print("\n========== DATASET CHECK ==========\n")

total_csv = 0

for label in sorted(os.listdir(DATASET_PATH)):

    folder_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(folder_path):
        continue

    csv_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".csv")
    ]

    print(f"\n[{label}]")
    print(f"Number of CSV files: {len(csv_files)}")

    for csv_file in sorted(csv_files)[:3]:

        path = os.path.join(folder_path, csv_file)

        df = pd.read_csv(path)

        print(
            f"  {csv_file}: "
            f"rows={len(df)}, "
            f"columns={len(df.columns)}"
        )

        print(f"  columns = {list(df.columns)}")

        if "frame" in df.columns:
            print(f"  frames = {df['frame'].nunique()}")

        if "landmark" in df.columns:
            print(f"  landmarks = {df['landmark'].nunique()}")

        print()

    total_csv += len(csv_files)

print("\n===================================")
print("TOTAL CSV FILES:", total_csv)
print("===================================\n")