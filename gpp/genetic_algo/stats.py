import os
from typing import Any, Dict, List, Tuple, Optional
from collections import defaultdict, Counter
import csv
from math import floor
from pathlib import Path

def load_summary(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find summary file at {path}")
    with path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))
    
def normalize_pct(value: Optional[str]) -> Optional[float]:
    """Convert stored ratios/percentages into 0–100 scale."""
    if value in (None, ""):
        return None
    try:
        pct = float(value)
    except ValueError:
        return None
    return pct * 100 if pct <= 1 else pct

def main():
    csv_path = "/home/utn/firi22ka/Desktop/jenga/GeneticPatchPruning/compare_0_0.3_full/summary.csv"
    res_path = Path(csv_path)
    records = []
    # keep_pct = []
    # num_img = []
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                records = list(csv.DictReader(f))
        except Exception as exc:
            print(f"Failed to parse {csv_path}: {exc}")
    else:
        print(f"Summary file {csv_path} not found.")
        return  # bail early, nothing else to do

    if not records:
        print(f"{csv_path} is empty or invalid.")
        return

    print("Columns:", ", ".join(records[0].keys()))


    


    # # round_decimals = 2
    # pct_counter = Counter(round(float(r['keep_pct']))
    #                       for r in records if 'keep_pct' in r and isinstance(r['keep_pct'], (int, float)))
    
    # if pct_counter:
    #     pct_rows = sorted(pct_counter.items(), key=lambda x: x[0])
    #     print('\n=== Table: grouped by keep_pct (rounded) ===')
    #     print(f'{"keep_pct":>10} | {"num_images":>10}')
    #     print('-' * 27)
    #     for kp, n in pct_rows:
    #         print(f'{kp:>10} | {n:>10}')

    pct_counter = Counter()

    bucket_counts = Counter()

    og_avg_acc = 0.0
    ga_avg_acc = 0.0

    for row in records:
        if row.get("gt") == row.get("og_top1"):
            og_avg_acc += 1.0
        if row.get("gt") == row.get("ga_top1"):
            ga_avg_acc += 1.0   
    og_avg_acc /= len(records)
    ga_avg_acc /= len(records)

    print(f"\nOriginal Average Accuracy: {og_avg_acc*100:.2f}%")
    print(f"Genetic Algorithm Average Accuracy: {ga_avg_acc*100:.2f}%\n")

    
    for row in records:
        if row.get("selected_count"):
            pct_value = int(row.get("selected_count")) / 196
        else:
            pct_value = row.get("selected_pct") or row.get("keep_pct")
        pct = normalize_pct(pct_value)
        if pct is not None:
            pct_counter[round(pct)] += 1

        bucket_start = int(floor(pct / 10) * 10)
        bucket_end = bucket_start + 10
        bucket_label = f"{bucket_start:02d}-{bucket_end:02d}%"
        bucket_counts[bucket_label] += 1

    if pct_counter:
        print('\n=== Table: grouped by selected_pct (rounded) ===')
        print(f'{"pct":>10} | {"num_images":>10}')
        print('-' * 27)
        for pct_value, count in sorted(pct_counter.items()):
            print(f'{pct_value:>10} | {count:>10}')

    if bucket_counts:
        print("\n=== Table: grouped by 10% buckets ===")
        print(f'{"range":>10} | {"num_images":>10}')
        print('-' * 29)
        for label, count in sorted(bucket_counts.items()):
            print(f'{label:>10} | {count:>10}')

    print(f"\nTotal records: {len(records)}")

    output_path = res_path.with_name("summary_bucket_counts.csv")
    with output_path.open("w", encoding="utf-8", newline="") as out_csv:
        writer = csv.writer(out_csv)
        writer.writerow(["range", "num_images"])
        for label, count in sorted(bucket_counts.items()):
            writer.writerow([label, count])

    print(f"\nBucket counts saved to {output_path}")

if __name__ == "__main__":
    main()




