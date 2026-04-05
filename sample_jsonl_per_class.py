#!/usr/bin/env python3
import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple
import random


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield line


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sample a fixed number of records per class from a JSONL file."
    )
    parser.add_argument("input", help="Input JSONL file")
    parser.add_argument(
        "--class-key",
        default="gt",
        help="Key to use as class label (default: gt)",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        nargs="+",
        required=True,
        help="One or more sample counts per class (e.g., 50 100 150)",
    )
    parser.add_argument(
        "--outputs",
        nargs="+",
        required=True,
        help="Output JSONL paths, one per samples-per-class value",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)",
    )
    parser.add_argument(
        "--preserve-input-order",
        action="store_true",
        help="Write samples in the original file order",
    )
    args = parser.parse_args()

    if len(args.samples_per_class) != len(args.outputs):
        print(
            "error: --samples-per-class and --outputs must have the same length",
            file=sys.stderr,
        )
        return 2

    if not os.path.exists(args.input):
        print(f"error: missing input file: {args.input}", file=sys.stderr)
        return 2

    sizes = list(args.samples_per_class)
    rng = random.Random(args.seed)

    # For each size -> class -> list[(idx, obj)]
    reservoirs: Dict[int, Dict[int, List[Tuple[int, dict]]]] = {
        n: defaultdict(list) for n in sizes
    }
    counts: Dict[int, int] = defaultdict(int)

    for idx, line in enumerate(iter_jsonl(args.input)):
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cls = obj.get(args.class_key)
        if cls is None:
            continue
        try:
            cls = int(cls)
        except (TypeError, ValueError):
            continue

        counts[cls] += 1
        seen = counts[cls]

        for n in sizes:
            bucket = reservoirs[n][cls]
            if len(bucket) < n:
                bucket.append((idx, obj))
            else:
                j = rng.randint(1, seen)
                if j <= n:
                    bucket[j - 1] = (idx, obj)

    for n, out_path in zip(sizes, args.outputs):
        selected: List[Tuple[int, dict]] = []
        for items in reservoirs[n].values():
            selected.extend(items)
        if args.preserve_input_order:
            selected.sort(key=lambda x: x[0])

        with open(out_path, "w", encoding="utf-8") as out:
            for _, obj in selected:
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")

        print(
            f"wrote {len(selected)} records to {out_path} "
            f"(samples_per_class={n}, classes={len(reservoirs[n])})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
