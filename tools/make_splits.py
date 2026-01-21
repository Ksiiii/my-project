# tools/make_splits.py
# 固化 train/val/test 切分：输出 train_idx.npy / val_idx.npy / test_idx.npy
# 用法示例：
# python tools/make_splits.py --data_path data/02/VDI2230_vdiStrict_ideal_10000.csv --seed 0 --out_dir splits/02/ideal/seed_0
# python tools/make_splits.py --data_path data/02/VDI2230_vdiStrict_perturbed_10000.csv --seed 0 --out_dir splits/02/perturbed/seed_0

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def save_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    ap = argparse.ArgumentParser(description="Create reproducible train/val/test splits and save indices.")
    ap.add_argument("--data_path", required=True, help="CSV path, e.g. data/02/xxx.csv")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for split.")
    ap.add_argument("--out_dir", required=True, help="Output dir, e.g. splits/02/ideal/seed_0")
    ap.add_argument("--test_size", type=float, default=0.2, help="Test proportion of full data.")
    ap.add_argument("--val_size_in_trainval", type=float, default=0.2,
                    help="Val proportion inside trainval (after removing test). "
                         "Default 0.2 -> 0.16 of total if test_size=0.2.")
    ap.add_argument("--shuffle", action="store_true", default=True, help="Shuffle before split (default True).")
    ap.add_argument("--no_shuffle", action="store_true", help="Disable shuffle (rarely needed).")
    args = ap.parse_args()

    shuffle = True
    if args.no_shuffle:
        shuffle = False

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取数据，仅用于获得行数并确保 CSV 可读
    df = pd.read_csv(args.data_path)
    n = len(df)
    idx = np.arange(n)

    # 先切 test，再从 trainval 切 val
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=shuffle,
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=args.val_size_in_trainval,
        random_state=args.seed,
        shuffle=shuffle,
    )

    # 保存为 npy
    np.save(os.path.join(args.out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(args.out_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(args.out_dir, "test_idx.npy"), test_idx)

    # 同时保存一个可读的 meta.txt
    meta = []
    meta.append(f"data_path={args.data_path}")
    meta.append(f"seed={args.seed}")
    meta.append(f"shuffle={shuffle}")
    meta.append(f"n_total={n}")
    meta.append(f"test_size={args.test_size}")
    meta.append(f"val_size_in_trainval={args.val_size_in_trainval}")
    meta.append(f"n_train={len(train_idx)}")
    meta.append(f"n_val={len(val_idx)}")
    meta.append(f"n_test={len(test_idx)}")
    meta.append("")  # newline
    save_text(os.path.join(args.out_dir, "meta.txt"), "\n".join(meta))

    print("✅ Splits saved to:", args.out_dir)
    print("   train:", len(train_idx), "val:", len(val_idx), "test:", len(test_idx))
    print("   files:", "train_idx.npy, val_idx.npy, test_idx.npy, meta.txt")


if __name__ == "__main__":
    main()
