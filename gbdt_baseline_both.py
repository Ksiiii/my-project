# gbdt_baseline_both.py - v02 Grid Runner Friendly Version (WP0 Split-Fixed Enhanced)
# 目标：
# - 支持数据集 {ideal, perturbed}
# - 支持目标模式 {fm, k_meas, delta_k}
# - 统一在 FM_N 空间评估并保存结果（txt + scatter + 可选 summary CSV）
# - ✅ 支持 --split_dir：读取固化 train/val/test 切分（WP0 必需）

import os
import math
import argparse
from typing import Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# ----------------- 默认配置 -----------------
DEFAULT_DATA_VERSION = "02"
DEFAULT_DATASETS = {
    "ideal": "VDI2230_vdiStrict_ideal_10000.csv",
    "perturbed": "VDI2230_vdiStrict_perturbed_10000.csv",
}

TARGET_COL_MAP = {
    "fm": "FM_N",
    "k_meas": "k_meas",
    "delta_k": "delta_k",
}

BASE_EXCLUDE_COLS: List[str] = [
    "pass_fail",
    "fail_reason",
]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def split_df(df: pd.DataFrame, seed: int, test_size: float = 0.2, val_size_in_trainval: float = 0.2
            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    返回 train/val/test 三份：
    - test 占总量 test_size（默认 0.2）
    - val 占总量 0.8 * val_size_in_trainval（默认 0.16）
    - train 占总量 0.64
    """
    trainval_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(trainval_df, test_size=val_size_in_trainval, random_state=seed)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_splits(df: pd.DataFrame, split_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    从 split_dir 读取固化的 train/val/test 索引（npy 文件），并切分 df。
    split_dir 需要包含：
      - train_idx.npy
      - val_idx.npy
      - test_idx.npy
    """
    train_idx = np.load(os.path.join(split_dir, "train_idx.npy"))
    val_idx = np.load(os.path.join(split_dir, "val_idx.npy"))
    test_idx = np.load(os.path.join(split_dir, "test_idx.npy"))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    return train_df, val_df, test_df


def load_dataset(tag: str, data_dir: str, datasets_map: Dict[str, str]) -> pd.DataFrame:
    path = os.path.join(data_dir, datasets_map[tag])
    if not os.path.exists(path):
        raise FileNotFoundError(f"数据集不存在: {path}")
    return pd.read_csv(path)


def build_features_and_labels(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, np.ndarray, List[str], str]:
    """根据 mode 构造特征 X、标签 y 以及特征列名列表。"""
    if mode not in TARGET_COL_MAP:
        raise ValueError(f"不支持的 mode: {mode}. 允许: {list(TARGET_COL_MAP.keys())}")

    target_col = TARGET_COL_MAP[mode]

    # 排除标签相关列，避免信息泄露
    exclude_cols = BASE_EXCLUDE_COLS + [
        "FM_N",
        "k_meas",
        "delta_k",
        target_col,
        "FM_target_N",
    ]

    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].copy()
    y = df[target_col].astype(float).values

    # 把字符串类别列（series, class 等）编码成数字，避免 sklearn 报错
    for col in feature_cols:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category").cat.codes.astype(float)

    return X, y, feature_cols, target_col


def preds_to_FM(df_subset: pd.DataFrame, preds: np.ndarray, mode: str) -> np.ndarray:
    """把模型输出（可能是 FM_N / k_meas / delta_k）统一转换成 FM_N 预测值。"""
    if mode == "fm":
        return preds
    if mode == "k_meas":
        return preds * df_subset["F_proof_N"].values
    if mode == "delta_k":
        k_pred = df_subset["k_vdi"].values + preds
        return k_pred * df_subset["F_proof_N"].values
    raise ValueError(f"不支持的 mode: {mode}")


def evaluate_in_FM_space(
    *,
    tag: str,
    mode: str,
    seed: int,
    test_df: pd.DataFrame,
    y_pred_raw: np.ndarray,
    out_dir: str,
    run_id: str,
) -> Dict[str, Any]:
    """在 FM_N 空间下计算误差指标，并保存结果（txt + scatter）。"""
    y_true_FM = test_df["FM_N"].values.astype(float)
    y_pred_FM = preds_to_FM(test_df, y_pred_raw, mode)

    mae = float(mean_absolute_error(y_true_FM, y_pred_FM))
    rmse = float(math.sqrt(mean_squared_error(y_true_FM, y_pred_FM)))
    r2 = float(r2_score(y_true_FM, y_pred_FM))

    eps = 1e-12
    valid = np.abs(y_true_FM) > eps
    mape = float(np.mean(np.abs((y_pred_FM[valid] - y_true_FM[valid]) / y_true_FM[valid])) * 100.0)

    rel_err = (y_pred_FM - y_true_FM) / np.maximum(np.abs(y_true_FM), eps)
    within_5 = float(np.mean(np.abs(rel_err) <= 0.05))
    within_10 = float(np.mean(np.abs(rel_err) <= 0.10))
    under_tight = float(np.mean(y_pred_FM < y_true_FM))
    over_tight = float(np.mean(y_pred_FM > y_true_FM))

    result = {
        "model": "GBDT",
        "dataset": tag,
        "mode": mode,
        "seed": seed,
        "MAE_FM": mae,
        "RMSE_FM": rmse,
        "R2_FM": r2,
        "MAPE_FM(%)": mape,
        "within_5%": within_5,
        "within_10%": within_10,
        "under_tight_ratio": under_tight,
        "over_tight_ratio": over_tight,
        "n_test": int(len(test_df)),
        "run_id": run_id,
    }

    # 保存 txt
    txt_path = os.path.join(out_dir, f"{run_id}_metrics_FM.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    # 保存散点图
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_FM, y_pred_FM, alpha=0.3, s=5)
    lims = [min(y_true_FM.min(), y_pred_FM.min()), max(y_true_FM.max(), y_pred_FM.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True FM_N [N]")
    plt.ylabel("Predicted FM_N [N]")
    plt.title(f"GBDT ({tag}) - True vs Pred FM (mode={mode}, seed={seed})")
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"{run_id}_true_vs_pred_FM.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()

    return result


def run_one(
    *,
    tag: str,
    mode: str,
    seed: int,
    data_dir: str,
    datasets_map: Dict[str, str],
    out_dir: str,
    split_dir: str = "",
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 3,
    subsample: float = 0.8,
) -> Dict[str, Any]:
    """运行单个 (dataset, mode, seed) 设置，返回 FM 空间指标。"""
    df = load_dataset(tag, data_dir, datasets_map)

    # ✅ WP0：优先使用固化切分
    if split_dir:
        train_df, val_df, test_df = load_splits(df, split_dir)
    else:
        train_df, val_df, test_df = split_df(df, seed=seed)

    X_train, y_train, feature_cols, target_col = build_features_and_labels(train_df, mode)
    X_test, y_test, _, _ = build_features_and_labels(test_df, mode)

    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        random_state=seed,
    )
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)

    # 目标空间指标（辅助调试/表格对齐）
    base_mae = float(mean_absolute_error(y_test, y_pred_test))
    base_rmse = float(math.sqrt(mean_squared_error(y_test, y_pred_test)))
    base_r2 = float(r2_score(y_test, y_pred_test))

    run_id = f"gbdt_{tag}_mode-{mode}_seed-{seed}"
    metrics = evaluate_in_FM_space(
        tag=tag, mode=mode, seed=seed,
        test_df=test_df, y_pred_raw=y_pred_test,
        out_dir=out_dir, run_id=run_id,
    )
    metrics.update({
        "target_col": target_col,
        "MAE_target": base_mae,
        "RMSE_target": base_rmse,
        "R2_target": base_r2,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, default=DEFAULT_DATA_VERSION)
    parser.add_argument("--dataset", type=str, default="all", choices=["ideal", "perturbed", "all"])
    parser.add_argument("--mode", type=str, default="all", choices=["fm", "k_meas", "delta_k", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--summary_csv", type=str, default="")  # 可选：追加写入汇总表

    # ✅ WP0：固化 split 目录
    parser.add_argument("--split_dir", type=str, default="",
                        help="Fixed split dir, e.g. splits/02/ideal/seed_0")

    # 模型超参（保持与你原版一致，支持命令行覆盖）
    parser.add_argument("--n_estimators", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--max_depth", type=int, default=3)
    parser.add_argument("--subsample", type=float, default=0.8)

    args = parser.parse_args()

    data_dir = os.path.join("data", args.data_version)
    out_dir = args.out_dir or os.path.join("results", f"gbdt_v{args.data_version}")
    ensure_dir(out_dir)

    selected_datasets = ["ideal", "perturbed"] if args.dataset == "all" else [args.dataset]
    selected_modes = ["fm", "k_meas", "delta_k"] if args.mode == "all" else [args.mode]

    all_rows: List[Dict[str, Any]] = []
    for tag in selected_datasets:
        for mode in selected_modes:
            # 对每个 dataset 自动推导 split_dir：如果用户传的是空，就退回随机切分
            auto_split_dir = args.split_dir
            if (not auto_split_dir) and args.dataset != "all":
                auto_split_dir = args.split_dir

            print(f"\n========== 运行 GBDT ({tag}) mode={mode} seed={args.seed} ==========")
            row = run_one(
                tag=tag, mode=mode, seed=args.seed,
                data_dir=data_dir, datasets_map=DEFAULT_DATASETS,
                out_dir=out_dir,
                split_dir=auto_split_dir,
                n_estimators=args.n_estimators,
                learning_rate=args.learning_rate,
                max_depth=args.max_depth,
                subsample=args.subsample,
            )
            all_rows.append(row)
            print("FM-space metrics:", {k: row[k] for k in ["MAE_FM", "RMSE_FM", "R2_FM", "MAPE_FM(%)", "within_10%"]})

    if args.summary_csv:
        # 追加写入
        summary_path = args.summary_csv
        ensure_dir(os.path.dirname(summary_path) or ".")
        df_sum = pd.DataFrame(all_rows)
        if os.path.exists(summary_path):
            df_sum.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            df_sum.to_csv(summary_path, index=False)
        print(f"\n已追加写入汇总表: {summary_path}")

    print("\n全部 GBDT 运行完成。")


if __name__ == "__main__":
    main()
