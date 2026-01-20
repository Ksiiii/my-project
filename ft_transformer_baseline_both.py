# ft_transformer_baseline_both.py - v02 Grid Runner Friendly Version
# 目标：
# - 支持数据集 {ideal, perturbed}
# - 支持目标模式 {fm, k_meas, delta_k}
# - 统一在 FM_N 空间评估并保存结果（txt + scatter + 可选 summary CSV）
#
# 用法示例：
#   python ft_transformer_baseline_both.py --dataset ideal --mode k_meas --seed 42 --epochs 50
#   python ft_transformer_baseline_both.py --dataset all --mode all --seed 42 --epochs 50
#
# 说明：
# - 依赖 pytorch_tabular
# - 输出默认写入 results/ft_v02
#
# 注意：
# - 你当前环境里 rich / lightning 的进度条存在兼容性问题；此脚本保留你原来的“补丁”逻辑。

import os
import math
import argparse
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, TrainerConfig, OptimizerConfig
from pytorch_tabular.models.ft_transformer import FTTransformerConfig

# ==== 打补丁 0：修复 rich 进度条 clear_live 的 IndexError ====
import rich.console as rich_console

_old_clear_live = rich_console.Console.clear_live

def _safe_clear_live(self, *args, **kwargs):
    try:
        return _old_clear_live(self, *args, **kwargs)
    except IndexError:
        # live_stack 为空时直接忽略，避免 "pop from empty list"
        return None

rich_console.Console.clear_live = _safe_clear_live

# ==== 打补丁 1：禁止 TabularModel 自动加载“最佳模型” checkpoint ====
import pytorch_tabular.tabular_model as pt_tabular_model

def _noop_load_best_model(self, *args, **kwargs):
    return None

pt_tabular_model.TabularModel.load_best_model = _noop_load_best_model
pt_tabular_model.TabularModel._load_best_model = _noop_load_best_model

# ==== 打补丁 2：禁用 Lightning 的 sanity check，进一步减少与进度条的交互 ====
import pytorch_lightning as pl

def _noop_sanity_check(self, *args, **kwargs):
    return None

pl.Trainer._run_sanity_check = _noop_sanity_check


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

CATEGORICAL_COLS_DEFAULT = ["series", "class"]


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


def load_dataset(tag: str, data_dir: str, datasets_map: Dict[str, str]) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, datasets_map[tag])
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到数据文件: {csv_path}")
    return pd.read_csv(csv_path)


def build_column_config(df: pd.DataFrame, mode: str) -> Tuple[List[str], List[str], str]:
    """根据数据集与 mode 构造 continuous/categorical/target 列。"""
    if mode not in TARGET_COL_MAP:
        raise ValueError(f"不支持的 mode: {mode}. 允许: {list(TARGET_COL_MAP.keys())}")

    target_col = TARGET_COL_MAP[mode]

    exclude_cols = BASE_EXCLUDE_COLS + [
        "FM_N",
        "k_meas",
        "delta_k",
        target_col,
        "FM_target_N",
    ]

    all_cols = df.columns.tolist()
    cat_cols = [c for c in CATEGORICAL_COLS_DEFAULT if c in all_cols]
    cont_cols = [c for c in all_cols if c not in exclude_cols and c not in cat_cols]
    return cont_cols, cat_cols, target_col


def preds_to_FM(df_subset: pd.DataFrame, preds: np.ndarray, mode: str) -> np.ndarray:
    """把模型输出（FM_N / k_meas / delta_k）统一转换为 FM_N 预测值。"""
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
    """在 FM_N 空间下计算工程指标，并保存结果与散点图。"""
    y_true_FM = test_df["FM_N"].values.astype(float)
    y_pred_FM = preds_to_FM(test_df, y_pred_raw, mode)

    mae = float(np.mean(np.abs(y_pred_FM - y_true_FM)))
    rmse = float(math.sqrt(np.mean((y_pred_FM - y_true_FM) ** 2)))

    ss_tot = float(np.sum((y_true_FM - np.mean(y_true_FM)) ** 2))
    ss_res = float(np.sum((y_true_FM - y_pred_FM) ** 2))
    r2 = float(1.0 - ss_res / ss_tot)

    eps = 1e-12
    mask = np.abs(y_true_FM) > eps
    mape = float(np.mean(np.abs((y_pred_FM[mask] - y_true_FM[mask]) / y_true_FM[mask])) * 100.0)

    rel_err = (y_pred_FM - y_true_FM) / np.maximum(np.abs(y_true_FM), eps)
    within_5 = float(np.mean(np.abs(rel_err) <= 0.05))
    within_10 = float(np.mean(np.abs(rel_err) <= 0.10))

    under_tight = float(np.mean(y_pred_FM < y_true_FM))
    over_tight = float(np.mean(y_pred_FM > y_true_FM))

    result = {
        "model": "FT",
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

    txt_path = os.path.join(out_dir, f"{run_id}_metrics_FM.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for k, v in result.items():
            f.write(f"{k}: {v}\n")

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true_FM, y_pred_FM, alpha=0.3, s=5)
    lims = [min(y_true_FM.min(), y_pred_FM.min()), max(y_true_FM.max(), y_pred_FM.max())]
    plt.plot(lims, lims, "k--", linewidth=1)
    plt.xlabel("True FM_N [N]")
    plt.ylabel("Predicted FM_N [N]")
    plt.title(f"FT-Transformer ({tag}) - True vs Pred FM (mode={mode}, seed={seed})")
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
    epochs: int = 50,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """运行单个 (dataset, mode, seed) 设置，返回 FM 空间指标。"""
    # 保证可复现（Lightning + numpy + python）
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)

    df = load_dataset(tag, data_dir, datasets_map)
    cont_cols, cat_cols, target_col = build_column_config(df, mode)

    data_config = DataConfig(
        target=[target_col],
        continuous_cols=cont_cols,
        categorical_cols=cat_cols,
    )

    model_config = FTTransformerConfig(
        task="regression",
        learning_rate=1e-3,
        num_heads=4,
        num_attn_blocks=3,
        input_embed_dim=64,
        embedding_dropout=0.1,
        attn_dropout=0.1,
        ff_dropout=0.1,
    )

    trainer_config = TrainerConfig(
        max_epochs=epochs,
        batch_size=batch_size,
    )

    optimizer_config = OptimizerConfig()

    model = TabularModel(
        data_config=data_config,
        model_config=model_config,
        optimizer_config=optimizer_config,
        trainer_config=trainer_config,
    )

    train_df, val_df, test_df = split_df(df, seed=seed)

    # 训练（显式提供 validation，避免内部自动切分导致与 GBDT 不一致）
    model.fit(train=train_df, validation=val_df)

    # 目标空间评估（供调试参考；不同版本返回结构可能不同，所以做成字符串保存即可）
    try:
        eval_res = model.evaluate(test_df)
    except Exception as e:
        eval_res = {"evaluate_error": str(e)}

    run_id = f"ft_{tag}_mode-{mode}_seed-{seed}"

    # 预测（predict 会在传入 df 上追加 *_prediction 列）
    pred_df = model.predict(test_df).reset_index(drop=True)
    test_df_eval = test_df.reset_index(drop=True)

    target_pred_name = f"{target_col}_prediction"
    pred_col = None
    if target_pred_name in pred_df.columns:
        pred_col = target_pred_name
    else:
        # 兼容旧/不同版本的列命名
        for c in pred_df.columns:
            if c.startswith("prediction"):
                pred_col = c
                break
        if pred_col is None and target_col in pred_df.columns:
            pred_col = target_col

    if pred_col is None:
        debug_path = os.path.join(out_dir, f"{run_id}_pred_debug.csv")
        pred_df.to_csv(debug_path, index=False)
        raise RuntimeError(f"找不到 prediction 列，已将 pred_df 保存在 {debug_path} 以便检查。")

    y_pred_raw = pred_df[pred_col].values.astype(float)

    metrics = evaluate_in_FM_space(
        tag=tag, mode=mode, seed=seed,
        test_df=test_df_eval, y_pred_raw=y_pred_raw,
        out_dir=out_dir, run_id=run_id,
    )

    # 写入目标空间评估（调试用）
    eval_path = os.path.join(out_dir, f"{run_id}_metrics_target_space.txt")
    with open(eval_path, "w", encoding="utf-8") as f:
        f.write(str(eval_res))

    metrics.update({
        "target_col": target_col,
        "n_train": int(len(train_df)),
        "n_val": int(len(val_df)),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, default=DEFAULT_DATA_VERSION)
    parser.add_argument("--dataset", type=str, default="all", choices=["ideal", "perturbed", "all"])
    parser.add_argument("--mode", type=str, default="all", choices=["fm", "k_meas", "delta_k", "all"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--summary_csv", type=str, default="")  # 可选：追加写入汇总表
    args = parser.parse_args()

    data_dir = os.path.join("data", args.data_version)
    out_dir = args.out_dir or os.path.join("results", f"ft_v{args.data_version}")
    ensure_dir(out_dir)

    selected_datasets = ["ideal", "perturbed"] if args.dataset == "all" else [args.dataset]
    selected_modes = ["fm", "k_meas", "delta_k"] if args.mode == "all" else [args.mode]

    all_rows: List[Dict[str, Any]] = []
    for tag in selected_datasets:
        for mode in selected_modes:
            print(f"\n========== 运行 FT ({tag}) mode={mode} seed={args.seed} epochs={args.epochs} ==========")
            row = run_one(
                tag=tag, mode=mode, seed=args.seed,
                data_dir=data_dir, datasets_map=DEFAULT_DATASETS,
                out_dir=out_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
            )
            all_rows.append(row)
            print("FM-space metrics:", {k: row[k] for k in ["MAE_FM", "RMSE_FM", "R2_FM", "MAPE_FM(%)", "within_10%"]})

    if args.summary_csv:
        summary_path = args.summary_csv
        ensure_dir(os.path.dirname(summary_path) or ".")
        df_sum = pd.DataFrame(all_rows)
        if os.path.exists(summary_path):
            df_sum.to_csv(summary_path, mode="a", header=False, index=False)
        else:
            df_sum.to_csv(summary_path, index=False)
        print(f"\n已追加写入汇总表: {summary_path}")

    print("\n全部 FT-Transformer 运行完成。")


if __name__ == "__main__":
    main()
