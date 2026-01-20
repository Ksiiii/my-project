# run_full_grid.py
# 一键跑完：GBDT + FT 在 (ideal/perturbed) × (fm/k_meas/delta_k) 的全组合
#
# 用法：
#   python run_full_grid.py --seed 42 --ft_epochs 50
#
# 输出：
# - 每个 run 的 txt + scatter 图分别落在 results/gbdt_v02 与 results/ft_v02
# - 额外生成一个总汇总表：results/grid_v02_summary.csv

import os
import argparse
import pandas as pd

import gbdt_baseline_both as gbdt
import ft_transformer_baseline_both as ft


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_version", type=str, default="02")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--datasets", type=str, default="ideal,perturbed",
                        help="逗号分隔，如: ideal,perturbed")
    parser.add_argument("--modes", type=str, default="fm,k_meas,delta_k",
                        help="逗号分隔，如: fm,k_meas,delta_k")
    parser.add_argument("--models", type=str, default="gbdt,ft",
                        help="逗号分隔: gbdt,ft 或只跑一个")
    parser.add_argument("--ft_epochs", type=int, default=50)
    parser.add_argument("--ft_batch_size", type=int, default=256)

    # gbdt 超参（可选）
    parser.add_argument("--gbdt_n_estimators", type=int, default=300)
    parser.add_argument("--gbdt_learning_rate", type=float, default=0.05)
    parser.add_argument("--gbdt_max_depth", type=int, default=3)
    parser.add_argument("--gbdt_subsample", type=float, default=0.8)

    parser.add_argument("--summary_csv", type=str, default=os.path.join("results", "grid_v02_summary.csv"))
    parser.add_argument("--stop_on_error", action="store_true", help="若设置，则任一组合报错即停止")
    args = parser.parse_args()

    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    models = [x.strip().lower() for x in args.models.split(",") if x.strip()]

    data_dir = os.path.join("data", args.data_version)
    gbdt_out = os.path.join("results", f"gbdt_v{args.data_version}")
    ft_out = os.path.join("results", f"ft_v{args.data_version}")
    ensure_dir(os.path.dirname(args.summary_csv) or ".")

    rows = []
    for tag in datasets:
        for mode in modes:
            if "gbdt" in models:
                try:
                    r = gbdt.run_one(
                        tag=tag, mode=mode, seed=args.seed,
                        data_dir=data_dir, datasets_map=gbdt.DEFAULT_DATASETS,
                        out_dir=gbdt_out,
                        n_estimators=args.gbdt_n_estimators,
                        learning_rate=args.gbdt_learning_rate,
                        max_depth=args.gbdt_max_depth,
                        subsample=args.gbdt_subsample,
                    )
                    rows.append(r)
                except Exception as e:
                    if args.stop_on_error:
                        raise
                    rows.append({"model": "GBDT", "dataset": tag, "mode": mode, "seed": args.seed, "error": str(e)})

            if "ft" in models:
                try:
                    r = ft.run_one(
                        tag=tag, mode=mode, seed=args.seed,
                        data_dir=data_dir, datasets_map=ft.DEFAULT_DATASETS,
                        out_dir=ft_out,
                        epochs=args.ft_epochs,
                        batch_size=args.ft_batch_size,
                    )
                    rows.append(r)
                except Exception as e:
                    if args.stop_on_error:
                        raise
                    rows.append({"model": "FT", "dataset": tag, "mode": mode, "seed": args.seed, "error": str(e)})

    df = pd.DataFrame(rows)
    df.to_csv(args.summary_csv, index=False)
    print(f"\n全组合运行结束，汇总表已保存：{args.summary_csv}")
    print(df[["model", "dataset", "mode", "seed", "MAE_FM", "RMSE_FM", "R2_FM", "MAPE_FM(%)", "within_10%", "error"]]
          if "error" in df.columns else df)


if __name__ == "__main__":
    main()
