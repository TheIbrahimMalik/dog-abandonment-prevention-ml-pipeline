from __future__ import annotations

import argparse
from pathlib import Path

from dog_abandonment.return_labels import build_return_risk_dataset, ReturnLabelConfig
from dog_abandonment.train import TrainConfig, train
from dog_abandonment.predict import predict_batch


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="dog-abandonment",
        description="Dog abandonment prevention ML pipeline (risk, reason, interventions).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pp = sub.add_parser("preprocess", help="Build adoption→return modelling dataset.")
    pp.add_argument("--intakes", default="data/raw/aac_intakes.csv", help="Path to intakes CSV.")
    pp.add_argument("--outcomes", default="data/raw/aac_outcomes.csv", help="Path to outcomes CSV.")
    pp.add_argument("--out", default="data/processed/return_risk_dataset.csv", help="Output CSV path.")
    pp.add_argument("--horizons", default="30,60,90", help="Comma-separated day horizons, e.g. 30,60,90")

    tr = sub.add_parser("train", help="Train binary + reason models on the processed dataset.")
    tr.add_argument("--dataset", default="data/processed/return_risk_dataset.csv", help="Processed dataset CSV.")
    tr.add_argument("--horizon", type=int, default=90, help="Return horizon in days.")
    tr.add_argument("--outdir", default="artifacts", help="Directory to write models/metrics.")
    tr.add_argument("--min-reason-count", type=int, default=50, help="Collapse rarer reasons into 'Other'.")
    tr.add_argument("--top-n-breeds", type=int, default=50, help="Top-N breeds to keep; others -> Other.")
    tr.add_argument("--top-n-colours", type=int, default=20, help="Top-N colours to keep; others -> Other.")

    pr = sub.add_parser("predict", help="Run inference on adoption episodes CSV.")
    pr.add_argument("--input", required=True, help="Input CSV with adoption episode fields.")
    pr.add_argument("--out", default="outputs/predictions.csv", help="Output CSV path.")
    pr.add_argument("--horizon", type=int, default=90, help="Return horizon in days.")
    pr.add_argument("--threshold", type=float, default=None, help="Risk threshold; default from validation metrics.")
    pr.add_argument("--binary-model", default=None, help="Path to binary model joblib.")
    pr.add_argument("--reason-model", default=None, help="Path to reason model joblib.")
    pr.add_argument("--binary-metrics", default=None, help="Path to binary metrics JSON (for default threshold).")

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.command == "preprocess":
        horizons = tuple(int(x.strip()) for x in args.horizons.split(",") if x.strip())
        cfg = ReturnLabelConfig(return_days=horizons)

        df = build_return_risk_dataset(args.intakes, args.outcomes, cfg=cfg)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

        print(f"Wrote: {out_path} (rows={len(df)})")
        return 0

    if args.command == "train":
        cfg = TrainConfig(
            dataset_csv=args.dataset,
            horizon_days=args.horizon,
            out_dir=args.outdir,
            min_reason_count=args.min_reason_count,
            top_n_breeds=args.top_n_breeds,
            top_n_colours=args.top_n_colours,
        )
        train(cfg)
        return 0

    if args.command == "predict":
        predict_batch(
            args.input,
            horizon_days=args.horizon,
            threshold=args.threshold,
            binary_model_path=args.binary_model,
            reason_model_path=args.reason_model,
            binary_metrics_path=args.binary_metrics,
            output_csv=args.out,
        )
        print(f"Wrote: {args.out}")
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
