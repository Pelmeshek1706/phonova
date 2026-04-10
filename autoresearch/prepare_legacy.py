import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_DIR = BASE_DIR / "artifacts"

LEGACY_TARGETS = ("Depression_label", "PTSD_label")
LEGACY_MODES = ("experiment", "final", "custom")
LEGACY_COMMANDS = ("build-datasets", "train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parse legacy dataset-builder and training arguments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_build = subparsers.add_parser(
        "build-datasets",
        help="Parse legacy dataset build arguments.",
    )
    p_build.add_argument("--detailed-labels-path", type=Path, required=True)
    p_build.add_argument("--data-dir", type=Path, required=True)
    p_build.add_argument("--out-dir", type=Path, required=True)
    p_build.add_argument("--language", default="eng")
    p_build.add_argument("--iteration", default="3")

    p_train = subparsers.add_parser(
        "train",
        help="Parse legacy model training arguments.",
    )
    p_train.add_argument("--dataset-path", type=Path, required=True)
    p_train.add_argument("--target-col", choices=LEGACY_TARGETS, required=True)
    p_train.add_argument("--mode", default="experiment", choices=LEGACY_MODES)
    p_train.add_argument("--train-values", nargs="*", default=None)
    p_train.add_argument("--eval-value", default=None)
    p_train.add_argument("--save-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR)
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument("--importance-top-k", type=int, default=5)
    p_train.add_argument("--importance-corr-thr", type=float, default=0.90)
    p_train.add_argument("--no-importance-selection", action="store_true")
    p_train.add_argument("--no-importance", action="store_true")
    p_train.add_argument("--no-tree-tuning", action="store_true")
    p_train.add_argument("--keep-demographics", action="store_true")

    args = parser.parse_args()

    if args.command == "train" and args.mode == "custom":
        if not args.train_values or args.eval_value is None:
            raise ValueError("For mode=custom you must provide --train-values and --eval-value.")
        args.train_values = tuple(args.train_values)
    elif args.command == "train":
        if args.mode == "experiment":
            args.train_values = ("train",)
            args.eval_value = "dev"
        elif args.mode == "final":
            args.train_values = ("train", "dev")
            args.eval_value = "test"

    return args


def main():
    args = parse_args()
    print(json.dumps(vars(args), indent=2, default=str))


if __name__ == "__main__":
    main()
