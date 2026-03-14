import argparse
from pathlib import Path

from utils.plotting import update_training_plot


def iter_logdirs(root: Path, recursive: bool):
    patterns = ["overall_log.csv"] if not recursive else ["overall_log.csv", "**/overall_log.csv"]
    seen = set()

    for pattern in patterns:
        for log_file in root.glob(pattern):
            logdir = log_file.parent.resolve()
            if logdir in seen:
                continue
            seen.add(logdir)
            yield logdir


def main():
    parser = argparse.ArgumentParser(description="Regenerate training plots from overall_log.csv files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="One or more log directories or run directories to scan.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan subdirectories recursively for repetition folders.",
    )
    args = parser.parse_args()

    generated = 0
    missing = []

    for raw_path in args.paths:
        root = Path(raw_path).expanduser()
        if not root.exists():
            missing.append(str(root))
            continue

        for logdir in sorted(iter_logdirs(root, args.recursive)):
            print(f"[PLOT] Regenerating plots in {logdir}")
            update_training_plot(str(logdir))
            generated += 1

    if missing:
        print("[PLOT] Missing paths:")
        for path in missing:
            print(f"  - {path}")

    print(f"[PLOT] Processed {generated} log director{'y' if generated == 1 else 'ies'}.")


if __name__ == "__main__":
    main()