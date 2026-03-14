#!/usr/bin/env python3
import argparse
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Print rerun command(s) from a job manifest.")
    parser.add_argument("--job-dir", required=True, help="Path to job_<id> directory")
    parser.add_argument("--task-id", default="<TASK_ID>", help="Task id to substitute in template")
    args = parser.parse_args()

    job_dir = Path(args.job_dir).expanduser().resolve()
    manifest_md_path = job_dir / "JOB_MANIFEST.md"
    exact_path = job_dir / "exact_command.txt"
    run_sol_copy_path = job_dir / "run_sol_used.sh"
    master_csv = job_dir / "repetition_master.csv"

    if not manifest_md_path.exists() and not exact_path.exists():
        raise FileNotFoundError(
            f"Neither JOB_MANIFEST.md nor exact_command.txt found in {job_dir}"
        )

    template = None
    hint = "sbatch run_sol.sh"
    if manifest_md_path.exists():
        content = manifest_md_path.read_text(encoding="utf-8")
        template_match = re.search(r"^- Template: `(.+?)`", content, flags=re.MULTILINE)
        if template_match:
            template = template_match.group(1)

    exact = None
    if exact_path.exists():
        exact = exact_path.read_text(encoding="utf-8").strip()

    print("=== Re-run Info ===")
    if exact:
        print(f"Exact command from captured run:\n{exact}\n")
    if template:
        print("Per-repetition template:")
        print(template.replace("<TASK_ID>", str(args.task_id)))
        print()
    if hint:
        print("Job-array hint:")
        print(hint)
        print()

    print("Artifacts:")
    print(f"- Exact command file: {exact_path}")
    print(f"- Used run_sol.sh copy: {run_sol_copy_path}")
    print(f"- Repetition master CSV: {master_csv}")


if __name__ == "__main__":
    main()
