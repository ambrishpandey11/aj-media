"""
run_all.py — Single entry point for the full DTC Marketing Analytics pipeline.
Usage:
    python run_all.py              # Run all 4 tasks
    python run_all.py --task 1     # Run only Task 1
    python run_all.py --task 2     # Run only Task 2
    python run_all.py --task 3     # Run only Task 3
    python run_all.py --task 4     # Run only Task 4 (bonus)
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent / "src"))

def banner(title: str):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print(f"{'='*62}\n")

def run_task_direct(task_num: int):
    """Import and run module directly (handles numbered filenames)."""
    import importlib.util
    filename_map = {
        1: "01_data_quality.py",
        2: "02_performance_analysis.py",
        3: "03_recommendations.py",
        4: "04_predictive_model.py",
    }
    src_dir = Path(__file__).parent / "src"
    script  = src_dir / filename_map[task_num]
    spec    = importlib.util.spec_from_file_location(f"task{task_num}", script)
    mod     = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.run()

def main():
    parser = argparse.ArgumentParser(description="DTC Marketing Analytics Pipeline")
    parser.add_argument("--task", type=int, choices=[1,2,3,4],
                        help="Run a specific task only (1-4). Default: run all.")
    parser.add_argument("--skip-bonus", action="store_true",
                        help="Skip Task 4 (bonus predictive model)")
    args = parser.parse_args()

    banner("DTC FITNESS SUPPLEMENTS - MARKETING ANALYTICS PIPELINE")
    print("  Tasks: Data Quality -> Performance Analysis -> Recommendations -> Predictive Model")
    print(f"  Output directory: outputs/\n")

    # Check data exists
    from pathlib import Path as P
    data_dir   = P(__file__).parent / "data"
    xlsx_files = list(data_dir.glob("*.xlsx")) + list(data_dir.glob("*.xls"))
    if not xlsx_files:
        print("  WARNING: No Excel file found in data/")
        print("  Please place your marketing dataset (Excel) in the data/ folder.")
        print("  Expected: data/marketing_data.xlsx")
        print("\n  Exiting. Add your data file and re-run.\n")
        sys.exit(1)
    else:
        print(f"  Dataset found: {xlsx_files[0].name}")

    tasks_to_run = [args.task] if args.task else ([1,2,3] if args.skip_bonus else [1,2,3,4])

    results = {}
    total_start = time.time()

    for task_num in tasks_to_run:
        task_names = {
            1: "Data Quality & Cleaning",
            2: "Performance Analysis",
            3: "Strategic Recommendations",
            4: "Predictive Modelling (Bonus)",
        }
        banner(f"TASK {task_num}: {task_names[task_num]}")
        start = time.time()
        try:
            run_task_direct(task_num)
            elapsed = time.time() - start
            results[task_num] = ("PASSED", f"{elapsed:.1f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[task_num] = (f"FAILED: {e}", "-")

    # Summary
    total_elapsed = time.time() - total_start
    banner("PIPELINE COMPLETE - SUMMARY")
    for task_num, (status, elapsed) in results.items():
        print(f"  Task {task_num}: {status}  ({elapsed})")
    print(f"\n  Total runtime: {total_elapsed:.1f}s")
    print(f"\n  Outputs saved to:")
    print(f"    outputs/figures/   <- All charts (.png)")
    print(f"    outputs/reports/   <- CSV summaries + executive_summary.docx")
    print()

if __name__ == "__main__":
    main()
