from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone HomerWings plotting/summary workflow. "
            "This does not run or import the main HomerWings.py application."
        )
    )
    parser.add_argument("master_folder", type=Path, help="Folder containing condition subfolders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to a timestamped folder under the master folder.",
    )
    parser.add_argument(
        "--plots",
        choices=["none", "basic", "full"],
        default="full",
        help="Plot suite to generate. Use 'none' to only write CSV summaries.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Alias for --plots none.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plot_suite = "none" if args.data_only else args.plots

    from .analyzer import HomerwingsDataAnalyzer

    analyzer = HomerwingsDataAnalyzer(master_folder=args.master_folder, output_root=args.output)
    analyzer.analyze()
    analyzer.save_results()

    enhanced = analyzer.calculate_wing_metrics()
    enhanced_path = analyzer.output_root / "homerwings_analysis_results_with_calculations.csv"
    enhanced.to_csv(enhanced_path, index=False)
    print(f"Enhanced results saved to: {enhanced_path}")
    analyzer.write_condition_summary()

    if plot_suite in {"basic", "full"}:
        plots_root = analyzer.output_root / "plots"
        analyzer.plot_wing_area_binned_vs_temperature(output_dir=plots_root / "binned")
        analyzer.plot_wing_area_vs_avg_cell_area_per_condition(output_dir=plots_root / "per_condition")
        if plot_suite == "full":
            analyzer.plot_size_relationships_with_fits(output_root=plots_root / "size_relationships")
    else:
        print("Plot generation skipped.")

    analyzer.write_run_manifest(plot_suite=plot_suite)

    print(f"\nDone. Everything saved under:\n{analyzer.output_root}")
    return 0
