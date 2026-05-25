from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import HomerwingsDataAnalyzer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run HomerWings CSV analysis.")
    parser.add_argument("master_folder", type=Path, help="Folder containing condition subfolders.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output folder. Defaults to a timestamped folder under the master folder.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only save calculated CSV outputs.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    analyzer = HomerwingsDataAnalyzer(master_folder=args.master_folder, output_root=args.output)
    analyzer.analyze()
    analyzer.save_results()

    enhanced = analyzer.calculate_wing_metrics()
    enhanced_path = analyzer.output_root / "homerwings_analysis_results_with_calculations.csv"
    enhanced.to_csv(enhanced_path, index=False)
    print(f"Enhanced results saved to: {enhanced_path}")

    if not args.skip_plots:
        plots_root = analyzer.output_root / "plots"
        analyzer.plot_wing_area_binned_vs_temperature(output_dir=plots_root / "binned")
        analyzer.plot_wing_area_vs_avg_cell_area_per_condition(output_dir=plots_root / "per_condition")

    print(f"\nDone. Everything saved under:\n{analyzer.output_root}")
    return 0
