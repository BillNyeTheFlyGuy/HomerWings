from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

np = None
pd = None
stats = None


# =============================================================================
# EDIT THESE SETTINGS IF YOU WANT TO RUN THE SCRIPT DIRECTLY FROM YOUR EDITOR
# =============================================================================

# Folder containing condition folders named like: mutation_temperature_sex
# Example: r"C:\Users\Arthu\OneDrive - The University of Liverpool\Transfers\Oregon_Results"
MASTER_FOLDER = r"C:\path\to\Oregon_Results"

# Leave as None to create:
# <MASTER_FOLDER>\HomerWings_Analysis_Output\<timestamp>\
OUTPUT_ROOT = None

# Options:
#   "full"  = CSVs + condition summary + all plots in this script
#   "basic" = CSVs + condition summary + lighter starter plots
#   "none"  = CSVs + condition summary only
PLOT_SUITE = "full"


def load_dependencies() -> None:
    global np, pd, stats

    if np is not None and pd is not None and stats is not None:
        return

    try:
        import numpy as _np
        import pandas as _pd
        from scipy import stats as _stats
    except ModuleNotFoundError as exc:
        missing = exc.name
        raise SystemExit(
            f"Missing required package: {missing}\n"
            "Install dependencies with:\n"
            "  python -m pip install -r requirements.txt"
        ) from exc

    np = _np
    pd = _pd
    stats = _stats


class HomerwingsDataAnalyzer:
    def __init__(self, master_folder: str | Path, output_root: str | Path | None = None):
        load_dependencies()

        self.master_folder = Path(master_folder)

        if output_root is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_root = self.master_folder / "HomerWings_Analysis_Output" / stamp
        else:
            self.output_root = Path(output_root)

        self.output_root.mkdir(parents=True, exist_ok=True)
        self.data = {
            "mutation": [],
            "temperature": [],
            "sex": [],
            "wing_id": [],
            "region_1_area": [],
            "region_1_avg_cell_area": [],
            "region_2_area": [],
            "region_2_avg_cell_area": [],
            "region_3_area": [],
            "region_3_avg_cell_area": [],
            "region_4plus_area": [],
            "region_4plus_avg_cell_area": [],
            "region_4plus_needs_verification": [],
            "wing_area_um2": [],
            "aspect_ratio": [],
        }

    def _ensure_dir(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _regress_label(self, x, y, label_prefix: str = "fit"):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x = x[mask]
        y = y[mask]

        if x.size < 3 or np.unique(x).size < 2:
            return None

        slope, intercept, r_value, _p_value, _std_err = stats.linregress(x, y)
        r2 = r_value**2
        label = f"{label_prefix}: y={slope:.4g}x+{intercept:.4g}, R^2={r2:.3f}"
        return slope, intercept, r2, label

    def _sex_marker(self, sex: str) -> str:
        return {"Male": "o", "Female": "s"}.get(str(sex), "^")

    def parse_folder_name(self, folder_name: str):
        parts = folder_name.split("_")
        if len(parts) >= 3:
            mutation = parts[0]
            temperature = parts[1]
            sex = parts[2]
            return mutation, temperature, sex
        return None

    def extract_wing_id(self, filename: str):
        series_match = re.search(r"(processednew-\d+)", filename, re.IGNORECASE)
        slice_match = re.search(r"_s(\d+)", filename, re.IGNORECASE)

        if series_match and slice_match:
            return f"{series_match.group(1)}_s{slice_match.group(1)}"
        if slice_match:
            return f"s{slice_match.group(1)}"
        return None

    def process_voronoi_file(self, filepath: str | Path):
        filepath = Path(filepath)
        df = pd.read_csv(filepath, comment="#", on_bad_lines="skip", engine="python")

        print(f"  Voronoi file shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First few rows:\n{df.head()}")

        if df.empty:
            raise ValueError("Voronoi file is empty")

        has_headers = any("region" in str(col).lower() for col in df.columns)
        if not has_headers:
            df = pd.read_csv(filepath, header=None, comment="#", on_bad_lines="skip", engine="python")
            print(f"  Re-reading without headers. Shape: {df.shape}")

        regions = {}
        region_2_area = None

        for idx, row in df.iterrows():
            try:
                if has_headers:
                    region_col = None
                    area_col = None
                    cell_area_col = None

                    for col in df.columns:
                        col_lower = str(col).lower()
                        if "region" in col_lower and "name" in col_lower:
                            region_col = col
                        elif "region" in col_lower and "area" in col_lower:
                            area_col = col
                        elif "average" in col_lower and "cell" in col_lower and "area" in col_lower:
                            cell_area_col = col

                    if region_col is None or area_col is None or cell_area_col is None:
                        if df.shape[1] < 4:
                            continue
                        region_name = str(row.iloc[1]).strip()
                        region_area = float(row.iloc[2])
                        avg_cell_area = float(row.iloc[3])
                    else:
                        region_name = str(row[region_col]).strip()
                        region_area = float(row[area_col])
                        avg_cell_area = float(row[cell_area_col])
                else:
                    if (
                        df.shape[1] < 4
                        or pd.isna(row.iloc[1])
                        or pd.isna(row.iloc[2])
                        or pd.isna(row.iloc[3])
                    ):
                        continue
                    region_name = str(row.iloc[1]).strip()
                    region_area = float(row.iloc[2])
                    avg_cell_area = float(row.iloc[3])

                region_match = re.search(r"region[_\s-]*(\d+)", region_name, re.IGNORECASE)
                if region_match:
                    region_num = int(region_match.group(1))
                    regions[region_num] = {
                        "area": region_area,
                        "avg_cell_area": avg_cell_area,
                    }
                    print(
                        f"  Found region {region_num}: "
                        f"area={region_area}, avg_cell_area={avg_cell_area}"
                    )

                    if region_num == 2:
                        region_2_area = region_area
            except (ValueError, TypeError, KeyError) as exc:
                print(f"  Error parsing row {idx}: {exc}")
                continue

        result = {
            "region_1_area": None,
            "region_1_avg_cell_area": None,
            "region_2_area": None,
            "region_2_avg_cell_area": None,
            "region_3_area": None,
            "region_3_avg_cell_area": None,
            "region_4plus_area": None,
            "region_4plus_avg_cell_area": None,
            "region_4plus_needs_verification": False,
        }

        print(f"  Total regions found: {len(regions)} - {sorted(regions.keys())}")

        for region_num in [1, 2, 3]:
            if region_num in regions:
                result[f"region_{region_num}_area"] = regions[region_num]["area"]
                result[f"region_{region_num}_avg_cell_area"] = regions[region_num]["avg_cell_area"]
            else:
                print(f"  WARNING: Region {region_num} not found!")

        regions_4plus = []
        needs_verification = False

        for region_num in sorted(regions.keys()):
            if region_num >= 4:
                regions_4plus.append(regions[region_num])
                print(f"  Region {region_num} added to 4+ category")

        if len(regions) == 4 and region_2_area is not None:
            for region_num in regions.keys():
                if region_num >= 4 and regions[region_num]["area"] >= 2 * region_2_area:
                    needs_verification = True
                    print(
                        f"  Region {region_num} flagged - area is 2x region 2 "
                        "(NEEDS VERIFICATION)"
                    )

        if regions_4plus:
            total_area = sum(region["area"] for region in regions_4plus)
            weighted_sum = sum(region["area"] * region["avg_cell_area"] for region in regions_4plus)
            avg_cell_area = weighted_sum / total_area if total_area > 0 else np.nan

            result["region_4plus_area"] = total_area
            result["region_4plus_avg_cell_area"] = avg_cell_area
            result["region_4plus_needs_verification"] = needs_verification
            print(
                f"  Region 4+ combined: area={total_area}, "
                f"weighted avg_cell_area={avg_cell_area}"
            )

        return result

    def process_wing_metrics_file(self, filepath: str | Path):
        filepath = Path(filepath)
        df = pd.read_csv(filepath, comment="#", on_bad_lines="skip", engine="python")

        print(f"  Wing metrics file shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")

        if df.empty:
            raise ValueError("Wing metrics file is empty")

        wing_area = None
        aspect_ratio = None

        if df.shape[1] >= 2:
            for idx, row in df.iterrows():
                try:
                    metric_name = str(row.iloc[0]).lower()
                    value = row.iloc[1]

                    if (
                        "wing_area_microns_squared" in metric_name
                        or "wing area microns squared" in metric_name.replace("_", " ")
                    ):
                        wing_area = float(value)
                        print(f"  Found wing area: {wing_area} (row {idx})")
                    elif (
                        "wing_bbox_aspect_ratio" in metric_name
                        or "wing ellipse aspect ratio" in metric_name.replace("_", " ")
                    ):
                        aspect_ratio = float(value)
                        print(f"  Found aspect ratio: {aspect_ratio} (row {idx})")

                    if wing_area is not None and aspect_ratio is not None:
                        break
                except (ValueError, TypeError):
                    continue

        if wing_area is None or aspect_ratio is None:
            print("  Trying fixed row positions (5 and 9)...")
            df_no_header = pd.read_csv(
                filepath,
                header=None,
                comment="#",
                on_bad_lines="skip",
                engine="python",
            )

            if df_no_header.shape[0] >= 9 and df_no_header.shape[1] >= 2:
                wing_area = float(df_no_header.iloc[4, 1])
                aspect_ratio = float(df_no_header.iloc[8, 1])
                print(f"  Wing area from row 5: {wing_area}")
                print(f"  Aspect ratio from row 9: {aspect_ratio}")

        if wing_area is None or aspect_ratio is None:
            raise ValueError("Could not find wing area or aspect ratio in metrics file")

        return {
            "wing_area_um2": wing_area,
            "aspect_ratio": aspect_ratio,
        }

    def process_condition_folder(
        self,
        folder_path: str | Path,
        mutation: str,
        temperature: str,
        sex: str,
    ):
        folder_path = Path(folder_path)
        csv_files = list(folder_path.glob("*.csv"))
        wing_data = {}

        for csv_file in csv_files:
            wing_id = self.extract_wing_id(csv_file.name)
            if wing_id is None:
                continue

            if wing_id not in wing_data:
                wing_data[wing_id] = {}

            name_lower = csv_file.name.lower()

            if "voronoi_average_cell_area_normal" in name_lower:
                wing_data[wing_id]["voronoi"] = csv_file
                wing_data[wing_id]["analysis_mode"] = "normal"
            elif "voronoi_average_cell_area_pentagone" in name_lower:
                wing_data[wing_id]["voronoi"] = csv_file
                wing_data[wing_id]["analysis_mode"] = "pentagone"
            elif "wing_metrics_normal" in name_lower:
                wing_data[wing_id]["metrics"] = csv_file
                wing_data[wing_id]["analysis_mode"] = wing_data[wing_id].get(
                    "analysis_mode", "normal"
                )
            elif "wing_metrics_pentagone" in name_lower:
                wing_data[wing_id]["metrics"] = csv_file
                wing_data[wing_id]["analysis_mode"] = wing_data[wing_id].get(
                    "analysis_mode", "pentagone"
                )

        for wing_id, files in wing_data.items():
            if "voronoi" not in files or "metrics" not in files:
                continue

            try:
                mode = files.get("analysis_mode", "unknown")
                print(f"\n  Processing wing {wing_id} ({mode})...")

                voronoi_data = self.process_voronoi_file(files["voronoi"])
                metrics_data = self.process_wing_metrics_file(files["metrics"])

                self.data["mutation"].append(mutation)
                self.data["temperature"].append(temperature)
                self.data["sex"].append(sex)
                self.data["wing_id"].append(wing_id)

                for key in [
                    "region_1_area",
                    "region_1_avg_cell_area",
                    "region_2_area",
                    "region_2_avg_cell_area",
                    "region_3_area",
                    "region_3_avg_cell_area",
                    "region_4plus_area",
                    "region_4plus_avg_cell_area",
                    "region_4plus_needs_verification",
                ]:
                    self.data[key].append(voronoi_data[key])

                self.data["wing_area_um2"].append(metrics_data["wing_area_um2"])
                self.data["aspect_ratio"].append(metrics_data["aspect_ratio"])

                print(f"  Successfully processed: {mutation}_{temperature}_{sex} - {wing_id} ({mode})")
            except Exception as exc:
                print(f"  Error processing {wing_id} in {mutation}_{temperature}_{sex}: {exc}")

    def analyze(self):
        if not self.master_folder.exists():
            raise ValueError(f"Master folder not found: {self.master_folder}")

        for subfolder in self.master_folder.iterdir():
            if not subfolder.is_dir():
                continue

            parsed = self.parse_folder_name(subfolder.name)
            if parsed:
                mutation, temperature, sex = parsed
                print(f"\n{'=' * 60}")
                print(f"Processing condition: {mutation}_{temperature}_{sex}")
                print(f"{'=' * 60}")
                self.process_condition_folder(subfolder, mutation, temperature, sex)
            else:
                print(f"Skipping folder with invalid format: {subfolder.name}")

        print(f"\n{'=' * 60}")
        print(f"Analysis complete! Processed {len(self.data['wing_id'])} wings.")
        print(f"{'=' * 60}")

        verification_count = sum(self.data["region_4plus_needs_verification"])
        if verification_count > 0:
            print(f"{verification_count} wings flagged for verification (region 4+ 2x rule)")

    def get_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def save_results(self, output_path: str | Path | None = None):
        df = self.get_dataframe()

        if output_path is None:
            output_path = self.output_root / "homerwings_analysis_results.csv"
        else:
            output_path = Path(output_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")

    def write_run_manifest(self, plot_suite: str = "basic") -> Path:
        manifest_path = self.output_root / "RUN_SUMMARY.md"
        df = self.get_dataframe()

        lines = [
            "# HomerWings Plotting Run",
            "",
            f"- Input folder: `{self.master_folder}`",
            f"- Output folder: `{self.output_root}`",
            f"- Wings processed: `{len(df)}`",
            f"- Plot suite: `{plot_suite}`",
            "",
            "## Generated Files",
            "",
            "- `homerwings_analysis_results.csv`",
            "- `homerwings_analysis_results_with_calculations.csv`",
            "- `condition_summary.csv`",
        ]

        if plot_suite != "none":
            lines.append("- `plots/`")
            lines.append("- `Temp_Individual_Histograms/`")

        manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Run summary saved to: {manifest_path}")
        return manifest_path

    def write_condition_summary(self, output_path: str | Path | None = None) -> Path:
        df = self.calculate_wing_metrics().copy()
        if output_path is None:
            output_path = self.output_root / "condition_summary.csv"
        else:
            output_path = Path(output_path)

        summary = (
            df.groupby(["mutation", "temperature", "sex"], dropna=False)
            .agg(
                n=("wing_id", "count"),
                wing_area_mean=("wing_area_um2", "mean"),
                wing_area_sem=(
                    "wing_area_um2",
                    lambda x: stats.sem(x, nan_policy="omit") if len(x.dropna()) > 1 else np.nan,
                ),
                wing_avg_cell_area_mean=("wing_avg_cell_area", "mean"),
                wing_avg_cell_area_sem=(
                    "wing_avg_cell_area",
                    lambda x: stats.sem(x, nan_policy="omit") if len(x.dropna()) > 1 else np.nan,
                ),
                total_estimated_cells_mean=("total_estimated_cells", "mean"),
                total_estimated_cells_sem=(
                    "total_estimated_cells",
                    lambda x: stats.sem(x, nan_policy="omit") if len(x.dropna()) > 1 else np.nan,
                ),
                region_coverage_fraction_mean=("region_coverage_fraction", "mean"),
                verification_flags=("region_4plus_needs_verification", "sum"),
            )
            .reset_index()
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
        print(f"Condition summary saved to: {output_path}")
        return output_path

    def _configure_plot_style(self):
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "figure.dpi": 120,
                "savefig.dpi": 300,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "legend.frameon": True,
            }
        )

    @staticmethod
    def _safe_filename(text: object) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")

    def calculate_wing_metrics(self) -> pd.DataFrame:
        df = self.get_dataframe().copy()

        region_info = [
            ("region_1_area", "region_1_avg_cell_area", "region_1_estimated_cells"),
            ("region_2_area", "region_2_avg_cell_area", "region_2_estimated_cells"),
            ("region_3_area", "region_3_avg_cell_area", "region_3_estimated_cells"),
            ("region_4plus_area", "region_4plus_avg_cell_area", "region_4plus_estimated_cells"),
        ]

        for area_col, cell_col, est_col in region_info:
            df[est_col] = df[area_col] / df[cell_col]

        area_cols = [item[0] for item in region_info]
        df["total_region_area"] = df[area_cols].sum(axis=1, skipna=True)

        weighted_sum = pd.Series(0.0, index=df.index)
        weight_sum = pd.Series(0.0, index=df.index)

        for area_col, cell_col, _ in region_info:
            valid = df[area_col].notna() & df[cell_col].notna()
            weighted_sum += (df[area_col] * df[cell_col]).where(valid, 0)
            weight_sum += df[area_col].where(valid, 0)

        df["wing_avg_cell_area"] = weighted_sum / weight_sum
        df.loc[weight_sum == 0, "wing_avg_cell_area"] = np.nan
        df["wing_avg_cell_area_weighted"] = df["wing_avg_cell_area"]

        df["total_estimated_cells"] = df["wing_area_um2"] / df["wing_avg_cell_area"]
        df["region_coverage_fraction"] = df["total_region_area"] / df["wing_area_um2"]
        df["low_region_coverage_flag"] = df["region_coverage_fraction"] < 0.90

        df["region_1_area_ratio"] = df["region_1_area"] / df["wing_area_um2"]
        df["region_2_area_ratio"] = df["region_2_area"] / df["wing_area_um2"]
        df["region_3_area_ratio"] = df["region_3_area"] / df["wing_area_um2"]
        df["region_4plus_area_ratio"] = df["region_4plus_area"] / df["wing_area_um2"]

        df["region_1_cell_density"] = 1 / df["region_1_avg_cell_area"]
        df["region_2_cell_density"] = 1 / df["region_2_avg_cell_area"]
        df["region_3_cell_density"] = 1 / df["region_3_avg_cell_area"]
        df["region_4plus_cell_density"] = 1 / df["region_4plus_avg_cell_area"]
        df["overall_cell_density"] = df["total_estimated_cells"] / df["wing_area_um2"]

        return df

    def perform_trend_analysis(self, metric_col: str, df: pd.DataFrame | None = None):
        if df is None:
            df = self.calculate_wing_metrics()

        df = df.copy()
        df["temperature_numeric"] = pd.to_numeric(df["temperature"], errors="coerce")
        df_clean = df.dropna(subset=["temperature_numeric", metric_col])

        results = {
            "trends": {},
            "slope_comparisons": {},
            "sex_comparisons": {},
        }

        for mutation in df_clean["mutation"].unique():
            results["trends"][mutation] = {}

            for sex in df_clean["sex"].unique():
                df_group = df_clean[
                    (df_clean["mutation"] == mutation) & (df_clean["sex"] == sex)
                ]

                if len(df_group) < 3:
                    continue

                x = df_group["temperature_numeric"].values
                y = df_group[metric_col].values

                if len(np.unique(x)) < 2:
                    print(
                        f"  Warning: Skipping {mutation} - {sex} for {metric_col}: "
                        "only one temperature value"
                    )
                    continue

                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                results["trends"][mutation][sex] = {
                    "slope": slope,
                    "intercept": intercept,
                    "r_squared": r_value**2,
                    "p_value": p_value,
                    "std_err": std_err,
                    "n": len(df_group),
                }

        return results

    def plot_wing_area_binned_vs_temperature(self, output_dir: str | Path = "plots"):
        import matplotlib.pyplot as plt

        self._configure_plot_style()
        output_dir = self._ensure_dir(output_dir)

        df = self.calculate_wing_metrics().copy()
        df["temperature_numeric"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["temperature_numeric", "wing_area_um2", "mutation", "sex"])

        sexes = sorted(df["sex"].unique())
        mutations = sorted(df["mutation"].unique())

        for sex in sexes:
            df_sex = df[df["sex"] == sex]
            if df_sex.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))

            for mutation in mutations:
                df_mut = df_sex[df_sex["mutation"] == mutation]
                if df_mut.empty:
                    continue

                ax.scatter(
                    df_mut["temperature_numeric"],
                    df_mut["wing_area_um2"],
                    alpha=0.25,
                    s=35,
                    label=None,
                )

                temp_stats = (
                    df_mut.groupby("temperature_numeric")["wing_area_um2"]
                    .agg(
                        mean="mean",
                        sem=lambda x: stats.sem(x, nan_policy="omit")
                        if len(x.dropna()) > 1
                        else np.nan,
                    )
                    .reset_index()
                )

                if len(temp_stats) == 0:
                    continue

                ax.errorbar(
                    temp_stats["temperature_numeric"],
                    temp_stats["mean"],
                    yerr=temp_stats["sem"],
                    marker="o",
                    linewidth=2.5,
                    capsize=5,
                    label=f"{mutation} (binned)",
                )

            ax.set_xlabel("Temperature (deg C)")
            ax.set_ylabel("Wing Area (um^2)")
            ax.set_title(f"Wing Area vs Temperature (Binned) - {sex}")
            ax.grid(True, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(fontsize=9, loc="best")

            plt.tight_layout()
            out = output_dir / f"wing_area_binned_vs_temp_{sex}.png"
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out}")

    def plot_wing_area_vs_avg_cell_area_per_condition(self, output_dir: str | Path = "plots"):
        import matplotlib.pyplot as plt

        self._configure_plot_style()
        output_dir = self._ensure_dir(output_dir)

        df = self.calculate_wing_metrics().copy()
        df["temperature_numeric"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(
            subset=["wing_area_um2", "wing_avg_cell_area", "mutation", "temperature", "sex"]
        )
        if df.empty:
            print("No data available for wing_area_vs_avg_cell_area_per_condition")
            return

        df["condition"] = (
            df["mutation"].astype(str)
            + "_"
            + df["temperature"].astype(str)
            + "C_"
            + df["sex"].astype(str)
        )

        grouped = (
            df.groupby("condition")
            .agg(
                mutation=("mutation", "first"),
                temperature=("temperature", "first"),
                sex=("sex", "first"),
                wing_area_mean=("wing_area_um2", "mean"),
                wing_area_sem=(
                    "wing_area_um2",
                    lambda x: stats.sem(x, nan_policy="omit")
                    if len(x.dropna()) > 1
                    else np.nan,
                ),
                cell_area_mean=("wing_avg_cell_area", "mean"),
                cell_area_sem=(
                    "wing_avg_cell_area",
                    lambda x: stats.sem(x, nan_policy="omit")
                    if len(x.dropna()) > 1
                    else np.nan,
                ),
                n=("wing_area_um2", "count"),
            )
            .reset_index()
        )

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.errorbar(
            grouped["cell_area_mean"],
            grouped["wing_area_mean"],
            xerr=grouped["cell_area_sem"],
            yerr=grouped["wing_area_sem"],
            fmt="none",
            alpha=0.6,
            capsize=4,
            linewidth=1.5,
        )
        ax.scatter(grouped["cell_area_mean"], grouped["wing_area_mean"], s=70, alpha=0.9)

        for _, row in grouped.iterrows():
            ax.text(
                row["cell_area_mean"],
                row["wing_area_mean"],
                f"{row['condition']} (n={int(row['n'])})",
                fontsize=7,
                alpha=0.7,
            )

        ax.set_xlabel("Wing Average Cell Area (um^2)")
        ax.set_ylabel("Wing Area (um^2)")
        ax.set_title("Wing Area vs Wing Average Cell Area (Condition Means +/- SEM)")
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        out = output_dir / "wing_area_vs_avg_cell_area_condition_means.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")

    def plot_size_relationships_with_fits(self, output_root: str | Path = "plots/size_relationships"):
        import matplotlib.pyplot as plt

        self._configure_plot_style()
        output_root = self._ensure_dir(output_root)

        df = self.calculate_wing_metrics().copy()
        df["temperature_numeric"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["mutation", "sex", "temperature_numeric"])
        if df.empty:
            print("No data available for size relationship plots")
            return

        relationships = [
            (
                "wing_area_um2",
                "wing_avg_cell_area",
                "Wing Area (um^2)",
                "Wing Average Cell Area (um^2)",
                "cell_area_vs_wing_area",
            ),
            (
                "wing_area_um2",
                "total_estimated_cells",
                "Wing Area (um^2)",
                "Estimated Cell Number",
                "cell_number_vs_wing_area",
            ),
            (
                "wing_avg_cell_area",
                "total_estimated_cells",
                "Wing Average Cell Area (um^2)",
                "Estimated Cell Number",
                "cell_number_vs_cell_area",
            ),
        ]

        for x_col, y_col, xlabel, ylabel, filename_base in relationships:
            dfx = df.dropna(subset=[x_col, y_col]).copy()
            if dfx.empty:
                print(f"No data for {filename_base}")
                continue

            for temperature in sorted(dfx["temperature_numeric"].dropna().unique()):
                sub_t = dfx[dfx["temperature_numeric"] == temperature].copy()
                if sub_t.empty:
                    continue

                fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
                mutations = sorted(sub_t["mutation"].dropna().unique())
                cmap = plt.colormaps.get_cmap("tab20").resampled(max(1, len(mutations)))
                mutation_colors = {mutation: cmap(i) for i, mutation in enumerate(mutations)}

                for mutation in mutations:
                    sub_m = sub_t[sub_t["mutation"] == mutation]
                    if sub_m.empty:
                        continue

                    for sex in sorted(sub_m["sex"].dropna().unique()):
                        sub_ms = sub_m[sub_m["sex"] == sex]
                        marker = self._sex_marker(sex)
                        color = mutation_colors[mutation]
                        ax.scatter(
                            sub_ms[x_col],
                            sub_ms[y_col],
                            alpha=0.7,
                            s=52,
                            marker=marker,
                            color=color,
                            edgecolors="white",
                            linewidths=0.6,
                            label=f"{mutation} - {sex}",
                        )

                        regression = self._regress_label(
                            sub_ms[x_col].values,
                            sub_ms[y_col].values,
                            label_prefix=f"{mutation} - {sex}",
                        )
                        if regression is None:
                            continue

                        slope, intercept, _r2, label = regression
                        x_min = np.nanmin(sub_ms[x_col].values)
                        x_max = np.nanmax(sub_ms[x_col].values)
                        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
                            continue

                        x_line = np.array([x_min, x_max], dtype=float)
                        y_line = slope * x_line + intercept
                        ax.plot(
                            x_line,
                            y_line,
                            color=color,
                            linewidth=2,
                            linestyle="-" if str(sex) == "Male" else "--",
                            label=label,
                        )

                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"{ylabel} vs {xlabel} at {temperature:g} C")
                ax.legend(fontsize=7, loc="center left", bbox_to_anchor=(1.02, 0.5))

                out = output_root / f"{filename_base}_{self._safe_filename(temperature)}C.png"
                fig.savefig(out, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {out}")

    def plot_temp_individual_histograms(
        self,
        output_root: str | Path = "Temp_Individual_Histograms",
    ):
        import math
        import matplotlib.pyplot as plt

        self._configure_plot_style()
        output_root = self._ensure_dir(output_root)

        df = self.calculate_wing_metrics().copy()
        df["temperature_numeric"] = pd.to_numeric(df["temperature"], errors="coerce")
        df = df.dropna(subset=["mutation", "sex", "temperature_numeric"])
        if df.empty:
            print("No data available for temperature individual histograms")
            return

        metrics = [
            (
                "wing_avg_cell_area",
                "Average Cell Area (um^2)",
                "average_cell_area_histograms",
            ),
            (
                "total_estimated_cells",
                "Estimated Cell Number",
                "cell_number_histograms",
            ),
            (
                "wing_area_um2",
                "Wing Area (um^2)",
                "wing_area_histograms",
            ),
        ]

        sex_colors = {
            "Male": "#2563eb",
            "Female": "#dc2626",
        }
        fallback_colors = ["#16a34a", "#9333ea", "#ea580c", "#0891b2"]

        for temperature in sorted(df["temperature_numeric"].dropna().unique()):
            df_temp = df[df["temperature_numeric"] == temperature].copy()
            mutations = sorted(df_temp["mutation"].dropna().unique())
            if not mutations:
                continue

            temp_label = f"{temperature:g}C"
            temp_dir = self._ensure_dir(output_root / self._safe_filename(temp_label))
            ncols = 2
            nrows = max(1, math.ceil(len(mutations) / ncols))

            for metric_col, xlabel, filename_base in metrics:
                df_metric = df_temp.dropna(subset=[metric_col]).copy()
                if df_metric.empty:
                    print(f"No data for {filename_base} at {temp_label}")
                    continue

                figure_values = df_metric[metric_col].dropna()
                if figure_values.nunique() > 1:
                    bin_count = min(16, max(6, int(np.sqrt(len(figure_values))) + 2))
                    bins = np.histogram_bin_edges(figure_values, bins=bin_count)
                    x_min, x_max = float(bins[0]), float(bins[-1])
                else:
                    value = float(figure_values.iloc[0])
                    pad = abs(value) * 0.05 if value != 0 else 1.0
                    bins = np.linspace(value - pad, value + pad, 6)
                    x_min, x_max = float(bins[0]), float(bins[-1])

                fig, axes = plt.subplots(
                    nrows,
                    ncols,
                    figsize=(7.2 * ncols, 4.8 * nrows),
                    squeeze=False,
                    constrained_layout=True,
                )
                axes = axes.flatten()

                for panel_index, mutation in enumerate(mutations):
                    ax = axes[panel_index]
                    df_mutation = df_metric[df_metric["mutation"] == mutation].copy()

                    if df_mutation.empty:
                        ax.set_visible(False)
                        continue

                    values_all = df_mutation[metric_col].dropna()
                    if values_all.empty:
                        ax.set_visible(False)
                        continue

                    sexes = sorted(df_mutation["sex"].dropna().unique())
                    for sex_index, sex in enumerate(sexes):
                        values = df_mutation[df_mutation["sex"] == sex][metric_col].dropna()
                        if values.empty:
                            continue

                        color = sex_colors.get(
                            str(sex),
                            fallback_colors[sex_index % len(fallback_colors)],
                        )
                        ax.hist(
                            values,
                            bins=bins,
                            alpha=0.52,
                            color=color,
                            edgecolor="white",
                            linewidth=0.7,
                            label=f"{sex} (n={len(values)})",
                        )

                    ax.set_title(str(mutation), fontsize=11, fontweight="bold")
                    ax.set_xlabel(xlabel)
                    ax.set_ylabel("Wings")
                    ax.set_xlim(x_min, x_max)
                    ax.legend(fontsize=8)

                for unused_index in range(len(mutations), len(axes)):
                    axes[unused_index].set_visible(False)

                fig.suptitle(
                    f"{xlabel} by Mutation at {temperature:g} C\n"
                    "Male and female distributions overlaid",
                    fontsize=14,
                    fontweight="bold",
                )

                out = temp_dir / f"{filename_base}_{self._safe_filename(temp_label)}.png"
                fig.savefig(out, dpi=300, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved: {out}")


def run_analysis(
    master_folder: str | Path,
    output_root: str | Path | None = None,
    plot_suite: str = "full",
) -> Path:
    load_dependencies()

    analyzer = HomerwingsDataAnalyzer(master_folder=master_folder, output_root=output_root)
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
            analyzer.plot_temp_individual_histograms(
                output_root=analyzer.output_root / "Temp_Individual_Histograms"
            )
    else:
        print("Plot generation skipped.")

    analyzer.write_run_manifest(plot_suite=plot_suite)
    print(f"\nDone. Everything saved under:\n{analyzer.output_root}")
    return analyzer.output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone HomerWings plotting script. "
            "This file is separate from HomerWings.py."
        )
    )
    parser.add_argument(
        "master_folder",
        nargs="?",
        default=MASTER_FOLDER,
        help="Folder containing condition subfolders. Defaults to MASTER_FOLDER in this file.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_ROOT,
        help="Optional output folder. Defaults to OUTPUT_ROOT in this file.",
    )
    parser.add_argument(
        "--plots",
        choices=["none", "basic", "full"],
        default=PLOT_SUITE,
        help="Plot suite to generate. Defaults to PLOT_SUITE in this file.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Alias for --plots none.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    selected_plot_suite = "none" if args.data_only else args.plots

    if not args.master_folder or args.master_folder == r"C:\path\to\Oregon_Results":
        raise SystemExit(
            "Please edit MASTER_FOLDER near the top of HomerWings_Plotting.py, "
            "or pass the folder path on the command line."
        )

    run_analysis(
        master_folder=args.master_folder,
        output_root=args.output,
        plot_suite=selected_plot_suite,
    )
