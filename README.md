# HomerWings Plotting

Standalone analysis and plotting tools for HomerWings wing-region CSV outputs.

This is deliberately separate from the main `HomerWings.py` script. The main script can keep doing image analysis/GUI work, while this plotting workflow consumes already-generated CSV files and writes summary tables and plots.

You can work on, run, and review the plotting code without importing or changing `HomerWings.py`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Run

```powershell
homerwings-plotting "C:\path\to\Oregon_Results"
```

By default, outputs are written under:

```text
<master-folder>\HomerWings_Analysis_Output\<timestamp>\
```

You can choose a different output folder:

```powershell
homerwings-plotting "C:\path\to\Oregon_Results" --output "C:\path\to\analysis-output"
```

You can also run it without installing the editable package:

```powershell
python run_homerwings_plotting.py "C:\path\to\Oregon_Results"
```

The default run creates CSV summaries and the full starter plot suite. To only write CSV summaries:

```powershell
homerwings-plotting "C:\path\to\Oregon_Results" --data-only
```

To make only the two lighter starter plots:

```powershell
homerwings-plotting "C:\path\to\Oregon_Results" --plots basic
```

## Expected Input Layout

Each condition folder should be named:

```text
<mutation>_<temperature>_<sex>
```

For example:

```text
control_25_Male
```

Within each condition folder, matching Voronoi and wing metric CSV files are paired by wing id patterns like `processednew-01_s3`.

## Notes

Generated CSVs, plots, and timestamped output folders are ignored by Git so analysis results do not accidentally get committed.

Current separation:

- `HomerWings.py`: main HomerWings application/script.
- `src/homerwings_plotting/`: standalone plotting and CSV-summary package.
- `run_homerwings_plotting.py`: convenience launcher for the plotting workflow.

Typical outputs:

- `homerwings_analysis_results.csv`: raw parsed per-wing results.
- `homerwings_analysis_results_with_calculations.csv`: parsed results plus wing-level calculations.
- `condition_summary.csv`: grouped condition means/SEM.
- `RUN_SUMMARY.md`: short manifest for the run.
- `plots/`: generated figures when plots are enabled.
