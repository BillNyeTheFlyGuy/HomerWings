# HomerWings Plotting

Analysis and plotting tools for HomerWings wing-region CSV outputs.

The project currently packages the main `HomerwingsDataAnalyzer` workflow so it can be versioned, tested, and extended on GitHub rather than maintained as one loose script.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m homerwings "C:\path\to\Oregon_Results"
```

By default, outputs are written under:

```text
<master-folder>\HomerWings_Analysis_Output\<timestamp>\
```

You can choose a different output folder:

```powershell
python -m homerwings "C:\path\to\Oregon_Results" --output "C:\path\to\analysis-output"
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
