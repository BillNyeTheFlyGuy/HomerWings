#!/usr/bin/env python3
"""
DAPI nuclei pipeline for HomerWings.

This module scans experiment folders for Ilastik probability maps and matching
raw TIFF stacks, segments nuclei using probability heuristics, and exports
annotated overlays and summaries. It is designed to be callable from the CLI or
from the GUI entry point.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from scipy import ndimage as ndi
from skimage import exposure, measure, morphology, segmentation, util

logger = logging.getLogger(__name__)


@dataclass
class DapiNucleiPipelineConfig:
    """Configuration for DAPI nuclei processing."""

    input_root: Path
    output_root: Path
    probability_dataset: str = "exported_data"
    nucleus_channel: int = 0
    probability_threshold: float = 0.35
    slope_threshold: float = 0.08
    neighbour_radius: int = 24
    min_area: int = 24
    large_output_enabled: bool = True
    overlay_cmap: str = "magma"
    raw_stack_glob: str = "*.tif*"
    probability_glob: str = "*.h5"

    def ensure_ready(self) -> None:
        self.input_root = Path(self.input_root)
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)


@dataclass
class NucleusSummary:
    slice_index: int
    disc: str
    area: int
    slope: float
    is_low_confidence: bool


@dataclass
class SliceResult:
    index: int
    total_area: int
    modal_disc: str
    summaries: List[NucleusSummary] = field(default_factory=list)
    overlay_path: Optional[Path] = None


@dataclass
class ExperimentResult:
    name: str
    slice_results: List[SliceResult]
    modal_overlay: Optional[Path]
    coverage_table_path: Path


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------

def _load_probability_map(probability_path: Path, dataset: str, channel: int) -> np.ndarray:
    with h5py.File(probability_path, "r") as f:
        if dataset not in f:
            raise KeyError(f"Dataset '{dataset}' not found in {probability_path}")
        data = f[dataset][...]
    data = np.squeeze(data)
    if data.ndim == 4:
        # Possible layouts: Z x C x Y x X or Z x Y x X x C
        if data.shape[1] < 10:  # assume channels second
            data = np.moveaxis(data, 1, -1)
    if data.ndim != 4:
        raise ValueError(f"Unexpected probability shape {data.shape} in {probability_path}")
    if channel >= data.shape[-1]:
        raise IndexError(
            f"Requested channel {channel} but map has {data.shape[-1]} channels"
        )
    return data[..., channel]


def _load_raw_stack(raw_path: Optional[Path]) -> Optional[np.ndarray]:
    if raw_path is None:
        return None
    stack = imageio.volread(raw_path)
    stack = util.img_as_float32(stack)
    return stack


# -----------------------------------------------------------------------------
# Segmentation helpers
# -----------------------------------------------------------------------------

def _region_probability_slope(prob_slice: np.ndarray, region_mask: np.ndarray) -> float:
    values = prob_slice[region_mask]
    if values.size == 0:
        return 0.0
    high = np.percentile(values, 90)
    low = np.percentile(values, 10)
    gradient = np.mean(np.abs(np.gradient(values))) if values.ndim > 1 else 0
    return float((high - low) + gradient)


def _segment_slice(prob_slice: np.ndarray, cfg: DapiNucleiPipelineConfig) -> Tuple[np.ndarray, Dict[int, float]]:
    smoothed = ndi.gaussian_filter(prob_slice, sigma=1.0)
    threshold = max(cfg.probability_threshold, float(smoothed.mean()))
    mask = smoothed > threshold
    mask = morphology.remove_small_objects(mask, cfg.min_area)
    labeled = measure.label(mask)

    slopes: Dict[int, float] = {}
    for region in measure.regionprops(labeled, intensity_image=prob_slice):
        slopes[region.label] = _region_probability_slope(prob_slice, labeled == region.label)
    return labeled, slopes


def _replace_low_confidence_with_hull(
    label_image: np.ndarray, slopes: Dict[int, float], cfg: DapiNucleiPipelineConfig
) -> Tuple[np.ndarray, Dict[int, bool], Dict[int, float]]:
    low_confidence: Dict[int, bool] = {}
    adjusted = np.zeros_like(label_image)
    slope_map: Dict[int, float] = {}
    current_label = 1
    for region in measure.regionprops(label_image):
        region_mask = label_image == region.label
        slope = slopes.get(region.label, 0.0)
        if slope < cfg.slope_threshold:
            hull_mask = morphology.convex_hull_image(region_mask)
            low_confidence[current_label] = True
            adjusted[hull_mask] = current_label
        else:
            low_confidence[current_label] = False
            adjusted[region_mask] = current_label
        slope_map[current_label] = slope
        current_label += 1
    return adjusted, low_confidence, slope_map


def _discard_isolated_low_confidence(
    label_image: np.ndarray,
    low_confidence: Dict[int, bool],
    slopes: Dict[int, float],
    cfg: DapiNucleiPipelineConfig,
) -> Tuple[np.ndarray, Dict[int, bool], Dict[int, float]]:
    final_labels = np.zeros_like(label_image)
    current_label = 1
    labelled_high = np.isin(label_image, [idx for idx, is_low in low_confidence.items() if not is_low])
    high_neighbourhood = morphology.binary_dilation(labelled_high, morphology.disk(cfg.neighbour_radius))

    new_low_flags: Dict[int, bool] = {}
    new_slopes: Dict[int, float] = {}
    for region in measure.regionprops(label_image):
        mask = label_image == region.label
        is_low = low_confidence.get(region.label, False)
        if is_low and not np.any(high_neighbourhood & mask):
            continue
        final_labels[mask] = current_label
        new_low_flags[current_label] = is_low
        new_slopes[current_label] = slopes.get(region.label, 0.0)
        current_label += 1
    return final_labels, new_low_flags, new_slopes


def _assign_discs(label_image: np.ndarray) -> Dict[int, str]:
    assignments: Dict[int, str] = {}
    centroids = [region.centroid[1] for region in measure.regionprops(label_image)]
    if not centroids:
        return assignments
    median_x = float(np.median(centroids))
    for region in measure.regionprops(label_image):
        assignments[region.label] = "disc_right" if region.centroid[1] >= median_x else "disc_left"
    return assignments


def _false_color_overlay(prob_slice: np.ndarray, label_image: np.ndarray, cmap_name: str) -> np.ndarray:
    background = exposure.rescale_intensity(prob_slice, in_range="image", out_range=(0, 1))
    colored = np.stack([background, background, background], axis=-1)
    if label_image.max() == 0:
        return colored
    areas = {region.label: region.area for region in measure.regionprops(label_image)}
    area_map = np.zeros_like(label_image, dtype=float)
    for label, area in areas.items():
        area_map[label_image == label] = area
    norm = colors.Normalize(vmin=area_map[area_map > 0].min(), vmax=area_map.max())
    cmap = cm.get_cmap(cmap_name)
    overlay = cmap(norm(area_map))
    overlay[..., 3] = (area_map > 0) * 0.45
    blended = colored.copy()
    blended[area_map > 0] = overlay[..., :3][area_map > 0]
    return blended


# -----------------------------------------------------------------------------
# Processing
# -----------------------------------------------------------------------------

def _process_slice(
    prob_slice: np.ndarray, cfg: DapiNucleiPipelineConfig
) -> Tuple[np.ndarray, Dict[int, bool], Dict[int, float]]:
    labeled, slopes = _segment_slice(prob_slice, cfg)
    hull_labels, low_confidence, slope_map = _replace_low_confidence_with_hull(labeled, slopes, cfg)
    final_labels, final_low_flags, final_slopes = _discard_isolated_low_confidence(
        hull_labels, low_confidence, slope_map, cfg
    )
    return final_labels, final_low_flags, final_slopes


def _summaries_for_slice(
    label_image: np.ndarray,
    low_flags: Dict[int, bool],
    disc_assignments: Dict[int, str],
    slopes: Dict[int, float],
    slice_idx: int,
) -> List[NucleusSummary]:
    summaries: List[NucleusSummary] = []
    for region in measure.regionprops(label_image):
        disc = disc_assignments.get(region.label, "disc_left")
        summaries.append(
            NucleusSummary(
                slice_index=slice_idx,
                disc=disc,
                area=region.area,
                slope=slopes.get(region.label, 0.0),
                is_low_confidence=low_flags.get(region.label, False),
            )
        )
    return summaries


def _save_overlay(
    output_path: Path,
    prob_slice: np.ndarray,
    label_image: np.ndarray,
    disc_assignments: Dict[int, str],
    low_flags: Dict[int, bool],
    title: str,
    cfg: DapiNucleiPipelineConfig,
) -> None:
    overlay = _false_color_overlay(prob_slice, label_image, cfg.overlay_cmap)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)
    outlines = segmentation.find_boundaries(label_image)
    ax.imshow(np.ma.masked_where(~outlines, outlines), cmap="autumn", alpha=0.7)
    for region in measure.regionprops(label_image):
        cy, cx = region.centroid
        disc = disc_assignments.get(region.label, "disc_left")
        text = f"{region.label}\n{disc.split('_')[1]}"
        if low_flags.get(region.label, False):
            text += "\nlow"
        ax.text(cx, cy, text, color="white", fontsize=6, ha="center")
    ax.set_title(title)
    ax.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _coverage_summary(results: List[SliceResult], output_path: Path) -> None:
    header = ["slice", "disc", "area", "low_confidence", "slope"]
    lines: List[str] = [",".join(header)]
    for result in results:
        for summary in result.summaries:
            lines.append(
                f"{summary.slice_index},{summary.disc},{summary.area},{int(summary.is_low_confidence)},{summary.slope:.4f}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _modal_slice(results: List[SliceResult]) -> Optional[SliceResult]:
    if not results:
        return None
    return max(results, key=lambda s: s.total_area)


def process_experiment(experiment_folder: Path, cfg: DapiNucleiPipelineConfig) -> ExperimentResult:
    probability_files = sorted(experiment_folder.glob(cfg.probability_glob))
    raw_files = sorted(experiment_folder.glob(cfg.raw_stack_glob))
    probability_path = probability_files[0] if probability_files else None
    if probability_path is None:
        raise FileNotFoundError(f"No probability map found in {experiment_folder}")
    raw_path = raw_files[0] if raw_files else None

    prob_stack = _load_probability_map(probability_path, cfg.probability_dataset, cfg.nucleus_channel)
    _ = _load_raw_stack(raw_path)
    output_dir = cfg.output_root / experiment_folder.name
    output_dir.mkdir(parents=True, exist_ok=True)

    slice_results: List[SliceResult] = []
    for idx in range(prob_stack.shape[0]):
        prob_slice = prob_stack[idx]
        labels, low_flags, slopes = _process_slice(prob_slice, cfg)
        disc_assignments = _assign_discs(labels)
        summaries = _summaries_for_slice(labels, low_flags, disc_assignments, slopes, idx)
        total_area = sum(s.area for s in summaries)
        overlay_path = None
        if cfg.large_output_enabled:
            title = f"Slice {idx} nuclei"
            overlay_path = output_dir / f"slice_{idx:03d}_overlay.png"
            _save_overlay(overlay_path, prob_slice, labels, disc_assignments, low_flags, title, cfg)
        slice_results.append(
            SliceResult(index=idx, total_area=total_area, modal_disc="mixed", summaries=summaries, overlay_path=overlay_path)
        )

    modal = _modal_slice(slice_results)
    modal_overlay = None
    if modal is not None:
        prob_slice = prob_stack[modal.index]
        labels, low_flags, _ = _process_slice(prob_slice, cfg)
        disc_assignments = _assign_discs(labels)
        modal_overlay = output_dir / f"modal_slice_{modal.index:03d}.png"
        _save_overlay(modal_overlay, prob_slice, labels, disc_assignments, low_flags, "Modal z-slice", cfg)

    coverage_path = output_dir / "nuclei_coverage.csv"
    _coverage_summary(slice_results, coverage_path)

    return ExperimentResult(
        name=experiment_folder.name,
        slice_results=slice_results,
        modal_overlay=modal_overlay,
        coverage_table_path=coverage_path,
    )


def run_dapi_nuclei_pipeline(cfg: DapiNucleiPipelineConfig) -> List[ExperimentResult]:
    cfg.ensure_ready()
    experiment_folders = [p for p in cfg.input_root.iterdir() if p.is_dir()]
    results: List[ExperimentResult] = []
    for folder in experiment_folders:
        try:
            logger.info("Processing experiment %s", folder.name)
            result = process_experiment(folder, cfg)
            results.append(result)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed processing %s: %s", folder, exc)
    return results


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def run_dapi_pipeline_cli(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run the DAPI nuclei pipeline across experiments")
    parser.add_argument("--input-root", required=True, help="Folder containing experiment subdirectories")
    parser.add_argument("--output-root", required=True, help="Folder to write outputs")
    parser.add_argument("--probability-threshold", type=float, default=0.35, help="Probability threshold for nuclei detection")
    parser.add_argument("--slope-threshold", type=float, default=0.08, help="Minimum slope for high-confidence nuclei")
    parser.add_argument("--neighbour-radius", type=int, default=24, help="Neighbourhood radius used to rescue low-confidence nuclei")
    parser.add_argument("--min-area", type=int, default=24, help="Minimum nucleus area in pixels")
    parser.add_argument("--disable-large-output", action="store_true", help="Suppress auxiliary figures and emit only the modal overlay")
    parser.add_argument("--probability-dataset", default="exported_data", help="Dataset name inside the Ilastik H5 file")
    parser.add_argument("--nucleus-channel", type=int, default=0, help="Probability channel index for nuclei")
    parser.add_argument("--raw-glob", default="*.tif*", help="Glob to discover raw TIFF stacks")
    parser.add_argument("--probability-glob", default="*.h5", help="Glob to discover probability maps")

    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = DapiNucleiPipelineConfig(
        input_root=Path(args.input_root),
        output_root=Path(args.output_root),
        probability_dataset=args.probability_dataset,
        nucleus_channel=args.nucleus_channel,
        probability_threshold=args.probability_threshold,
        slope_threshold=args.slope_threshold,
        neighbour_radius=args.neighbour_radius,
        min_area=args.min_area,
        large_output_enabled=not args.disable_large_output,
        raw_stack_glob=args.raw_glob,
        probability_glob=args.probability_glob,
    )

    results = run_dapi_nuclei_pipeline(cfg)
    logger.info("Completed DAPI nuclei pipeline for %d experiments", len(results))


__all__ = [
    "DapiNucleiPipelineConfig",
    "ExperimentResult",
    "run_dapi_nuclei_pipeline",
    "run_dapi_pipeline_cli",
]
