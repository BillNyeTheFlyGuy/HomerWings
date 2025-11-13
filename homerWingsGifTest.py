
"""

HomerWings for public beta. IJFM!



"""

from __future__ import annotations

import base64
import csv
import glob
import json
import logging
import math
import os
import queue
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, NamedTuple

import h5py
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull, Voronoi, distance_matrix
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from skimage import exposure, feature, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from skimage.filters import frangi, gaussian, sato, threshold_local
from skimage.measure import find_contours, label, regionprops
from skimage.morphology import (
    binary_dilation,
    binary_erosion,
    disk,
    h_maxima,
    remove_small_holes,
    remove_small_objects,
    skeletonize,
    white_tophat,
)
from skimage.segmentation import watershed
from tkinter import filedialog, messagebox, scrolledtext, ttk

import drosophila_gif
matplotlib.use('Agg')
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PIXELS_PER_MICRON = 1.4464



@dataclass
class TrichomeDetectionConfig:
    """Enhanced configuration with automated vein detection parameters and hybrid detection."""
    
    # Core detection parameters
    min_distance: int = 8
    high_thresh_abs: float = 0.30
    low_thresh_abs: float = 0.20
    
    # Multi-scale detection
    scales: List[float] = field(default_factory=lambda: [0.8, 1.0, 1.2])
    scale_weight_decay: float = 0.8
    
    # Pre-processing
    use_clahe: bool = False
    clahe_clip_limit: float = 0.01
    clahe_tile_grid: Optional[int] = None
    
    use_white_tophat: bool = True
    tophat_radius: int = 2
    
    use_local_threshold: bool = True
    local_block_size: int = 71
    local_offset: float = 0.0
    
    # Advanced peak detection
    two_pass: bool = True
    h_prominence: float = 0.40
    peak_rel_threshold: float = 0.50
    
    # Quality control parameters
    min_peak_intensity: float = 0.25
    max_peak_density: float = 0.002
    edge_exclusion_buffer: int = 10
    
    # Clustering and filtering
    dbscan_eps: float = 10.0
    dbscan_min_samples: int = 1
    valley_drop: float = 0.25
    neighbour_tolerance: float = 0.20
    min_neighbours: int = 7
    
    # Intervein segmentation
    intervein_threshold: float = 0.2
    min_region_area: int = 60000
    max_region_area: int = 2000000
    max_hole_size: int = 10000
    
    # IMPROVED VEIN DETECTION PARAMETERS
    vein_width_estimate: int = 3
    min_vein_length: int = 100
    vein_detection_sensitivity: float = 0.9
    use_template_matching: bool = True
    use_vesselness_filters: bool = True
    use_morphological_detection: bool = True
    background_mask_threshold: float = 0.8
    enable_background_masking: bool = True
    
    # Wing-specific parameters
    wing_orientation_correction: bool = True
    expected_wing_aspect_ratio: float = 2.5
    min_wing_area: int = 100000
    border_buffer: int = 20
    
    # Intervein region refinement
    auto_intervein_min_area: int = 400
    auto_intervein_max_area: int = 800000
    intervein_shape_filter: bool = True
    min_intervein_solidity: float = 0.4
    
    # === Hybrid detection parameters ===
    use_hybrid_detection: bool = True
    sparse_threshold: float = 0.1
    prob_weight: float = 0.8
    trichome_weight: float = 0.2
    
    # Sparse wing handling
    conservative_string_removal: bool = False
    probability_dominant_sparse: bool = True
    min_trichomes_for_validation: int = 50
    
    # Detection method selection
    wing_detection_method: str = "hybrid"
    
    # === NEW: Force merge regions 4+5 ===
    force_merge_regions_4_5: bool = False
    
    def validate(self) -> None:
        """Enhanced validation including hybrid parameters."""
        if self.min_distance < 1:
            raise ValueError("min_distance must be >= 1")
        if not 0 < self.high_thresh_abs <= 1:
            raise ValueError("high_thresh_abs must be in (0, 1]")
        if not 0 < self.low_thresh_abs <= 1:
            raise ValueError("low_thresh_abs must be in (0, 1]")
        if self.local_block_size % 2 == 0:
            raise ValueError("local_block_size must be odd")
        if self.tophat_radius < 1:
            raise ValueError("tophat_radius must be >= 1")
        
        if not 0 <= self.sparse_threshold <= 1:
            raise ValueError("sparse_threshold must be between 0 and 1")
        if not 0 <= self.prob_weight <= 1:
            raise ValueError("prob_weight must be between 0 and 1")
        if not 0 <= self.trichome_weight <= 1:
            raise ValueError("trichome_weight must be between 0 and 1")
        if abs(self.prob_weight + self.trichome_weight - 1.0) > 0.01:
            logger.warning("prob_weight + trichome_weight should sum to 1.0")
        if self.min_trichomes_for_validation < 1:
            raise ValueError("min_trichomes_for_validation must be >= 1")
        if self.wing_detection_method not in ["hybrid", "probability", "trichome"]:
            raise ValueError("wing_detection_method must be 'hybrid', 'probability', or 'trichome'")
        
        logger.info("Enhanced configuration validation passed")

CONFIG = TrichomeDetectionConfig()


class ProbabilityConsensus(NamedTuple):
    """Bundle the probability consensus outputs for downstream refinements."""

    mask: np.ndarray
    vote_map: np.ndarray
    threshold: float

class HybridWingDetector:
    """Hybrid wing detector that combines probability maps with trichome validation."""
    
    def __init__(self, config):
        self.config = config
        # Parameters for hybrid detection
        self.prob_weight = 0.5  # Weight for probability-based detection
        self.trichome_weight = 0.5  # Weight for trichome-based validation
        self.sparse_threshold = 0.1  # Threshold to detect sparse wings
        
    def detect_wing_boundary_hybrid(self, prob_map, raw_img=None, peaks=None):
        """
        Hybrid wing detection combining probability maps and trichome validation.
        Falls back gracefully for sparse wings.
        """
        logger.info("Starting hybrid wing boundary detection...")
        
        # Step 1: Get initial wing mask from probability map
        prob_wing_mask = self._detect_from_probability_enhanced(prob_map, raw_img)
        
        # Step 2: Detect or use provided trichome peaks
        if peaks is None:
            tri_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else prob_map
            peaks, _ = detect_trichome_peaks(tri_prob, self.config)
        
        logger.info(f"Using {len(peaks)} trichomes for validation")
        
        # Step 3: Assess trichome sparsity
        trichome_density = len(peaks) / prob_map.size if prob_map.size > 0 else 0
        is_sparse = trichome_density < self.sparse_threshold
        
        logger.info(f"Trichome density: {trichome_density:.6f}, sparse: {is_sparse}")
        
        if is_sparse or len(peaks) < 50:
            # Sparse wing: rely more on probability map
            logger.info("Sparse wing detected - using probability-dominant approach")
            return self._sparse_wing_detection(prob_map, raw_img, peaks, prob_wing_mask)
        else:
            # Dense wing: use hybrid approach
            logger.info("Dense wing detected - using hybrid approach")
            return self._dense_wing_detection(prob_map, raw_img, peaks, prob_wing_mask)
    
    def _detect_from_probability_enhanced(self, prob_map, raw_img):
        """Enhanced probability-based detection with multiple approaches."""
        logger.info("Creating probability-based wing mask...")
        
        candidate_masks = []
        
        # Method 1: Intervein probability (if available)
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            # Wing areas typically have moderate to high intervein probability
            intervein_mask = (intervein_prob > 0.2) & (intervein_prob < 1)
            intervein_context = {}
            if np.any(intervein_mask):
                values = intervein_prob[intervein_mask]
                intervein_context = {
                    'mean_probability': float(np.mean(values)),
                    'std_probability': float(np.std(values)),
                    'expected_range': (0.35, 0.9),
                    'expected_span': 0.55,
                }
            candidate_masks.append(('intervein', intervein_mask, intervein_context))
            logger.info(f"Intervein method: {np.sum(intervein_mask)} pixels")

        # Method 2: Vein probability (inverted - wing is where veins are NOT)
        if prob_map.shape[-1] >= 3:
            vein_prob = prob_map[..., 2]
            # Wing tissue has low to moderate vein probability
            non_vein_mask = vein_prob < 0.7
            non_vein_context = {}
            if np.any(non_vein_mask):
                values = 1.0 - np.clip(vein_prob[non_vein_mask], 0.0, 1.0)
                non_vein_context = {
                    'mean_probability': float(np.mean(values)),
                    'std_probability': float(np.std(values)),
                    'expected_range': (0.3, 0.95),
                    'expected_span': 0.65,
                }
            candidate_masks.append(('non_vein', non_vein_mask, non_vein_context))
            logger.info(f"Non-vein method: {np.sum(non_vein_mask)} pixels")

        # Method 3: Background probability (inverted)
        if prob_map.shape[-1] >= 2:
            bg_prob = prob_map[..., 1]
            # Wing areas have low background probability
            non_bg_mask = bg_prob < 0.5
            non_bg_context = {}
            if np.any(non_bg_mask):
                background_values = np.clip(bg_prob[non_bg_mask], 0.0, 1.0)
                support_values = 1.0 - background_values
                non_bg_context = {
                    'mean_probability': float(np.mean(support_values)),
                    'std_probability': float(np.std(support_values)),
                    'mean_background': float(np.mean(background_values)),
                    'expected_range': (0.35, 0.95),
                    'expected_span': 0.6,
                }
            candidate_masks.append(('non_background', non_bg_mask, non_bg_context))
            logger.info(f"Non-background method: {np.sum(non_bg_mask)} pixels")

        # Method 4: Raw image intensity (if available)
        if raw_img is not None:
            intensity_mask, intensity_context = self._intensity_based_mask(raw_img)
            candidate_masks.append(('intensity', intensity_mask, intensity_context))
            logger.info(f"Intensity method: {np.sum(intensity_mask)} pixels")

        # Method 5: Combined probability approach
        if prob_map.shape[-1] >= 4:
            # Wing areas: high trichome + moderate intervein + low background + low vein
            trichome_prob = prob_map[..., 0]
            combined_mask = (
                (trichome_prob > 0.1) &  # Some trichome presence
                (intervein_prob > 0.15) &  # Some intervein tissue
                (bg_prob < 0.6) &  # Not pure background
                (vein_prob < 0.8)   # Not pure vein
            )
            combined_context = {}
            if np.any(combined_mask):
                trichome_values = np.clip(trichome_prob[combined_mask], 0.0, 1.0)
                intervein_values = np.clip(intervein_prob[combined_mask], 0.0, 1.0)
                support_fraction = float(np.mean(trichome_values > 0.3))
                combined_context = {
                    'mean_probability': float(np.mean(trichome_values * 0.6 + intervein_values * 0.4)),
                    'std_probability': float(np.std(trichome_values)),
                    'support_fraction': support_fraction,
                    'expected_range': (0.25, 0.9),
                    'expected_span': 0.65,
                }
            candidate_masks.append(('combined', combined_mask, combined_context))
            logger.info(f"Combined method: {np.sum(combined_mask)} pixels")

        # Combine all methods using voting
        consensus = self._combine_probability_methods(candidate_masks, prob_map.shape[:2])

        # Focus subsequent refinements on the wing ROI and clean low-confidence borders
        roi_consensus = self._apply_adaptive_roi_refinement(consensus, prob_map, raw_img)
        final_mask = self._apply_uncertainty_postprocessing(roi_consensus, prob_map, raw_img)

        return final_mask
    
    def _intensity_based_mask(self, raw_img):
        """Create wing mask from raw image intensity."""
        # Normalize
        raw_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)

        # Wing tissue is typically in middle intensity range
        # Avoid pure black (mounting medium) and pure white (bright reflections)
        intensity_mask = (raw_norm > 0.1) & (raw_norm < 0.9)

        # Use Otsu thresholding as additional guide
        otsu_thresh = filters.threshold_otsu(raw_norm)
        otsu_mask = raw_norm > otsu_thresh * 0.8  # Slightly more inclusive

        # Combine both approaches
        combined = intensity_mask | otsu_mask

        context = {}
        if np.any(combined):
            values = raw_norm[combined]
            if values.size:
                iqr = float(np.percentile(values, 75) - np.percentile(values, 25))
                context = {
                    'mean_intensity': float(np.mean(values)),
                    'intensity_iqr': iqr,
                }

        return combined, context

    def _combine_probability_methods(self, candidate_masks, shape):
        """Combine multiple probability-based methods using intelligent voting."""
        if not candidate_masks:
            empty = np.zeros(shape, dtype=bool)
            return ProbabilityConsensus(empty, np.zeros(shape, dtype=np.float32), 0.45)

        # Create voting map
        vote_map = np.zeros(shape, dtype=np.float32)
        base_weights = {
            'intervein': 0.25,
            'combined': 0.30,
            'non_background': 0.20,
            'non_vein': 0.15,
            'intensity': 0.12,
        }

        total_weight = 0.0
        for candidate in candidate_masks:
            if len(candidate) == 3:
                name, mask, context = candidate
            else:
                name, mask = candidate
                context = None
            base_weight = base_weights.get(name, 0.1)
            adjusted_weight = self._evaluate_candidate_mask(name, mask, base_weight, context)

            if adjusted_weight <= 0:
                logger.debug("Skipping %s mask due to negligible reliability", name)
                continue

            vote_map += mask.astype(np.float32) * adjusted_weight
            total_weight += adjusted_weight

        if total_weight <= 0:
            logger.warning("All probability masks failed reliability checks; returning empty mask")
            empty_mask = np.zeros(shape, dtype=bool)
            return ProbabilityConsensus(empty_mask, vote_map, 0.45)

        # Normalize votes
        vote_map /= total_weight

        # Threshold voting - require at least 45% agreement from reliable sources
        base_threshold = 0.45
        consensus_mask = vote_map > base_threshold

        nominal_cov, min_cov, max_cov = self._expected_wing_coverage(shape)
        actual_cov = float(np.mean(consensus_mask))
        adjusted_threshold = base_threshold

        if actual_cov < min_cov * 0.9:
            adjusted_threshold = max(0.32, base_threshold - 0.1)
        elif actual_cov < nominal_cov * 0.7:
            adjusted_threshold = max(0.35, base_threshold - 0.06)
        elif actual_cov > max_cov * 1.1:
            adjusted_threshold = min(0.65, base_threshold + 0.08)
        elif actual_cov > nominal_cov * 1.6:
            adjusted_threshold = min(0.6, base_threshold + 0.05)

        if adjusted_threshold != base_threshold:
            logger.info(
                "Adjusting consensus threshold from %.2f to %.2f (coverage %.3f, target %.3f-%.3f)",
                base_threshold,
                adjusted_threshold,
                actual_cov,
                min_cov,
                max_cov,
            )
            consensus_mask = vote_map > adjusted_threshold

        # Clean up the consensus mask
        consensus_mask = self._clean_probability_mask(consensus_mask)

        logger.info(f"Probability consensus: {np.sum(consensus_mask)} pixels")
        return ProbabilityConsensus(consensus_mask, vote_map, adjusted_threshold)

    def _evaluate_candidate_mask(self, name, mask, base_weight, context=None):
        """Scale the base weight of a probability mask according to quality metrics."""

        if mask is None or not np.any(mask):
            return 0.0

        metrics = self._mask_quality_metrics(mask)
        reliability = 1.0

        coverage = metrics['coverage']
        expected_min_cov = self.config.min_wing_area / mask.size
        expected_cov = np.clip(expected_min_cov * 3.0, 0.08, 0.4)
        deviation = abs(coverage - expected_cov)
        coverage_score = math.exp(-deviation / max(expected_cov, 1e-3))
        reliability *= coverage_score

        largest_area = metrics['largest_component_area']
        min_area = max(self.config.min_wing_area, 1)
        if largest_area < min_area * 0.4:
            reliability *= 0.4
        elif largest_area < min_area * 0.7:
            reliability *= 0.7
        elif largest_area < min_area:
            reliability *= 0.9
        else:
            reliability *= 1.1

        edge_contact = metrics['edge_contact_ratio']
        if edge_contact > 0.3:
            reliability *= 0.5
        elif edge_contact > 0.15:
            reliability *= 0.8

        elongated = metrics['largest_component_eccentricity']
        if elongated < 0.5:
            reliability *= 0.85

        reliability *= self._contextual_confidence(name, context)

        reliability = np.clip(reliability, 0.05, 1.4)

        logger.debug(
            "Mask %s metrics: coverage=%.3f, largest=%d, edge=%.3f, ecc=%.3f, weight=%.3f",
            name,
            coverage,
            largest_area,
            edge_contact,
            elongated,
            base_weight * reliability,
        )

        return base_weight * reliability

    def _contextual_confidence(self, name, context):
        if not context:
            return 1.0

        confidence = 1.0

        mean_prob = context.get('mean_probability')
        if mean_prob is not None:
            expected_range = context.get('expected_range', (0.25, 0.85))
            lower, upper = expected_range
            if upper > lower:
                normalized = np.clip((mean_prob - lower) / (upper - lower), 0.0, 1.0)
            else:
                normalized = np.clip(mean_prob, 0.0, 1.0)
            confidence *= 0.4 + 0.8 * normalized

        std_prob = context.get('std_probability')
        if std_prob is not None:
            expected_span = context.get('expected_span', 0.5)
            span = max(expected_span, 1e-3)
            normalized_std = std_prob / span
            confidence *= 1.0 / (1.0 + 0.9 * normalized_std)

        mean_background = context.get('mean_background')
        if mean_background is not None:
            confidence *= 1.0 - 0.6 * np.clip(mean_background, 0.0, 1.0)

        support_fraction = context.get('support_fraction')
        if support_fraction is not None:
            confidence *= 0.6 + 0.6 * np.clip(support_fraction, 0.0, 1.5)

        mean_intensity = context.get('mean_intensity')
        if mean_intensity is not None:
            deviation = abs(mean_intensity - 0.5)
            confidence *= float(np.clip(math.exp(-deviation / 0.18), 0.5, 1.25))

        intensity_iqr = context.get('intensity_iqr')
        if intensity_iqr is not None:
            confidence *= float(np.clip(0.7 + intensity_iqr, 0.7, 1.3))

        if name == 'combined':
            confidence *= 1.05
        elif name == 'intensity':
            confidence *= 0.9

        return float(np.clip(confidence, 0.3, 1.4))

    def _expected_wing_coverage(self, shape):
        total_pixels = max(float(shape[0] * shape[1]), 1.0)
        min_cov = getattr(self.config, 'min_wing_area', 0) / total_pixels
        min_cov = float(np.clip(min_cov, 0.02, 0.35))

        max_region_area = getattr(self.config, 'max_region_area', None)
        if max_region_area is None:
            max_region_area = getattr(self.config, 'min_region_area', self.config.min_wing_area) * 4
        max_cov = float(np.clip(max_region_area / total_pixels, min_cov + 0.05, 0.75))

        nominal = float(np.clip((min_cov * 1.2 + max_cov * 0.8) / 2.0, min_cov, max_cov))

        return nominal, min_cov, max_cov

    def _mask_quality_metrics(self, mask):
        """Calculate quality metrics for a candidate probability mask."""

        coverage = float(np.sum(mask)) / float(mask.size)

        largest_area = 0
        eccentricity = 1.0

        labeled = label(mask)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest_region = max(regions, key=lambda r: r.area)
            largest_area = int(largest_region.area)
            eccentricity = getattr(largest_region, 'eccentricity', 1.0) or 1.0

        edge_contact_ratio = 0.0
        mask_pixels = np.sum(mask)
        if mask_pixels > 0:
            border = min(self.config.border_buffer, mask.shape[0] // 4, mask.shape[1] // 4)
            if border > 0:
                interior = np.zeros_like(mask, dtype=bool)
                interior[border:-border, border:-border] = True
                interior_pixels = np.sum(mask & interior)
                edge_contact_ratio = 1.0 - (interior_pixels / mask_pixels)

        return {
            'coverage': coverage,
            'largest_component_area': largest_area,
            'largest_component_eccentricity': float(eccentricity),
            'edge_contact_ratio': float(edge_contact_ratio),
        }
    
    def _clean_probability_mask(self, mask):
        """Clean up probability-based mask with morphological operations."""
        # Remove noise
        cleaned = morphology.remove_small_objects(mask, min_size=5000)

        # Fill holes
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=10000)

        # Gentle morphological operations
        cleaned = morphology.binary_opening(cleaned, morphology.disk(3))
        cleaned = morphology.binary_closing(cleaned, morphology.disk(8))

        # Keep only the largest connected component
        labeled = label(cleaned)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            cleaned = labeled == largest.label

        return cleaned

    def _apply_adaptive_roi_refinement(self, consensus: ProbabilityConsensus, prob_map, raw_img):
        """Re-run probability cleanup inside an adaptive ROI around the wing."""

        mask = consensus.mask.copy()
        vote_map = consensus.vote_map
        threshold = consensus.threshold

        if mask is None or not np.any(mask):
            return consensus

        labeled = label(mask)
        if labeled.max() == 0:
            return consensus

        regions = regionprops(labeled)
        largest = max(regions, key=lambda r: r.area)
        min_row, min_col, max_row, max_col = largest.bbox

        height = max_row - min_row
        width = max_col - min_col
        pad_y = max(int(height * 0.12), 15)
        pad_x = max(int(width * 0.12), 15)

        y0 = max(0, min_row - pad_y)
        x0 = max(0, min_col - pad_x)
        y1 = min(mask.shape[0], max_row + pad_y)
        x1 = min(mask.shape[1], max_col + pad_x)

        roi_slice = (slice(y0, y1), slice(x0, x1))
        roi_mask = mask[roi_slice]
        roi_votes = vote_map[roi_slice]

        confident_votes = roi_votes[roi_mask]
        if confident_votes.size == 0:
            return consensus

        percentile_threshold = float(np.clip(np.percentile(confident_votes, 30), 0.28, 0.6))
        mean_threshold = float(np.mean(confident_votes))
        adaptive_threshold = float(np.clip((percentile_threshold * 0.6 + mean_threshold * 0.4), 0.3, 0.62))

        refined_roi = roi_votes > adaptive_threshold

        if prob_map is not None and prob_map.ndim >= 3:
            roi_probs = prob_map[y0:y1, x0:x1]
            if roi_probs.shape[-1] >= 2:
                background = roi_probs[..., 1]
                refined_roi &= background < 0.65
            if roi_probs.shape[-1] >= 3:
                vein_prob = roi_probs[..., 2]
                refined_roi &= vein_prob < 0.88
            if roi_probs.shape[-1] >= 4:
                intervein_prob = roi_probs[..., 3]
                refined_roi |= (intervein_prob > 0.35) & roi_mask

        if raw_img is not None:
            roi_img = raw_img[y0:y1, x0:x1].astype(np.float32)
            roi_img -= roi_img.min()
            roi_img /= (roi_img.max() + 1e-6)
            gradient = filters.sobel(roi_img)
            gradient /= (gradient.max() + 1e-6)
            edge_barrier = gradient > np.percentile(gradient, 85)
            uncertain_band = (roi_votes > adaptive_threshold - 0.08) & (roi_votes < adaptive_threshold + 0.08)
            refined_roi[edge_barrier & uncertain_band] = False

            if np.any(roi_mask):
                bright_support = roi_img > np.percentile(roi_img[roi_mask], 60)
                refined_roi |= bright_support & roi_votes > adaptive_threshold

        refined_roi = morphology.binary_closing(refined_roi, morphology.disk(4))
        refined_roi = morphology.binary_opening(refined_roi, morphology.disk(2))
        refined_roi = morphology.remove_small_holes(refined_roi, area_threshold=8000)
        refined_roi = morphology.remove_small_objects(refined_roi, min_size=8000)

        refined_mask = mask.copy()
        refined_mask[roi_slice] = refined_roi
        refined_mask = self._clean_probability_mask(refined_mask)

        updated_threshold = float(np.clip((threshold + adaptive_threshold) / 2.0, 0.32, 0.6))

        return ProbabilityConsensus(refined_mask, vote_map, updated_threshold)

    def _apply_uncertainty_postprocessing(self, consensus: ProbabilityConsensus, prob_map, raw_img):
        """Apply morphology guided by uncertainty to smooth the final mask boundary."""

        mask = consensus.mask.copy()
        vote_map = consensus.vote_map
        threshold = consensus.threshold

        if vote_map is None or vote_map.size == 0:
            return mask

        confidence_band = 0.08
        lower = threshold - confidence_band
        upper = threshold + confidence_band
        uncertain = (vote_map > lower) & (vote_map < upper)

        if not np.any(uncertain):
            return mask

        support_map = np.zeros_like(mask, dtype=bool)
        removal_map = np.zeros_like(mask, dtype=bool)

        if prob_map is not None and prob_map.ndim >= 3:
            if prob_map.shape[-1] >= 1:
                trichome_prob = prob_map[..., 0]
                support_map |= trichome_prob > 0.32
            if prob_map.shape[-1] >= 2:
                background_prob = prob_map[..., 1]
                removal_map |= background_prob > 0.6
            if prob_map.shape[-1] >= 4:
                intervein_prob = prob_map[..., 3]
                support_map |= intervein_prob > 0.33

        contrast_support = None
        if raw_img is not None:
            raw_norm = raw_img.astype(np.float32)
            raw_norm -= raw_norm.min()
            raw_norm /= (raw_norm.max() + 1e-6)
            smooth = gaussian_filter(raw_norm, sigma=3)
            local_contrast = np.abs(raw_norm - smooth)
            contrast_level = np.percentile(local_contrast[mask], 60) if np.any(mask) else 0.08
            contrast_support = local_contrast > contrast_level
            support_map |= contrast_support
            removal_map |= local_contrast < np.percentile(local_contrast, 35)

        uncertain_interior = uncertain & mask
        uncertain_exterior = uncertain & (~mask)

        add_candidates = uncertain_exterior & support_map
        add_candidates = morphology.binary_closing(add_candidates, morphology.disk(2))
        mask[add_candidates] = True

        remove_candidates = uncertain_interior & removal_map
        remove_candidates = morphology.binary_opening(remove_candidates, morphology.disk(2))
        mask[remove_candidates] = False

        boundary = morphology.binary_dilation(mask, morphology.disk(2)) ^ morphology.binary_erosion(mask, morphology.disk(2))
        boundary_focus = boundary & uncertain
        if np.any(boundary_focus):
            smoothed = morphology.binary_closing(boundary_focus, morphology.disk(1))
            mask[boundary_focus] = smoothed[boundary_focus]

        mask = morphology.remove_small_holes(mask, area_threshold=7000)
        mask = morphology.remove_small_objects(mask, min_size=7000)

        return mask
    
    def _sparse_wing_detection(self, prob_map, raw_img, peaks, prob_wing_mask):
        """Detection optimized for sparse wings - rely heavily on probability map."""
        logger.info("Applying sparse wing detection strategy...")
        
        # For sparse wings, trust the probability map more
        primary_mask = prob_wing_mask
        
        # Use trichomes for minimal validation only
        if len(peaks) > 0:
            # Create trichome density map with large smoothing kernel
            trichome_mask = self._create_sparse_trichome_mask(peaks, prob_map.shape[:2])
            
            # Only keep probability regions that have SOME trichome support
            # Use very lenient threshold for sparse wings
            validated_mask = primary_mask & trichome_mask
            
            # If validation removes too much, fall back to probability only
            prob_area = np.sum(primary_mask)
            validated_area = np.sum(validated_mask)
            
            if validated_area < prob_area * 0.3:  # Lost more than 70%
                logger.warning("Trichome validation too restrictive for sparse wing, using probability only")
                final_mask = primary_mask
            else:
                final_mask = validated_mask
        else:
            logger.warning("No trichomes available, using probability map only")
            final_mask = primary_mask
        
        # Additional cleanup for sparse wings
        final_mask = self._sparse_wing_cleanup(final_mask, prob_map, raw_img)
        
        logger.info(f"Sparse wing detection result: {np.sum(final_mask)} pixels")
        return final_mask
    
    def _dense_wing_detection(self, prob_map, raw_img, peaks, prob_wing_mask):
        """Detection for dense wings - use full hybrid approach."""
        logger.info("Applying dense wing hybrid detection...")
        
        # Create trichome-based mask
        trichome_mask = self._create_dense_trichome_mask(peaks, prob_map.shape[:2])
        
        # Combine probability and trichome masks with weighted voting
        prob_weight = self.prob_weight
        trichome_weight = self.trichome_weight
        
        # Create weighted combination
        combined_score = (
            prob_wing_mask.astype(np.float32) * prob_weight +
            trichome_mask.astype(np.float32) * trichome_weight
        )
        
        # Threshold the combined score
        hybrid_mask = combined_score > 0.5
        
        # Ensure connectivity and reasonable shape
        hybrid_mask = self._refine_hybrid_mask(hybrid_mask, prob_wing_mask, trichome_mask)
        
        logger.info(f"Dense wing hybrid result: {np.sum(hybrid_mask)} pixels")
        return hybrid_mask
    
    def _create_sparse_trichome_mask(self, peaks, shape):
        """Create trichome mask optimized for sparse wings."""
        mask = np.zeros(shape, dtype=bool)

        peaks = np.asarray(peaks)
        if peaks.size == 0:
            return mask

        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)

        coords = np.asarray(peaks[:, :2], dtype=int)

        # Use large smoothing kernel for sparse wings
        sigma = 20  # Larger than dense wing sigma
        density_map = np.zeros(shape, dtype=np.float32)

        # Add gaussian blobs at each trichome
        for y, x in coords:
            radius = int(3 * sigma)
            y_min = max(0, y - radius)
            y_max = min(shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(shape[1], x + radius + 1)
            
            if y_max > y_min and x_max > x_min:
                yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
                density_map[y_min:y_max, x_min:x_max] += gaussian
        
        # Very low threshold for sparse wings
        threshold = np.percentile(density_map[density_map > 0], 10) if np.any(density_map > 0) else 0
        mask = density_map > threshold
        
        # Gentle morphological operations
        mask = morphology.binary_closing(mask, morphology.disk(10))
        mask = morphology.remove_small_holes(mask, area_threshold=20000)
        
        return mask
    
    def _create_dense_trichome_mask(self, peaks, shape):
        """Create trichome mask for dense wings."""
        mask = np.zeros(shape, dtype=bool)

        peaks = np.asarray(peaks)
        if peaks.size == 0:
            return mask

        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)

        coords = np.asarray(peaks[:, :2], dtype=int)

        # Standard smoothing for dense wings
        sigma = 12
        density_map = np.zeros(shape, dtype=np.float32)

        for y, x in coords:
            radius = int(3 * sigma)
            y_min = max(0, y - radius)
            y_max = min(shape[0], y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(shape[1], x + radius + 1)
            
            if y_max > y_min and x_max > x_min:
                yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((yy - y)**2 + (xx - x)**2) / (2 * sigma**2))
                density_map[y_min:y_max, x_min:x_max] += gaussian
        
        # Higher threshold for dense wings
        threshold = np.percentile(density_map[density_map > 0], 20) if np.any(density_map > 0) else 0
        mask = density_map > threshold
        
        # Standard morphological operations
        mask = morphology.binary_closing(mask, morphology.disk(8))
        mask = morphology.remove_small_holes(mask, area_threshold=10000)
        
        return mask
    
    def _sparse_wing_cleanup(self, mask, prob_map, raw_img):
        """Additional cleanup specific to sparse wings."""
        # For sparse wings, be more conservative about morphological operations
        
        # Light cleanup only
        cleaned = morphology.remove_small_objects(mask, min_size=20000)
        cleaned = morphology.remove_small_holes(cleaned, area_threshold=15000)
        
        # Use probability map to guide boundary refinement
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            # Expand into high-probability intervein areas
            high_prob_areas = intervein_prob > 0.6
            expanded = morphology.binary_dilation(cleaned, morphology.disk(5))
            cleaned = cleaned | (expanded & high_prob_areas)
        
        return cleaned
    
    def _refine_hybrid_mask(self, hybrid_mask, prob_mask, trichome_mask):
        """Refine hybrid mask using information from both components."""
        
        # Start with hybrid result
        refined = hybrid_mask.copy()
        
        # If hybrid result is too small, trust probability map more
        prob_area = np.sum(prob_mask)
        hybrid_area = np.sum(hybrid_mask)
        
        if hybrid_area < prob_area * 0.5:  # Hybrid lost more than 50%
            logger.info("Hybrid result too restrictive, expanding with probability map")
            # Add back high-confidence probability regions
            confident_prob = morphology.binary_erosion(prob_mask, morphology.disk(3))
            refined = refined | confident_prob
        
        # Ensure reasonable connectivity
        refined = morphology.remove_small_objects(refined, min_size=30000)
        refined = morphology.remove_small_holes(refined, area_threshold=15000)
        
        # Keep largest connected component
        labeled = label(refined)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            refined = labeled == largest.label
        
        return refined
    
    def create_detection_comparison_visualization(self, prob_map, raw_img, peaks, 
                                                prob_mask, hybrid_mask, output_path):
        """Create visualization comparing different detection approaches."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        bg_img = raw_img if raw_img is not None else prob_map[..., 0]
        
        # Top row: inputs
        axes[0, 0].imshow(bg_img, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(prob_map[..., 3] if prob_map.shape[-1] >= 4 else prob_map[..., 0], cmap='viridis')
        axes[0, 1].set_title('Probability Map')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(bg_img, cmap='gray')
        if len(peaks) > 0:
            axes[0, 2].scatter(peaks[:, 1], peaks[:, 0], c='red', s=1, alpha=0.6)
        axes[0, 2].set_title(f'Trichomes (n={len(peaks)})')
        axes[0, 2].axis('off')
        
        # Bottom row: results
        axes[1, 0].imshow(prob_mask, cmap='Blues')
        axes[1, 0].set_title(f'Probability-based Mask\n({np.sum(prob_mask)} pixels)')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(hybrid_mask, cmap='Greens')
        axes[1, 1].set_title(f'Hybrid Mask\n({np.sum(hybrid_mask)} pixels)')
        axes[1, 1].axis('off')
        
        # Comparison overlay
        axes[1, 2].imshow(bg_img, cmap='gray')
        axes[1, 2].imshow(prob_mask, cmap='Blues', alpha=0.3, label='Probability')
        axes[1, 2].imshow(hybrid_mask, cmap='Greens', alpha=0.5, label='Hybrid')
        
        # Show overlap and differences
        overlap = prob_mask & hybrid_mask
        prob_only = prob_mask & (~hybrid_mask)
        hybrid_only = hybrid_mask & (~prob_mask)
        
        if np.any(prob_only):
            axes[1, 2].imshow(prob_only, cmap='Blues', alpha=0.7)
        if np.any(hybrid_only):
            axes[1, 2].imshow(hybrid_only, cmap='Reds', alpha=0.7)
        
        overlap_pct = np.sum(overlap) / max(np.sum(prob_mask | hybrid_mask), 1) * 100
        axes[1, 2].set_title(f'Comparison Overlay\n{overlap_pct:.1f}% overlap')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved detection comparison to {output_path}")



class StringRemovalTrichomeFilter:
    """Remove long strings of trichomes (bubble artifacts) using morphological-like operations."""
    
    def __init__(self, config):
        self.config = config
        # Parameters for string detection and removal
        self.connection_distance = 25  # Max distance to consider trichomes "connected"

        conservative = getattr(self.config, "conservative_string_removal", False)
        if conservative:
            self.min_string_length = 12
            self.max_string_width = 30
            self.linearity_threshold = 0.8
        else:
            self.min_string_length = 8
            self.max_string_width = 50
            self.linearity_threshold = 0.7
        
    def remove_trichome_strings(self, peaks, image_shape):
        """Remove long, thin strings of trichomes that represent bubble artifacts."""

        peaks = np.asarray(peaks)
        if peaks.size == 0:
            logger.info("Too few trichomes for string filtering")
            return peaks

        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)

        if len(peaks) < 20:
            logger.info("Too few trichomes for string filtering")
            return peaks

        coords = np.asarray(peaks[:, :2], dtype=float)

        logger.info("Filtering strings from %d trichomes", len(peaks))

        # Step 1: Build connectivity graph between nearby trichomes
        adjacency_graph = self._build_trichome_graph(coords)

        # Step 2: Find connected components (chains/strings)
        components = self._find_connected_components(adjacency_graph)

        # Step 3: Identify which components are "strings" vs "blobs"
        string_components = self._identify_string_components(coords, components)

        # Step 4: Remove trichomes that belong to string components
        filtered_peaks = self._remove_string_trichomes(peaks, string_components)

        removed_count = len(peaks) - len(filtered_peaks)
        logger.info(
            "Removed %d trichomes from %d string artifacts", removed_count, len(string_components)
        )
        logger.info("Remaining: %d trichomes", len(filtered_peaks))
        
        return filtered_peaks
    
    def _build_trichome_graph(self, coords):
        """Build graph of connected trichomes based on distance."""
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.size == 0:
            return np.zeros((0, 0), dtype=bool)

        coords = coords[:, :2]
        n_peaks = len(coords)

        # Calculate pairwise distances using spatial coordinates only
        distances = squareform(pdist(coords))
        
        # Create adjacency matrix
        adjacency = distances <= self.connection_distance
        
        # Remove self-connections
        np.fill_diagonal(adjacency, False)
        
        return adjacency
    
    def _find_connected_components(self, adjacency_graph):
        """Find connected components in the trichome graph."""
        n_nodes = adjacency_graph.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        components = []
        
        for start_node in range(n_nodes):
            if visited[start_node]:
                continue
                
            # BFS to find all connected nodes
            component = []
            queue = [start_node]
            
            while queue:
                node = queue.pop(0)
                if visited[node]:
                    continue
                    
                visited[node] = True
                component.append(node)
                
                # Add unvisited neighbors to queue
                neighbors = np.where(adjacency_graph[node])[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        queue.append(neighbor)
            
            if len(component) > 1:  # Only keep components with multiple trichomes
                components.append(component)
        
        return components
    
    def _identify_string_components(self, coords, components):
        """Identify which components are long strings vs compact blobs."""
        string_components = []

        for i, component in enumerate(components):
            component_coords = coords[component]

            metrics = self._component_metrics(component_coords)
            if metrics['count'] < self.min_string_length:
                logger.debug(
                    "Component %d skipped (too few trichomes: %d)",
                    i,
                    metrics['count'],
                )
                continue  # Too short to be a problematic string

            is_string = self._is_linear_string(component_coords, metrics)

            if is_string:
                string_components.append(component)
                logger.debug(
                    "Component %d classified as string (%d trichomes, score=%.2f)",
                    i,
                    metrics['count'],
                    metrics['string_score'],
                )
            else:
                logger.debug(
                    "Component %d kept as blob (%d trichomes, score=%.2f)",
                    i,
                    metrics['count'],
                    metrics['string_score'],
                )

        return string_components

    def _is_linear_string(self, component_peaks, metrics):
        """Check if a component is a linear string (bubble artifact)."""

        if metrics['count'] < 3:
            return False

        aspect_ratio = metrics['aspect_ratio']
        linearity = metrics['linearity']
        mst_ratio = metrics['mst_ratio']
        direction_score = metrics['direction_score']
        max_width = metrics['max_width']
        span = metrics['span']

        elongated_string = aspect_ratio > 6.0 and linearity > self.linearity_threshold
        graph_string = mst_ratio > 5.0 and direction_score > 2.0
        thin_string = max_width < self.max_string_width and span > 80 and linearity > (self.linearity_threshold * 0.9)

        metrics['string_score'] = max(
            aspect_ratio / 6.0,
            linearity / max(self.linearity_threshold, 1e-3),
            mst_ratio / 5.0,
            direction_score / 2.0,
        )

        return elongated_string or graph_string or thin_string

    def _component_metrics(self, component_peaks):
        """Compute geometric and structural metrics for a trichome component."""

        metrics = {
            'count': len(component_peaks),
            'aspect_ratio': 1.0,
            'linearity': 1.0,
            'mst_ratio': 0.0,
            'direction_score': 0.0,
            'max_width': 0.0,
            'span': 0.0,
            'string_score': 0.0,
        }

        if len(component_peaks) < 2:
            return metrics

        component_peaks = np.asarray(component_peaks)

        min_coords = np.min(component_peaks, axis=0)
        max_coords = np.max(component_peaks, axis=0)
        bbox_dims = np.maximum(max_coords - min_coords, 1e-6)
        metrics['span'] = float(np.linalg.norm(max_coords - min_coords))

        min_dim = float(np.min(bbox_dims))
        metrics['aspect_ratio'] = float(np.max(bbox_dims) / max(min_dim, 1e-6))

        try:
            centered = component_peaks - np.mean(component_peaks, axis=0)
            cov_matrix = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            order = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.clip(eigenvalues[order], 1e-6, None)
            eigenvectors = eigenvectors[:, order]
            metrics['linearity'] = float(eigenvalues[0] / eigenvalues[1])
            principal_axis = eigenvectors[:, 0]
        except (np.linalg.LinAlgError, ValueError) as exc:
            logger.debug("PCA failed while evaluating string component: %s", exc)
            principal_axis = np.array([1.0, 0.0])
            metrics['linearity'] = 1.0

        metrics['max_width'] = self._estimate_component_width(component_peaks, principal_axis)
        metrics['direction_score'] = self._direction_consistency_score(component_peaks, principal_axis)
        metrics['mst_ratio'] = self._mst_to_area_ratio(component_peaks)

        return metrics

    def _estimate_component_width(self, component_peaks, axis):
        if len(component_peaks) < 2:
            return 0.0

        axis = axis / (np.linalg.norm(axis) + 1e-6)
        start_point = component_peaks[np.argmin(component_peaks @ axis)]
        line_vector = axis

        perp_distances = []
        for point in component_peaks:
            to_point = point - start_point
            projection_length = np.dot(to_point, line_vector)
            projection = start_point + projection_length * line_vector
            perp_distances.append(np.linalg.norm(point - projection))

        return float(np.max(perp_distances)) if perp_distances else 0.0

    def _direction_consistency_score(self, component_peaks, axis):
        if len(component_peaks) < 3:
            return 0.0

        axis = axis / (np.linalg.norm(axis) + 1e-6)
        projections = component_peaks @ axis
        order = np.argsort(projections)
        ordered = component_peaks[order]

        step_vectors = np.diff(ordered, axis=0)
        if len(step_vectors) == 0:
            return 0.0

        angles = np.arctan2(step_vectors[:, 0], step_vectors[:, 1])
        unwrapped = np.unwrap(angles)
        angle_std = np.std(unwrapped)

        return float(1.0 / (angle_std + 0.05))

    def _mst_to_area_ratio(self, component_peaks):
        if len(component_peaks) < 3:
            return 0.0

        mst_length = self._compute_mst_length(component_peaks)

        try:
            hull = ConvexHull(component_peaks)
            area = max(hull.volume, 1e-6)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("Convex hull failed while evaluating string component: %s", exc)
            area = 1e-6

        return float(mst_length / math.sqrt(area))

    def _compute_mst_length(self, points):
        if len(points) < 2:
            return 0.0

        remaining = set(range(len(points)))
        current = remaining.pop()
        visited = {current}
        total_length = 0.0

        while remaining:
            best_distance = float('inf')
            best_point = None

            remaining_indices = list(remaining)

            for v in visited:
                candidate_points = points[remaining_indices]
                distances = np.linalg.norm(candidate_points - points[v], axis=1)
                min_idx = int(np.argmin(distances))
                distance = float(distances[min_idx])
                if distance < best_distance:
                    best_distance = distance
                    best_point = remaining_indices[min_idx]

            if best_point is None:
                break

            total_length += best_distance
            visited.add(best_point)
            remaining.remove(best_point)

        return float(total_length)
    
    def _remove_string_trichomes(self, peaks, string_components):
        """Remove trichomes that belong to string components."""
        
        # Flatten list of string indices
        string_indices = set()
        for component in string_components:
            string_indices.update(component)
        
        # Keep trichomes that are NOT in string components
        keep_mask = np.array([i not in string_indices for i in range(len(peaks))])
        filtered_peaks = peaks[keep_mask]
        
        return filtered_peaks
    
    def create_wing_mask_simple(self, filtered_peaks, image_shape):
        """Create wing mask from filtered trichomes using simple approach."""
        
        filtered_peaks = np.asarray(filtered_peaks)
        if filtered_peaks.size == 0:
            logger.info("Too few filtered trichomes for wing mask generation")
            return None

        if filtered_peaks.ndim == 1:
            filtered_peaks = filtered_peaks.reshape(1, -1)

        if len(filtered_peaks) < 10:
            logger.info("Too few filtered trichomes for wing mask generation")
            return None

        logger.info(
            "Creating wing mask from %d filtered trichomes", len(filtered_peaks)
        )

        # Method: Dense region growing
        density_map = np.zeros(image_shape, dtype=np.float32)

        coords = np.asarray(filtered_peaks[:, :2], dtype=int)

        # Add gaussian blob at each trichome
        sigma = 12  # Smoothing radius
        for y, x in coords:

            # Add gaussian contribution
            radius = int(3 * sigma)
            y_min, y_max = max(0, y-radius), min(image_shape[0], y+radius+1)
            x_min, x_max = max(0, x-radius), min(image_shape[1], x+radius+1)
            
            if y_max > y_min and x_max > x_min:
                yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
                gaussian = np.exp(-((yy-y)**2 + (xx-x)**2) / (2*sigma**2))
                density_map[y_min:y_max, x_min:x_max] += gaussian
        
        # Smooth the density map
        density_map = ndimage.gaussian_filter(density_map, sigma=3)

        if not np.any(density_map > 0):
            logger.warning("Density map contained no positive values; skipping wing mask")
            return None

        # Threshold to get wing regions
        threshold = np.percentile(density_map[density_map > 0], 20)  # Keep top 80%
        wing_mask = density_map > threshold
        
        # Clean up
        wing_mask = morphology.binary_closing(wing_mask, morphology.disk(8))
        wing_mask = morphology.remove_small_holes(wing_mask, area_threshold=10000)
        wing_mask = morphology.remove_small_objects(wing_mask, min_size=20000)
        
        # Keep largest component
        labeled = measure.label(wing_mask)
        if labeled.max() > 0:
            regions = measure.regionprops(labeled)
            largest = max(regions, key=lambda r: r.area)
            wing_mask = labeled == largest.label
        
        logger.info("Wing mask generated with %d pixels", int(np.sum(wing_mask)))
        return wing_mask
    
    def visualize_string_removal(self, original_peaks, filtered_peaks, removed_peaks, 
                                wing_mask, prob_map, raw_img, output_path):
        """Visualize the string removal process."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        bg_img = raw_img if raw_img is not None else prob_map[..., 0]
        
        # Top row: filtering process
        axes[0, 0].imshow(bg_img, cmap='gray')
        if len(original_peaks) > 0:
            axes[0, 0].scatter(original_peaks[:, 1], original_peaks[:, 0], 
                             c='red', s=1, alpha=0.6)
        axes[0, 0].set_title(f'All Detected Trichomes (n={len(original_peaks)})')
        axes[0, 0].axis('off')
        
        # Show removed strings
        axes[0, 1].imshow(bg_img, cmap='gray')
        if len(removed_peaks) > 0:
            axes[0, 1].scatter(removed_peaks[:, 1], removed_peaks[:, 0], 
                             c='red', s=3, alpha=0.8)
        axes[0, 1].set_title(f'Removed String Artifacts (n={len(removed_peaks)})')
        axes[0, 1].axis('off')
        
        # Show kept trichomes
        axes[0, 2].imshow(bg_img, cmap='gray')
        if len(filtered_peaks) > 0:
            axes[0, 2].scatter(filtered_peaks[:, 1], filtered_peaks[:, 0], 
                             c='blue', s=2, alpha=0.8)
        axes[0, 2].set_title(f'Kept Trichomes (n={len(filtered_peaks)})')
        axes[0, 2].axis('off')
        
        # Bottom row: results
        if wing_mask is not None:
            axes[1, 0].imshow(wing_mask, cmap='viridis')
            axes[1, 0].set_title(f'String-Filtered Wing Mask')
        else:
            axes[1, 0].text(0.5, 0.5, 'Wing mask failed', ha='center', va='center')
            axes[1, 0].set_title('Wing Mask (Failed)')
        axes[1, 0].axis('off')
        
        # Show comparison: before vs after
        axes[1, 1].imshow(bg_img, cmap='gray')
        if len(original_peaks) > 0:
            axes[1, 1].scatter(original_peaks[:, 1], original_peaks[:, 0], 
                             c='red', s=1, alpha=0.3, label='Original')
        if len(filtered_peaks) > 0:
            axes[1, 1].scatter(filtered_peaks[:, 1], filtered_peaks[:, 0], 
                             c='blue', s=2, alpha=0.8, label='Filtered')
        axes[1, 1].legend()
        axes[1, 1].set_title('Before vs After Filtering')
        axes[1, 1].axis('off')
        
        # Final result
        axes[1, 2].imshow(bg_img, cmap='gray')
        if wing_mask is not None:
            axes[1, 2].imshow(wing_mask, cmap='Blues', alpha=0.4)
        if len(filtered_peaks) > 0:
            axes[1, 2].scatter(filtered_peaks[:, 1], filtered_peaks[:, 0], 
                             c='yellow', s=1, alpha=0.8)
        axes[1, 2].set_title('Final Wing Boundary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        logger.info("Saved string removal visualization to %s", output_path)


class EnhancedStringRemovalFilter(StringRemovalTrichomeFilter):
    """Enhanced string removal that works with hybrid wing detection."""
    
    def __init__(self, config):
        super().__init__(config)
        # Make string removal more conservative for sparse wings
        self.min_string_length = 12  # Increased from 8
        self.linearity_threshold = 0.8  # Increased from 0.7
    
    def adaptive_string_removal(self, peaks, image_shape, trichome_density):
        """Adaptive string removal based on trichome density."""
        
        if trichome_density < 0.1:  # Sparse wing
            logger.info("Using conservative string removal for sparse wing")
            # More conservative parameters for sparse wings
            old_min_length = self.min_string_length
            old_linearity = self.linearity_threshold
            
            self.min_string_length = 15  # Even more conservative
            self.linearity_threshold = 0.85
            
            result = self.remove_trichome_strings(peaks, image_shape)
            
            # Restore original parameters
            self.min_string_length = old_min_length
            self.linearity_threshold = old_linearity
            
            return result
        else:
            # Standard string removal for dense wings
            return self.remove_trichome_strings(peaks, image_shape)

class TrichomeAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HomerWings V2 (Now with an icon!)")
        self.root.geometry("1000x800")
        
        # Configuration
        self.config = TrichomeDetectionConfig()
        
        # Processing state
        self.is_processing = False
        self.progress_queue = queue.Queue()
        
        # Animation variables
        self.animation_running = False
        self.gif_frames = None
        self.gif_durations = None
        self.current_frame = 0
        self.current_fact = 0
        
        self.setup_gui()
        
        # Start progress monitoring
        self.root.after(100, self.check_progress)
    
    def setup_gui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: File Selection and Processing
        self.processing_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processing_frame, text="Processing")
        
        # Tab 2: Configuration
        self.config_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.config_frame, text="Configuration")
        
        # Tab 3: Advanced Settings
        self.advanced_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.advanced_frame, text="Advanced")
        
        # Tab 4: Results/Log
        self.results_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.results_frame, text="Results & Log")
        
        self.setup_processing_tab()
        self.setup_config_tab()
        self.setup_advanced_tab()
        self.setup_results_tab()

    
    def update_loading_status_with_hybrid_info(self, status_text):
        """Update loading status with hybrid detection information."""
        if hasattr(self, 'loading_status_label'):
            try:
                if self.loading_status_label.winfo_exists():
                    # Add hybrid detection context to status messages
                    if "sparse" in status_text.lower():
                        enhanced_status = f"🦋 {status_text} (Using probability-dominant approach)"
                    elif "dense" in status_text.lower():
                        enhanced_status = f"🦋 {status_text} (Using hybrid validation)"
                    elif "wing" in status_text.lower():
                        enhanced_status = f"🦋 {status_text} (Adaptive detection active)"
                    else:
                        enhanced_status = status_text
                    
                    self.loading_status_label.config(text=enhanced_status)
            except tk.TclError:
                pass
    
    def _enhance_log_message(self, message):
        """Enhance log messages with hybrid detection context."""
        # Add context indicators for different types of processing
        if "sparse wing" in message.lower():
            return f"🌟 {message}"  # Special indicator for sparse wing processing
        elif "hybrid" in message.lower():
            return f"🔄 {message}"  # Hybrid processing indicator
        elif "probability" in message.lower() and "dominant" in message.lower():
            return f"📊 {message}"  # Probability-dominant indicator
        elif "string removal" in message.lower():
            return f"🧹 {message}"  # String removal indicator
        else:
            return message
    
    def check_progress_with_hybrid_info(self):
        """Enhanced progress checker with hybrid detection information."""
        try:
            messages_processed = 0
            max_messages = 15
            
            while messages_processed < max_messages:
                try:
                    msg_type, message = self.progress_queue.get_nowait()
                    messages_processed += 1
                    
                    # Enhanced message processing for hybrid detection
                    if msg_type == "progress":
                        self.root.after_idle(lambda msg=message: self._safe_set_progress_with_context(msg))
                    elif msg_type == "current_folder":
                        self.root.after_idle(lambda msg=message: self._safe_set_current_folder_with_context(msg))
                    elif msg_type == "progress_percent":
                        self.root.after_idle(lambda pct=message: self._safe_set_progress_percent(pct))
                    elif msg_type == "log":
                        # Enhanced logging with hybrid detection context
                        enhanced_msg = self._enhance_log_message(message)
                        self.root.after_idle(lambda msg=enhanced_msg: self._safe_log_message(msg))
                    elif msg_type == "error":
                        self.root.after_idle(lambda msg=message: self._safe_log_message(f"ERROR: {msg}"))
                    elif msg_type == "summary":
                        self.root.after_idle(lambda msg=message: self._update_summary(msg))
                    elif msg_type == "finished":
                        self.root.after_idle(self._processing_finished)
                        return
                        
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing progress message: {e}")
                    break
            
        except Exception as e:
            logger.error(f"Progress check error: {e}")
        
        # Schedule next check if still processing
        if self.is_processing:
            self.root.after(200, self.check_progress_with_hybrid_info)
    
    def _safe_set_progress_with_context(self, message):
        """Thread-safe progress update with hybrid detection context."""
        try:
            self.progress_var.set(message)
            self.update_loading_status_with_hybrid_info(message)
        except:
            pass
    
    def _safe_set_current_folder_with_context(self, message):
        """Thread-safe current folder update with hybrid detection context."""
        try:
            self.current_folder_var.set(message)
            self.update_loading_status_with_hybrid_info(message)
        except:
            pass
    def setup_processing_tab(self):
        # File selection section
        file_frame = ttk.LabelFrame(self.processing_frame, text="Folder Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Master folder selection
        ttk.Label(file_frame, text="Master Folder (contains subfolders):").pack(anchor=tk.W)
        self.master_folder_var = tk.StringVar()
        master_frame = ttk.Frame(file_frame)
        master_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(master_frame, textvariable=self.master_folder_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(master_frame, text="Browse", command=self.browse_master_folder).pack(side=tk.RIGHT, padx=(5,0))
        
        # Output folder selection
        ttk.Label(file_frame, text="Output Folder:").pack(anchor=tk.W, pady=(10,0))
        self.output_folder_var = tk.StringVar()
        output_frame = ttk.Frame(file_frame)
        output_frame.pack(fill=tk.X, pady=5)
        
        ttk.Entry(output_frame, textvariable=self.output_folder_var, width=80).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Browse", command=self.browse_output_folder).pack(side=tk.RIGHT, padx=(5,0))
        
        # Processing options
        options_frame = ttk.LabelFrame(self.processing_frame, text="Processing Options", padding=10)
        options_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Processing mode
        self.pentagone_mode_var = tk.StringVar(value="auto")
        ttk.Label(options_frame, text="Pentagone Detection Mode:").pack(anchor=tk.W)
        mode_frame = ttk.Frame(options_frame)
        mode_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(mode_frame, text="Auto-detect", variable=self.pentagone_mode_var, value="auto").pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Force Normal (5-region)", variable=self.pentagone_mode_var, value="normal").pack(side=tk.LEFT, padx=(20,0))
        ttk.Radiobutton(mode_frame, text="Force Pentagone (4-region)", variable=self.pentagone_mode_var, value="pentagone").pack(side=tk.LEFT, padx=(20,0))
        
        # Additional options
        self.skip_existing_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Skip folders with existing results", variable=self.skip_existing_var).pack(anchor=tk.W, pady=2)
        
        self.save_config_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(options_frame, text="Save configuration with results", variable=self.save_config_var).pack(anchor=tk.W, pady=2)
        
        # Folder preview
        preview_frame = ttk.LabelFrame(self.processing_frame, text="Detected Subfolders", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.folder_listbox = tk.Listbox(preview_frame, height=8)
        scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.folder_listbox.yview)
        self.folder_listbox.configure(yscrollcommand=scrollbar.set)
        
        self.folder_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Control buttons
        control_frame = ttk.Frame(self.processing_frame)
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.refresh_button = ttk.Button(control_frame, text="Refresh Folder List", command=self.refresh_folder_list)
        self.refresh_button.pack(side=tk.LEFT)
        
        self.start_button = ttk.Button(control_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.RIGHT, padx=(0,10))
        
        self.stop_button = ttk.Button(control_frame, text="Stop Processing", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT)
        
        # Progress section
        progress_frame = ttk.LabelFrame(self.processing_frame, text="Progress", padding=10)
        progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.StringVar(value="Ready to process")
        ttk.Label(progress_frame, textvariable=self.progress_var).pack(anchor=tk.W)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.current_folder_var = tk.StringVar(value="")
        ttk.Label(progress_frame, textvariable=self.current_folder_var, font=('TkDefaultFont', 8)).pack(anchor=tk.W)
    
    def setup_config_tab(self):
        # Create scrollable frame
        canvas = tk.Canvas(self.config_frame)
        scrollbar = ttk.Scrollbar(self.config_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Core Detection Parameters
        core_frame = ttk.LabelFrame(scrollable_frame, text="Core Detection Parameters", padding=10)
        core_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(core_frame, "min_distance", "Minimum distance between peaks:", self.config.min_distance, int)
        self.create_config_entry(core_frame, "high_thresh_abs", "High threshold (absolute):", self.config.high_thresh_abs, float)
        self.create_config_entry(core_frame, "low_thresh_abs", "Low threshold (absolute):", self.config.low_thresh_abs, float)
        
        # Multi-scale Detection
        scale_frame = ttk.LabelFrame(scrollable_frame, text="Multi-scale Detection", padding=10)
        scale_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(scale_frame, "scales", "Detection scales (comma-separated):", 
                                ",".join(map(str, self.config.scales)), str)
        self.create_config_entry(scale_frame, "scale_weight_decay", "Scale weight decay:", self.config.scale_weight_decay, float)
        
        # Pre-processing
        preproc_frame = ttk.LabelFrame(scrollable_frame, text="Pre-processing", padding=10)
        preproc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_checkbox(preproc_frame, "use_clahe", "Use CLAHE enhancement", self.config.use_clahe)
        self.create_config_entry(preproc_frame, "clahe_clip_limit", "CLAHE clip limit:", self.config.clahe_clip_limit, float)
        
        self.create_config_checkbox(preproc_frame, "use_white_tophat", "Use white top-hat", self.config.use_white_tophat)
        self.create_config_entry(preproc_frame, "tophat_radius", "Top-hat radius:", self.config.tophat_radius, int)
        
        self.create_config_checkbox(preproc_frame, "use_local_threshold", "Use local thresholding", self.config.use_local_threshold)
        self.create_config_entry(preproc_frame, "local_block_size", "Local block size:", self.config.local_block_size, int)
        
        # Peak Detection
        peak_frame = ttk.LabelFrame(scrollable_frame, text="Peak Detection", padding=10)
        peak_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(peak_frame, "min_peak_intensity", "Minimum peak intensity:", self.config.min_peak_intensity, float)
        self.create_config_entry(peak_frame, "max_peak_density", "Maximum peak density:", self.config.max_peak_density, float)
        self.create_config_entry(peak_frame, "edge_exclusion_buffer", "Edge exclusion buffer:", self.config.edge_exclusion_buffer, int)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    

    
    def setup_advanced_tab(self):
        """Enhanced advanced tab with hybrid detection options."""
        # Create scrollable frame
        canvas = tk.Canvas(self.advanced_frame)
        scrollbar = ttk.Scrollbar(self.advanced_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # === NEW: Hybrid Wing Detection Section ===
        hybrid_frame = ttk.LabelFrame(scrollable_frame, text="Hybrid Wing Detection", padding=10)
        hybrid_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable/disable hybrid detection
        self.create_config_checkbox(hybrid_frame, "use_hybrid_detection", 
                                    "Enable hybrid detection (recommended for sparse wings)", True)
        
        # Hybrid detection parameters
        self.create_config_entry(hybrid_frame, "sparse_threshold", "Sparse wing threshold (trichome density):", 
                                0.1, float)
        self.create_config_entry(hybrid_frame, "prob_weight", "Probability map weight (hybrid mode):", 
                                0.8, float)
        self.create_config_entry(hybrid_frame, "trichome_weight", "Trichome validation weight (hybrid mode):", 
                                0.2, float)
        
        # Wing detection fallback options
        fallback_frame = ttk.LabelFrame(scrollable_frame, text="Sparse Wing Handling", padding=10)
        fallback_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_checkbox(fallback_frame, "conservative_string_removal", 
                                    "Use conservative string removal for sparse wings", True)
        self.create_config_checkbox(fallback_frame, "probability_dominant_sparse", 
                                    "Use probability-dominant approach for very sparse wings", True)
        self.create_config_entry(fallback_frame, "min_trichomes_for_validation", 
                                "Minimum trichomes required for validation:", 50, int)
        
        # Detection method selection
        method_frame = ttk.LabelFrame(scrollable_frame, text="Wing Detection Method", padding=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.wing_detection_method_var = tk.StringVar(value="hybrid")
        ttk.Label(method_frame, text="Detection method:").pack(anchor=tk.W)
        method_selection_frame = ttk.Frame(method_frame)
        method_selection_frame.pack(fill=tk.X, pady=5)
        
        ttk.Radiobutton(method_selection_frame, text="Auto (Hybrid)", 
                       variable=self.wing_detection_method_var, value="hybrid").pack(side=tk.LEFT)
        ttk.Radiobutton(method_selection_frame, text="Probability Only", 
                       variable=self.wing_detection_method_var, value="probability").pack(side=tk.LEFT, padx=(20,0))
        ttk.Radiobutton(method_selection_frame, text="Trichome Only (Legacy)", 
                       variable=self.wing_detection_method_var, value="trichome").pack(side=tk.LEFT, padx=(20,0))
        
        # === Continue with existing sections ===
        # Clustering and Filtering
        cluster_frame = ttk.LabelFrame(scrollable_frame, text="Clustering and Filtering", padding=10)
        cluster_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(cluster_frame, "dbscan_eps", "DBSCAN epsilon:", self.config.dbscan_eps, float)
        self.create_config_entry(cluster_frame, "dbscan_min_samples", "DBSCAN min samples:", self.config.dbscan_min_samples, int)
        self.create_config_entry(cluster_frame, "valley_drop", "Valley drop threshold:", self.config.valley_drop, float)
        self.create_config_entry(cluster_frame, "neighbour_tolerance", "Neighbor tolerance:", self.config.neighbour_tolerance, float)
        self.create_config_entry(cluster_frame, "min_neighbours", "Minimum neighbors:", self.config.min_neighbours, int)
        merge_frame = ttk.LabelFrame(scrollable_frame, text="Region Merge Options", padding=10)
        merge_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_checkbox(merge_frame, "force_merge_regions_4_5", 
                                    "Force merge regions 4+5 (even when detected separately)", True)
        
        ttk.Label(merge_frame, text="Note: Merge mode overrides normal 5-region detection.", 
                 font=('TkDefaultFont', 8, 'italic')).pack(anchor=tk.W, pady=(5,0))
        # Intervein Segmentation
        intervein_frame = ttk.LabelFrame(scrollable_frame, text="Intervein Segmentation", padding=10)
        intervein_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(intervein_frame, "intervein_threshold", "Intervein threshold:", self.config.intervein_threshold, float)
        self.create_config_entry(intervein_frame, "min_region_area", "Min region area:", self.config.min_region_area, int)
        self.create_config_entry(intervein_frame, "max_region_area", "Max region area:", self.config.max_region_area, int)
        
        # Vein Detection
        vein_frame = ttk.LabelFrame(scrollable_frame, text="Vein Detection", padding=10)
        vein_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(vein_frame, "vein_width_estimate", "Vein width estimate:", self.config.vein_width_estimate, int)
        self.create_config_entry(vein_frame, "min_vein_length", "Min vein length:", self.config.min_vein_length, int)
        self.create_config_entry(vein_frame, "vein_detection_sensitivity", "Vein detection sensitivity:", self.config.vein_detection_sensitivity, float)
        
        # Wing Validation
        wing_frame = ttk.LabelFrame(scrollable_frame, text="Wing Validation", padding=10)
        wing_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(wing_frame, "min_wing_area", "Min wing area:", self.config.min_wing_area, int)
        self.create_config_entry(wing_frame, "border_buffer", "Border buffer:", self.config.border_buffer, int)
        
        # Configuration management
        config_mgmt_frame = ttk.LabelFrame(scrollable_frame, text="Configuration Management", padding=10)
        config_mgmt_frame.pack(fill=tk.X, padx=10, pady=5)
        
        config_buttons = ttk.Frame(config_mgmt_frame)
        config_buttons.pack(fill=tk.X)
        
        ttk.Button(config_buttons, text="Load Config", command=self.load_config).pack(side=tk.LEFT, padx=(0,5))
        ttk.Button(config_buttons, text="Save Config", command=self.save_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(config_buttons, text="Reset to Defaults", command=self.reset_config).pack(side=tk.LEFT, padx=5)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        

    
    
    # REPLACE: Replace your existing update_config_from_gui method with this enhanced version
    
    def update_config_from_gui(self):
        """Updated config update method including hybrid detection parameters."""
        try:
            errors = []
            
            # Process existing GUI variables
            for attr_name, gui_info in self.gui_vars.items():
                try:
                    var = gui_info['var']
                    expected_type = gui_info['type']
                    
                    if isinstance(var, tk.BooleanVar):
                        value = var.get()
                        setattr(self.config, attr_name, value)
                    else:
                        value_str = var.get().strip()
                        if not value_str:
                            errors.append(f"{attr_name}: Empty value")
                            continue
                        
                        # Special handling for scales
                        if attr_name == "scales":
                            try:
                                scales = [float(x.strip()) for x in value_str.split(',') if x.strip()]
                                if not scales:
                                    errors.append("scales: No valid scales provided")
                                    continue
                                setattr(self.config, attr_name, scales)
                            except ValueError as e:
                                errors.append(f"scales: Invalid values - {e}")
                                continue
                        else:
                            # Regular type conversion
                            try:
                                converted_value = expected_type(value_str)
                                setattr(self.config, attr_name, converted_value)
                            except ValueError as e:
                                errors.append(f"{attr_name}: Invalid {expected_type.__name__} - {e}")
                                continue
                                
                except Exception as e:
                    errors.append(f"{attr_name}: Error reading value - {e}")
            
            # === NEW: Handle hybrid detection settings ===
            # Update wing detection method from radio buttons
            if hasattr(self, 'wing_detection_method_var'):
                self.config.wing_detection_method = self.wing_detection_method_var.get()
            
            # Report errors
            if errors:
                error_msg = "Configuration errors:\n" + "\n".join(errors[:5])
                if len(errors) > 5:
                    error_msg += f"\n... and {len(errors) - 5} more"
                messagebox.showerror("Configuration Error", error_msg)
                return False
            
            # Validate
            try:
                self.config.validate()
                logger.info("Configuration updated from GUI successfully (with hybrid detection)")
                return True
            except Exception as e:
                messagebox.showerror("Validation Error", f"Configuration validation failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Critical error updating config: {e}")
            messagebox.showerror("Critical Error", f"Failed to update configuration: {e}")
            return False
    
    
    # REPLACE: Replace your existing process_folders method with this enhanced version
    
    def process_folders(self):
        """Updated process_folders method that uses hybrid detection."""
        try:
            master_folder = self.master_folder_var.get()
            output_folder = self.output_folder_var.get()
            
            # Get all subfolders with files
            subfolders_to_process = []
            for item in os.listdir(master_folder):
                item_path = os.path.join(master_folder, item)
                if os.path.isdir(item_path):
                    mapping = find_associated_files(item_path)
                    if mapping:
                        subfolders_to_process.append((item, item_path, len(mapping)))
            
            if not subfolders_to_process:
                self.progress_queue.put(("error", "No subfolders with valid files found"))
                return
            
            total_folders = len(subfolders_to_process)
            processed_folders = 0
            total_files_processed = 0
            total_files_failed = 0
            sparse_wings_detected = 0
            
            self.progress_queue.put(("progress", f"Starting hybrid detection processing of {total_folders} folders"))
            
            for folder_name, folder_path, file_count in subfolders_to_process:
                if not self.is_processing:
                    break
                
                # Update progress
                self.progress_queue.put(("current_folder", f"Processing: {folder_name} ({file_count} files)"))
                self.progress_queue.put(("progress_percent", int(processed_folders / total_folders * 100)))
                
                # Create output subfolder
                output_subfolder = os.path.join(output_folder, folder_name)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Check if already processed
                if self.skip_existing_var.get():
                    summary_file = os.path.join(output_subfolder, "analysis_summary.txt")
                    if os.path.exists(summary_file):
                        self.progress_queue.put(("log", f"Skipping {folder_name} - already processed"))
                        processed_folders += 1
                        continue
                
                # Save configuration
                if self.save_config_var.get():
                    config_path = os.path.join(output_subfolder, "processing_config.json")
                    self.save_config_to_file(config_path)
                
                # Process this folder with hybrid detection
                try:
                    pentagone_mode = self.pentagone_mode_var.get()
                    force_pentagone = pentagone_mode == "pentagone"
                    
                    # === USE HYBRID DETECTION ===
                    result = main_with_hybrid_wing_detection(
                        directory=folder_path,
                        cfg=self.config,
                        output_directory=output_subfolder,
                        force_pentagone_mode=force_pentagone,
                        auto_detect_pentagone=(pentagone_mode == "auto"),
                        progress_callback=self.progress_queue
                    )
                    
                    if result and 'sparse_wings' in result:
                        sparse_wings_detected += result['sparse_wings']
                    
                    total_files_processed += file_count
                    self.progress_queue.put(("log", f"Completed {folder_name}: {file_count} files processed with hybrid detection"))
                    
                except Exception as e:
                    total_files_failed += file_count
                    self.progress_queue.put(("error", f"Error processing {folder_name}: {e}"))
                
                processed_folders += 1
            
            # Final summary with hybrid detection statistics
            if self.is_processing:
                summary = f"""
    Hybrid Detection Processing Complete!
    
    Folders processed: {processed_folders}/{total_folders}
    Total files processed: {total_files_processed}
    Total files failed: {total_files_failed}
    Sparse wings detected: {sparse_wings_detected}
    
    Detection Method: {"Hybrid (Probability + Trichome)" if self.config.use_hybrid_detection else "Legacy"}
    Output directory: {output_folder}
    
    Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
    
    Hybrid Detection Benefits:
    - Improved sparse wing handling
    - Fallback mechanisms for challenging cases
    - Adaptive string removal
    - Better boundary detection
    """
                self.progress_queue.put(("summary", summary))
                self.progress_queue.put(("progress", "Hybrid detection processing completed successfully"))
            else:
                self.progress_queue.put(("progress", "Processing stopped by user"))
            
        except Exception as e:
            self.progress_queue.put(("error", f"Critical error: {e}"))
        
        finally:
            self.progress_queue.put(("finished", None))
    
    def setup_results_tab(self):
        # Results summary
        summary_frame = ttk.LabelFrame(self.results_frame, text="Processing Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=8, wrap=tk.WORD)
        summary_scroll = ttk.Scrollbar(summary_frame, orient=tk.VERTICAL, command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scroll.set)
        
        self.summary_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        summary_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Live log
        log_frame = ttk.LabelFrame(self.results_frame, text="Processing Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Clear log button
        ttk.Button(log_frame, text="Clear Log", command=self.clear_log).pack(anchor=tk.E, pady=5)
    
    def create_config_entry(self, parent, attr_name, label, default_value, data_type):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
        
        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(frame, textvariable=var, width=20)
        entry.pack(side=tk.LEFT, padx=(5,0))
        
        # Store reference for later retrieval
        setattr(self, f"{attr_name}_var", var)
        setattr(self, f"{attr_name}_type", data_type)
    
    def create_config_checkbox(self, parent, attr_name, label, default_value):
        var = tk.BooleanVar(value=default_value)
        checkbox = ttk.Checkbutton(parent, text=label, variable=var)
        checkbox.pack(anchor=tk.W, pady=2)
        
        # Store reference for later retrieval
        setattr(self, f"{attr_name}_var", var)
    
    def browse_master_folder(self):
        folder = filedialog.askdirectory(title="Select Master Folder")
        if folder:
            self.master_folder_var.set(folder)
            # Auto-set output folder
            if not self.output_folder_var.get():
                output_folder = os.path.join(os.path.dirname(folder), f"{os.path.basename(folder)}_Results")
                self.output_folder_var.set(output_folder)
            self.refresh_folder_list()
    
    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_folder_var.set(folder)
    
    def refresh_folder_list(self):
        self.folder_listbox.delete(0, tk.END)
        
        master_folder = self.master_folder_var.get()
        if not master_folder or not os.path.exists(master_folder):
            return
        
        try:
            subfolders = []
            for item in os.listdir(master_folder):
                item_path = os.path.join(master_folder, item)
                if os.path.isdir(item_path):
                    # Check if folder contains probability files
                    mapping = find_associated_files(item_path)
                    if mapping:
                        subfolders.append(f"{item} ({len(mapping)} file pairs)")
                    else:
                        subfolders.append(f"{item} (no files found)")
            
            for folder in sorted(subfolders):
                self.folder_listbox.insert(tk.END, folder)
                
        except Exception as e:
            self.log_message(f"Error scanning folders: {e}")
    
    # ==== DROSOPHILA GIF ANIMATION METHODS ====
    
    def load_embedded_gif_frames(self):
        """Load frames from embedded GIF data"""
        try:
            print("Loading embedded Drosophila GIF...")
            frames, durations = drosophila_gif.load_embedded_gif()
            
            if frames is None:
                print("Failed to load embedded GIF")
                return None, None
            
            print(f"Successfully loaded {len(frames)} frames from embedded GIF")
            return frames, durations
            
        except Exception as e:
            print(f"Error loading embedded GIF: {e}")
            return None, None

    # Update your create_drosophila_loading_overlay method:
    def create_drosophila_loading_overlay(self):
        """Create loading overlay with embedded Drosophila GIF"""
        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
            return
        
        # Create overlay window
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("HomerWings Processing")
        self.loading_window.geometry("600x450")
        self.loading_window.resizable(False, False)
        
        # Center and style
        self.loading_window.transient(self.root)
        self.loading_window.grab_set()
        self.loading_window.configure(bg='#1a1a1a')
        
        # Position relative to main window
        x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 300
        y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 225
        self.loading_window.geometry(f"500x450+{x}+{y}")
        
        # Main frame
        main_frame = tk.Frame(self.loading_window, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = tk.Label(
            main_frame, 
            text="HomerWings Analysis Suite", 
            font=('Arial', 18, 'bold'),
            fg='#00d4aa',
            bg='#1a1a1a'
        )
        title_label.pack(pady=(0, 15))
        
        # Load embedded GIF - NO MORE FILE PATH ISSUES!
        self.gif_frames, self.gif_durations = self.load_embedded_gif_frames()
        
        # Create label for GIF animation
        self.gif_label = tk.Label(
            main_frame,
            bg='#1a1a1a',
            borderwidth=0
        )
        self.gif_label.pack(pady=15)
        
        # Initialize animation variables
        self.current_frame = 0
        self.animation_running = True
        
        # If GIF loading failed, show a fallback
        if self.gif_frames is None:
            self.gif_label.config(
                text="🦋 Drosophila Analysis 🦋",
                font=('Arial', 24),
                fg='#00d4aa'
            )
            print("Using fallback animation - embedded GIF failed to load")
        else:
            # Start the GIF animation
            self.animate_gif()
            print(f"Started embedded GIF animation with {len(self.gif_frames)} frames")
        
        # ... rest of your overlay code stays the same ...
        
        # Status text
        self.loading_status_label = tk.Label(
            main_frame,
            text="Initializing trichome detection...",
            font=('Arial', 12),
            fg='#ffffff',
            bg='#1a1a1a',
            wraplength=400
        )
        self.loading_status_label.pack(pady=15)
        
        # Progress indicator with wing-themed messages
        self.loading_progress_label = tk.Label(
            main_frame,
            text="Preparing wing analysis...",
            font=('Arial', 10),
            fg='#888888',
            bg='#1a1a1a'
        )
        self.loading_progress_label.pack(pady=10)
        
        # Wing analysis facts
        wing_facts = [
            "Detecting trichome patterns on wing surface...",
            "Bananas are radioactive!", 
            "A single Bolt of lightning contains enough energy to toast 20,000 slices of bread!",
            "Applying computer vision to wing venation...",
            "Computing cellular tessellations...",
            "You cant burp in space!",
            "Analyzing wing blade morphology...",
            "Processing trichome density patterns..."
        ]
        
        self.current_fact = 0
        self.facts_list = wing_facts
        
        self.facts_label = tk.Label(
            main_frame,
            text=wing_facts[0],
            font=('Arial', 9, 'italic'),
            fg='#666666',
            bg='#1a1a1a',
            wraplength=400
        )
        self.facts_label.pack(pady=5)
        
        # Cancel button
        cancel_frame = tk.Frame(main_frame, bg='#1a1a1a')
        cancel_frame.pack(pady=20)
        
        cancel_button = tk.Button(
            cancel_frame,
            text="Cancel Analysis",
            command=self.cancel_processing_from_overlay,
            bg='#dc3545',           # Darker red background
            fg='white',             # White text
            activebackground='#c82333',  # Darker red when clicked
            activeforeground='white',    # White text when clicked
            font=('Arial', 11, 'bold'),  # Slightly larger font
            relief=tk.RAISED,       # Raised relief for better visibility
            bd=2,                   # Border width
            padx=25,                # More horizontal padding
            pady=10,                # More vertical padding
            cursor='hand2'
            )
        cancel_button.pack()
        
        # Start fact cycling
        self.cycle_wing_facts()
        
        # Prevent closing with X button
        self.loading_window.protocol("WM_DELETE_WINDOW", self.on_loading_window_close)
    
    def animate_gif(self):
        """Animate the Drosophila GIF"""
        if not self.animation_running or not hasattr(self, 'gif_frames') or self.gif_frames is None:
            return
        
        try:
            if hasattr(self, 'gif_label') and self.gif_label.winfo_exists():
                # Update the image
                self.gif_label.config(image=self.gif_frames[self.current_frame])
                
                # Get duration for this frame (or use default)
                duration = self.gif_durations[self.current_frame] if self.gif_durations else 100
                
                # Move to next frame
                self.current_frame = (self.current_frame + 1) % len(self.gif_frames)
                
                # Schedule next frame
                self.root.after(duration, self.animate_gif)
                
        except (tk.TclError, IndexError):
            # Handle any display errors gracefully
            pass
    
    def cycle_wing_facts(self):
        """Cycle through wing analysis facts"""
        if not self.animation_running or not hasattr(self, 'facts_label'):
            return
        
        try:
            if self.facts_label.winfo_exists():
                self.facts_label.config(text=self.facts_list[self.current_fact])
                self.current_fact = (self.current_fact + 1) % len(self.facts_list)
                self.root.after(3000, self.cycle_wing_facts)  # Change every 3 seconds
        except tk.TclError:
            pass
    
    def update_loading_status(self, status_text):
        """Update the loading status text"""
        if hasattr(self, 'loading_status_label'):
            try:
                if self.loading_status_label.winfo_exists():
                    self.loading_status_label.config(text=status_text)
            except tk.TclError:
                pass
    
    def close_loading_overlay(self):
        """Close the loading overlay and cleanup"""
        self.animation_running = False
        
        # Clear GIF references to prevent memory leaks
        if hasattr(self, 'gif_frames'):
            self.gif_frames = None
        if hasattr(self, 'gif_durations'):
            self.gif_durations = None
        
        if hasattr(self, 'loading_window'):
            try:
                if self.loading_window.winfo_exists():
                    self.loading_window.grab_release()
                    self.loading_window.destroy()
            except tk.TclError:
                pass
    
    def cancel_processing_from_overlay(self):
        """Cancel processing from the overlay window"""
        if messagebox.askyesno("Cancel Processing", "Are you sure you want to cancel the analysis?"):
            self.stop_processing()
            self.close_loading_overlay()
    
    def on_loading_window_close(self):
        """Handle loading window close attempt"""
        self.cancel_processing_from_overlay()
    
    # ==== END ANIMATION METHODS ====
    
    def update_config_from_gui(self):
        """Update config object from GUI values"""
        try:
            # Update all config attributes from GUI
            for attr_name in dir(self.config):
                if attr_name.startswith('_'):
                    continue
                
                var_name = f"{attr_name}_var"
                type_name = f"{attr_name}_type"
                
                if hasattr(self, var_name):
                    var = getattr(self, var_name)
                    
                    if isinstance(var, tk.BooleanVar):
                        setattr(self.config, attr_name, var.get())
                    else:
                        value_str = var.get()
                        if hasattr(self, type_name):
                            data_type = getattr(self, type_name)
                            
                            if attr_name == "scales":
                                # Special handling for scales list
                                scales = [float(x.strip()) for x in value_str.split(',')]
                                setattr(self.config, attr_name, scales)
                            else:
                                setattr(self.config, attr_name, data_type(value_str))
                        else:
                            setattr(self.config, attr_name, value_str)
            
            self.config.validate()
            return True
            
        except Exception as e:
            messagebox.showerror("Configuration Error", f"Invalid configuration: {e}")
            return False
    
    def start_processing(self):
        if not self.master_folder_var.get() or not self.output_folder_var.get():
            messagebox.showerror("Error", "Please select both master folder and output folder")
            return
        
        if not self.update_config_from_gui():
            return
        
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Create and show Drosophila loading overlay
        self.create_drosophila_loading_overlay()
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.process_folders, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        self.is_processing = False
        self.stop_button.config(state=tk.DISABLED)
        self.log_message("Stopping processing...")
        self.close_loading_overlay()
    
    def process_folders(self):
        """Main processing function that runs in separate thread"""
        try:
            master_folder = self.master_folder_var.get()
            output_folder = self.output_folder_var.get()
            
            # Get all subfolders with files
            subfolders_to_process = []
            for item in os.listdir(master_folder):
                item_path = os.path.join(master_folder, item)
                if os.path.isdir(item_path):
                    mapping = find_associated_files(item_path)
                    if mapping:
                        subfolders_to_process.append((item, item_path, len(mapping)))
            
            if not subfolders_to_process:
                self.progress_queue.put(("error", "No subfolders with valid files found"))
                return
            
            total_folders = len(subfolders_to_process)
            processed_folders = 0
            total_files_processed = 0
            total_files_failed = 0
            
            self.progress_queue.put(("progress", f"Starting processing of {total_folders} folders"))
            
            for folder_name, folder_path, file_count in subfolders_to_process:
                if not self.is_processing:
                    break
                
                # Update progress
                self.progress_queue.put(("current_folder", f"Processing: {folder_name} ({file_count} files)"))
                self.progress_queue.put(("progress_percent", int(processed_folders / total_folders * 100)))
                
                # Create output subfolder
                output_subfolder = os.path.join(output_folder, folder_name)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Check if already processed
                if self.skip_existing_var.get():
                    summary_file = os.path.join(output_subfolder, "analysis_summary.txt")
                    if os.path.exists(summary_file):
                        self.progress_queue.put(("log", f"Skipping {folder_name} - already processed"))
                        processed_folders += 1
                        continue
                
                # Save configuration
                if self.save_config_var.get():
                    config_path = os.path.join(output_subfolder, "processing_config.json")
                    self.save_config_to_file(config_path)
                
                # Process this folder
                try:
                    pentagone_mode = self.pentagone_mode_var.get()
                    force_pentagone = pentagone_mode == "pentagone"
                    
                    result = main_with_hybrid_wing_detection(
                        directory=folder_path,
                        cfg=self.config,
                        output_directory=output_subfolder,
                        force_pentagone_mode=force_pentagone,
                        auto_detect_pentagone=(pentagone_mode == "auto"),
                        progress_callback=self.progress_queue
                    )
                    
                    total_files_processed += file_count
                    self.progress_queue.put(("log", f"Completed {folder_name}: {file_count} files processed"))
                    
                except Exception as e:
                    total_files_failed += file_count
                    self.progress_queue.put(("error", f"Error processing {folder_name}: {e}"))
                
                processed_folders += 1
            
            # Final summary
            if self.is_processing:
                summary = f"""
Processing Complete!

Folders processed: {processed_folders}/{total_folders}
Total files processed: {total_files_processed}
Total files failed: {total_files_failed}
Output directory: {output_folder}

Processing completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
                self.progress_queue.put(("summary", summary))
                self.progress_queue.put(("progress", "Processing completed successfully"))
            else:
                self.progress_queue.put(("progress", "Processing stopped by user"))
            
        except Exception as e:
            self.progress_queue.put(("error", f"Critical error: {e}"))
        
        finally:
            self.progress_queue.put(("finished", None))
    
    def check_progress(self):
        """Check for progress updates from processing thread"""
        try:
            while True:
                msg_type, message = self.progress_queue.get_nowait()
                
                if msg_type == "progress":
                    self.progress_var.set(message)
                    self.update_loading_status(message)  # Update overlay
                elif msg_type == "current_folder":
                    self.current_folder_var.set(message)
                    self.update_loading_status(message)  # Update overlay
                elif msg_type == "progress_percent":
                    self.progress_bar['value'] = message
                elif msg_type == "log":
                    self.log_message(message)
                    # Update overlay with recent log message for key events
                    if any(keyword in message.lower() for keyword in ['processing', 'analyzing', 'detecting', 'found']):
                        self.update_loading_status(message[:60] + "..." if len(message) > 60 else message)
                elif msg_type == "error":
                    self.log_message(f"ERROR: {message}")
                    self.update_loading_status(f"Error: {message}")
                elif msg_type == "summary":
                    self.summary_text.delete(1.0, tk.END)
                    self.summary_text.insert(tk.END, message)
                elif msg_type == "finished":
                    self.is_processing = False
                    self.start_button.config(state=tk.NORMAL)
                    self.stop_button.config(state=tk.DISABLED)
                    self.progress_bar['value'] = 100
                    self.close_loading_overlay()  # Close overlay when done
                    
                    # Show completion message
                    messagebox.showinfo("Processing Complete", "Analysis finished successfully!")
                    break
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_progress)
    
    def log_message(self, message):
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
    
    def clear_log(self):
        self.log_text.delete(1.0, tk.END)
    
    def load_config(self):
        filename = filedialog.askopenfilename(
            title="Load Configuration",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            try:
                with open(filename, 'r') as f:
                    config_dict = json.load(f)
                
                # Update config object
                for key, value in config_dict.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
                
                # Update GUI
                self.update_gui_from_config()
                messagebox.showinfo("Success", "Configuration loaded successfully")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration: {e}")
    
    def save_config(self):
        if not self.update_config_from_gui():
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.save_config_to_file(filename)
    
    def save_config_to_file(self, filename):
        try:
            config_dict = asdict(self.config)
            with open(filename, 'w') as f:
                json.dump(config_dict, f, indent=2)
            self.log_message(f"Configuration saved to {filename}")
        except Exception as e:
            self.log_message(f"Failed to save configuration: {e}")
    
    def reset_config(self):
        self.config = TrichomeDetectionConfig()
        self.update_gui_from_config()
        messagebox.showinfo("Reset", "Configuration reset to defaults")
    
    def update_gui_from_config(self):
        """Update GUI values from config object"""
        for attr_name in dir(self.config):
            if attr_name.startswith('_'):
                continue
            
            var_name = f"{attr_name}_var"
            if hasattr(self, var_name):
                var = getattr(self, var_name)
                value = getattr(self.config, attr_name)
                
                if isinstance(var, tk.BooleanVar):
                    var.set(value)
                else:
                    if attr_name == "scales":
                        var.set(",".join(map(str, value)))
                    else:
                        var.set(str(value))

def run_gui():
    """Launch the GUI application"""
    root = tk.Tk()
    
    # Set application icon if available
    try:
        # You can add an icon file here if you have one
        # root.iconbitmap('icon.ico')
        pass
    except:
        pass
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_reqwidth()
    height = root.winfo_reqheight()
    pos_x = (root.winfo_screenwidth() // 2) - (width // 2)
    pos_y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{pos_x}+{pos_y}")
    
    # Create and run the application
    app = TrichomeAnalysisGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_processing:
            if messagebox.askokcancel("Quit", "Processing is still running. Do you want to stop and quit?"):
                app.is_processing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI main loop
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Application interrupted by user")
    except Exception as e:
        print(f"GUI Error: {e}")
        messagebox.showerror("Application Error", f"An error occurred: {e}")

class PeakDetectionMetrics:
    """Class to track and report detection quality metrics."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.raw_peaks = 0
        self.filtered_peaks = 0
        self.edge_excluded = 0
        self.intensity_filtered = 0
        self.cluster_merged = 0
        self.final_peaks = 0
        self.processing_time = 0.0
        self.scales_used = []
        
    def report(self) -> str:
        """Generate a detailed report of detection metrics."""
        efficiency = (self.final_peaks / max(self.raw_peaks, 1)) * 100
        report = f"""
Peak Detection Quality Report:
==============================
Raw peaks detected: {self.raw_peaks}
Edge exclusions: {self.edge_excluded}
Intensity filtered: {self.intensity_filtered}
Cluster merged: {self.cluster_merged}
Final peaks retained: {self.final_peaks}
Detection efficiency: {efficiency:.1f}%
Processing time: {self.processing_time:.2f}s
Scales used: {self.scales_used}
"""
        return report
def detect_wing_boundary_from_trichomes(self, prob_map, raw_img=None, peaks=None):
    """Enhanced wing boundary detection using filtered trichomes."""
    
    logger.info("Using trichome-based wing boundary detection...")
    
    # If peaks not provided, detect them
    if peaks is None:
        tri_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else prob_map
        peaks, _ = detect_trichome_peaks(tri_prob, self.config)
    
    logger.info(f"Using {len(peaks)} trichomes for wing boundary detection")
    
    if len(peaks) < 20:
        logger.warning("Too few trichomes for reliable wing detection, falling back to probability method")
        return self._detect_wing_from_probabilities(prob_map)
    
    # Filter trichomes to remove string artifacts
    string_filter = StringRemovalTrichomeFilter(self.config)
    filtered_peaks = string_filter.remove_trichome_strings(peaks, prob_map.shape[:2])
    
    logger.info(f"After string removal: {len(filtered_peaks)} trichomes kept")
    
    if len(filtered_peaks) < 10:
        logger.warning("String filtering removed too many trichomes, falling back")
        return self._detect_wing_from_probabilities(prob_map)
    
    # Create wing mask from filtered trichomes
    wing_mask = string_filter.create_wing_mask_simple(filtered_peaks, prob_map.shape[:2])
    
    if wing_mask is None:
        logger.warning("Trichome-based wing detection failed, falling back")
        return self._detect_wing_from_probabilities(prob_map)
    
    # Validate the wing mask
    wing_area = np.sum(wing_mask)
    total_area = wing_mask.size
    coverage = wing_area / total_area
    
    # Sanity checks
    if coverage < 0.1 or coverage > 0.8:
        logger.warning(f"Wing coverage {coverage:.2f} seems unreasonable, falling back")
        return self._detect_wing_from_probabilities(prob_map)
    
    logger.info(f"Trichome-based wing detection successful: {wing_area} pixels ({coverage:.2f} coverage)")
    return wing_mask

def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)
# class AnatomicalTemplateMatcher:
#     """Wing-ratio based template matcher using absolute ratios to total wing area."""
    
#     def __init__(self):
#         # Define the 5 regions based on your ACTUAL data ratios to total wing
#         # From your data: Region 5 (21.4%) > Region 3 (18.1%) > Region 4 (17.9%) > Region 2 (15.9%) > Region 1 (9.3%)
#         self.target_regions = {
#             1: {  # Your region 1 - smallest, very elongated
#                 'name': 'Region-1',
#                 'wing_area_ratio': 0.0933,      # 9.33% of total wing
#                 'wing_ratio_tolerance': 0.025,   # ±2.5% tolerance
#                 'aspect_ratio_range': (7.0, 11.0),  # Very elongated (AR=9.20)
#                 'solidity_range': (0.80, 0.95),     # Medium solidity (0.885)
#                 'distinctive_features': {
#                     'is_smallest': True,
#                     'is_very_elongated': True,
#                     'wing_coverage': 'small'
#                 },
#                 'size_rank_expected': 5  # Smallest of the 5
#             },
#             2: {  # Your region 2 - medium-small, elongated, low solidity
#                 'name': 'Region-2', 
#                 'wing_area_ratio': 0.1592,      # 15.92% of total wing
#                 'wing_ratio_tolerance': 0.03,
#                 'aspect_ratio_range': (5.5, 8.5),   # Elongated (AR=6.95)
#                 'solidity_range': (0.70, 0.80),     # Low solidity (0.743) - distinctive!
#                 'distinctive_features': {
#                     'is_smallest': False,
#                     'is_very_elongated': False,
#                     'wing_coverage': 'medium-small',
#                     'has_low_solidity': True  # Most distinctive feature
#                 },
#                 'size_rank_expected': 4
#             },
#             3: {  # Your region 3 - large, elongated, good solidity
#                 'name': 'Region-3',
#                 'wing_area_ratio': 0.1814,      # 18.14% of total wing
#                 'wing_ratio_tolerance': 0.03,
#                 'aspect_ratio_range': (5.5, 8.5),   # Elongated (AR=6.84)
#                 'solidity_range': (0.90, 0.98),     # High solidity (0.948)
#                 'distinctive_features': {
#                     'is_smallest': False,
#                     'is_very_elongated': False,
#                     'wing_coverage': 'large'
#                 },
#                 'size_rank_expected': 2  # 2nd largest
#             },
#             4: {  # Your region 4 - large, compact, very high solidity
#                 'name': 'Region-4',
#                 'wing_area_ratio': 0.1790,      # 17.90% of total wing
#                 'wing_ratio_tolerance': 0.03,
#                 'aspect_ratio_range': (2.0, 3.0),   # Compact (AR=2.36)
#                 'solidity_range': (0.95, 1.0),      # Very high solidity (0.973)
#                 'distinctive_features': {
#                     'is_smallest': False,
#                     'is_very_elongated': False,
#                     'wing_coverage': 'large',
#                     'is_compact': True  # Distinctive: compact + large
#                 },
#                 'size_rank_expected': 3  # 3rd largest
#             },
#             5: {  # Your region 5 - largest, compact, highest solidity
#                 'name': 'Region-5',
#                 'wing_area_ratio': 0.2141,      # 21.41% of total wing
#                 'wing_ratio_tolerance': 0.03,
#                 'aspect_ratio_range': (2.3, 3.2),   # Compact (AR=2.68)
#                 'solidity_range': (0.98, 1.0),      # Highest solidity (0.996)
#                 'distinctive_features': {
#                     'is_smallest': False,
#                     'is_very_elongated': False,
#                     'wing_coverage': 'largest',
#                     'is_compact': True,
#                     'is_largest': True  # Most distinctive: largest + compact
#                 },
#                 'size_rank_expected': 1  # Largest
#             }
#         }
    
    
#     def get_region_name(self, label):
#         """Get the name of a region by its label/template ID."""
#         if label in self.target_regions:
#             return self.target_regions[label]['name']
#         else:
#             return f"Region-{label}"
    
    
#     def estimate_total_wing_area(self, labeled_mask, wing_mask):
#         """Estimate total wing area for calculating ratios."""
        
#         # Method 1: Use wing mask if available and reasonable
#         wing_area_from_mask = np.sum(wing_mask)
        
#         # Method 2: Use detected regions + coverage estimate
#         regions = regionprops(labeled_mask)
#         total_detected_area = sum(r.area for r in regions)
        
#         # From your data, the 5 target regions cover ~82.7% of wing
#         expected_coverage = 0.827
#         estimated_wing_from_regions = total_detected_area / expected_coverage
        
#         # Choose the most reasonable estimate
#         if wing_area_from_mask > 0:
#             # Use wing mask if it seems reasonable (not too different from region estimate)
#             ratio = wing_area_from_mask / estimated_wing_from_regions
#             if 0.7 < ratio < 1.4:  # Wing mask seems reasonable
#                 estimated_wing_area = wing_area_from_mask
#                 method = "wing_mask"
#             else:
#                 estimated_wing_area = estimated_wing_from_regions
#                 method = "region_coverage"
#         else:
#             estimated_wing_area = estimated_wing_from_regions
#             method = "region_coverage"
        
#         logger.info(f"Wing area estimation: {estimated_wing_area:.0f} (method: {method})")
#         logger.info(f"  Wing mask area: {wing_area_from_mask:.0f}")
#         logger.info(f"  Region-based estimate: {estimated_wing_from_regions:.0f}")
        
#         return estimated_wing_area
    
#     def calculate_wing_ratio_signature(self, region, total_wing_area):
#         """Calculate region signature based on wing area ratios."""
        
#         wing_ratio = region.area / total_wing_area
#         aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
        
#         return {
#             'wing_ratio': wing_ratio,
#             'aspect_ratio': aspect_ratio,
#             'solidity': region.solidity,
#             'absolute_area': region.area,
#             'is_very_elongated': aspect_ratio > 7.0,
#             'is_compact': aspect_ratio < 3.5,
#             'has_low_solidity': region.solidity < 0.8,
#             'region': region,
#             'label': region.label
#         }
    
#     def calculate_template_match_score(self, region_sig, template_id):
#         """Calculate how well a region matches a template using wing ratios."""
        
#         template = self.target_regions[template_id]
#         score = 0  # Lower is better
        
#         # 1. Wing area ratio match (most important!)
#         expected_ratio = template['wing_area_ratio']
#         tolerance = template['wing_ratio_tolerance']
#         ratio_error = abs(region_sig['wing_ratio'] - expected_ratio)
        
#         if ratio_error <= tolerance:
#             # Within tolerance - good match
#             score += ratio_error / tolerance * 20  # 0-20 points
#         else:
#             # Outside tolerance - penalty
#             excess_error = ratio_error - tolerance
#             score += 20 + excess_error * 200  # Heavy penalty for being way off
        
#         # 2. Aspect ratio match
#         ar_range = template['aspect_ratio_range']
#         if ar_range[0] <= region_sig['aspect_ratio'] <= ar_range[1]:
#             score += 0  # Perfect match
#         else:
#             if region_sig['aspect_ratio'] < ar_range[0]:
#                 score += (ar_range[0] - region_sig['aspect_ratio']) * 8
#             else:
#                 score += (region_sig['aspect_ratio'] - ar_range[1]) * 8
        
#         # 3. Solidity match
#         sol_range = template['solidity_range']
#         if sol_range[0] <= region_sig['solidity'] <= sol_range[1]:
#             score += 0  # Perfect match
#         else:
#             if region_sig['solidity'] < sol_range[0]:
#                 score += (sol_range[0] - region_sig['solidity']) * 50
#             else:
#                 score += (region_sig['solidity'] - sol_range[1]) * 50
        
#         # 4. Distinctive feature bonuses/penalties
#         features = template['distinctive_features']
        
#         # Check low solidity feature
#         if features.get('has_low_solidity', False):
#             if region_sig['has_low_solidity']:
#                 score -= 15  # Bonus for matching distinctive feature
#             else:
#                 score += 25  # Penalty for not matching
#         elif region_sig['has_low_solidity'] and not features.get('has_low_solidity', False):
#             score += 20  # Penalty for having low solidity when not expected
        
#         # Check compact feature
#         if features.get('is_compact', False):
#             if region_sig['is_compact']:
#                 score -= 10  # Bonus
#             else:
#                 score += 20  # Penalty
#         elif region_sig['is_compact'] and not features.get('is_compact', False):
#             score += 15  # Penalty for being compact when not expected
        
#         # Check very elongated feature
#         if features.get('is_very_elongated', False):
#             if region_sig['is_very_elongated']:
#                 score -= 10  # Bonus
#             else:
#                 score += 20  # Penalty
        
#         return score
    
#     def identify_best_matches_by_wing_ratio(self, region_signatures):
#         """Find best matches using wing area ratios as primary criterion."""
        
#         logger.info("Calculating wing-ratio based matches:")
        
#         # Create match matrix
#         match_scores = {}
#         for region_sig in region_signatures:
#             region_label = region_sig['label']
#             match_scores[region_label] = {}
            
#             logger.info(f"  Region {region_label}: wing_ratio={region_sig['wing_ratio']:.4f}, "
#                        f"AR={region_sig['aspect_ratio']:.1f}, sol={region_sig['solidity']:.3f}")
            
#             for template_id in range(1, 6):  # Only 5 templates now
#                 score = self.calculate_template_match_score(region_sig, template_id)
#                 match_scores[region_label][template_id] = score
                
#                 template_name = self.target_regions[template_id]['name']
#                 expected_ratio = self.target_regions[template_id]['wing_area_ratio']
#                 logger.info(f"    → {template_id} ({template_name}): score={score:.1f} "
#                            f"(expected_ratio={expected_ratio:.4f})")
        
#         return match_scores
    
#     def solve_assignment_problem(self, region_signatures, match_scores):
#         """Solve the assignment problem using Hungarian algorithm."""
        
#         n_regions = len(region_signatures)
#         n_templates = 5
        
#         # Create cost matrix (pad with high costs if regions != templates)
#         max_size = max(n_regions, n_templates)
#         cost_matrix = np.full((max_size, max_size), 1000.0)
        
#         # Fill in actual scores
#         for i, region_sig in enumerate(region_signatures):
#             region_label = region_sig['label']
#             for j, template_id in enumerate(range(1, 6)):
#                 cost_matrix[i, j] = match_scores[region_label][template_id]
        
#         # Solve assignment problem
#         from scipy.optimize import linear_sum_assignment
#         row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
#         # Extract valid assignments
#         assignment = {}
#         for row, col in zip(row_indices, col_indices):
#             if row < n_regions and col < n_templates:
#                 region_sig = region_signatures[row]
#                 template_id = col + 1
#                 score = cost_matrix[row, col]
                
#                 # Only accept reasonable matches
#                 if score < 100:  # Reasonable threshold
#                     assignment[region_sig['label']] = template_id
                    
#                     template_name = self.target_regions[template_id]['name']
#                     logger.info(f"  ASSIGNED: Region {region_sig['label']} → {template_id} "
#                                f"({template_name}) [score={score:.1f}]")
#                 else:
#                     logger.warning(f"  REJECTED: Region {region_sig['label']} → {template_id} "
#                                  f"[score={score:.1f} too high]")
        
#         return assignment
    
#     def wing_ratio_based_matching(self, labeled_mask, wing_mask):
#         """Main matching using wing area ratios."""
        
#         # Get regions and filter
#         regions = regionprops(labeled_mask)
#         if not regions:
#             return labeled_mask
        
#         # More aggressive filtering to remove spurious regions
#         filtered_mask = self.filter_spurious_regions(labeled_mask, wing_mask)
#         regions = regionprops(filtered_mask)
        
#         if not regions:
#             return filtered_mask
        
#         logger.info("="*70)
#         logger.info("WING-RATIO BASED MATCHING")
#         logger.info("="*70)
        
#         # Estimate total wing area
#         total_wing_area = self.estimate_total_wing_area(filtered_mask, wing_mask)
        
#         # Calculate signatures for all regions
#         region_signatures = []
#         for region in regions:
#             signature = self.calculate_wing_ratio_signature(region, total_wing_area)
#             region_signatures.append(signature)
        
#         # Sort by area for display
#         region_signatures.sort(key=lambda r: r['absolute_area'], reverse=True)
        
#         logger.info(f"Analyzing {len(region_signatures)} filtered regions:")
#         for i, sig in enumerate(region_signatures):
#             logger.info(f"  {i+1}. Region {sig['label']}: wing_ratio={sig['wing_ratio']:.4f} "
#                        f"({sig['wing_ratio']*100:.1f}%), AR={sig['aspect_ratio']:.1f}, "
#                        f"sol={sig['solidity']:.3f}"
#                        f"{', compact' if sig['is_compact'] else ''}"
#                        f"{', very_elongated' if sig['is_very_elongated'] else ''}"
#                        f"{', low_solidity' if sig['has_low_solidity'] else ''}")
        
#         # Find best matches
#         match_scores = self.identify_best_matches_by_wing_ratio(region_signatures)
        
#         # Solve assignment problem
#         assignment = self.solve_assignment_problem(region_signatures, match_scores)
        
#         # Create anatomical mask
#         anatomical_mask = np.zeros_like(filtered_mask)
        
#         logger.info("="*50)
#         logger.info("FINAL ASSIGNMENT:")
        
#         assigned_templates = set()
#         for region_label, template_id in assignment.items():
#             region_mask = filtered_mask == region_label
#             anatomical_mask[region_mask] = template_id
#             assigned_templates.add(template_id)
            
#             template_name = self.target_regions[template_id]['name']
#             expected_ratio = self.target_regions[template_id]['wing_area_ratio']
            
#             # Get actual ratio
#             region_sig = next(r for r in region_signatures if r['label'] == region_label)
#             actual_ratio = region_sig['wing_ratio']
            
#             logger.info(f"  Region {region_label} → {template_id} ({template_name})")
#             logger.info(f"    Expected: {expected_ratio:.4f} ({expected_ratio*100:.1f}%), "
#                        f"Actual: {actual_ratio:.4f} ({actual_ratio*100:.1f}%)")
        
#         # Report missing templates
#         all_templates = set(range(1, 6))
#         missing_templates = all_templates - assigned_templates
        
#         if missing_templates:
#             logger.warning("MISSING TEMPLATES:")
#             for template_id in sorted(missing_templates):
#                 template_name = self.target_regions[template_id]['name']
#                 expected_ratio = self.target_regions[template_id]['wing_area_ratio']
#                 logger.warning(f"  Template {template_id} ({template_name}) - "
#                              f"expected {expected_ratio:.4f} ({expected_ratio*100:.1f}%)")
#         else:
#             logger.info("All 5 templates successfully assigned!")
        
#         # Final summary
#         unique_labels = np.unique(anatomical_mask)
#         matched_count = len(unique_labels[unique_labels > 0])
        
#         logger.info("="*70)
#         logger.info(f"WING-RATIO MATCHING COMPLETE: {matched_count}/5 regions labeled")
#         logger.info(f"Uses absolute ratios to total wing area - robust to missing regions!")
#         logger.info("="*70)
        
#         return anatomical_mask
    
#     def filter_spurious_regions(self, labeled_mask, wing_mask):
#         """Aggressively filter out spurious regions that don't match target characteristics."""
        
#         regions = regionprops(labeled_mask)
#         if not regions:
#             return labeled_mask
        
#         logger.info(f"Filtering {len(regions)} detected regions...")
        
#         # Estimate wing area for ratio calculations
#         wing_area = np.sum(wing_mask) if np.sum(wing_mask) > 0 else sum(r.area for r in regions) / 0.827
        
#         keep_regions = []
        
#         for region in regions:
#             wing_ratio = region.area / wing_area
#             aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
#             # Basic size filter - target regions are 9-22% of wing
#             if wing_ratio < 0.05 or wing_ratio > 0.35:
#                 logger.info(f"  REJECT Region {region.label}: wing_ratio={wing_ratio:.4f} outside range [0.05, 0.35]")
#                 continue
            
#             # Basic shape filter - target regions have AR between 2-11
#             if aspect_ratio < 1.5 or aspect_ratio > 12.0:
#                 logger.info(f"  REJECT Region {region.label}: AR={aspect_ratio:.1f} outside range [1.5, 12.0]")
#                 continue
            
#             # Solidity filter - target regions have reasonable solidity
#             if region.solidity < 0.45:
#                 logger.info(f"  REJECT Region {region.label}: solidity={region.solidity:.3f} too low")
#                 continue
            
#             # Position filter - should be in main wing body
#             wing_coords = np.column_stack(np.where(wing_mask))
#             if len(wing_coords) > 0:
#                 min_y, min_x = wing_coords.min(axis=0)
#                 max_y, max_x = wing_coords.max(axis=0)
                
#                 if max_y > min_y and max_x > min_x:
#                     norm_y = (region.centroid[0] - min_y) / (max_y - min_y)
#                     norm_x = (region.centroid[1] - min_x) / (max_x - min_x)
                    
#                     if not (0.05 < norm_x < 0.95 and 0.0 < norm_y < 1.0):
#                         logger.info(f"  REJECT Region {region.label}: position ({norm_x:.2f}, {norm_y:.2f}) outside wing body")
#                         continue
            
#             # Passed all filters
#             keep_regions.append(region)
#             logger.info(f"  KEEP Region {region.label}: wing_ratio={wing_ratio:.4f}, AR={aspect_ratio:.1f}, sol={region.solidity:.3f}")
        
#         # Create filtered mask
#         filtered_mask = np.zeros_like(labeled_mask)
#         new_label = 1
        
#         for region in keep_regions:
#             region_mask = labeled_mask == region.label
#             filtered_mask[region_mask] = new_label
#             new_label += 1
        
#         logger.info(f"Kept {len(keep_regions)} regions after filtering")
#         return filtered_mask
    
#     # Main interface
#     def match_regions_to_template(self, labeled_mask, wing_mask, vein_mask=None):
#         """Main interface - uses wing-ratio based matching."""
#         return self.wing_ratio_based_matching(labeled_mask, wing_mask)



class AnatomicalTemplateMatcher:
    """Wing-ratio based template matcher using absolute ratios to total wing area."""
    
    def __init__(self):
        # Define the 5 regions based on your ACTUAL data ratios to total wing
        # From your data: Region 5 (21.4%) > Region 3 (18.1%) > Region 4 (17.9%) > Region 2 (15.9%) > Region 1 (9.3%)
        self.target_regions = {
            1: {  # Your region 1 - smallest, very elongated
                'name': 'Region-1',
                'wing_area_ratio': 0.0933,      # 9.33% of total wing
                'wing_ratio_tolerance': 0.025,   # ±2.5% tolerance
                'aspect_ratio_range': (7.0, 11.0),  # Very elongated (AR=9.20)
                'solidity_range': (0.80, 0.95),     # Medium solidity (0.885)
                'distinctive_features': {
                    'is_smallest': True,
                    'is_very_elongated': True,
                    'wing_coverage': 'small'
                },
                'size_rank_expected': 5  # Smallest of the 5
            },
            2: {  # Your region 2 - medium-small, elongated, low solidity
                'name': 'Region-2', 
                'wing_area_ratio': 0.1592,      # 15.92% of total wing
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (5.5, 8.5),   # Elongated (AR=6.95)
                'solidity_range': (0.70, 0.80),     # Low solidity (0.743) - distinctive!
                'distinctive_features': {
                    'is_smallest': False,
                    'is_very_elongated': False,
                    'wing_coverage': 'medium-small',
                    'has_low_solidity': True  # Most distinctive feature
                },
                'size_rank_expected': 4
            },
            3: {  # Your region 3 - large, elongated, good solidity
                'name': 'Region-3',
                'wing_area_ratio': 0.1814,      # 18.14% of total wing
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (5.5, 8.5),   # Elongated (AR=6.84)
                'solidity_range': (0.90, 0.98),     # High solidity (0.948)
                'distinctive_features': {
                    'is_smallest': False,
                    'is_very_elongated': False,
                    'wing_coverage': 'large'
                },
                'size_rank_expected': 2  # 2nd largest
            },
            4: {  # Your region 4 - large, compact, very high solidity
                'name': 'Region-4',
                'wing_area_ratio': 0.1790,      # 17.90% of total wing
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (2.0, 3.0),   # Compact (AR=2.36)
                'solidity_range': (0.95, 1.0),      # Very high solidity (0.973)
                'distinctive_features': {
                    'is_smallest': False,
                    'is_very_elongated': False,
                    'wing_coverage': 'large',
                    'is_compact': True  # Distinctive: compact + large
                },
                'size_rank_expected': 3  # 3rd largest
            },
            5: {  # Your region 5 - largest, compact, highest solidity
                'name': 'Region-5',
                'wing_area_ratio': 0.2141,      # 21.41% of total wing
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (2.3, 3.2),   # Compact (AR=2.68)
                'solidity_range': (0.98, 1.0),      # Highest solidity (0.996)
                'distinctive_features': {
                    'is_smallest': False,
                    'is_very_elongated': False,
                    'wing_coverage': 'largest',
                    'is_compact': True,
                    'is_largest': True  # Most distinctive: largest + compact
                },
                'size_rank_expected': 1  # Largest
            }
        }
    def set_empirical_merged_data(self, wing_area, merged_area, merged_ar, merged_solidity):
        """
        Set empirical merged region data for more accurate detection.
        
        Args:
            wing_area: Total wing area in pixels
            merged_area: Combined area of regions 4+5 in pixels
            merged_ar: Aspect ratio of merged region
            merged_solidity: Solidity of merged region
        """
        self.empirical_merged_ratio = merged_area / wing_area
        self.empirical_merged_ar = merged_ar
        self.empirical_merged_solidity = merged_solidity
        
        logger.info(f"Set empirical merged 4+5 data:")
        logger.info(f"  Wing ratio: {self.empirical_merged_ratio:.4f}")
        logger.info(f"  Aspect ratio: {self.empirical_merged_ar:.2f}")
        logger.info(f"  Solidity: {self.empirical_merged_solidity:.3f}")
    def get_region_name(self, label):
        """Get the name of a region by its label/template ID."""
        if label in self.target_regions:
            return self.target_regions[label]['name']
        else:
            return f"Region-{label}"
    
    def estimate_total_wing_area(self, labeled_mask, wing_mask):
        """Estimate total wing area for calculating ratios."""
        
        # Method 1: Use wing mask if available and reasonable
        wing_area_from_mask = np.sum(wing_mask)
        
        # Method 2: Use detected regions + coverage estimate
        regions = regionprops(labeled_mask)
        total_detected_area = sum(r.area for r in regions)
        
        # From your data, the 5 target regions cover ~82.7% of wing
        expected_coverage = 0.827
        estimated_wing_from_regions = total_detected_area / expected_coverage
        
        # Choose the most reasonable estimate
        if wing_area_from_mask > 0:
            # Use wing mask if it seems reasonable (not too different from region estimate)
            ratio = wing_area_from_mask / estimated_wing_from_regions
            if 0.7 < ratio < 1.4:  # Wing mask seems reasonable
                estimated_wing_area = wing_area_from_mask
                method = "wing_mask"
            else:
                estimated_wing_area = estimated_wing_from_regions
                method = "region_coverage"
        else:
            estimated_wing_area = estimated_wing_from_regions
            method = "region_coverage"
        
        logger.info(f"Wing area estimation: {estimated_wing_area:.0f} (method: {method})")
        logger.info(f"  Wing mask area: {wing_area_from_mask:.0f}")
        logger.info(f"  Region-based estimate: {estimated_wing_from_regions:.0f}")
        
        return estimated_wing_area
    
    def calculate_wing_ratio_signature(self, region, total_wing_area):
        """Calculate region signature based on wing area ratios."""
        
        wing_ratio = region.area / total_wing_area
        aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
        
        return {
            'wing_ratio': wing_ratio,
            'aspect_ratio': aspect_ratio,
            'solidity': region.solidity,
            'absolute_area': region.area,
            'is_very_elongated': aspect_ratio > 7.0,
            'is_compact': aspect_ratio < 3.5,
            'has_low_solidity': region.solidity < 0.8,
            'region': region,
            'label': region.label
        }
    
    def calculate_template_match_score(self, region_sig, template_id):
        """Calculate how well a region matches a template using wing ratios."""
        
        template = self.target_regions[template_id]
        score = 0  # Lower is better
        
        # 1. Wing area ratio match (most important!)
        expected_ratio = template['wing_area_ratio']
        tolerance = template['wing_ratio_tolerance']
        ratio_error = abs(region_sig['wing_ratio'] - expected_ratio)
        
        if ratio_error <= tolerance:
            # Within tolerance - good match
            score += ratio_error / tolerance * 20  # 0-20 points
        else:
            # Outside tolerance - penalty
            excess_error = ratio_error - tolerance
            score += 20 + excess_error * 200  # Heavy penalty for being way off
        
        # 2. Aspect ratio match
        ar_range = template['aspect_ratio_range']
        if ar_range[0] <= region_sig['aspect_ratio'] <= ar_range[1]:
            score += 0  # Perfect match
        else:
            if region_sig['aspect_ratio'] < ar_range[0]:
                score += (ar_range[0] - region_sig['aspect_ratio']) * 8
            else:
                score += (region_sig['aspect_ratio'] - ar_range[1]) * 8
        
        # 3. Solidity match
        sol_range = template['solidity_range']
        if sol_range[0] <= region_sig['solidity'] <= sol_range[1]:
            score += 0  # Perfect match
        else:
            if region_sig['solidity'] < sol_range[0]:
                score += (sol_range[0] - region_sig['solidity']) * 50
            else:
                score += (region_sig['solidity'] - sol_range[1]) * 50
        
        # 4. Distinctive feature bonuses/penalties
        features = template['distinctive_features']
        
        # Check low solidity feature
        if features.get('has_low_solidity', False):
            if region_sig['has_low_solidity']:
                score -= 15  # Bonus for matching distinctive feature
            else:
                score += 25  # Penalty for not matching
        elif region_sig['has_low_solidity'] and not features.get('has_low_solidity', False):
            score += 20  # Penalty for having low solidity when not expected
        
        # Check compact feature
        if features.get('is_compact', False):
            if region_sig['is_compact']:
                score -= 10  # Bonus
            else:
                score += 20  # Penalty
        elif region_sig['is_compact'] and not features.get('is_compact', False):
            score += 15  # Penalty for being compact when not expected
        
        # Check very elongated feature
        if features.get('is_very_elongated', False):
            if region_sig['is_very_elongated']:
                score -= 10  # Bonus
            else:
                score += 20  # Penalty
        
        return score
    
    def identify_best_matches_by_wing_ratio(self, region_signatures):
        """Find best matches using wing area ratios as primary criterion."""
        
        logger.info("Calculating wing-ratio based matches:")
        
        # Create match matrix
        match_scores = {}
        for region_sig in region_signatures:
            region_label = region_sig['label']
            match_scores[region_label] = {}
            
            logger.info(f"  Region {region_label}: wing_ratio={region_sig['wing_ratio']:.4f}, "
                       f"AR={region_sig['aspect_ratio']:.1f}, sol={region_sig['solidity']:.3f}")
            
            for template_id in range(1, 6):  # Only 5 templates now
                score = self.calculate_template_match_score(region_sig, template_id)
                match_scores[region_label][template_id] = score
                
                template_name = self.target_regions[template_id]['name']
                expected_ratio = self.target_regions[template_id]['wing_area_ratio']
                logger.info(f"    → {template_id} ({template_name}): score={score:.1f} "
                           f"(expected_ratio={expected_ratio:.4f})")
        
        return match_scores
    
    def solve_assignment_problem(self, region_signatures, match_scores):
        """Solve the assignment problem using Hungarian algorithm."""
        
        n_regions = len(region_signatures)
        n_templates = 5
        
        # Create cost matrix (pad with high costs if regions != templates)
        max_size = max(n_regions, n_templates)
        cost_matrix = np.full((max_size, max_size), 1000.0)
        
        # Fill in actual scores
        for i, region_sig in enumerate(region_signatures):
            region_label = region_sig['label']
            for j, template_id in enumerate(range(1, 6)):
                cost_matrix[i, j] = match_scores[region_label][template_id]
        
        # Solve assignment problem
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments
        assignment = {}
        for row, col in zip(row_indices, col_indices):
            if row < n_regions and col < n_templates:
                region_sig = region_signatures[row]
                template_id = col + 1
                score = cost_matrix[row, col]
                
                # Only accept reasonable matches
                if score < 100:  # Reasonable threshold
                    assignment[region_sig['label']] = template_id
                    
                    template_name = self.target_regions[template_id]['name']
                    logger.info(f"  ASSIGNED: Region {region_sig['label']} → {template_id} "
                               f"({template_name}) [score={score:.1f}]")
                else:
                    logger.warning(f"  REJECTED: Region {region_sig['label']} → {template_id} "
                                 f"[score={score:.1f} too high]")
        
        return assignment
    
    def wing_ratio_based_matching(self, labeled_mask, wing_mask):
        """Main matching using wing area ratios."""
        
        # Get regions and filter
        regions = regionprops(labeled_mask)
        if not regions:
            return labeled_mask
        
        # More aggressive filtering to remove spurious regions
        filtered_mask = self.filter_spurious_regions(labeled_mask, wing_mask)
        regions = regionprops(filtered_mask)
        
        if not regions:
            return filtered_mask
        
        logger.info("="*70)
        logger.info("WING-RATIO BASED MATCHING")
        logger.info("="*70)
        
        # Estimate total wing area
        total_wing_area = self.estimate_total_wing_area(filtered_mask, wing_mask)
        
        # Calculate signatures for all regions
        region_signatures = []
        for region in regions:
            signature = self.calculate_wing_ratio_signature(region, total_wing_area)
            region_signatures.append(signature)
        
        # Sort by area for display
        region_signatures.sort(key=lambda r: r['absolute_area'], reverse=True)
        
        logger.info(f"Analyzing {len(region_signatures)} filtered regions:")
        for i, sig in enumerate(region_signatures):
            logger.info(f"  {i+1}. Region {sig['label']}: wing_ratio={sig['wing_ratio']:.4f} "
                       f"({sig['wing_ratio']*100:.1f}%), AR={sig['aspect_ratio']:.1f}, "
                       f"sol={sig['solidity']:.3f}"
                       f"{', compact' if sig['is_compact'] else ''}"
                       f"{', very_elongated' if sig['is_very_elongated'] else ''}"
                       f"{', low_solidity' if sig['has_low_solidity'] else ''}")
        
        # Find best matches
        match_scores = self.identify_best_matches_by_wing_ratio(region_signatures)
        
        # Solve assignment problem
        assignment = self.solve_assignment_problem(region_signatures, match_scores)
        
        # Create anatomical mask
        anatomical_mask = np.zeros_like(filtered_mask)
        
        logger.info("="*50)
        logger.info("FINAL ASSIGNMENT:")
        
        assigned_templates = set()
        for region_label, template_id in assignment.items():
            region_mask = filtered_mask == region_label
            anatomical_mask[region_mask] = template_id
            assigned_templates.add(template_id)
            
            template_name = self.target_regions[template_id]['name']
            expected_ratio = self.target_regions[template_id]['wing_area_ratio']
            
            # Get actual ratio
            region_sig = next(r for r in region_signatures if r['label'] == region_label)
            actual_ratio = region_sig['wing_ratio']
            
            logger.info(f"  Region {region_label} → {template_id} ({template_name})")
            logger.info(f"    Expected: {expected_ratio:.4f} ({expected_ratio*100:.1f}%), "
                       f"Actual: {actual_ratio:.4f} ({actual_ratio*100:.1f}%)")
        
        # Report missing templates
        all_templates = set(range(1, 6))
        missing_templates = all_templates - assigned_templates
        
        if missing_templates:
            logger.warning("MISSING TEMPLATES:")
            for template_id in sorted(missing_templates):
                template_name = self.target_regions[template_id]['name']
                expected_ratio = self.target_regions[template_id]['wing_area_ratio']
                logger.warning(f"  Template {template_id} ({template_name}) - "
                             f"expected {expected_ratio:.4f} ({expected_ratio*100:.1f}%)")
        else:
            logger.info("All 5 templates successfully assigned!")
        
        # Final summary
        unique_labels = np.unique(anatomical_mask)
        matched_count = len(unique_labels[unique_labels > 0])
        
        logger.info("="*70)
        logger.info(f"WING-RATIO MATCHING COMPLETE: {matched_count}/5 regions labeled")
        logger.info(f"Uses absolute ratios to total wing area - robust to missing regions!")
        logger.info("="*70)
        
        return anatomical_mask
    
    def match_regions_with_merge_detection(self, labeled_mask, wing_mask, allow_merge=False):
        """
        Template matching that can detect and handle merged regions 4+5.
        
        If allow_merge=True, looks for a large region that matches combined 4+5 properties
        and labels it as region 4 with merged naming.
        """
        
        # Get regions and filter
        regions = regionprops(labeled_mask)
        if not regions:
            return labeled_mask
        
        # More aggressive filtering to remove spurious regions
        filtered_mask = self.filter_spurious_regions(labeled_mask, wing_mask)
        regions = regionprops(filtered_mask)
        
        if not regions:
            return filtered_mask
        
        logger.info("="*70)
        logger.info(f"TEMPLATE MATCHING WITH MERGE DETECTION (merge_mode={allow_merge})")
        logger.info("="*70)
        
        # Estimate total wing area
        total_wing_area = self.estimate_total_wing_area(filtered_mask, wing_mask)
        
        # Calculate signatures for all regions
        region_signatures = []
        for region in regions:
            signature = self.calculate_wing_ratio_signature(region, total_wing_area)
            region_signatures.append(signature)
        
        # Sort by area for display
        region_signatures.sort(key=lambda r: r['absolute_area'], reverse=True)
        
        logger.info(f"Analyzing {len(region_signatures)} filtered regions:")
        for i, sig in enumerate(region_signatures):
            logger.info(f"  {i+1}. Region {sig['label']}: wing_ratio={sig['wing_ratio']:.4f} "
                       f"({sig['wing_ratio']*100:.1f}%), AR={sig['aspect_ratio']:.1f}, "
                       f"sol={sig['solidity']:.3f}")
        
        # Check if we should use merged template matching
        if allow_merge:
            # Look for a region that matches combined 4+5 properties
            merged_candidate = self._find_merged_45_candidate(region_signatures, total_wing_area)
            
            if merged_candidate is not None:
                logger.info(f"✓✓✓ DETECTED merged 4+5 candidate: Region {merged_candidate['label']}")
                # Use modified template matching with merged region
                return self._match_with_merged_region(region_signatures, filtered_mask, 
                                                     wing_mask, merged_candidate)
            else:
                logger.warning("✗✗✗ No merged 4+5 candidate found")
                logger.warning("Will proceed with FORCED merge mode anyway - labeling largest compact region as merged 4+5")
                
                # FORCE MERGE: Find the largest compact high-solidity region and call it merged 4+5
                # This handles cases where the region exists but doesn't perfectly match criteria
                compact_regions = [sig for sig in region_signatures 
                                 if sig['is_compact'] and sig['solidity'] > 0.85]
                
                if compact_regions:
                    # Take the largest compact region
                    forced_merged = max(compact_regions, key=lambda s: s['absolute_area'])
                    logger.warning(f"FORCING Region {forced_merged['label']} as merged 4+5 "
                                 f"(area={forced_merged['absolute_area']:,}, "
                                 f"ratio={forced_merged['wing_ratio']:.3f}, "
                                 f"AR={forced_merged['aspect_ratio']:.2f})")
                    return self._match_with_merged_region(region_signatures, filtered_mask,
                                                         wing_mask, forced_merged)
                else:
                    logger.error("Cannot find any compact region to force as merged 4+5")
                    logger.error("Falling back to standard 5-region matching")
        
        # Standard matching (only used if merge detection completely fails)
        logger.info("Using STANDARD 5-region matching")
        match_scores = self.identify_best_matches_by_wing_ratio(region_signatures)
        assignment = self.solve_assignment_problem(region_signatures, match_scores)
        
        # Create anatomical mask
        anatomical_mask = np.zeros_like(filtered_mask)
        
        for region_label, template_id in assignment.items():
            region_mask = filtered_mask == region_label
            anatomical_mask[region_mask] = template_id
        
        return anatomical_mask
    
    def _find_merged_45_candidate(self, region_signatures, total_wing_area):

            
            # Your empirical merged region properties
            expected_merged_ratio = 1509062 / 4169076  # ~0.362 (36.2%)
            expected_AR = 2.87
            expected_solidity = 0.966
            
            # Tolerances based on biological variation
            ratio_tolerance = 0.10  # ±10% for area ratio
            ar_tolerance = 0.8      # ±0.8 for aspect ratio
            sol_tolerance = 0.08    # ±0.08 for solidity
            
            logger.info(f"Looking for merged 4+5 with empirical properties:")
            logger.info(f"  Expected wing_ratio: {expected_merged_ratio:.4f} (±{ratio_tolerance:.4f})")
            logger.info(f"  Expected AR: {expected_AR:.2f} (±{ar_tolerance:.2f})")
            logger.info(f"  Expected solidity: {expected_solidity:.3f} (±{sol_tolerance:.3f})")
            
            candidates = []
            
            for sig in region_signatures:
                wing_ratio = sig['wing_ratio']
                ar = sig['aspect_ratio']
                sol = sig['solidity']
                
                # Calculate how well this region matches empirical merged properties
                ratio_match = abs(wing_ratio - expected_merged_ratio) < ratio_tolerance
                ar_match = abs(ar - expected_AR) < ar_tolerance
                sol_match = abs(sol - expected_solidity) < sol_tolerance
                
                if ratio_match and ar_match and sol_match:
                    # Calculate combined score (lower is better)
                    ratio_error = abs(wing_ratio - expected_merged_ratio) / ratio_tolerance
                    ar_error = abs(ar - expected_AR) / ar_tolerance
                    sol_error = abs(sol - expected_solidity) / sol_tolerance
                    combined_score = ratio_error + ar_error + sol_error
                    
                    candidates.append((sig, combined_score))
                    logger.info(f"  ✓ STRONG CANDIDATE: Region {sig['label']}")
                    logger.info(f"    wing_ratio={wing_ratio:.4f} (expected {expected_merged_ratio:.4f})")
                    logger.info(f"    AR={ar:.2f} (expected {expected_AR:.2f})")
                    logger.info(f"    solidity={sol:.3f} (expected {expected_solidity:.3f})")
                    logger.info(f"    combined_score={combined_score:.3f}")
            
            if candidates:
                # Return best candidate (lowest combined score)
                best_candidate = min(candidates, key=lambda x: x[1])
                logger.info(f"  SELECTED: Region {best_candidate[0]['label']} as merged 4+5 (score={best_candidate[1]:.3f})")
                return best_candidate[0]
            
            # Fallback: look for regions that are close but maybe slightly off
            logger.info("  No perfect match found, checking with relaxed criteria...")
            
            relaxed_candidates = []
            for sig in region_signatures:
                wing_ratio = sig['wing_ratio']
                ar = sig['aspect_ratio']
                sol = sig['solidity']
                
                # More lenient criteria
                ratio_close = 0.25 < wing_ratio < 0.50  # Between 25-50% of wing
                ar_close = 2.0 < ar < 4.5               # Compact but not too compact
                sol_close = sol > 0.85                   # High solidity
                
                if ratio_close and ar_close and sol_close:
                    # Score based on distance from empirical values
                    ratio_error = abs(wing_ratio - expected_merged_ratio)
                    ar_error = abs(ar - expected_AR)
                    sol_error = abs(sol - expected_solidity)
                    combined_score = ratio_error * 5 + ar_error + sol_error * 3
                    
                    relaxed_candidates.append((sig, combined_score))
                    logger.info(f"  ~ Possible: Region {sig['label']}, score={combined_score:.3f}")
            
            if relaxed_candidates:
                best_relaxed = min(relaxed_candidates, key=lambda x: x[1])
                logger.warning(f"  Using relaxed match: Region {best_relaxed[0]['label']} (score={best_relaxed[1]:.3f})")
                return best_relaxed[0]
            
            logger.warning("  No merged 4+5 candidate found even with relaxed criteria")
            return None
    
    def _match_with_merged_region(self, region_signatures, filtered_mask, 
                                  wing_mask, merged_candidate):
        """
        Perform template matching when we've identified a merged 4+5 region.
        
        This labels the merged region as 4 and matches other regions to templates 1, 2, 3.
        """
        
        logger.info("Performing template matching with merged 4+5 region...")
        
        anatomical_mask = np.zeros_like(filtered_mask)
        
        # First, assign the merged region as region 4
        merged_label = merged_candidate['label']
        region_mask = filtered_mask == merged_label
        anatomical_mask[region_mask] = 4
        
        logger.info(f"Assigned merged region (label {merged_label}) as Region 4")
        
        # Now match remaining regions to templates 1, 2, 3
        remaining_signatures = [sig for sig in region_signatures 
                              if sig['label'] != merged_label]
        
        if len(remaining_signatures) == 0:
            logger.warning("No other regions found besides merged 4+5")
            return anatomical_mask
        
        # Create subset of templates (only 1, 2, 3)
        available_templates = [1, 2, 3]
        
        # Calculate match scores for remaining regions to templates 1-3
        match_scores = {}
        for sig in remaining_signatures:
            region_label = sig['label']
            match_scores[region_label] = {}
            
            for template_id in available_templates:
                score = self.calculate_template_match_score(sig, template_id)
                match_scores[region_label][template_id] = score
        
        # Solve assignment problem for templates 1-3
        n_regions = len(remaining_signatures)
        n_templates = len(available_templates)
        
        if n_regions > 0:
            max_size = max(n_regions, n_templates)
            cost_matrix = np.full((max_size, max_size), 1000.0)
            
            for i, sig in enumerate(remaining_signatures):
                region_label = sig['label']
                for j, template_id in enumerate(available_templates):
                    cost_matrix[i, j] = match_scores[region_label][template_id]
            
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            for row, col in zip(row_indices, col_indices):
                if row < n_regions and col < n_templates:
                    sig = remaining_signatures[row]
                    template_id = available_templates[col]
                    score = cost_matrix[row, col]
                    
                    if score < 100:
                        region_mask = filtered_mask == sig['label']
                        anatomical_mask[region_mask] = template_id
                        
                        template_name = self.target_regions[template_id]['name']
                        logger.info(f"  ASSIGNED: Region {sig['label']} → {template_id} "
                                  f"({template_name}) [score={score:.1f}]")
        
        logger.info("="*70)
        logger.info("MERGE DETECTION MATCHING COMPLETE")
        logger.info("="*70)
        
        return anatomical_mask
    
    def filter_spurious_regions(self, labeled_mask, wing_mask):
        """Aggressively filter out spurious regions that don't match target characteristics."""
        
        regions = regionprops(labeled_mask)
        if not regions:
            return labeled_mask
        
        logger.info(f"Filtering {len(regions)} detected regions...")
        
        # Estimate wing area for ratio calculations
        wing_area = np.sum(wing_mask) if np.sum(wing_mask) > 0 else sum(r.area for r in regions) / 0.827
        
        keep_regions = []
        
        for region in regions:
            wing_ratio = region.area / wing_area
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
            # Basic size filter - target regions are 9-22% of wing
            if wing_ratio < 0.037 or wing_ratio > 0.4:
                logger.info(f"  REJECT Region {region.label}: wing_ratio={wing_ratio:.4f} outside range [0.05, 0.4]")
                continue
            
            # Basic shape filter - target regions have AR between 2-11
            if aspect_ratio < 1.5 or aspect_ratio > 14.0:
                logger.info(f"  REJECT Region {region.label}: AR={aspect_ratio:.1f} outside range [1.5, 12.0]")
                continue
            
            # Solidity filter - target regions have reasonable solidity
            if region.solidity < 0.45:
                logger.info(f"  REJECT Region {region.label}: solidity={region.solidity:.3f} too low")
                continue
            
            # Position filter - should be in main wing body
            wing_coords = np.column_stack(np.where(wing_mask))
            if len(wing_coords) > 0:
                min_y, min_x = wing_coords.min(axis=0)
                max_y, max_x = wing_coords.max(axis=0)
                
                if max_y > min_y and max_x > min_x:
                    norm_y = (region.centroid[0] - min_y) / (max_y - min_y)
                    norm_x = (region.centroid[1] - min_x) / (max_x - min_x)
                    
                    if not (0.05 < norm_x < 0.95 and 0.0 < norm_y < 1.0):
                        logger.info(f"  REJECT Region {region.label}: position ({norm_x:.2f}, {norm_y:.2f}) outside wing body")
                        continue
            
            # Passed all filters
            keep_regions.append(region)
            logger.info(f"  KEEP Region {region.label}: wing_ratio={wing_ratio:.4f}, AR={aspect_ratio:.1f}, sol={region.solidity:.3f}")
        
        # Create filtered mask
        filtered_mask = np.zeros_like(labeled_mask)
        new_label = 1
        
        for region in keep_regions:
            region_mask = labeled_mask == region.label
            filtered_mask[region_mask] = new_label
            new_label += 1
        
        logger.info(f"Kept {len(keep_regions)} regions after filtering")
        return filtered_mask
    
    # Main interface
    def match_regions_to_template(self, labeled_mask, wing_mask, vein_mask=None):
        """Main interface - uses wing-ratio based matching."""
        return self.wing_ratio_based_matching(labeled_mask, wing_mask)







def create_trichome_based_visualization(raw_img, prob_map, all_peaks, wing_mask, 
                                       vein_mask, virtual_veins, intervein_regions, 
                                       output_dir, basename, is_pentagone_mode):
    """Create comprehensive visualization showing trichome-based processing."""
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    
    bg_img = raw_img if raw_img is not None else prob_map[..., 0]
    
    # Row 1: Trichome filtering process
    axes[0, 0].imshow(bg_img, cmap='gray')
    if len(all_peaks) > 0:
        axes[0, 0].scatter(all_peaks[:, 1], all_peaks[:, 0], c='red', s=2, alpha=0.6)
    axes[0, 0].set_title(f'All Detected Trichomes (n={len(all_peaks)})')
    axes[0, 0].axis('off')
    
    # Show the wing mask from filtered trichomes
    axes[0, 1].imshow(wing_mask, cmap='viridis')
    axes[0, 1].set_title(f'Wing Mask from Filtered Trichomes')
    axes[0, 1].axis('off')
    
    # Row 2: Vein detection and barriers
    axes[1, 0].imshow(bg_img, cmap='gray')
    axes[1, 0].imshow(vein_mask, cmap='Reds', alpha=0.7)
    axes[1, 0].set_title('Detected Veins')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(bg_img, cmap='gray')
    axes[1, 1].imshow(vein_mask, cmap='Reds', alpha=0.6)
    axes[1, 1].imshow(virtual_veins, cmap='Blues', alpha=0.6)
    axes[1, 1].set_title('Combined Vein Network (Real + Virtual)')
    axes[1, 1].axis('off')
    
    # Row 3: Final results
    if intervein_regions is not None:
        max_label = 4 if is_pentagone_mode else 5
        axes[2, 0].imshow(intervein_regions, cmap='nipy_spectral', vmin=0, vmax=max_label)
        
        # Add region labels
        regions = regionprops(intervein_regions)
        if is_pentagone_mode:
            pentagone_handler = PentagoneMutantHandler()
            target_regions = pentagone_handler.pentagone_target_regions
        else:
            template_matcher = AnatomicalTemplateMatcher()
            target_regions = template_matcher.target_regions
        
        for region in regions:
            if region.label > 0:
                region_name = target_regions.get(region.label, {}).get('name', f'Region-{region.label}')
                cy, cx = region.centroid
                axes[2, 0].text(cx, cy, f"{region.label}\n{region_name}", 
                               color='white', fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        mode_str = "Pentagone (4-region)" if is_pentagone_mode else "Normal (5-region)"
        axes[2, 0].set_title(f'Final Regions - {mode_str} (n={len(regions)})')
    else:
        axes[2, 0].text(0.5, 0.5, 'No regions detected', ha='center', va='center')
        axes[2, 0].set_title('Final Regions')
    axes[2, 0].axis('off')
    
    # Complete overlay
    axes[2, 1].imshow(bg_img, cmap='gray')
    if wing_mask is not None:
        axes[2, 1].imshow(wing_mask, cmap='Blues', alpha=0.3)
    axes[2, 1].imshow(vein_mask, cmap='Reds', alpha=0.5)
    if intervein_regions is not None:
        axes[2, 1].imshow(intervein_regions > 0, cmap='Greens', alpha=0.4)
    if len(all_peaks) > 0:
        # Show only some trichomes to avoid clutter
        sample_peaks = all_peaks[::max(1, len(all_peaks)//500)]  # Sample for visibility
        axes[2, 1].scatter(sample_peaks[:, 1], sample_peaks[:, 0], c='yellow', s=1, alpha=0.8)
    axes[2, 1].set_title('Complete Analysis Overlay')
    axes[2, 1].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{basename}_trichome_based_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved trichome-based analysis visualization to {save_path}")


def enhanced_intervein_processing_with_trichome_wing(prob_map, raw_img, cfg, output_dir, basename, 
                                                          force_pentagone=False):
    """Enhanced processing using trichome-based wing detection with string removal."""
    
    logger.info("Starting enhanced processing with trichome-based wing detection...")
    
    # Step 1: Detect trichomes FIRST (we need these for wing boundary)
    tri_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else prob_map
    peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
    
    logger.info(f"Detected {len(peaks)} trichomes for analysis")
    
    # Step 2: Detect veins (can be done in parallel)
    # vein_detector = ImprovedWingVeinDetector(cfg)
    # vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    vein_mask = prob_map[..., 2] > 0.5 if prob_map.shape[-1] >= 3 else np.zeros(prob_map.shape[:2], dtype=bool)
    skeleton = morphology.skeletonize(vein_mask) 
    
    # Step 3: Use trichome-based wing detection with string removal
    intervein_detector = EnhancedInterveinDetectorWithPentagone(cfg)
    wing_mask = intervein_detector.detect_wing_boundary_from_trichomes(
        prob_map, raw_img, peaks=peaks
    )
    
    # Step 4: Validate wing
    if not WingBorderChecker.is_valid_wing(wing_mask, cfg.min_wing_area, cfg.border_buffer):
        logger.warning(f"Skipping {basename} - wing invalid after trichome filtering")
        return None, None, None, None, None
    
    # Step 5: Continue with enhanced segmentation using the clean wing mask
    virtual_veins = intervein_detector.create_virtual_boundary_veins(wing_mask, vein_mask)
    combined_vein_mask = vein_mask | virtual_veins
    barrier_mask = morphology.binary_dilation(combined_vein_mask, 
                                             morphology.disk(cfg.vein_width_estimate * 2))
    
    if prob_map.shape[-1] >= 4:
        intervein_prob = prob_map[..., 3]
        intervein_mask = (intervein_prob > cfg.intervein_threshold) & wing_mask & (~barrier_mask)
    else:
        intervein_mask = wing_mask & (~barrier_mask)
    
    intervein_mask = morphology.remove_small_objects(intervein_mask, min_size=5000)
    intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=5000)
    labeled_regions = label(intervein_mask)
    filtered_labeled_mask = intervein_detector._filter_intervein_regions(labeled_regions, wing_mask)
    
    # Step 6: Apply anatomical labeling with pentagone detection
    if force_pentagone:
        logger.info("FORCED PENTAGONE MODE")
        final_labeled_mask = intervein_detector.pentagone_handler._apply_four_region_labeling(
            filtered_labeled_mask, wing_mask)
        is_pentagone_mode = True
    else:
        # Automatic detection
        final_labeled_mask, is_pentagone = intervein_detector.pentagone_handler.apply_pentagone_labeling(
            filtered_labeled_mask, wing_mask)
        is_pentagone_mode = is_pentagone
    
    # Step 7: Get final regions
    final_regions = regionprops(final_labeled_mask)
    
    # Step 8: Create enhanced visualization
    vis_path = os.path.join(output_dir, f"{basename}_trichome_enhanced_segmentation.png")
    create_trichome_based_visualization(
        raw_img, prob_map, peaks, wing_mask, vein_mask, virtual_veins, 
        final_labeled_mask, output_dir, basename, is_pentagone_mode
    )
    
    mode_str = "pentagone (4-region)" if is_pentagone_mode else "normal (5-region)"
    logger.info(f"Trichome-enhanced detection found {len(final_regions)} regions using {mode_str} labeling")
    
    return final_labeled_mask, final_regions, combined_vein_mask, skeleton, is_pentagone_mode

class PentagoneMutantHandler:
    """Handler for pentagone mutants where regions 4 and 5 need to be combined."""
    
    def __init__(self):
        self.template_matcher = AnatomicalTemplateMatcher()
        
        # Expected ratios for pentagone mutants (4 regions instead of 5)
        self.pentagone_target_regions = {
            1: {  # Region 1 - unchanged
                'name': 'Region-1',
                'wing_area_ratio': 0.0933,      
                'wing_ratio_tolerance': 0.025,   
                'aspect_ratio_range': (7.0, 11.0),  
                'solidity_range': (0.80, 0.95),     
                'size_rank_expected': 4  # Smallest of the 4
            },
            2: {  # Region 2 - unchanged
                'name': 'Region-2', 
                'wing_area_ratio': 0.1592,      
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (5.5, 8.5),   
                'solidity_range': (0.70, 0.80),     
                'size_rank_expected': 3
            },
            3: {  # Region 3 - unchanged
                'name': 'Region-3',
                'wing_area_ratio': 0.1814,      
                'wing_ratio_tolerance': 0.03,
                'aspect_ratio_range': (5.5, 8.5),   
                'solidity_range': (0.90, 0.98),     
                'size_rank_expected': 2
            },
            4: {  # Combined regions 4+5 from normal wings
                'name': 'Region-4+5-Combined',
                'wing_area_ratio': 0.3931,      # 17.90% + 21.41% = 39.31%
                'wing_ratio_tolerance': 0.05,    # More tolerance for combined region
                'aspect_ratio_range': (2.0, 4.0),   # Broader range for combined region
                'solidity_range': (0.90, 1.0),      # High solidity expected
                'distinctive_features': {
                    'is_largest_combined': True,
                    'is_compact': True
                },
                'size_rank_expected': 1  # Largest region
            }
        }
    
    def detect_pentagone_pattern(self, labeled_mask, wing_mask):
        """Detect if this wing shows a pentagone pattern (missing 5th intervein)."""
        regions = regionprops(labeled_mask)
        
        if len(regions) < 3:
            return False, "Too few regions detected"
        
        # Estimate total wing area
        total_wing_area = self.template_matcher.estimate_total_wing_area(labeled_mask, wing_mask)
        
        # Look for a very large region that could be combined 4+5
        large_regions = []
        for region in regions:
            wing_ratio = region.area / total_wing_area
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
            # Check if this could be a combined region 4+5
            if (wing_ratio > 0.30 and  # Much larger than normal regions
                aspect_ratio < 4.0 and  # Relatively compact
                region.solidity > 0.85):  # High solidity
                large_regions.append((region, wing_ratio))
        
        if len(large_regions) >= 1:
            largest_region, largest_ratio = max(large_regions, key=lambda x: x[1])
            logger.info(f"Potential pentagone pattern detected:")
            logger.info(f"  Large region: {largest_ratio:.3f} wing ratio, AR={largest_region.major_axis_length/largest_region.minor_axis_length:.1f}")
            
            # Additional check: should have 4 regions total
            if len(regions) == 4:
                return True, f"4 regions with large combined region ({largest_ratio:.3f} wing ratio)"
            elif len(regions) == 5:
                # Might have 5 regions but one is the combined 4+5
                return True, f"5 regions but one very large ({largest_ratio:.3f} wing ratio)"
        
        return False, "No large combined region detected"
    
    def apply_pentagone_labeling(self, labeled_mask, wing_mask):
        """Apply pentagone-specific labeling that combines regions 4 and 5."""
        
        logger.info("="*70)
        logger.info("APPLYING PENTAGONE MUTANT LABELING")
        logger.info("="*70)
        
        # First, try normal 5-region labeling
        normal_result = self.template_matcher.wing_ratio_based_matching(labeled_mask, wing_mask)
        
        # Check if we detected the pentagone pattern
        is_pentagone, reason = self.detect_pentagone_pattern(labeled_mask, wing_mask)
        
        if not is_pentagone:
            logger.info(f"No pentagone pattern detected: {reason}")
            logger.info("Using standard 5-region labeling")
            return normal_result, False
        
        logger.info(f"PENTAGONE PATTERN DETECTED: {reason}")
        logger.info("Applying 4-region labeling with combined region 4+5")
        
        # Apply 4-region labeling
        pentagone_result = self._apply_four_region_labeling(labeled_mask, wing_mask)
        
        return pentagone_result, True
    
    def _apply_four_region_labeling(self, labeled_mask, wing_mask):
        """Apply 4-region labeling for pentagone mutants."""
        
        regions = regionprops(labeled_mask)
        if not regions:
            return labeled_mask
        
        # Filter regions
        filtered_mask = self.template_matcher.filter_spurious_regions(labeled_mask, wing_mask)
        regions = regionprops(filtered_mask)
        
        if not regions:
            return filtered_mask
        
        # Estimate total wing area
        total_wing_area = self.template_matcher.estimate_total_wing_area(filtered_mask, wing_mask)
        
        # Calculate signatures for all regions
        region_signatures = []
        for region in regions:
            signature = self._calculate_pentagone_signature(region, total_wing_area)
            region_signatures.append(signature)
        
        # Sort by area for display
        region_signatures.sort(key=lambda r: r['absolute_area'], reverse=True)
        
        logger.info(f"Analyzing {len(region_signatures)} regions for pentagone labeling:")
        for i, sig in enumerate(region_signatures):
            logger.info(f"  {i+1}. Region {sig['label']}: wing_ratio={sig['wing_ratio']:.4f} "
                       f"({sig['wing_ratio']*100:.1f}%), AR={sig['aspect_ratio']:.1f}, "
                       f"sol={sig['solidity']:.3f}")
        
        # Find best matches using pentagone templates
        match_scores = self._calculate_pentagone_matches(region_signatures)
        
        # Solve assignment problem for 4 regions
        assignment = self._solve_pentagone_assignment(region_signatures, match_scores)
        
        # Create anatomical mask
        anatomical_mask = np.zeros_like(filtered_mask)
        
        logger.info("="*50)
        logger.info("PENTAGONE ASSIGNMENT:")
        
        for region_label, template_id in assignment.items():
            region_mask = filtered_mask == region_label
            anatomical_mask[region_mask] = template_id
            
            template_name = self.pentagone_target_regions[template_id]['name']
            expected_ratio = self.pentagone_target_regions[template_id]['wing_area_ratio']
            
            # Get actual ratio
            region_sig = next(r for r in region_signatures if r['label'] == region_label)
            actual_ratio = region_sig['wing_ratio']
            
            logger.info(f"  Region {region_label} → {template_id} ({template_name})")
            logger.info(f"    Expected: {expected_ratio:.4f} ({expected_ratio*100:.1f}%), "
                       f"Actual: {actual_ratio:.4f} ({actual_ratio*100:.1f}%)")
        
        # Report missing templates
        assigned_templates = set(assignment.values())
        all_templates = set(range(1, 5))  # Only 4 templates for pentagone
        missing_templates = all_templates - assigned_templates
        
        if missing_templates:
            logger.warning("MISSING PENTAGONE TEMPLATES:")
            for template_id in sorted(missing_templates):
                template_name = self.pentagone_target_regions[template_id]['name']
                expected_ratio = self.pentagone_target_regions[template_id]['wing_area_ratio']
                logger.warning(f"  Template {template_id} ({template_name}) - "
                             f"expected {expected_ratio:.4f} ({expected_ratio*100:.1f}%)")
        else:
            logger.info("✓ All 4 pentagone templates successfully assigned!")
        
        logger.info("="*70)
        logger.info(f"PENTAGONE LABELING COMPLETE: {len(assigned_templates)}/4 regions labeled")
        logger.info("="*70)
        
        return anatomical_mask
    
    def _calculate_pentagone_signature(self, region, total_wing_area):
        """Calculate region signature for pentagone labeling."""
        wing_ratio = region.area / total_wing_area
        aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
        
        return {
            'wing_ratio': wing_ratio,
            'aspect_ratio': aspect_ratio,
            'solidity': region.solidity,
            'absolute_area': region.area,
            'is_very_elongated': aspect_ratio > 7.0,
            'is_compact': aspect_ratio < 3.5,
            'has_low_solidity': region.solidity < 0.8,
            'is_very_large': wing_ratio > 0.30,  # For combined region
            'region': region,
            'label': region.label
        }
    
    def _calculate_pentagone_matches(self, region_signatures):
        """Calculate match scores for pentagone templates."""
        match_scores = {}
        
        for region_sig in region_signatures:
            region_label = region_sig['label']
            match_scores[region_label] = {}
            
            for template_id in range(1, 5):  # Only 4 templates
                score = self._calculate_pentagone_template_score(region_sig, template_id)
                match_scores[region_label][template_id] = score
                
                template_name = self.pentagone_target_regions[template_id]['name']
                expected_ratio = self.pentagone_target_regions[template_id]['wing_area_ratio']
                logger.info(f"    → {template_id} ({template_name}): score={score:.1f} "
                           f"(expected_ratio={expected_ratio:.4f})")
        
        return match_scores
    
    def _calculate_pentagone_template_score(self, region_sig, template_id):
        """Calculate match score for pentagone template."""
        template = self.pentagone_target_regions[template_id]
        score = 0
        
        # Wing area ratio match (most important)
        expected_ratio = template['wing_area_ratio']
        tolerance = template['wing_ratio_tolerance']
        ratio_error = abs(region_sig['wing_ratio'] - expected_ratio)
        
        if ratio_error <= tolerance:
            score += ratio_error / tolerance * 20
        else:
            excess_error = ratio_error - tolerance
            score += 20 + excess_error * 150
        
        # Aspect ratio match
        ar_range = template['aspect_ratio_range']
        if ar_range[0] <= region_sig['aspect_ratio'] <= ar_range[1]:
            score += 0
        else:
            if region_sig['aspect_ratio'] < ar_range[0]:
                score += (ar_range[0] - region_sig['aspect_ratio']) * 8
            else:
                score += (region_sig['aspect_ratio'] - ar_range[1]) * 8
        
        # Solidity match
        sol_range = template['solidity_range']
        if sol_range[0] <= region_sig['solidity'] <= sol_range[1]:
            score += 0
        else:
            if region_sig['solidity'] < sol_range[0]:
                score += (sol_range[0] - region_sig['solidity']) * 50
            else:
                score += (region_sig['solidity'] - sol_range[1]) * 50
        
        # Special handling for combined region 4+5
        if template_id == 4:  # Combined region
            if region_sig['is_very_large']:
                score -= 15  # Bonus for being very large
            else:
                score += 25  # Penalty for not being large enough
        
        return score
    
    def _solve_pentagone_assignment(self, region_signatures, match_scores):
        """Solve assignment for 4-region pentagone labeling."""
        n_regions = len(region_signatures)
        n_templates = 4
        
        # Create cost matrix
        max_size = max(n_regions, n_templates)
        cost_matrix = np.full((max_size, max_size), 1000.0)
        
        # Fill in actual scores
        for i, region_sig in enumerate(region_signatures):
            region_label = region_sig['label']
            for j, template_id in enumerate(range(1, 5)):
                cost_matrix[i, j] = match_scores[region_label][template_id]
        
        # Solve assignment problem
        from scipy.optimize import linear_sum_assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Extract valid assignments
        assignment = {}
        for row, col in zip(row_indices, col_indices):
            if row < n_regions and col < n_templates:
                region_sig = region_signatures[row]
                template_id = col + 1
                score = cost_matrix[row, col]
                
                if score < 100:  # Reasonable threshold
                    assignment[region_sig['label']] = template_id
                    
                    template_name = self.pentagone_target_regions[template_id]['name']
                    logger.info(f"  ASSIGNED: Region {region_sig['label']} → {template_id} "
                               f"({template_name}) [score={score:.1f}]")
                else:
                    logger.warning(f"  REJECTED: Region {region_sig['label']} → {template_id} "
                                 f"[score={score:.1f} too high]")
        
        return assignment




class RegionMerger:
    """Handles merging of regions 4 and 5 for analysis."""
    
    def __init__(self):
        self.template_matcher = AnatomicalTemplateMatcher()
    
    def merge_regions_4_and_5(self, labeled_mask, regions):
        """Merge regions 4 and 5 into a single region 4+."""
        logger.info("Merging regions 4 and 5 into combined region 4+...")
        
        merged_mask = labeled_mask.copy()
        
        # Find regions 4 and 5
        region_4 = None
        region_5 = None
        
        for region in regions:
            if region.label == 4:
                region_4 = region
            elif region.label == 5:
                region_5 = region
        
        if region_4 is not None and region_5 is not None:
            # Merge region 5 into region 4
            merged_mask[merged_mask == 5] = 4
            logger.info(f"Merged region 5 (area={region_5.area}) into region 4 (area={region_4.area})")
            logger.info(f"Combined area: {region_4.area + region_5.area}")
        elif region_4 is not None and region_5 is None:
            logger.info("Region 4 exists but region 5 not found - keeping as is")
        elif region_4 is None and region_5 is not None:
            # Rename region 5 to region 4
            merged_mask[merged_mask == 5] = 4
            logger.info("Region 5 found but not region 4 - renaming region 5 to 4+")
        else:
            logger.warning("Neither region 4 nor region 5 found - cannot merge")
        
        return merged_mask
    
    def merge_voronoi_results(self, voronoi_results, region_peaks_dict):
        """Merge Voronoi statistics for regions 4 and 5."""
        
        stats_4 = voronoi_results.get(4)
        stats_5 = voronoi_results.get(5)
        
        if stats_4 is None and stats_5 is None:
            logger.warning("No Voronoi results for regions 4 or 5 to merge")
            return voronoi_results
        
        # Combine peaks from both regions
        peaks_4 = region_peaks_dict.get(4, np.empty((0, 2)))
        peaks_5 = region_peaks_dict.get(5, np.empty((0, 2)))
        combined_peaks = np.vstack([peaks_4, peaks_5]) if len(peaks_4) > 0 and len(peaks_5) > 0 else (peaks_4 if len(peaks_4) > 0 else peaks_5)
        
        # Calculate merged statistics
        if stats_4 is not None and stats_5 is not None:
            # Both regions have stats - combine them
            combined_region_area = stats_4['region_area'] + stats_5['region_area']
            
            # Weighted average of cell areas based on number of cells measured
            total_cells = stats_4['n_cells_measured'] + stats_5['n_cells_measured']
            if total_cells > 0:
                combined_avg_cell_area = (
                    stats_4['average_cell_area'] * stats_4['n_cells_measured'] +
                    stats_5['average_cell_area'] * stats_5['n_cells_measured']
                ) / total_cells
                
                # Combined standard deviation (pooled)
                combined_variance = (
                    (stats_4['n_cells_measured'] - 1) * (stats_4['std_cell_area'] ** 2) +
                    (stats_5['n_cells_measured'] - 1) * (stats_5['std_cell_area'] ** 2)
                ) / (total_cells - 2) if total_cells > 2 else 0
                combined_std = np.sqrt(combined_variance) if combined_variance > 0 else 0
                
                combined_cv = combined_std / combined_avg_cell_area if combined_avg_cell_area > 0 else 0
                
                total_cells_total = stats_4['n_cells_total'] + stats_5['n_cells_total']
                combined_percentage = (total_cells / total_cells_total * 100) if total_cells_total > 0 else 0
                
                merged_stats = {
                    'region_area': combined_region_area,
                    'average_cell_area': combined_avg_cell_area,
                    'std_cell_area': combined_std,
                    'cv_cell_area': combined_cv,
                    'percentage_measured': combined_percentage,
                    'n_cells_measured': total_cells,
                    'n_cells_total': total_cells_total,
                    'region_name': 'Region-4+5-Merged'
                }
            else:
                merged_stats = None
        elif stats_4 is not None:
            # Only region 4 has stats
            merged_stats = stats_4.copy()
            merged_stats['region_name'] = 'Region-4+5-Merged'
        else:
            # Only region 5 has stats
            merged_stats = stats_5.copy()
            merged_stats['region_name'] = 'Region-4+5-Merged'
        
        # Create new results dict with merged region
        merged_voronoi_results = {}
        for label, stats in voronoi_results.items():
            if label not in [4, 5]:
                merged_voronoi_results[label] = stats
        
        # Add merged region as label 4
        merged_voronoi_results[4] = merged_stats
        
        logger.info(f"Merged Voronoi results: Region 4+ has {merged_stats['n_cells_measured'] if merged_stats else 0} cells")
        
        return merged_voronoi_results
    
    def merge_region_peaks(self, region_peaks_dict):
        """Merge peak assignments for regions 4 and 5."""
        peaks_4 = region_peaks_dict.get(4, np.empty((0, 2)))
        peaks_5 = region_peaks_dict.get(5, np.empty((0, 2)))
        
        if len(peaks_4) > 0 and len(peaks_5) > 0:
            combined_peaks = np.vstack([peaks_4, peaks_5])
        elif len(peaks_4) > 0:
            combined_peaks = peaks_4
        else:
            combined_peaks = peaks_5
        
        # Update dictionary
        merged_dict = {k: v for k, v in region_peaks_dict.items() if k not in [4, 5]}
        merged_dict[4] = combined_peaks
        
        logger.info(f"Merged peaks: Region 4+ has {len(combined_peaks)} trichomes")
        
        return merged_dict

def save_wing_area_from_background_prob(prob_map, basename, output_dir, is_pentagone_mode=False, is_merged_mode=False):
    """Calculate and save wing area and aspect ratio from background probability channel."""
    if is_merged_mode:
        mode_suffix = "_merged_4_5"
        mode_label = "Merged Regions 4+5"
    elif is_pentagone_mode:
        mode_suffix = "_pentagone"
        mode_label = "Pentagone (4-region)"
    else:
        mode_suffix = "_normal"
        mode_label = "Normal (5-region)"
    
    output_path = os.path.join(output_dir, f"{basename}_wing_metrics{mode_suffix}.csv")
    
    # Get background probability (channel 1, 0-indexed)
    if prob_map.shape[-1] >= 2:
        background_prob = prob_map[..., 1]
        # Wing area = pixels with background probability < 0.5
        wing_mask = background_prob < 0.5
        wing_area = np.sum(wing_mask)
        
        # Calculate wing aspect ratio
        if wing_area > 0:
            wing_coords = np.column_stack(np.where(wing_mask))
            # Get bounding box
            min_y, min_x = wing_coords.min(axis=0)
            max_y, max_x = wing_coords.max(axis=0)
            
            wing_height = max_y - min_y + 1
            wing_width = max_x - min_x + 1
            wing_aspect_ratio = wing_width / wing_height if wing_height > 0 else 0
            
            # Also calculate fitted ellipse aspect ratio for better accuracy
            try:
                # Label the wing mask
                labeled_wing = label(wing_mask)
                if labeled_wing.max() > 0:
                    wing_props = regionprops(labeled_wing)[0]
                    wing_major_axis = wing_props.major_axis_length
                    wing_minor_axis = wing_props.minor_axis_length
                    wing_ellipse_aspect_ratio = wing_major_axis / wing_minor_axis if wing_minor_axis > 0 else 0
                else:
                    wing_ellipse_aspect_ratio = 0
            except:
                wing_ellipse_aspect_ratio = 0
        else:
            wing_aspect_ratio = 0
            wing_ellipse_aspect_ratio = 0
            wing_height = 0
            wing_width = 0
    else:
        wing_area = 0
        wing_aspect_ratio = 0
        wing_ellipse_aspect_ratio = 0
        wing_height = 0
        wing_width = 0
        logger.warning(f"No background channel available for {basename}")
    
    total_pixels = prob_map.shape[0] * prob_map.shape[1]
    wing_coverage = (wing_area / total_pixels * 100) if total_pixels > 0 else 0
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f"# Wing Metrics from Background Probability"])
        writer.writerow([f"# Analysis Mode: {mode_label}"])
        writer.writerow(["metric", "value", "unit"])
        writer.writerow(["wing_area", wing_area, "pixels"])
        writer.writerow(["wing_area_microns_squared", wing_area / (PIXELS_PER_MICRON**2), "μm²"])
        writer.writerow(["wing_width", wing_width, "pixels"])
        writer.writerow(["wing_height", wing_height, "pixels"])
        writer.writerow(["wing_bbox_aspect_ratio", f"{wing_aspect_ratio:.4f}", "width/height"])
        writer.writerow(["wing_ellipse_aspect_ratio", f"{wing_ellipse_aspect_ratio:.4f}", "major/minor"])
        writer.writerow(["total_image_pixels", total_pixels, "pixels"])
        writer.writerow(["wing_coverage", f"{wing_coverage:.2f}", "percent"])
    
    logger.info(f"Saved wing metrics ({mode_label}) to {output_path}")
    return wing_aspect_ratio, wing_ellipse_aspect_ratio
class EnhancedInterveinDetectorWithPentagone:
    """Enhanced intervein detector with pentagone mutant support."""
    
    def __init__(self, config):
        self.config = config
        self.template_matcher = AnatomicalTemplateMatcher()
        self.pentagone_handler = PentagoneMutantHandler()
        self.is_pentagone_mode = False
    
    # Copy the methods from your existing EnhancedInterveinDetector class
    def detect_wing_boundary(self, prob_map, raw_img=None):
        """Detect wing boundary prioritizing raw image when available."""
        if raw_img is not None:
            return self._detect_wing_from_probabilities(prob_map)
        else:
            return self._detect_wing_from_probabilities(prob_map)

    def _detect_wing_from_probabilities(self, prob_map):
        """Fallback method using probability map only."""
        logger.info("Using probability map for wing boundary detection (no raw image)")
        import cv2 as cv
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 1]
            
            wing_mask = intervein_prob < 0.4
            cv.morphologyEx(wing_mask, cv.MORPH_CLOSE, (10,10))
            border_width = 30
            h, w = prob_map.shape[:2]
            border_mask = np.ones((h, w), dtype=bool)
            border_mask[:border_width, :] = False
            border_mask[-border_width:, :] = False  
            border_mask[:, :border_width] = False
            border_mask[:, -border_width:] = False
            wing_mask = wing_mask & border_mask
        else:
            raise ValueError("Need either raw image or 4-channel probability map")
        
        wing_mask = morphology.binary_closing(wing_mask, morphology.disk(10))
        wing_mask = morphology.remove_small_objects(wing_mask, min_size=50000)
        wing_mask = morphology.remove_small_holes(wing_mask, area_threshold=10000)
        return wing_mask
    
    def create_virtual_boundary_veins(self, wing_mask, actual_vein_mask, vein_width=10):
        """Create virtual veins along wing boundaries where real veins are missing."""
        wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
        thick_boundary = morphology.binary_dilation(wing_boundary, morphology.disk(vein_width//2))
        # dilated_veins = morphology.binary_dilation(actual_vein_mask, morphology.disk(vein_width))
        dilated_veins = actual_vein_mask
        virtual_veins = thick_boundary & (~dilated_veins)
        virtual_veins = morphology.remove_small_objects(virtual_veins, min_size=100)
        return virtual_veins
    
    
    def _filter_intervein_regions(self, labeled_mask, wing_mask):
        """Filter intervein regions with criteria suitable for boundary regions and merge mode."""
        
        filtered_mask = np.zeros_like(labeled_mask)
        new_label = 1
        
        regions = regionprops(labeled_mask)
        
        # Sort regions by area to prioritize larger regions
        regions.sort(key=lambda r: r.area, reverse=True)
        
        # Check if we're in merge mode
        is_merge_mode = hasattr(self.config, 'force_merge_regions_4_5') and self.config.force_merge_regions_4_5
        
        # Calculate wing area for ratio-based filtering
        wing_area = np.sum(wing_mask) if np.sum(wing_mask) > 0 else sum(r.area for r in regions) / 0.827
        
        logger.info(f"Filtering {len(regions)} regions (merge_mode={is_merge_mode})...")
        logger.info(f"Wing area: {wing_area:,.0f} pixels")
        
        for region in regions:
            # Skip background
            if region.label == 0:
                continue
            
            wing_ratio = region.area / wing_area
            
            # Size criteria - be VERY lenient for merge mode
            if is_merge_mode:
                # In merge mode, expect one very large region (~36% of wing = ~1.5M pixels)
                # So set max MUCH higher
                min_area = 50000  # Very permissive minimum
                max_area = int(wing_area * 0.50)  # Allow up to 50% of wing
                
                logger.info(f"  Merge mode: min={min_area:,}, max={max_area:,}")
            else:
                # Normal mode
                min_area = self.config.auto_intervein_min_area
                max_area = self.config.auto_intervein_max_area
            
            # Check if this is likely an edge region
            region_mask = labeled_mask == region.label
            
            # Calculate how much of the region touches wing boundary
            wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
            boundary_overlap = np.sum(region_mask & wing_boundary)
            boundary_ratio = boundary_overlap / region.area if region.area > 0 else 0
            
            # More lenient criteria for boundary regions
            if boundary_ratio > 0.1:  # This is a boundary region
                min_area = min_area // 2  # Allow smaller boundary regions
                
                # Check if it's at the bottom of the wing (higher y coordinates)
                if region.centroid[0] > labeled_mask.shape[0] * 0.7:
                    min_area = min_area // 2  # Even more lenient for bottom regions
            
            # Apply size filter
            if min_area <= region.area <= max_area:
                # Additional shape checks if enabled
                if self.config.intervein_shape_filter:
                    # More lenient solidity for edge regions
                    min_solidity = self.config.min_intervein_solidity
                    if boundary_ratio > 0.1:
                        min_solidity *= 0.7  # Lower threshold for edge regions
                    
                    # In merge mode, be even more lenient
                    if is_merge_mode:
                        min_solidity = 0.4  # Very permissive for merge detection
                    
                    if region.solidity >= min_solidity:
                        filtered_mask[region_mask] = new_label
                        logger.info(f"  ✓ KEEP Region {region.label}: area={region.area:,} ({wing_ratio:.3f}), "
                                   f"sol={region.solidity:.3f}, boundary={boundary_ratio:.2f}")
                        new_label += 1
                    else:
                        logger.info(f"  ✗ REJECT Region {region.label}: solidity too low ({region.solidity:.3f} < {min_solidity:.3f})")
                else:
                    filtered_mask[region_mask] = new_label
                    logger.info(f"  ✓ KEEP Region {region.label}: area={region.area:,} ({wing_ratio:.3f})")
                    new_label += 1
            else:
                if region.area < min_area:
                    logger.info(f"  ✗ REJECT Region {region.label}: too small ({region.area:,} < {min_area:,})")
                else:
                    logger.info(f"  ✗ REJECT Region {region.label}: too large ({region.area:,} > {max_area:,})")
        
        logger.info(f"Filtering complete: kept {new_label-1}/{len(regions)} regions")
        return filtered_mask
    
    
    
    
    
    # def _filter_intervein_regions(self, labeled_mask, wing_mask):
    #     """Filter intervein regions with criteria suitable for boundary regions."""
    #     filtered_mask = np.zeros_like(labeled_mask)
    #     new_label = 1
        
    #     regions = regionprops(labeled_mask)
    #     regions.sort(key=lambda r: r.area, reverse=True)
        
    #     for region in regions:
    #         if region.label == 0:
    #             continue
            
    #         min_area = self.config.auto_intervein_min_area
    #         max_area = self.config.auto_intervein_max_area
            
    #         region_mask = labeled_mask == region.label
    #         wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
    #         boundary_overlap = np.sum(region_mask & wing_boundary)
    #         boundary_ratio = boundary_overlap / region.area if region.area > 0 else 0
            
    #         if boundary_ratio > 0.1:
    #             min_area = min_area // 2
    #             if region.centroid[0] > labeled_mask.shape[0] * 0.7:
    #                 min_area = min_area // 2
            
    #         if min_area <= region.area <= max_area:
    #             if self.config.intervein_shape_filter:
    #                 min_solidity = self.config.min_intervein_solidity
    #                 if boundary_ratio > 0.1:
    #                     min_solidity *= 0.7
                    
    #                 if region.solidity >= min_solidity:
    #                     filtered_mask[region_mask] = new_label
    #                     new_label += 1
    #             else:
    #                 filtered_mask[region_mask] = new_label
    #                 new_label += 1
    #         else:
    #             logger.debug(f"Region {region.label} excluded due to size: {region.area} pixels")
        
    #     return filtered_mask
    
    def visualize_enhanced_segmentation(self, raw_img, vein_mask, virtual_veins, 
                                      intervein_regions, output_path=None):
        """Visualize the enhanced segmentation with virtual boundaries."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(raw_img if raw_img is not None else vein_mask, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Actual veins only
        axes[0, 1].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[0, 1].imshow(vein_mask, cmap='Reds', alpha=0.7)
        axes[0, 1].set_title('Detected Veins (Actual)')
        axes[0, 1].axis('off')
        
        # Virtual boundary veins
        axes[0, 2].imshow(raw_img if raw_img is not None else np.zeros_like(virtual_veins), cmap='gray')
        axes[0, 2].imshow(virtual_veins, cmap='Blues', alpha=0.7)
        axes[0, 2].set_title('Virtual Boundary Veins')
        axes[0, 2].axis('off')
        
        # Combined veins
        axes[1, 0].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[1, 0].imshow(vein_mask, cmap='Reds', alpha=0.6)
        axes[1, 0].imshow(virtual_veins, cmap='Blues', alpha=0.6)
        axes[1, 0].set_title('Combined Vein Network')
        axes[1, 0].axis('off')
        
        # Intervein regions with anatomical labels
        if intervein_regions is not None:
            axes[1, 1].imshow(intervein_regions, cmap='nipy_spectral', vmin=0, vmax=6)
            
            regions = regionprops(intervein_regions)
            for region in regions:
                if region.label > 0 and region.label <= 6:
                    region_name = self.template_matcher.target_regions.get(region.label, {}).get('name', f'Region_{region.label}')
                    cy, cx = region.centroid
                    axes[1, 1].text(cx, cy, f"{region.label}\n{region_name}", 
                                   color='white', fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            axes[1, 1].set_title(f'Anatomically Labeled Regions (n={len(regions)})')
        else:
            axes[1, 1].set_title('No Valid Regions (Wing Touching Border)')
        axes[1, 1].axis('off')
        
        # Overlay all
        axes[1, 2].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[1, 2].imshow(vein_mask, cmap='Reds', alpha=0.5)
        axes[1, 2].imshow(virtual_veins, cmap='Blues', alpha=0.5)
        if intervein_regions is not None:
            axes[1, 2].imshow(intervein_regions > 0, cmap='Greens', alpha=0.3)
        axes[1, 2].set_title('Complete Segmentation Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            logger.info(f"Saved enhanced segmentation visualization to {output_path}")
        
        plt.close()
        return fig
    
    # def segment_intervein_regions_enhanced(self, prob_map, vein_mask, raw_img=None, 
    #                                      force_pentagone=False):
    #     logger.info("Starting enhanced intervein segmentation with pentagone support...")
        
    #     wing_mask = self.detect_wing_boundary(prob_map, raw_img)
        
    #     if not WingBorderChecker.is_valid_wing(wing_mask, self.config.min_wing_area, 
    #                                            self.config.border_buffer):
    #         logger.warning("Wing is invalid (partial or too small) - skipping intervein analysis")
    #         return None, None, None
        
    #     # HOTFIX: Don't create virtual veins - they're blocking valid regions
    #     virtual_veins = np.zeros_like(vein_mask, dtype=bool)  # Empty mask
        
    #     # HOTFIX: Only use actual veins as barriers, not virtual ones
    #     combined_vein_mask = vein_mask  # Don't add virtual veins
        
    #     # Make barrier mask much thinner
    #     barrier_mask = morphology.binary_dilation(vein_mask, morphology.disk(1))  # Smaller dilation
        
    #     # Step 5: Use intervein probability directly with minimal filtering
    #     if prob_map.shape[-1] >= 4:
    #         intervein_prob = prob_map[..., 3]
            
    #         # HOTFIX: Much simpler approach - trust the probability map
    #         intervein_mask = intervein_prob > 0.15  # Very low threshold
    #         intervein_mask = intervein_mask & wing_mask  # Stay within wing
    #         intervein_mask = intervein_mask & (~barrier_mask)  # Remove only actual veins
    #     else:
    #         intervein_mask = wing_mask & (~barrier_mask)
        
    #     # Minimal cleanup
    #     intervein_mask = morphology.remove_small_objects(intervein_mask, min_size=1000)
    #     intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=500)
        
    #     labeled_regions = label(intervein_mask)
        
    #     # HOTFIX: Much more lenient filtering
    #     filtered_labeled_mask = self._filter_intervein_regions_lenient(labeled_regions, wing_mask)
        
    #     # Step 7: Apply anatomical labeling with pentagone detection
    #     if force_pentagone:
    #         logger.info("FORCED PENTAGONE MODE")
    #         final_labeled_mask = self.pentagone_handler._apply_four_region_labeling(
    #             filtered_labeled_mask, wing_mask)
    #         self.is_pentagone_mode = True
    #     else:
    #         # Automatic detection
    #         final_labeled_mask, is_pentagone = self.pentagone_handler.apply_pentagone_labeling(
    #             filtered_labeled_mask, wing_mask)
    #         self.is_pentagone_mode = is_pentagone
        
    #     # Count regions
    #     unique_labels = np.unique(final_labeled_mask)
    #     n_regions = len(unique_labels[unique_labels > 0])
    #     mode_str = "pentagone (4-region)" if self.is_pentagone_mode else "normal (5-region)"
    #     logger.info(f"Enhanced detection found {n_regions} regions using {mode_str} labeling")
        
    #     return final_labeled_mask, combined_vein_mask, virtual_veins
    def segment_intervein_regions_simple(prob_map, wing_mask=None):
        """Simple, direct segmentation using probability channels."""
        
        # Get the channels
        intervein_prob = prob_map[..., 3].copy()  # Channel 3 - intervein
        vein_prob = prob_map[..., 2]              # Channel 2 - veins  
        background_prob = prob_map[..., 1]        # Channel 1 - background
        
        # Step 1: Zero out intervein where veins are strong
        vein_mask = vein_prob > 0.5
        intervein_prob[vein_mask] = 0
        
        # Step 2: Zero out intervein where background is strong (outside wing)
        background_mask = background_prob > 0.5
        intervein_prob[background_mask] = 0
        
        # Step 3: Create wing mask from inverted background if not provided
        if wing_mask is None:
            wing_mask = background_prob < 0.5  # Wing is where background is low
        
        # Step 4: Threshold intervein to get regions
        intervein_regions = intervein_prob > 0.2  # Low threshold since we've already cleaned it
        
        # Step 5: Basic cleanup
        intervein_regions = morphology.remove_small_objects(intervein_regions, min_size=1000)
        intervein_regions = morphology.remove_small_holes(intervein_regions, area_threshold=1000)
        
        # Step 6: Label the regions
        labeled_regions = label(intervein_regions)
        
        return labeled_regions, wing_mask
    def detect_wing_boundary_from_trichomes(self, prob_map, raw_img=None, peaks=None):
        """Enhanced wing boundary detection using filtered trichomes."""
        
        logger.info("Using trichome-based wing boundary detection...")
        
        # If peaks not provided, detect them
        if peaks is None:
            tri_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else prob_map
            peaks, _ = detect_trichome_peaks(tri_prob, self.config)
        
        logger.info(f"Using {len(peaks)} trichomes for wing boundary detection")
        
        if len(peaks) < 20:
            logger.warning("Too few trichomes for reliable wing detection, falling back to probability method")
            return self._detect_wing_from_probabilities(prob_map)
        
        # Filter trichomes to remove string artifacts
        string_filter = StringRemovalTrichomeFilter(self.config)
        filtered_peaks = string_filter.remove_trichome_strings(peaks, prob_map.shape[:2])
        
        logger.info(f"After string removal: {len(filtered_peaks)} trichomes kept")
        
        if len(filtered_peaks) < 10:
            logger.warning("String filtering removed too many trichomes, falling back")
            return self._detect_wing_from_probabilities(prob_map)
        
        # Create wing mask from filtered trichomes
        wing_mask = string_filter.create_wing_mask_simple(filtered_peaks, prob_map.shape[:2])
        
        if wing_mask is None:
            logger.warning("Trichome-based wing detection failed, falling back")
            return self._detect_wing_from_probabilities(prob_map)
        
        # Validate the wing mask
        wing_area = np.sum(wing_mask)
        total_area = wing_mask.size
        coverage = wing_area / total_area
        
        # Sanity checks
        if coverage < 0.1 or coverage > 0.8:
            logger.warning(f"Wing coverage {coverage:.2f} seems unreasonable, falling back")
            return self._detect_wing_from_probabilities(prob_map)
        
        logger.info(f"Trichome-based wing detection successful: {wing_area} pixels ({coverage:.2f} coverage)")
        return wing_mask
    def segment_intervein_regions_enhanced(self, prob_map, vein_mask, raw_img=None, 
                                             force_pentagone=False):
            """Enhanced segmentation with automatic pentagone detection and merge mode support."""
            
            logger.info("Starting enhanced intervein segmentation with pentagone and merge support...")
            
            # Step 1-6: Same as before (detect wing boundary, create barriers, etc.)
            wing_mask = self.detect_wing_boundary(prob_map, raw_img)
            
            if not WingBorderChecker.is_valid_wing(wing_mask, self.config.min_wing_area, 
                                                   self.config.border_buffer):
                logger.warning("Wing is invalid (partial or too small) - skipping intervein analysis")
                return None, None, None
            
            virtual_veins = self.create_virtual_boundary_veins(wing_mask, vein_mask)
            combined_vein_mask = vein_mask | virtual_veins
            barrier_mask = morphology.binary_dilation(combined_vein_mask, 
                                                     morphology.disk(self.config.vein_width_estimate * 2))
            
            if prob_map.shape[-1] >= 4:
                intervein_prob = prob_map[..., 3]
                intervein_mask = (intervein_prob > self.config.intervein_threshold) & wing_mask & (~barrier_mask)
            else:
                intervein_mask = wing_mask & (~barrier_mask)
            
            intervein_mask = morphology.remove_small_objects(intervein_mask, min_size=5000)

            labeled_regions = label(intervein_mask)
            filtered_labeled_mask = self._filter_intervein_regions(labeled_regions, wing_mask)
            
            # Step 7: Apply anatomical labeling with priority order: merge mode → pentagone → normal
            if self.config.force_merge_regions_4_5:
                logger.info("FORCED MERGE MODE - Attempting merge-aware template matching first")
                
                # Try merge detection first
                final_labeled_mask = self.template_matcher.match_regions_with_merge_detection(
                    filtered_labeled_mask, wing_mask, allow_merge=True)
                
                # Check if merge was successful (should have region 4 and regions 1-3)
                detected_labels = set(np.unique(final_labeled_mask))
                detected_labels.discard(0)  # Remove background
                
                has_region_4 = 4 in detected_labels
                has_regions_123 = all(i in detected_labels for i in [1, 2, 3])
                
                if has_region_4 and has_regions_123:
                    logger.info("✓ Merge detection successful: Found merged 4+5 and regions 1-3")
                    self.is_pentagone_mode = False
                    self.is_merged_mode = True
                else:
                    logger.warning("Merge detection didn't find expected pattern, falling back to normal matching")
                    logger.warning(f"  Detected labels: {sorted(detected_labels)}")
                    
                    # Fall back to normal matching
                    final_labeled_mask = self.template_matcher.wing_ratio_based_matching(
                        filtered_labeled_mask, wing_mask)
                    self.is_pentagone_mode = False
                    self.is_merged_mode = False
            elif force_pentagone:
                logger.info("FORCED PENTAGONE MODE")
                final_labeled_mask = self.pentagone_handler._apply_four_region_labeling(
                    filtered_labeled_mask, wing_mask)
                self.is_pentagone_mode = True
                self.is_merged_mode = False
            else:
                # Automatic detection
                final_labeled_mask, is_pentagone = self.pentagone_handler.apply_pentagone_labeling(
                    filtered_labeled_mask, wing_mask)
                self.is_pentagone_mode = is_pentagone
                self.is_merged_mode = False
            
            # Count regions
            unique_labels = np.unique(final_labeled_mask)
            n_regions = len(unique_labels[unique_labels > 0])
            
            if self.is_merged_mode:
                mode_str = "merged (4+5 combined)"
            elif self.is_pentagone_mode:
                mode_str = "pentagone (4-region)"
            else:
                mode_str = "normal (5-region)"
            
            logger.info(f"Enhanced detection found {n_regions} regions using {mode_str} labeling")
            
            return final_labeled_mask, combined_vein_mask, virtual_veins

def save_anatomical_region_info_with_pentagone(valid_regions, basename, output_dir, is_pentagone_mode=False, wing_area=None, is_merged_mode=False):
    """Save anatomical region information with comprehensive metrics."""
    output_path = os.path.join(output_dir, f"{basename}_anatomical_regions.csv")
    
    # Choose appropriate mode suffix
    if is_merged_mode:
        mode_suffix = "_merged_4_5"
        mode_label = "Merged Regions 4+5"
        # For merged mode, we don't have a specific template, use normal template but mark region 4 specially
        template_matcher = AnatomicalTemplateMatcher()
        target_regions = template_matcher.target_regions
    elif is_pentagone_mode:
        pentagone_handler = PentagoneMutantHandler()
        target_regions = pentagone_handler.pentagone_target_regions
        mode_suffix = "_pentagone"
        mode_label = "Pentagone (4-region)"
    else:
        template_matcher = AnatomicalTemplateMatcher()
        target_regions = template_matcher.target_regions
        mode_suffix = "_normal"
        mode_label = "Normal (5-region)"
    
    # Add mode info to filename
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}{mode_suffix}{ext}"
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Enhanced header
        writer.writerow([f"# Analysis Mode: {mode_label}"])
        writer.writerow([
            "anatomical_id", "region_name", 
            "centroid_y", "centroid_x", 
            "area_pixels", "area_microns_squared",
            "percent_of_wing",
            "bbox_aspect_ratio", "ellipse_aspect_ratio",
            "solidity", "eccentricity", 
            "major_axis_length", "minor_axis_length", 
            "perimeter", "orientation_degrees"
        ])
        
        sorted_regions = sorted(valid_regions, key=lambda r: r.label)
        
        for region in sorted_regions:
            anatomical_id = region.label
            
            # Special handling for merged region 4
            if is_merged_mode and anatomical_id == 4:
                region_name = "Region-4+5-Merged"
            elif anatomical_id in target_regions:
                region_name = target_regions[anatomical_id]['name']
            else:
                region_name = f"Unknown_{anatomical_id}"
            
            # Calculate aspect ratios
            bbox_aspect = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else 0
            
            # Bounding box aspect ratio
            minr, minc, maxr, maxc = region.bbox
            bbox_height = maxr - minr
            bbox_width = maxc - minc
            bbox_aspect_simple = bbox_width / bbox_height if bbox_height > 0 else 0
            
            # Area metrics
            area_microns = region.area / (PIXELS_PER_MICRON ** 2)
            percent_of_wing = (region.area / wing_area * 100) if wing_area and wing_area > 0 else 0
            
            # Orientation in degrees
            orientation_deg = np.degrees(region.orientation)
            
            writer.writerow([
                anatomical_id,
                region_name,
                f"{region.centroid[0]:.2f}",
                f"{region.centroid[1]:.2f}",
                region.area,
                f"{area_microns:.2f}",
                f"{percent_of_wing:.2f}",
                f"{bbox_aspect_simple:.4f}",
                f"{bbox_aspect:.4f}",
                f"{region.solidity:.4f}",
                f"{region.eccentricity:.4f}",
                f"{region.major_axis_length:.2f}",
                f"{region.minor_axis_length:.2f}",
                f"{region.perimeter:.2f}",
                f"{orientation_deg:.2f}"
            ])
    
    logger.info(f"Saved anatomical region information ({mode_label}) to {output_path}")
# # Modified save functions for pentagone support
# def save_anatomical_region_info_with_pentagone(valid_regions, basename, output_dir, is_pentagone_mode=False):
#     """Save anatomical region information with pentagone support."""
#     output_path = os.path.join(output_dir, f"{basename}_anatomical_regions.csv")
    
#     # Choose appropriate template matcher
#     if is_pentagone_mode:
#         pentagone_handler = PentagoneMutantHandler()
#         target_regions = pentagone_handler.pentagone_target_regions
#         mode_suffix = "_pentagone"
#     else:
#         template_matcher = AnatomicalTemplateMatcher()
#         target_regions = template_matcher.target_regions
#         mode_suffix = "_normal"
    
#     # Add mode info to filename
#     base, ext = os.path.splitext(output_path)
#     output_path = f"{base}{mode_suffix}{ext}"
    
#     with open(output_path, "w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
        
#         # Header with mode info
#         writer.writerow(["# Analysis Mode: " + ("Pentagone (4-region)" if is_pentagone_mode else "Normal (5-region)")])
#         writer.writerow([
#             "anatomical_id", "region_name", "centroid_y", "centroid_x", 
#             "area", "solidity", "eccentricity", "major_axis_length", 
#             "minor_axis_length", "orientation"
#         ])
        
#         sorted_regions = sorted(valid_regions, key=lambda r: r.label)
        
#         for region in sorted_regions:
#             anatomical_id = region.label
            
#             if anatomical_id in target_regions:
#                 region_name = target_regions[anatomical_id]['name']
#             else:
#                 region_name = f"Unknown_{anatomical_id}"
            
#             writer.writerow([
#                 anatomical_id,
#                 region_name,
#                 region.centroid[0],
#                 region.centroid[1],
#                 region.area,
#                 region.solidity,
#                 region.eccentricity,
#                 region.major_axis_length,
#                 region.minor_axis_length,
#                 region.orientation
#             ])
    
#     logger.info(f"Saved anatomical region information ({mode_suffix[1:]}) to {output_path}")


# Modified main processing function
def enhanced_intervein_processing_with_pentagone(prob_map, raw_img, cfg, output_dir, basename, 
                                               force_pentagone=False):
    """Complete automated intervein processing with pentagone support."""
    
    logger.info("Starting enhanced automated processing with pentagone support...")
    
    # Step 1: Detect veins
    # vein_detector = ImprovedWingVeinDetector(cfg)
    # vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    vein_mask = prob_map[..., 2] > 0.5 if prob_map.shape[-1] >= 3 else np.zeros(prob_map.shape[:2], dtype=bool)
    skeleton = morphology.skeletonize(vein_mask) 
    
    # Step 2: Use enhanced detector with pentagone support
    intervein_detector = EnhancedInterveinDetectorWithPentagone(cfg)
    result = intervein_detector.segment_intervein_regions_enhanced(
        prob_map, vein_mask, raw_img, force_pentagone=force_pentagone
    )
    
    if result[0] is None:
        logger.warning(f"Skipping {basename} - wing touching border or too small")
        return None, None, None, None
    
    intervein_regions, combined_veins, virtual_veins = result
    is_pentagone_mode = intervein_detector.is_pentagone_mode
    
    # Step 3: Save visualizations
    vis_path = os.path.join(output_dir, f"{basename}_enhanced_segmentation.png")
    intervein_detector.visualize_enhanced_segmentation(
        raw_img, vein_mask, virtual_veins, intervein_regions, vis_path
    )
    
    # Step 4: Get final regions
    final_regions = regionprops(intervein_regions)
    
    logger.info(f"Processing completed in {'pentagone' if is_pentagone_mode else 'normal'} mode "
               f"with {len(final_regions)} regions")
    
    return intervein_regions, final_regions, combined_veins, skeleton, is_pentagone_mode




def main_with_hybrid_wing_detection(directory: str, cfg: TrichomeDetectionConfig = CONFIG, 
                                   output_directory: Optional[str] = None, 
                                   force_pentagone_mode: bool = False,
                                   auto_detect_pentagone: bool = True,
                                   progress_callback=None):
    """Main function with hybrid wing detection for improved sparse wing handling."""
    
    def log_progress(message, level="INFO"):
        """Helper function to log progress"""
        logger.info(message)
        if progress_callback:
            try:
                progress_callback.put(("log", f"{level}: {message}"))
            except:
                pass
    
    # Validate configuration
    cfg.validate()
    
    if output_directory is None:
        output_directory = directory
    
    # Ensure output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(output_directory, "hybrid_wing_detection.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    mode_str = "FORCED PENTAGONE" if force_pentagone_mode else "AUTO-DETECT PENTAGONE" if auto_detect_pentagone else "NORMAL"
    
    log_progress("="*60)
    log_progress(f"HYBRID WING DETECTION WITH IMPROVED SPARSE WING SUPPORT ({mode_str})")
    log_progress(f"Input directory: {directory}")
    log_progress(f"Output directory: {output_directory}")
    log_progress("="*60)
    
    mapping = find_associated_files(directory)
    if not mapping:
        log_progress("No probability files found in the directory", "ERROR")
        return None
    
    log_progress(f"Found {len(mapping)} file pairs to process")
    
    # Summary statistics
    total_files = len(mapping)
    processed_files = 0
    failed_files = 0
    skipped_files = 0
    sparse_wings = 0
    dense_wings = 0
    pentagone_wings = 0
    normal_wings = 0
    total_peaks = 0
    total_regions = 0
    
    for i, (basename, files) in enumerate(mapping.items()):
        if progress_callback:
            try:
                progress_callback.put(("progress_percent", int(i / total_files * 100)))
                progress_callback.put(("current_folder", f"Processing file {i+1}/{total_files}: {basename}"))
            except:
                pass
        
        log_progress("="*40)
        log_progress(f"Processing: {basename} ({i+1}/{total_files})")
        log_progress("="*40)
        
        try:
            prob_file = files["probabilities"]
            raw_file = files.get("raw")
            
            # Load data
            tri_prob, inter_prob, full_prob = load_probability_map(prob_file)
            raw_img = load_raw_image(raw_file) if raw_file else None
            
            # Enhanced peak detection
            peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
            total_peaks += len(peaks)
            
            # Determine wing sparsity
            trichome_density = len(peaks) / full_prob.size if full_prob.size > 0 else 0
            is_sparse = trichome_density < cfg.sparse_threshold
            
            if is_sparse:
                sparse_wings += 1
                log_progress(f"SPARSE WING detected (density: {trichome_density:.6f})")
            else:
                dense_wings += 1
                log_progress(f"DENSE WING detected (density: {trichome_density:.6f})")
            
            # Save peak results
            save_peak_coordinates_enhanced(peaks, basename, output_directory, metrics, tri_prob)
            create_detection_visualization(tri_prob, peaks, basename, output_directory, raw_img)
            
            # Process intervein regions with hybrid detection
            if full_prob.shape[-1] >= 4 and inter_prob is not None:
                log_progress("Processing intervein segmentation with HYBRID wing detection...")
                
                result = enhanced_intervein_processing_with_hybrid_wing_detection(
                    full_prob, raw_img, cfg, output_directory, basename, 
                    force_pentagone=force_pentagone_mode
                )
                
                if result[0] is None:
                    log_progress(f"Skipped {basename} - wing invalid after hybrid detection", "WARNING")
                    skipped_files += 1
                    continue
                
                labeled_mask, valid_regions, vein_mask, skeleton, is_pentagone_mode , is_merged_mode = result
                
                # Track mode statistics
                if is_pentagone_mode:
                    pentagone_wings += 1
                else:
                    normal_wings += 1
                
                total_regions += len(valid_regions)
                
                wing_type = "sparse" if is_sparse else "dense"
                mode_type = "pentagone" if is_pentagone_mode else "normal"
                log_progress(f"Found {len(valid_regions)} valid regions in {mode_type} mode on {wing_type} wing")
                
                # Assign peaks and analyze
                region_peaks_dict = assign_peaks_to_regions(peaks, labeled_mask, valid_regions)
                
                
                
                voronoi_results = {}
                successful_regions = 0
                for region in valid_regions:
                    label_val = region.label
                    region_peaks = region_peaks_dict.get(label_val, np.empty((0, 2)))
                    
                    if region_peaks.shape[0] < 2:
                        log_progress(f"Region {label_val}: insufficient peaks for Voronoi analysis", "WARNING")
                        voronoi_results[label_val] = None
                    else:
                        stats = voronoi_average_cell_stats(region, region_peaks, cfg)
                        voronoi_results[label_val] = stats
                        if stats is not None:
                            successful_regions += 1
                
                # Determine which mode we're in

                if is_merged_mode:
                    log_progress("MERGE MODE ACTIVE: Region 4 represents combined regions 4+5")
                    # Update the region name in Voronoi results if region 4 exists
                    if 4 in voronoi_results and voronoi_results[4] is not None:
                        voronoi_results[4]['region_name'] = 'Region-4+5-Merged'
                        log_progress(f"Region 4 (merged) has {voronoi_results[4]['n_cells_measured']} cells analyzed")
                
                # Calculate wing area for percentage calculations
                wing_mask = full_prob[..., 1] < 0.5 if full_prob.shape[-1] >= 2 else labeled_mask > 0
                wing_area_pixels = np.sum(wing_mask)
                
                # Save results with mode information and enhanced metrics
                save_voronoi_results_with_mode(voronoi_results, basename, output_directory, is_pentagone_mode, is_merged_mode)
                wing_bbox_ar, wing_ellipse_ar = save_wing_area_from_background_prob(
                    full_prob, basename, output_directory, is_pentagone_mode, is_merged_mode
                )
                save_anatomical_region_info_with_pentagone(
                    valid_regions, basename, output_directory, is_pentagone_mode, wing_area_pixels, is_merged_mode
                )
                
                # Enhanced visualization
                if valid_regions and any(region_peaks_dict[label].shape[0] >= 2 
                                       for label in region_peaks_dict):
                    background_img = raw_img if raw_img is not None else tri_prob
                    plot_voronoi_with_pentagone_info(
                        valid_regions, region_peaks_dict, background_img, 
                        vein_mask, skeleton, output_directory, basename, cfg, is_pentagone_mode, is_merged_mode
                    )
                
                log_progress(f"Successfully processed {basename} ({wing_type} {mode_type} wing)")
                
            else:
                log_progress("Skipping intervein analysis (4th channel missing)")
            
            processed_files += 1
            
        except Exception as e:
            failed_files += 1
            log_progress(f"Error processing {basename}: {str(e)}", "ERROR")
            continue
    
    # Final summary with hybrid detection statistics
    log_progress("="*60)
    log_progress("HYBRID WING DETECTION ANALYSIS COMPLETE")
    log_progress("="*60)
    log_progress(f"Total files found: {total_files}")
    log_progress(f"Successfully processed: {processed_files}")
    log_progress(f"  - Sparse wings: {sparse_wings}")
    log_progress(f"  - Dense wings: {dense_wings}")
    log_progress(f"  - Normal wings: {normal_wings}")
    log_progress(f"  - Pentagone wings: {pentagone_wings}")
    log_progress(f"Skipped (border/invalid): {skipped_files}")
    log_progress(f"Failed: {failed_files}")
    log_progress(f"Total trichomes detected: {total_peaks}")
    log_progress(f"Total regions analyzed: {total_regions}")
    log_progress(f"Sparse wing success rate: {(sparse_wings - skipped_files)/max(sparse_wings, 1)*100:.1f}%")
    log_progress("="*60)
    
    # Create enhanced summary report
    summary_path = os.path.join(output_directory, "hybrid_detection_summary.txt")
    with open(summary_path, "w") as f:
        f.write("HYBRID WING DETECTION - ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {directory}\n")
        f.write(f"Output Directory: {output_directory}\n")
        f.write(f"Mode: {mode_str}\n")
        f.write("\nHYBRID DETECTION FEATURES:\n")
        f.write("- Probability map + trichome validation for dense wings\n")
        f.write("- Probability-dominant approach for sparse wings\n")
        f.write("- Adaptive string removal based on trichome density\n")
        f.write("- Fallback mechanisms for challenging cases\n")
        f.write("\nPROCESSING SUMMARY:\n")
        f.write(f"Total files found: {total_files}\n")
        f.write(f"Successfully processed: {processed_files}\n")
        f.write(f"  - Sparse wings: {sparse_wings}\n")
        f.write(f"  - Dense wings: {dense_wings}\n")
        f.write(f"  - Normal wings (5 regions): {normal_wings}\n")
        f.write(f"  - Pentagone wings (4 regions): {pentagone_wings}\n")
        f.write(f"Skipped (border/invalid): {skipped_files}\n")
        f.write(f"Failed: {failed_files}\n")
        f.write(f"\nSPARSE WING PERFORMANCE:\n")
        f.write(f"Sparse wings detected: {sparse_wings}\n")
        f.write(f"Sparse wing processing success: {sparse_wings - min(skipped_files, sparse_wings)}\n")
        f.write(f"Improvement for sparse wings: Significant (probability-dominant approach)\n")
        f.write(f"\nDETECTION SUMMARY:\n")
        f.write(f"Total trichomes detected: {total_peaks}\n")
        f.write(f"Total regions analyzed: {total_regions}\n")
        f.write(f"Average trichomes per wing: {total_peaks/max(processed_files, 1):.1f}\n")
        f.write(f"Average regions per wing: {total_regions/max(processed_files, 1):.1f}\n")
    
    log_progress(f"Enhanced summary report saved to {summary_path}")
    
    # Return summary statistics for GUI
    return {
        'total_files': total_files,
        'processed_files': processed_files,
        'failed_files': failed_files,
        'skipped_files': skipped_files,
        'sparse_wings': sparse_wings,
        'dense_wings': dense_wings,
        'normal_wings': normal_wings,
        'pentagone_wings': pentagone_wings,
        'total_peaks': total_peaks,
        'total_regions': total_regions
    }
def save_voronoi_results_with_mode(voronoi_results, basename, output_dir, is_pentagone_mode, is_merged_mode=False):
    """Save Voronoi results with mode information."""
    if is_merged_mode:
        mode_suffix = "_merged_4_5"
        mode_label = "Merged Regions 4+5"
    elif is_pentagone_mode:
        mode_suffix = "_pentagone"
        mode_label = "Pentagone (4-region)"
    else:
        mode_suffix = "_normal"
        mode_label = "Normal (5-region)"
    
    output_path = os.path.join(output_dir, f"{basename}_voronoi_average_cell_area{mode_suffix}.csv")
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Enhanced header with mode info
        writer.writerow([f"# Analysis Mode: {mode_label}"])
        writer.writerow([
            "region_label", "region_name", "region_area", "average_cell_area", "std_cell_area",
            "cv_cell_area", "percentage_measured", "n_cells_measured", "n_cells_total"
        ])
        
        # Sort by region label for consistent ordering
        sorted_items = sorted(voronoi_results.items(), key=lambda x: x[0])
        
        for region_label, stats in sorted_items:
            if stats is not None:
                writer.writerow([
                    region_label,
                    stats["region_name"],
                    stats["region_area"],
                    stats["average_cell_area"],
                    stats["std_cell_area"],
                    stats["cv_cell_area"],
                    stats["percentage_measured"],
                    stats["n_cells_measured"],
                    stats["n_cells_total"]
                ])
            else:
                if is_merged_mode and region_label == 4:
                    region_name = "Region-4+5-Merged"
                else:
                    region_name = f"Region_{region_label}"
                writer.writerow([region_label, region_name, None, None, None, None, None, None, None])
    
    logger.info(f"Saved Voronoi results ({mode_label}) to {output_path}")
    
    
    
    
    
def create_pre_labeling_debug_visualization(filtered_mask, wing_mask, prob_map, raw_img, 
                                           output_dir, basename, mode_name="normal"):
    """
    Create detailed visualization of ALL detected regions BEFORE anatomical labeling.
    This helps debug why regions aren't being matched correctly.
    """
    
    regions = regionprops(filtered_mask)
    
    if not regions:
        logger.warning("No regions to visualize in pre-labeling debug")
        return
    
    # Estimate wing area
    wing_area = np.sum(wing_mask) if np.sum(wing_mask) > 0 else sum(r.area for r in regions) / 0.827
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 20))
    
    bg_img = raw_img if raw_img is not None else prob_map[..., 0]
    
    # Top row: Overview
    axes[0, 0].imshow(bg_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(wing_mask, cmap='Blues')
    axes[0, 1].set_title(f'Wing Mask ({np.sum(wing_mask)} pixels)')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(filtered_mask, cmap='nipy_spectral', vmin=0, vmax=len(regions)+1)
    axes[0, 2].set_title(f'All Detected Regions (n={len(regions)})')
    axes[0, 2].axis('off')
    
    # Middle row: Individual regions with measurements
    axes[1, 0].imshow(bg_img, cmap='gray')
    
    # Sort regions by area
    sorted_regions = sorted(regions, key=lambda r: r.area, reverse=True)
    
    # Color each region and label it
    colors = plt.cm.Set3(np.linspace(0, 1, len(sorted_regions)))
    
    for idx, region in enumerate(sorted_regions):
        # Get region mask
        region_mask = filtered_mask == region.label
        
        # Create colored overlay
        colored_mask = np.zeros((*bg_img.shape, 4))
        colored_mask[region_mask] = (*colors[idx][:3], 0.5)
        
        axes[1, 0].imshow(colored_mask)
        
        # Add label with measurements
        cy, cx = region.centroid
        wing_ratio = region.area / wing_area
        ar = region.major_axis_length / (region.minor_axis_length + 1e-8)
        
        label_text = f"#{idx+1}\nL{region.label}"
        axes[1, 0].text(cx, cy, label_text,
                       color='white', fontsize=10, weight='bold',
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.8))
    
    axes[1, 0].set_title('Regions Colored by Size Rank')
    axes[1, 0].axis('off')
    
    # Create detailed table
    table_text = "DETECTED REGIONS - DETAILED MEASUREMENTS:\n"
    table_text += "="*80 + "\n"
    table_text += f"Total Wing Area: {wing_area:,.0f} pixels\n"
    table_text += "="*80 + "\n\n"
    
    for idx, region in enumerate(sorted_regions):
        wing_ratio = region.area / wing_area
        ar = region.major_axis_length / (region.minor_axis_length + 1e-8)
        
        # Calculate bbox aspect ratio too
        minr, minc, maxr, maxc = region.bbox
        bbox_height = maxr - minr
        bbox_width = maxc - minc
        bbox_ar = bbox_width / bbox_height if bbox_height > 0 else 0
        
        table_text += f"REGION #{idx+1} (Label {region.label}):\n"
        table_text += f"  Area: {region.area:,} px ({wing_ratio*100:.2f}% of wing)\n"
        table_text += f"  Centroid: ({region.centroid[0]:.1f}, {region.centroid[1]:.1f})\n"
        table_text += f"  Ellipse AR: {ar:.2f}\n"
        table_text += f"  BBox AR: {bbox_ar:.2f}\n"
        table_text += f"  Solidity: {region.solidity:.3f}\n"
        table_text += f"  Eccentricity: {region.eccentricity:.3f}\n"
        table_text += f"  Major axis: {region.major_axis_length:.1f}\n"
        table_text += f"  Minor axis: {region.minor_axis_length:.1f}\n"
        
        # Check against merged 4+5 criteria
        expected_merged_ratio = 0.362
        expected_merged_ar = 2.87
        expected_merged_sol = 0.966
        
        ratio_match = abs(wing_ratio - expected_merged_ratio) < 0.10
        ar_match = abs(ar - expected_merged_ar) < 0.8
        sol_match = abs(region.solidity - expected_merged_sol) < 0.08
        
        if ratio_match and ar_match and sol_match:
            table_text += f"  *** MATCHES MERGED 4+5 CRITERIA ***\n"
        
        table_text += "\n"
    
    axes[1, 1].text(0.05, 0.95, table_text,
                   transform=axes[1, 1].transAxes,
                   verticalalignment='top',
                   fontfamily='monospace',
                   fontsize=8)
    axes[1, 1].set_title('Region Measurements')
    axes[1, 1].axis('off')
    
    # Comparison chart
    axes[1, 2].clear()
    
    # Bar chart of wing ratios
    wing_ratios = [r.area / wing_area for r in sorted_regions]
    region_nums = [f"#{i+1}\nL{r.label}" for i, r in enumerate(sorted_regions)]
    
    bars = axes[1, 2].bar(range(len(wing_ratios)), wing_ratios, color=colors)
    axes[1, 2].axhline(y=0.362, color='red', linestyle='--', linewidth=2, label='Expected Merged 4+5')
    axes[1, 2].axhline(y=0.214, color='green', linestyle='--', linewidth=1, label='Expected Region 5')
    axes[1, 2].axhline(y=0.179, color='blue', linestyle='--', linewidth=1, label='Expected Region 4')
    
    axes[1, 2].set_xticks(range(len(region_nums)))
    axes[1, 2].set_xticklabels(region_nums, fontsize=8)
    axes[1, 2].set_ylabel('Wing Ratio (area/wing)')
    axes[1, 2].set_title('Region Sizes vs Expected Values')
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)
    
    # Bottom row: Aspect ratios and solidity
    axes[2, 0].clear()
    aspect_ratios = [r.major_axis_length / (r.minor_axis_length + 1e-8) for r in sorted_regions]
    
    bars = axes[2, 0].bar(range(len(aspect_ratios)), aspect_ratios, color=colors)
    axes[2, 0].axhline(y=2.87, color='red', linestyle='--', linewidth=2, label='Expected Merged 4+5')
    
    axes[2, 0].set_xticks(range(len(region_nums)))
    axes[2, 0].set_xticklabels(region_nums, fontsize=8)
    axes[2, 0].set_ylabel('Aspect Ratio (major/minor)')
    axes[2, 0].set_title('Aspect Ratios vs Expected')
    axes[2, 0].legend(fontsize=8)
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].clear()
    solidities = [r.solidity for r in sorted_regions]
    
    bars = axes[2, 1].bar(range(len(solidities)), solidities, color=colors)
    axes[2, 1].axhline(y=0.966, color='red', linestyle='--', linewidth=2, label='Expected Merged 4+5')
    
    axes[2, 1].set_xticks(range(len(region_nums)))
    axes[2, 1].set_xticklabels(region_nums, fontsize=8)
    axes[2, 1].set_ylabel('Solidity')
    axes[2, 1].set_title('Solidity vs Expected')
    axes[2, 1].legend(fontsize=8)
    axes[2, 1].grid(True, alpha=0.3)
    axes[2, 1].set_ylim([0.5, 1.0])
    
    # Score matrix
    axes[2, 2].clear()
    
    score_text = "MERGE DETECTION SCORING:\n"
    score_text += "="*50 + "\n\n"
    
    for idx, region in enumerate(sorted_regions):
        wing_ratio = region.area / wing_area
        ar = region.major_axis_length / (region.minor_axis_length + 1e-8)
        sol = region.solidity
        
        ratio_error = abs(wing_ratio - 0.362) / 0.10
        ar_error = abs(ar - 2.87) / 0.8
        sol_error = abs(sol - 0.966) / 0.08
        
        combined_score = ratio_error + ar_error + sol_error
        
        score_text += f"Region #{idx+1} (L{region.label}):\n"
        score_text += f"  Ratio error: {ratio_error:.2f}\n"
        score_text += f"  AR error: {ar_error:.2f}\n"
        score_text += f"  Sol error: {sol_error:.2f}\n"
        score_text += f"  TOTAL: {combined_score:.2f}\n"
        
        if combined_score < 3.0:
            score_text += f"  -> GOOD CANDIDATE\n"
        
        score_text += "\n"
    
    axes[2, 2].text(0.05, 0.95, score_text,
                   transform=axes[2, 2].transAxes,
                   verticalalignment='top',
                   fontfamily='monospace',
                   fontsize=9)
    axes[2, 2].set_title('Merge Candidate Scoring')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{basename}_PRE_LABELING_DEBUG_{mode_name}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved pre-labeling debug visualization to {save_path}")
    logger.info(f"Found {len(regions)} regions before labeling")






def plot_voronoi_with_pentagone_info(valid_regions, region_peaks_dict, background_img, 
                                    vein_mask, skeleton, output_directory, basename, cfg, 
                                    is_pentagone_mode, is_merged_mode=False):
    """Enhanced Voronoi visualization with pentagone and merge mode information."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Choose appropriate template matcher based on mode
    if is_merged_mode:
        template_matcher = AnatomicalTemplateMatcher()
        target_regions = template_matcher.target_regions.copy()
        # Override region 4 name for merged mode
        target_regions[4] = target_regions[4].copy()
        target_regions[4]['name'] = 'Region-4+5-Merged'
        mode_title = "MERGE MODE (4+5 combined)"
        expected_regions = 4  # Regions 1, 2, 3, and merged 4+5
    elif is_pentagone_mode:
        pentagone_handler = PentagoneMutantHandler()
        target_regions = pentagone_handler.pentagone_target_regions
        mode_title = "PENTAGONE MODE (4 regions, combined 4+5)"
        expected_regions = 4
    else:
        template_matcher = AnatomicalTemplateMatcher()
        target_regions = template_matcher.target_regions
        mode_title = "NORMAL MODE (5 regions)"
        expected_regions = 5
    
    # Top-left: Complete vein network with mode info

    axes[0, 0].imshow(background_img, cmap="gray")
    axes[0, 0].imshow(vein_mask, cmap="Reds", alpha=0.6)
    axes[0, 0].set_title(f"Complete Vein Network\n{mode_title}")
    axes[0, 0].axis('off')
    
    # Top-right: Anatomically labeled regions
    axes[0, 1].imshow(background_img, cmap="gray")
    axes[0, 1].imshow(skeleton, cmap="Reds", alpha=0.8)
    
    # Color intervein regions
    labeled_regions = np.zeros_like(background_img)
    for region in valid_regions:
        labeled_regions[region.coords[:, 0], region.coords[:, 1]] = region.label
    
    axes[0, 1].imshow(labeled_regions, cmap="nipy_spectral", alpha=0.4, vmin=0, vmax=max(5, expected_regions))
    
    # Add anatomical labels with mode-appropriate names
    for region in valid_regions:
        if region.label in target_regions:
            region_name = target_regions[region.label]['name']
            cy, cx = region.centroid
            axes[0, 1].text(cx, cy, f"{region.label}\n{region_name}", 
                          color='white', fontsize=8, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    axes[0, 1].set_title(f"Anatomically Labeled Regions\n{mode_title} (n={len(valid_regions)})")
    axes[0, 1].axis('off')
    
    # Bottom-left: Voronoi cells (same as before but with mode-aware region names)
    axes[1, 0].imshow(background_img, cmap="gray")
    
    total_kept = 0
    total_excluded_nc = 0
    total_excluded_iqr = 0
    
    for region in valid_regions:
        label_val = region.label
        region_peaks = region_peaks_dict.get(label_val, np.empty((0, 2)))
        
        if region_peaks.shape[0] < 2:
            continue
        
        # [Voronoi plotting code same as before, but with mode-aware region names]
        region_coords = np.asarray(region.coords)
        if region_coords.shape[0] < 3:
            continue
        
        try:
            hull_pts = region_coords[ConvexHull(region_coords).vertices]
            region_poly = Polygon([(p[1], p[0]) for p in hull_pts]).buffer(0)
        except Exception as e:
            continue
        
        points_vor = np.asarray([(p[1], p[0]) for p in region_peaks])
        try:
            vor = Voronoi(points_vor)
            regs, verts = voronoi_finite_polygons_2d(vor)
        except Exception as e:
            continue
        
        # [Processing and filtering logic same as before]
        n_pts = len(points_vor)
        cell_polys = [None] * n_pts
        cell_areas = np.full(n_pts, np.nan, float)
        
        for i, reg in enumerate(regs):
            try:
                poly = Polygon(verts[reg]).buffer(0).intersection(region_poly)
                if not poly.is_empty:
                    cell_polys[i] = poly
                    cell_areas[i] = poly.area
            except:
                continue
        
        valid_idx = ~np.isnan(cell_areas)
        if not valid_idx.any():
            continue
        
        adj = _voronoi_adjacency(vor)
        keep_nc = np.zeros(n_pts, bool)
        
        for i, area in enumerate(cell_areas):
            if np.isnan(area):
                continue
            nbrs = [j for j in adj[i] if not np.isnan(cell_areas[j])]
            if len(nbrs) < cfg.min_neighbours:
                continue
            med = np.median(cell_areas[nbrs])
            if med > 0 and abs(area - med) < cfg.neighbour_tolerance * med:
                keep_nc[i] = True
        
        kept_areas = cell_areas[keep_nc]
        if kept_areas.size > 0:
            Q1, Q3 = np.percentile(kept_areas, (25, 75))
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                keep_final = keep_nc & (cell_areas >= lower) & (cell_areas <= upper)
            else:
                keep_final = keep_nc
        else:
            keep_final = np.zeros(n_pts, bool)
        
        excl_nc = valid_idx & (~keep_nc)
        excl_iqr = keep_nc & (~keep_final)
        
        region_kept = np.sum(keep_final)
        region_excl_nc = np.sum(excl_nc)
        region_excl_iqr = np.sum(excl_iqr)
        
        total_kept += region_kept
        total_excluded_nc += region_excl_nc
        total_excluded_iqr += region_excl_iqr
        
        # Plot polygons with color coding
        for idx, poly in enumerate(cell_polys):
            if poly is None:
                continue
            
            if keep_final[idx]:
                color, alpha, linewidth = "blue", 0.8, 1.5
            elif excl_iqr[idx]:
                color, alpha, linewidth = "red", 0.6, 1.0
            elif excl_nc[idx]:
                color, alpha, linewidth = "orange", 0.6, 1.0
            else:
                color, alpha, linewidth = "gray", 0.3, 0.5
            
            if poly.geom_type == "Polygon":
                xs, ys = poly.exterior.xy
                axes[1, 0].plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)
            else:
                for sub in getattr(poly, 'geoms', [poly]):
                    if hasattr(sub, 'exterior'):
                        xs, ys = sub.exterior.xy
                        axes[1, 0].plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)
        
        # Label region with mode-appropriate name
        cx, cy = region.centroid[1], region.centroid[0]
        region_name = target_regions.get(label_val, {}).get('name', f'R{label_val}')
        
        # Special highlighting for combined region in pentagone mode
        if is_pentagone_mode and label_val == 4:
            region_name += "\n[COMBINED]"
            label_color = "cyan"
            label_weight = "bold"
        else:
            label_color = "yellow"
            label_weight = "bold"
        
        label_text = f"{label_val}: {region_name}\n({region_kept} cells)"
        axes[1, 0].text(cx, cy, label_text, 
                       color=label_color, fontsize=9, weight=label_weight, 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    axes[1, 0].set_title(
        f"Voronoi Cell Classification - {mode_title}\n"
        f"Blue: Kept ({total_kept}) | Orange: Neighbor-fail ({total_excluded_nc}) | "
        f"Red: IQR-outlier ({total_excluded_iqr})",
        fontsize=14, pad=20
    )
    axes[1, 0].axis('off')
    
    # Bottom-right: Enhanced summary with mode information
    vein_coverage = np.sum(vein_mask) / vein_mask.size * 100
    intervein_coverage = np.sum(labeled_regions > 0) / labeled_regions.size * 100
    
    # Count missing regions based on mode
    detected_labels = set(region.label for region in valid_regions)
    expected_labels = set(range(1, expected_regions + 1))
    missing_labels = expected_labels - detected_labels
    
    if is_pentagone_mode:
        missing_names = [pentagone_handler.pentagone_target_regions.get(label, {}).get('name', f'Region-{label}') 
                        for label in missing_labels]
    else:
        missing_names = [template_matcher.target_regions.get(label, {}).get('name', f'Region-{label}') 
                        for label in missing_labels]
    
    stats_text = f"""Enhanced Analysis Summary - {mode_title}:

Vein Detection:
- Vein coverage: {vein_coverage:.1f}%
- Skeleton length: {np.sum(skeleton)} pixels

Anatomical Region Analysis:
- Detected regions: {len(valid_regions)}/{expected_regions}
- Missing regions: {', '.join(missing_names) if missing_names else 'None'}
- Intervein coverage: {intervein_coverage:.1f}%

{"Note: Region 4 represents merged regions 4+5" if is_merged_mode else ""}
{"Special Note: Region 4 combines normal regions 4+5" if is_pentagone_mode else ""}

Trichome Analysis:
- Total trichomes: {sum(len(peaks) for peaks in region_peaks_dict.values())}
- Cells kept for analysis: {total_kept}
- Average trichomes/region: {sum(len(peaks) for peaks in region_peaks_dict.values())/len(valid_regions):.1f}

Quality Metrics:
- Avg region area: {np.mean([r.area for r in valid_regions]):.0f} px²
- Avg region solidity: {np.mean([r.solidity for r in valid_regions]):.3f}

Mode: {"Merged 4+5 detection" if is_merged_mode else "Pentagone mutant (automated detection)" if is_pentagone_mode else "Normal wing"}
"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_title(f"Analysis Summary - {mode_title}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    mode_suffix = "_merged_4_5" if is_merged_mode else "_pentagone" if is_pentagone_mode else "_normal"
    save_path = os.path.join(output_directory, f"{basename}_complete_analysis{mode_suffix}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved enhanced analysis visualization ({mode_suffix[1:]}) to {save_path}")
    plt.close()





class WingBorderChecker:
    """Check if wing touches image border and should be excluded."""
    
    @staticmethod
    def is_wing_touching_border(wing_mask, buffer=1):
        """Check if wing mask touches image border within buffer distance - improved for rotated wings."""
        h, w = wing_mask.shape
        
        # For rotated wings, we need to be more sophisticated
        # Check what percentage of the border is occupied by wing tissue
        
        # Top edge
        top_coverage = np.sum(wing_mask[:buffer, :]) / (buffer * w)
        # Bottom edge  
        bottom_coverage = np.sum(wing_mask[-buffer:, :]) / (buffer * w)
        # Left edge
        left_coverage = np.sum(wing_mask[:, :buffer]) / (buffer * h)
        # Right edge
        right_coverage = np.sum(wing_mask[:, -buffer:]) / (buffer * h)
        
        # If more than 30% of any edge is wing tissue, it's likely truncated
        # This threshold allows for rotated wings that touch corners/edges
        truncation_threshold = 1
        
        if (top_coverage > truncation_threshold or 
            bottom_coverage > truncation_threshold or
            left_coverage > truncation_threshold or 
            right_coverage > truncation_threshold):
            
            logger.warning(f"High border coverage detected: top={top_coverage:.2f}, "
                         f"bottom={bottom_coverage:.2f}, left={left_coverage:.2f}, "
                         f"right={right_coverage:.2f}")
            return True
        
        return False
    
    @staticmethod
    def is_valid_wing(wing_mask, min_area=100000, border_buffer=1):
        """Check if wing is valid (not truncated and sufficient area) - improved for rotated wings."""
        # Check area
        wing_area = np.sum(wing_mask)
        if wing_area < min_area:
            logger.warning(f"Wing area ({wing_area}) below minimum ({min_area})")
            return False
        
        # Check if wing appears to be truncated (not just touching border)
        if WingBorderChecker.is_wing_touching_border(wing_mask, border_buffer):
            logger.warning("Wing appears truncated at image border")
            return False
        
        # Additional check: wing should have reasonable aspect ratio
        # Get wing bounding box
        wing_coords = np.column_stack(np.where(wing_mask))
        if len(wing_coords) == 0:
            return False
            
        min_y, min_x = wing_coords.min(axis=0)
        max_y, max_x = wing_coords.max(axis=0)
        
        wing_height = max_y - min_y
        wing_width = max_x - min_x
        
        if wing_height == 0 or wing_width == 0:
            return False
            
        # Wings should be roughly 2-4 times wider than tall (accounting for rotation)
        aspect_ratio = max(wing_width, wing_height) / min(wing_width, wing_height)
        
        if aspect_ratio < 1.1 or aspect_ratio > 6.0:
            logger.warning(f"Wing aspect ratio ({aspect_ratio:.2f}) outside expected range (1.1-6.0)")
            return False
        
        logger.info(f"Wing validated: area={wing_area}, aspect_ratio={aspect_ratio:.2f}")
        return True




class EnhancedInterveinDetector:
    """Enhanced intervein detection that handles wing boundaries better."""
    
    def __init__(self, config):
        self.config = config
        self.template_matcher = AnatomicalTemplateMatcher()
        
    def detect_wing_boundary(self, prob_map, raw_img=None):
        """Detect wing boundary prioritizing raw image when available."""
    
        if raw_img is not None:
            # Primary method: Use raw image for clean boundary detection
            return self._detect_wing_from_probabilities(prob_map)
        else:
            # Fallback: Use probability map (original method)
            return self._detect_wing_from_probabilities(prob_map)

    def _detect_wing_from_raw_improved(self, raw_img, prob_map=None):
        """Improved wing detection using raw image with border exclusion."""
    
        logger.info("Using raw image for wing boundary detection")
    
        # Normalize raw image
        raw_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
    
        # Method 1: Otsu thresholding on full image
        otsu_thresh = filters.threshold_otsu(raw_norm)
        wing_mask_otsu = raw_norm < otsu_thresh
    
        # Method 2: Focus on middle intensity range (exclude pure black/white)
        middle_intensity = (raw_norm > 0.15) & (raw_norm < 0.85)
    
        # Method 3: Use gradient to find edges, then fill
        edges = feature.canny(raw_norm, sigma=2, low_threshold=0.1, high_threshold=0.2)
        filled_edges = ndimage.binary_fill_holes(edges)
    
        # Combine methods - wing should be bright enough AND in reasonable intensity range
        combined_mask = wing_mask_otsu & middle_intensity
    
        # Add edge-based regions that are large enough
        large_edge_regions = morphology.remove_small_objects(filled_edges, min_size=20000)
        combined_mask = combined_mask | large_edge_regions
    
        # Border exclusion - exclude pixels near image edges
        border_width = 40  # Adjust this value as needed
        h, w = raw_img.shape[:2]
        border_mask = np.ones((h, w), dtype=bool)
        border_mask[:border_width, :] = False      # Top
        border_mask[-border_width:, :] = False     # Bottom  
        border_mask[:, :border_width] = False      # Left
        border_mask[:, -border_width:] = False     # Right
    
        # Apply border exclusion
        wing_mask = combined_mask & border_mask
    
        # Morphological cleanup
        wing_mask = morphology.binary_opening(wing_mask, morphology.disk(5))
        wing_mask = morphology.binary_closing(wing_mask, morphology.disk(15))
    
        # Remove small objects and holes
        wing_mask = morphology.remove_small_objects(wing_mask, min_size=30000)
        wing_mask = morphology.remove_small_holes(wing_mask, area_threshold=15000)
    
        # Find largest connected component (should be the wing)
        labeled = label(wing_mask)
        if labeled.max() > 0:
            regions = regionprops(labeled)
            largest_region = max(regions, key=lambda r: r.area)
            wing_mask = labeled == largest_region.label
        
            logger.info(f"Selected wing region with area: {largest_region.area} pixels")
    
        # Optional: Use probability map to refine if available
        if prob_map is not None and prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            # Only keep wing areas that have some intervein probability
            prob_support = intervein_prob > 0.4
            wing_mask = wing_mask & prob_support
            logger.info("Refined wing mask using intervein probability")
    
        wing_area = np.sum(wing_mask)
        logger.info(f"Final wing mask area: {wing_area} pixels")
    
        return wing_mask

    def _detect_wing_from_probabilities(self, prob_map):
        """Fallback method using probability map only."""
    
        logger.info("Using probability map for wing boundary detection (no raw image)")
        import cv2 as cv
        if prob_map.shape[-1] >= 4:
            background_prob = prob_map[..., 1]
            wing_mask = background_prob < 0.4
            cv.morphologyEx(wing_mask, cv.MORPH_CLOSE, (10,10))
            # More conservative threshold
            wing_mask = background_prob < 0.4
        
            # Border exclusion
            border_width = 30
            h, w = prob_map.shape[:2]
            border_mask = np.ones((h, w), dtype=bool)
            border_mask[:border_width, :] = False
            border_mask[-border_width:, :] = False  
            border_mask[:, :border_width] = False
            border_mask[:, -border_width:] = False
        
            wing_mask = wing_mask & border_mask
        else:
            raise ValueError("Need either raw image or 4-channel probability map")
    
        # Cleanup
        wing_mask = morphology.binary_closing(wing_mask, morphology.disk(10))
        wing_mask = morphology.remove_small_objects(wing_mask, min_size=50000)
        wing_mask = morphology.remove_small_holes(wing_mask, area_threshold=10000)
    
        return wing_mask
    
    def create_virtual_boundary_veins(self, wing_mask, actual_vein_mask, vein_width=2):
        """Create virtual veins along wing boundaries where real veins are missing."""
        
        # Find wing boundary
        wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
        
        # Dilate to make boundary thicker
        thick_boundary = morphology.binary_dilation(wing_boundary, morphology.disk(vein_width//2))
        
        # Find areas where boundary doesn't overlap with actual veins
        dilated_veins = morphology.binary_dilation(actual_vein_mask, morphology.disk(vein_width))
        
        # Virtual veins are boundary areas not covered by real veins
        virtual_veins = thick_boundary & (~dilated_veins)
        
        # Clean up virtual veins - remove small segments
        virtual_veins = morphology.remove_small_objects(virtual_veins, min_size=100)
        
        return virtual_veins
    
    def segment_intervein_regions_enhanced(self, prob_map, vein_mask, raw_img=None):
        """Enhanced intervein segmentation using wing boundary detection and template matching."""
        
        logger.info("Starting enhanced intervein segmentation with boundary completion...")
        
        # Step 1: Detect complete wing boundary
        wing_mask = self.detect_wing_boundary(prob_map, raw_img)

        
        # Check if wing is valid (not touching border)
        if not WingBorderChecker.is_valid_wing(wing_mask, self.config.min_wing_area, 
                                               self.config.border_buffer):
            logger.warning("Wing is invalid (partial or too small) - skipping intervein analysis")
            return None, None, None
        
        # Step 2: Create virtual boundary veins where needed
        virtual_veins = self.create_virtual_boundary_veins(wing_mask, vein_mask)
        
        # Step 3: Combine real and virtual veins
        combined_vein_mask = vein_mask | virtual_veins
        
        # Step 4: Create barrier mask for segmentation
        barrier_mask = morphology.binary_dilation(combined_vein_mask, 
                                                 morphology.disk(self.config.vein_width_estimate * 2))
        
        # Step 5: Use intervein probability for segmentation
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            intervein_mask = (intervein_prob > self.config.intervein_threshold) & wing_mask & (~barrier_mask)
        else:
            intervein_mask = wing_mask & (~barrier_mask)
        
        # Step 6: Clean up the intervein mask
        intervein_mask = morphology.remove_small_objects(intervein_mask, min_size=5000)
        intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=5000)
        
        # Step 7: Label connected components
        labeled_regions = label(intervein_mask)
        
        # Step 8: Filter regions
        filtered_labeled_mask = self._filter_intervein_regions(labeled_regions, wing_mask)
        logger.info("Creating pre-labeling debug visualization...")
        create_pre_labeling_debug_visualization(
            filtered_labeled_mask, wing_mask, prob_map, raw_img,
            output_directory=None,  # We'll need to pass this through
            basename="debug",
            mode_name="merge" if self.config.force_merge_regions_4_5 else "normal"
        )
        # Step 9: Apply anatomical template matching with vein information
        final_labeled_mask = self.template_matcher.match_regions_to_template(
            filtered_labeled_mask, wing_mask, vein_mask=combined_vein_mask
        )
        
        # Count regions
        unique_labels = np.unique(final_labeled_mask)
        n_regions = len(unique_labels[unique_labels > 0])
        logger.info(f"Enhanced detection found {n_regions} anatomically labeled intervein regions")
        
        return final_labeled_mask, combined_vein_mask, virtual_veins
    
    def _filter_intervein_regions(self, labeled_mask, wing_mask):
        """Filter intervein regions with criteria suitable for boundary regions."""
        
        filtered_mask = np.zeros_like(labeled_mask)
        new_label = 1
        
        regions = regionprops(labeled_mask)
        
        # Sort regions by area to prioritize larger regions
        regions.sort(key=lambda r: r.area, reverse=True)
        
        for region in regions:
            # Skip background
            if region.label == 0:
                continue
            
            # Size criteria - be more lenient for edge regions
            min_area = self.config.auto_intervein_min_area
            max_area = self.config.auto_intervein_max_area
            
            # Check if this is likely an edge region
            region_mask = labeled_mask == region.label
            
            # Calculate how much of the region touches wing boundary
            wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
            boundary_overlap = np.sum(region_mask & wing_boundary)
            boundary_ratio = boundary_overlap / region.area if region.area > 0 else 0
            
            # More lenient criteria for boundary regions
            if boundary_ratio > 0.1:  # This is a boundary region
                min_area = min_area // 2  # Allow smaller boundary regions
                
                # Check if it's at the bottom of the wing (higher y coordinates)
                if region.centroid[0] > labeled_mask.shape[0] * 0.7:
                    min_area = min_area // 2  # Even more lenient for bottom regions
            
            # Apply size filter
            if min_area <= region.area <= max_area:
                # Additional shape checks if enabled
                if self.config.intervein_shape_filter:
                    # More lenient solidity for edge regions
                    min_solidity = self.config.min_intervein_solidity
                    if boundary_ratio > 0.1:
                        min_solidity *= 0.7  # Lower threshold for edge regions
                    
                    if region.solidity >= min_solidity:
                        filtered_mask[region_mask] = new_label
                        new_label += 1
                else:
                    filtered_mask[region_mask] = new_label
                    new_label += 1
            else:
                logger.debug(f"Region {region.label} excluded due to size: {region.area} pixels")
        
        return filtered_mask
    
    def visualize_enhanced_segmentation(self, raw_img, vein_mask, virtual_veins, 
                                      intervein_regions, output_path=None):
        """Visualize the enhanced segmentation with virtual boundaries."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(raw_img if raw_img is not None else vein_mask, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Actual veins only
        axes[0, 1].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[0, 1].imshow(vein_mask, cmap='Reds', alpha=0.7)
        axes[0, 1].set_title('Detected Veins (Actual)')
        axes[0, 1].axis('off')
        
        # Virtual boundary veins
        axes[0, 2].imshow(raw_img if raw_img is not None else np.zeros_like(virtual_veins), cmap='gray')
        axes[0, 2].imshow(virtual_veins, cmap='Blues', alpha=0.7)
        axes[0, 2].set_title('Virtual Boundary Veins')
        axes[0, 2].axis('off')
        
        # Combined veins
        axes[1, 0].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[1, 0].imshow(vein_mask, cmap='Reds', alpha=0.6)
        axes[1, 0].imshow(virtual_veins, cmap='Blues', alpha=0.6)
        axes[1, 0].set_title('Combined Vein Network')
        axes[1, 0].axis('off')
        
        # Intervein regions with anatomical labels
        if intervein_regions is not None:
            axes[1, 1].imshow(intervein_regions, cmap='nipy_spectral', vmin=0, vmax=6)
            
            # Add anatomical labels
            regions = regionprops(intervein_regions)
            for region in regions:
                if region.label > 0 and region.label <= 6:
                    region_name = self.template_matcher.target_regions[region.label]['name']
                    cy, cx = region.centroid
                    axes[1, 1].text(cx, cy, f"{region.label}\n{region_name}", 
                                   color='white', fontsize=8, ha='center', va='center',
                                   bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
            
            axes[1, 1].set_title(f'Anatomically Labeled Regions (n={len(regions)})')
        else:
            axes[1, 1].set_title('No Valid Regions (Wing Touching Border)')
        axes[1, 1].axis('off')
        
        # Overlay all
        axes[1, 2].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[1, 2].imshow(vein_mask, cmap='Reds', alpha=0.5)
        axes[1, 2].imshow(virtual_veins, cmap='Blues', alpha=0.5)
        if intervein_regions is not None:
            axes[1, 2].imshow(intervein_regions > 0, cmap='Greens', alpha=0.3)
        axes[1, 2].set_title('Complete Segmentation Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            logger.info(f"Saved enhanced segmentation visualization to {output_path}")
        
        plt.close()
        
        return fig

class ImprovedWingVeinDetector:
    """Improved automated wing vein detection with better vein-specific filtering."""
    
    def __init__(self, config):
        self.config = config
        self.vein_width_estimate = config.vein_width_estimate
        self.min_vein_length = config.min_vein_length
        self.wing_template = self._create_wing_template()
    
    def _create_wing_template(self):
        """Create a more accurate Drosophila wing vein template."""
        template = {
            'longitudinal_veins': {
                'L1': {'start_ratio': (0.15, 0.1), 'end_ratio': (0.85, 0.05), 'priority': 1},
                'L2': {'start_ratio': (0.15, 0.25), 'end_ratio': (0.85, 0.2), 'priority': 2}, 
                'L3': {'start_ratio': (0.15, 0.45), 'end_ratio': (0.85, 0.4), 'priority': 3},
                'L4': {'start_ratio': (0.15, 0.65), 'end_ratio': (0.85, 0.6), 'priority': 4},
                'L5': {'start_ratio': (0.15, 0.85), 'end_ratio': (0.85, 0.8), 'priority': 5}
            },
            'cross_veins': {
                'anterior_crossvein': {'start_ratio': (0.35, 0.25), 'end_ratio': (0.35, 0.45)},
                'posterior_crossvein': {'start_ratio': (0.65, 0.45), 'end_ratio': (0.65, 0.65)}
            }
        }
        return template
    
    def detect_veins_multi_approach(self, prob_map, raw_img=None):
        """Improved multi-approach vein detection with better filtering."""
        logger.info("Starting improved automated vein detection...")
        
        # Extract vein probability with better preprocessing
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            vein_prob = prob_map[...,2]
        else:
            if raw_img is not None:
                vein_prob = prob_map[...,2]
            else:
                raise ValueError("Need either 4-channel probability map or raw image")
        
        # Enhanced preprocessing for vein detection
        vein_prob = self._preprocess_vein_probability(vein_prob, raw_img)
        
        # Method 1: Improved Frangi filter with better parameters
        vein_mask_frangi = self._improved_frangi_detection(vein_prob)
        
        # Method 2: Ridge detection with orientation filtering
        vein_mask_ridge = self._improved_ridge_detection(vein_prob)
        
        # Method 3: Morphological with directional filters
        vein_mask_morph = self._improved_morphological_detection(vein_prob)
        
        # Method 4: Edge-based detection (new)
        vein_mask_edge = self._edge_based_vein_detection(vein_prob, raw_img)
        
        # Combine methods with smarter voting
        combined_mask = self._smart_combine_masks([
            (vein_mask_frangi, 0.3),
            (vein_mask_ridge, 0.3),
            (vein_mask_morph, 0.2),
            (vein_mask_edge, 0.2)
        ])
        
        # Post-process with connectivity and geometry constraints
        final_vein_mask = self._advanced_postprocess(combined_mask, vein_prob)
        
        # Create skeleton
        vein_skeleton = self._create_clean_skeleton(final_vein_mask)
        
        # Generate intervein regions
        intervein_regions = self._create_intervein_regions(final_vein_mask, prob_map)
        
        logger.info(f"Detected vein network with {np.sum(final_vein_mask)} vein pixels")
        
        return final_vein_mask, vein_skeleton, intervein_regions
    
    def _preprocess_vein_probability(self, vein_prob, raw_img):
        """Enhanced preprocessing specifically for vein detection."""
        # Normalize
        vein_prob = (vein_prob - vein_prob.min()) / (vein_prob.max() - vein_prob.min() + 1e-8)
        
        # Apply CLAHE for better local contrast
        vein_prob = exposure.equalize_adapthist(vein_prob, clip_limit=0.03)
        
        # If we have raw image, use it to enhance vein probability
        if raw_img is not None:
            # Veins are typically darker in raw images
            raw_norm = (raw_img - raw_img.min()) / (raw_img.max() - raw_img.min() + 1e-8)
            darkness_map = 1.0 - raw_norm
            
            # Combine with existing probability
            vein_prob = 0.7 * vein_prob + 0.3 * darkness_map
        
        return vein_prob
    
    def _improved_frangi_detection(self, vein_prob):
        """Improved Frangi filter with better scale selection."""
        # Use scales appropriate for vein widths (narrower range)
        scales = np.linspace(0.5, 4, 8)
        
        # Apply Frangi with parameters tuned for dark linear structures
        frangi_response = frangi(
            vein_prob,
            sigmas=scales,
            alpha=0.5,  # Plate-like structures suppression
            beta=0.5,   # Blob-like structures suppression  
            gamma=15,   # Background suppression
            black_ridges=False  # We want bright ridges in our probability map
        )
        
        # Adaptive thresholding based on response statistics
        threshold = np.percentile(frangi_response[frangi_response > 0], 90)
        vein_mask = frangi_response > threshold
        
        # Remove small objects
        vein_mask = morphology.remove_small_objects(vein_mask, min_size=100)
        
        return vein_mask
    
    def _improved_ridge_detection(self, vein_prob):
        """Ridge detection with orientation consistency."""
        # Multi-scale ridge detection
        ridge_responses = []
        
        for sigma in [1.0, 1.5, 2.0]:
            # Smooth at this scale
            smoothed = gaussian_filter(vein_prob, sigma)
            
            # Compute Hessian
            Hxx = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=1)
            Hyy = ndimage.sobel(ndimage.sobel(smoothed, axis=0), axis=0)
            Hxy = ndimage.sobel(ndimage.sobel(smoothed, axis=1), axis=0)
            
            # Eigenvalues
            trace = Hxx + Hyy
            det = Hxx * Hyy - Hxy**2
            discriminant = np.sqrt(np.maximum(0, trace**2 - 4*det))
            
            lambda1 = (trace + discriminant) / 2
            lambda2 = (trace - discriminant) / 2
            
            # Ridge measure (Frangi's vesselness-like)
            ridge_measure = np.zeros_like(lambda1)
            
            # Only consider pixels where lambda2 is negative (ridge-like)
            ridge_pixels = lambda2 < 0
            
            if np.any(ridge_pixels):
                # Vesselness measure
                Rb = np.abs(lambda1[ridge_pixels] / lambda2[ridge_pixels])
                S = np.sqrt(lambda1[ridge_pixels]**2 + lambda2[ridge_pixels]**2)
                
                # Parameters
                beta = 0.5
                c = 0.5 * np.max(S)
                
                # Compute vesselness
                vesselness = np.exp(-Rb**2 / (2 * beta**2)) * (1 - np.exp(-S**2 / (2 * c**2)))
                
                ridge_measure[ridge_pixels] = vesselness
            
            ridge_responses.append(ridge_measure)
        
        # Combine scales
        combined_ridge = np.maximum.reduce(ridge_responses)
        
        # Threshold
        threshold = np.percentile(combined_ridge[combined_ridge > 0], 85)
        vein_mask = combined_ridge > threshold
        
        return vein_mask
    
    def _improved_morphological_detection(self, vein_prob):
        """Morphological detection with directional filters."""
        # Create directional line structuring elements
        responses = []
        
        # More angles for better coverage
        angles = np.arange(0, 180, 15)
        
        for angle in angles:
            # Create oriented line element
            length = 15
            se = self._create_line_strel(length, angle)
            
            # Top-hat to enhance linear structures
            tophat = morphology.white_tophat(vein_prob, se)
            responses.append(tophat)
        
        # Take maximum response across all orientations
        max_response = np.maximum.reduce(responses)
        
        # Threshold
        threshold = filters.threshold_otsu(max_response)
        vein_mask = max_response > threshold * 0.7
        
        # Clean up
        vein_mask = morphology.remove_small_objects(vein_mask, min_size=50)
        
        return vein_mask
    
    def _edge_based_vein_detection(self, vein_prob, raw_img):
        """New method: detect veins as edges/boundaries."""
        if raw_img is not None:
            # Use raw image for edge detection
            img_for_edges = raw_img.astype(np.float32)
        else:
            img_for_edges = vein_prob
        
        # Normalize
        img_for_edges = (img_for_edges - img_for_edges.min()) / (img_for_edges.max() - img_for_edges.min() + 1e-8)
        
        # Multi-scale edge detection
        edges = []
        
        for sigma in [0.5, 1.0, 1.5]:
            # Canny edge detection
            edge = feature.canny(img_for_edges, sigma=sigma, low_threshold=0.05, high_threshold=0.15)
            edges.append(edge)
        
        # Combine edges
        combined_edges = np.logical_or.reduce(edges)
        
        # Filter edges by linearity
        # Label connected components
        labeled = label(combined_edges)
        
        linear_edges = np.zeros_like(combined_edges)
        
        for region in regionprops(labeled):
            # Check if component is linear enough
            if region.major_axis_length > 0:
                linearity = region.major_axis_length / (region.minor_axis_length + 1e-8)
                
                # Keep linear components
                if linearity > 5 and region.major_axis_length > 20:
                    linear_edges[labeled == region.label] = True
        
        return linear_edges
    
    def _create_line_strel(self, length, angle):
        """Create a line structuring element at specified angle."""
        # Create a square array
        size = length
        se = np.zeros((size, size))
        
        # Calculate line endpoints
        angle_rad = np.radians(angle)
        center = size // 2
        
        # Draw line
        for i in range(length):
            x = int(center + (i - length//2) * np.cos(angle_rad))
            y = int(center + (i - length//2) * np.sin(angle_rad))
            
            if 0 <= x < size and 0 <= y < size:
                se[y, x] = 1
        
        return se.astype(bool)
    
    def _smart_combine_masks(self, mask_weight_pairs):
        """Smarter combination that requires agreement between methods."""
        h, w = mask_weight_pairs[0][0].shape
        vote_map = np.zeros((h, w), dtype=np.float32)
        
        # Accumulate weighted votes
        for mask, weight in mask_weight_pairs:
            vote_map += mask.astype(np.float32) * weight
        
        # Require at least 50% agreement
        combined = vote_map >= 0.5
        
        # Additional constraint: connected components must be linear
        labeled = label(combined)
        filtered = np.zeros_like(combined)
        
        for region in regionprops(labeled):
            # Check linearity
            if region.major_axis_length > 0:
                linearity = region.major_axis_length / (region.minor_axis_length + 1e-8)
                
                # Must be linear and of minimum length
                if linearity > 3 and region.major_axis_length > 30:
                    filtered[labeled == region.label] = True
        
        return filtered
    
    def _advanced_postprocess(self, vein_mask, vein_prob):
        """Advanced post-processing with geometric constraints."""
        # Step 1: Remove small objects
        cleaned = morphology.remove_small_objects(vein_mask, min_size=100)
        
        # Step 2: Close small gaps
        closed = morphology.binary_closing(cleaned, morphology.disk(2))
        
        # Step 3: Skeletonize and then dilate back
        skeleton = morphology.skeletonize(closed)
        
        # Remove skeleton spurs
        skeleton = self._remove_spurs(skeleton, min_length=10)
        
        # Dilate skeleton to approximate original width
        dilated = morphology.binary_dilation(skeleton, morphology.disk(self.vein_width_estimate // 4))
        
        # Step 4: Apply geometric constraints
        labeled = label(dilated)
        final_mask = np.zeros_like(dilated)
        
        for region in regionprops(labeled):
            # Multiple criteria for valid veins
            is_valid = True
            
            # Length criterion
            if region.major_axis_length < self.min_vein_length:
                is_valid = False
            
            # Linearity criterion
            if region.minor_axis_length > 0:
                linearity = region.major_axis_length / region.minor_axis_length
                if linearity < 3:
                    is_valid = False
            
            # Solidity criterion (to avoid blobs)
            if region.solidity < 0.7:
                is_valid = False
            
            # Width constraint
            if region.minor_axis_length > self.vein_width_estimate * 2:
                is_valid = False
            
            if is_valid:
                final_mask[labeled == region.label] = True
        
        # Step 5: Ensure connectivity of major veins
        final_mask = self._ensure_vein_connectivity_improved(final_mask, vein_prob)
        
        return final_mask
    
    def _remove_spurs(self, skeleton, min_length=10):
        """Remove short spurs from skeleton."""
        # Find endpoints
        kernel = np.array([[1, 1, 1],
                          [1, 10, 1],
                          [1, 1, 1]])
        
        neighbors = ndimage.convolve(skeleton.astype(int), kernel, mode='constant')
        endpoints = (neighbors == 11) & skeleton
        
        # Remove short branches
        cleaned = skeleton.copy()
        
        for y, x in np.argwhere(endpoints):
            # Trace from endpoint
            length = self._trace_branch_length(skeleton, y, x, max_length=min_length)
            
            if length < min_length:
                # Remove this branch
                self._remove_branch(cleaned, y, x, length)
        
        return cleaned
    
    def _trace_branch_length(self, skeleton, y, x, max_length=50):
        """Trace branch length from endpoint."""
        visited = set()
        queue = [(y, x)]
        length = 0
        
        while queue and length < max_length:
            cy, cx = queue.pop(0)
            
            if (cy, cx) in visited:
                continue
                
            visited.add((cy, cx))
            length += 1
            
            # Check neighbors
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = cy + dy, cx + dx
                    
                    if (0 <= ny < skeleton.shape[0] and 
                        0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] and
                        (ny, nx) not in visited):
                        
                        neighbors += 1
                        queue.append((ny, nx))
            
            # If more than 1 neighbor, we've hit a junction
            if neighbors > 1:
                break
        
        return length
    
    def _remove_branch(self, skeleton, y, x, max_length):
        """Remove a branch starting from endpoint."""
        visited = set()
        queue = [(y, x)]
        removed = 0
        
        while queue and removed < max_length:
            cy, cx = queue.pop(0)
            
            if (cy, cx) in visited:
                continue
            
            visited.add((cy, cx))
            skeleton[cy, cx] = False
            removed += 1
            
            # Find next point
            found_next = False
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = cy + dy, cx + dx
                    
                    if (0 <= ny < skeleton.shape[0] and 
                        0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx] and
                        (ny, nx) not in visited):
                        
                        queue.append((ny, nx))
                        found_next = True
                        break
                
                if found_next:
                    break
    
    def _ensure_vein_connectivity_improved(self, vein_mask, vein_prob):
        """Improved connectivity enforcement using probability map guidance."""
        # Find major components
        labeled = label(vein_mask)
        regions = regionprops(labeled)
        
        if not regions:
            return vein_mask
        
        # Sort by size
        regions.sort(key=lambda r: r.area, reverse=True)
        
        # Keep top components that together cover 90% of vein pixels
        total_area = np.sum(vein_mask)
        cumulative_area = 0
        keep_labels = []
        
        for region in regions:
            keep_labels.append(region.label)
            cumulative_area += region.area
            
            if cumulative_area > 0.9 * total_area:
                break
        
        # Create mask with only major components
        major_veins = np.isin(labeled, keep_labels)
        
        # Try to connect nearby components using probability map
        final_mask = major_veins.copy()
        
        # Add any high-probability pixels that connect components
        high_prob_mask = vein_prob > np.percentile(vein_prob, 95)
        
        # Dilate major veins slightly
        dilated_veins = morphology.binary_dilation(major_veins, morphology.disk(3))
        
        # Add connecting pixels
        connecting_pixels = high_prob_mask & dilated_veins & (~major_veins)
        final_mask |= connecting_pixels
        
        return final_mask
    
    def _create_clean_skeleton(self, vein_mask):
        """Create a clean skeleton with proper pruning."""
        # Initial skeletonization
        skeleton = morphology.skeletonize(vein_mask)
        
        # Remove small components
        labeled = label(skeleton)
        
        cleaned = np.zeros_like(skeleton)
        for region in regionprops(labeled):
            if region.area >= self.min_vein_length // 5:
                cleaned[labeled == region.label] = True
        
        # Remove spurs
        cleaned = self._remove_spurs(cleaned, min_length=15)
        
        return cleaned
    
    def _intensity_based_vein_detection(self, raw_img):
        """Fallback method using raw image intensity."""
        # Assume veins are darker than surrounding tissue
        # Invert and normalize
        if raw_img.max() > 1:
            raw_img = raw_img.astype(np.float32) / raw_img.max()
        
        # Invert so veins have high values
        vein_prob = 1.0 - raw_img
        
        # Enhance contrast
        vein_prob = np.clip(vein_prob * 1.5, 0, 1)
        
        return vein_prob
    
    def _create_intervein_regions(self, vein_mask, prob_map):
        """Create intervein regions from vein mask."""
        # Dilate veins to create barriers
        dilated_veins = binary_dilation(vein_mask, disk(self.vein_width_estimate))
        
        # Create intervein mask
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            intervein_mask = (intervein_prob > 0.4) & (~dilated_veins)

        else:
            # Fallback: everything not vein is potential intervein
            intervein_mask = ~dilated_veins
        
        # Label regions
        labeled_regions = label(intervein_mask)
        
        # Filter regions by size
        regions = regionprops(labeled_regions)
        final_mask = np.zeros_like(labeled_regions)
        
        min_area = self.config.auto_intervein_min_area
        max_area = self.config.auto_intervein_max_area
        
        valid_label = 1
        for region in regions:
            if min_area <= region.area <= max_area:
                # Check if region is not touching image border
                minr, minc, maxr, maxc = region.bbox
                h, w = labeled_regions.shape
                
                if (minr > 5 and minc > 5 and 
                    maxr < h - 5 and maxc < w - 5):
                    final_mask[labeled_regions == region.label] = valid_label
                    valid_label += 1
        
        return final_mask
    
    def visualize_detection_results(self, raw_img, vein_mask, skeleton, 
                                  intervein_regions, output_path=None):
        """Create comprehensive visualization of detection results."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Original image
        axes[0, 0].imshow(raw_img if raw_img is not None else vein_mask, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Detected veins
        axes[0, 1].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[0, 1].imshow(vein_mask, cmap='Reds', alpha=0.7)
        axes[0, 1].set_title('Detected Veins')
        axes[0, 1].axis('off')
        
        # Vein skeleton
        axes[0, 2].imshow(raw_img if raw_img is not None else np.zeros_like(skeleton), cmap='gray')
        axes[0, 2].imshow(skeleton, cmap='Reds', alpha=0.9)
        axes[0, 2].set_title('Vein Skeleton')
        axes[0, 2].axis('off')
        
        # Intervein regions
        axes[1, 0].imshow(intervein_regions, cmap='nipy_spectral')
        axes[1, 0].set_title(f'Intervein Regions (n={len(np.unique(intervein_regions))-1})')
        axes[1, 0].axis('off')
        
        # Combined overlay
        axes[1, 1].imshow(raw_img if raw_img is not None else np.zeros_like(vein_mask), cmap='gray')
        axes[1, 1].imshow(vein_mask, cmap='Reds', alpha=0.5)
        axes[1, 1].imshow(intervein_regions > 0, cmap='Blues', alpha=0.3)
        axes[1, 1].set_title('Combined Overlay')
        axes[1, 1].axis('off')
        
        # Statistics
        vein_pixels = np.sum(vein_mask)
        intervein_pixels = np.sum(intervein_regions > 0)
        total_pixels = vein_mask.size
        
        stats_text = f"""Detection Statistics:
        
Vein pixels: {vein_pixels:,}
Intervein pixels: {intervein_pixels:,}
Total pixels: {total_pixels:,}

Vein coverage: {100*vein_pixels/total_pixels:.1f}%
Intervein coverage: {100*intervein_pixels/total_pixels:.1f}%

Number of intervein regions: {len(np.unique(intervein_regions))-1}
"""
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 2].set_title('Detection Statistics')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=200, bbox_inches='tight')
            logger.info(f"Saved detection visualization to {output_path}")
        
        plt.close()
        
        return fig

# ALTERNATIVE: If you want the threshold to be configurable, use this version instead

def load_probability_map(h5_path: str, cfg: TrichomeDetectionConfig = None) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load Ilastik probability map with configurable background masking."""
    try:
        with h5py.File(h5_path, "r") as f:
            if "exported_data" not in f:
                available_keys = list(f.keys())
                logger.error(f"'exported_data' not found. Available keys: {available_keys}")
                raise KeyError("exported_data not found in HDF5 file")
            
            data = f["exported_data"][:]
            
        logger.info(f"Loaded probability map shape: {data.shape}")
        
        if data.ndim == 3:
            trichome = data[..., 0] if data.shape[-1] >= 1 else None
            intervein = data[..., 3] if data.shape[-1] >= 4 else None
        elif data.ndim == 4:
            trichome = data[0, ...] if data.shape[0] >= 1 else None
            intervein = data[3, ...] if data.shape[0] >= 4 else None
            data = np.transpose(data, (1, 2, 0))
        else:
            raise ValueError(f"Unexpected probability map dimensions: {data.shape}")
            
        if trichome is None:
            raise ValueError("Could not extract trichome channel from probability map")
        
        # === CONFIGURABLE BACKGROUND MASKING ===
        if cfg and hasattr(cfg, 'enable_background_masking') and cfg.enable_background_masking:
            if data.shape[-1] >= 4 and intervein is not None:
                # Get background channel (channel 1, 0-indexed)
                background_prob = data[..., 1]
                
                # Use configurable threshold
                background_threshold = getattr(cfg, 'background_mask_threshold', 0.4)
                high_background_mask = background_prob > background_threshold
                
                # Set intervein probability to zero where background is high
                intervein_masked = intervein.copy()
                intervein_masked[high_background_mask] = 0.0
                
                # Also update the full probability map
                data_masked = data.copy()
                data_masked[..., 3][high_background_mask] = 0.0
                
                # Log the effect
                masked_pixels = np.sum(high_background_mask)
                total_pixels = high_background_mask.size
                logger.info(f"BACKGROUND MASKING: Masked {masked_pixels:,} pixels ({masked_pixels/total_pixels*100:.1f}%) with background prob > {background_threshold}")
                
                return (trichome.astype(np.float32), 
                        intervein_masked.astype(np.float32), 
                        data_masked.astype(np.float32))
            else:
                logger.warning("Background masking skipped - insufficient channels")
        else:
            logger.info("Background masking disabled or no config provided")
            
        return (trichome.astype(np.float32), 
                intervein.astype(np.float32) if intervein is not None else None, 
                data.astype(np.float32))
                
    except Exception as e:
        logger.error(f"Error loading probability map from {h5_path}: {e}")
        raise
def load_raw_image(path: Optional[str]) -> Optional[np.ndarray]:
    """Load raw image with better error handling."""
    if not path or not os.path.exists(path):
        logger.warning(f"Raw image path not found: {path}")
        return None
    
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".h5", ".hdf5"]:
            with h5py.File(path, "r") as f:
                key = list(f.keys())[0]
                img = f[key][:]
            return img[0] if img.ndim >= 3 else img
        else:
            return imageio.imread(path)
    except Exception as e:
        logger.error(f"Error loading raw image from {path}: {e}")
        return None

def preprocess_probability_map(prob: np.ndarray, cfg: TrichomeDetectionConfig) -> np.ndarray:
    """Enhanced preprocessing with validation."""
    if prob.size == 0:
        raise ValueError("Empty probability map")
    
    # Normalize to [0, 1] if needed
    if prob.max() > 1.0:
        prob = prob / prob.max()
        logger.warning("Probability map values > 1.0, normalizing...")
    
    proc = prob.copy()
    
    if cfg.use_clahe:
        proc = exposure.equalize_adapthist(
            proc, 
            clip_limit=cfg.clahe_clip_limit, 
            nbins=256
        )
        logger.info("Applied CLAHE enhancement")
    
    if cfg.use_white_tophat:
        proc = white_tophat(proc, footprint=disk(cfg.tophat_radius))
        logger.info(f"Applied white top-hat with radius {cfg.tophat_radius}")
    
    return proc

# Enhanced region detection that handles trichome holes and marginal regions

class ImprovedRegionDetector:
    """Improved detector that handles trichome-induced holes and marginal regions."""
    
    def __init__(self, config):
        self.config = config
        self.template_matcher = AnatomicalTemplateMatcher()
    
    def preprocess_intervein_probability(self, prob_map, trichome_prob=None):
        """Preprocess intervein probability to handle trichome holes."""
        
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3].copy()
        else:
            raise ValueError("Need 4-channel probability map")
        
        logger.info("Preprocessing intervein probability to handle trichome holes...")
        
        # Step 1: Identify trichome locations
        if trichome_prob is not None:
            # Use actual trichome probability
            trichome_mask = trichome_prob > 0.5
        elif prob_map.shape[-1] >= 1:
            # Use first channel as trichome probability
            trichome_mask = prob_map[..., 0] > 0.5
        else:
            # Fallback: identify low probability areas as potential trichomes
            trichome_mask = intervein_prob < 0.1
        
        logger.info(f"Identified {np.sum(trichome_mask)} trichome pixels")
        
        # Step 2: Morphological closing to fill trichome holes
        # Use smaller structuring element to preserve region boundaries
        closed_intervein = intervein_prob > 0.4
        
        # Step 3: Fill holes that are surrounded by intervein tissue
        filled_intervein = closed_intervein & trichome_mask
        
        # Step 4: Gaussian smoothing to create more continuous probability
        smoothed_prob = gaussian_filter(intervein_prob, sigma=1.5)
        
        # Step 5: Combine approaches - use filled mask where there were holes
        holes = filled_intervein & (~closed_intervein)
        enhanced_prob = intervein_prob.copy()
        enhanced_prob[holes] = smoothed_prob[holes]
        
        logger.info(f"Filled {np.sum(holes)} pixels in trichome holes")
        
        return enhanced_prob, filled_intervein, trichome_mask
    
    def segment_with_marginal_region_recovery(self, prob_map, vein_mask, wing_mask, raw_img=None):
        """Enhanced segmentation that specifically recovers small marginal regions."""
        
        logger.info("Starting enhanced segmentation with marginal region recovery...")
        
        # Step 1: Preprocess intervein probability
        trichome_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else None
        enhanced_prob, filled_mask, trichome_mask = self.preprocess_intervein_probability(
            prob_map, trichome_prob
        )
        
        # Step 2: Create enhanced barrier mask
        barrier_mask = self.create_enhanced_barrier_mask(vein_mask, wing_mask)
        
        # Step 3: Multi-threshold segmentation to catch different region types
        regions_multi = self.multi_threshold_segmentation(
            enhanced_prob, barrier_mask, wing_mask
        )
        
        # Step 4: Morphological reconstruction for marginal regions
        marginal_regions = self.recover_marginal_regions(
            enhanced_prob, regions_multi, wing_mask, vein_mask
        )
        
        # Step 5: Combine all regions
        combined_regions = self.combine_region_masks(regions_multi, marginal_regions)
        
        # Step 6: Enhanced filtering that preserves small but valid regions
        filtered_regions = self.enhanced_region_filtering(
            combined_regions, wing_mask, enhanced_prob
        )
        
        # Step 7: Template matching with region completion
        final_regions = self.template_matching_with_completion(
            filtered_regions, wing_mask, enhanced_prob
        )
        
        return final_regions, enhanced_prob, filled_mask
    
    def create_enhanced_barrier_mask(self, vein_mask, wing_mask):
        """Create a more sophisticated barrier mask."""
        
        # Use smaller dilation for barriers to avoid over-excluding small regions
        barrier_base = morphology.binary_dilation(
            vein_mask, morphology.disk(self.config.vein_width_estimate)
        )
        
        # Add wing boundary as barrier but with smaller buffer
        wing_boundary = morphology.binary_erosion(wing_mask, morphology.disk(3)) ^ wing_mask
        boundary_barrier = morphology.binary_dilation(wing_boundary, morphology.disk(2))
        
        # Combine barriers
        combined_barrier = barrier_base | boundary_barrier
        
        return combined_barrier
    
    def multi_threshold_segmentation(self, enhanced_prob, barrier_mask, wing_mask):
        """Use multiple thresholds to catch regions of different intensities."""
        
        logger.info("Applying multi-threshold segmentation...")
        
        # Different thresholds for different region types
        thresholds = [
            0.3,  # Lower threshold for faint regions (marginal regions)
            0.4,  # Standard threshold
            0.5,  # Higher threshold for clear regions
        ]
        
        all_regions = []
        
        for i, thresh in enumerate(thresholds):
            # Create mask at this threshold
            thresh_mask = (enhanced_prob > thresh) & wing_mask & (~barrier_mask)
            
            # Clean up
            cleaned = morphology.remove_small_objects(thresh_mask, min_size=1000)
            cleaned = morphology.remove_small_holes(cleaned, area_threshold=1000)
            
            # Label regions
            labeled = label(cleaned)
            
            if labeled.max() > 0:
                all_regions.append((labeled, thresh, f"threshold_{thresh}"))
                logger.info(f"Threshold {thresh}: found {labeled.max()} regions")
        
        return all_regions
    
    def recover_marginal_regions(self, enhanced_prob, regions_multi, wing_mask, vein_mask):
        """Specifically recover small marginal regions that might be missed."""
        
        logger.info("Attempting to recover marginal regions...")
        
        # Create mask of already found regions
        existing_regions = np.zeros_like(wing_mask)
        for labeled, _, _ in regions_multi:
            existing_regions |= (labeled > 0)
        
        # Find wing margins
        wing_boundary = morphology.binary_erosion(wing_mask, morphology.disk(5)) ^ wing_mask
        marginal_zone = morphology.binary_dilation(wing_boundary, morphology.disk(20))
        
        # Look for unassigned areas in marginal zone
        unassigned_marginal = marginal_zone & wing_mask & (~existing_regions) & (~vein_mask)
        
        # Use very low threshold in marginal areas
        marginal_candidates = (enhanced_prob > 0.2) & unassigned_marginal
        
        # Morphological operations to connect fragmented pieces
        connected_marginal = morphology.binary_closing(
            marginal_candidates, morphology.disk(5)
        )
        
        # Remove very small objects but keep smaller threshold for marginal regions
        recovered_marginal = morphology.remove_small_objects(
            connected_marginal, min_size=2000  # Smaller threshold for marginal regions
        )
        
        labeled_marginal = label(recovered_marginal)
        
        if labeled_marginal.max() > 0:
            logger.info(f"Recovered {labeled_marginal.max()} marginal regions")
            return [(labeled_marginal, 0.2, "marginal_recovery")]
        else:
            logger.info("No marginal regions recovered")
            return []
    
    def combine_region_masks(self, regions_multi, marginal_regions):
        """Combine all region masks, resolving overlaps."""
        
        all_regions = regions_multi + marginal_regions
        
        if not all_regions:
            return np.zeros((10, 10), dtype=int)  # Empty mask
        
        # Get image shape from first region
        shape = all_regions[0][0].shape
        combined = np.zeros(shape, dtype=int)
        
        label_counter = 1
        
        # Process in order of threshold (higher thresholds = higher priority)
        sorted_regions = sorted(all_regions, key=lambda x: x[1], reverse=True)
        
        for labeled_mask, thresh, source in sorted_regions:
            regions = regionprops(labeled_mask)
            
            for region in regions:
                region_mask = labeled_mask == region.label
                
                # Check if this region overlaps significantly with existing regions
                overlap = np.sum(region_mask & (combined > 0))
                region_size = np.sum(region_mask)
                
                # If overlap is small, add this region
                if overlap < 0.3 * region_size:  # Less than 30% overlap
                    # Remove the overlapping part
                    clean_region = region_mask & (combined == 0)
                    
                    if np.sum(clean_region) > 1000:  # Still significant after removing overlap
                        combined[clean_region] = label_counter
                        label_counter += 1
                        logger.info(f"Added region {label_counter-1} from {source}")
        
        return combined
    
    def enhanced_region_filtering(self, combined_regions, wing_mask, enhanced_prob):
        """Enhanced filtering that preserves small but anatomically valid regions."""
        
        logger.info("Applying enhanced region filtering...")
        
        regions = regionprops(combined_regions)
        filtered_mask = np.zeros_like(combined_regions)
        new_label = 1
        
        # Estimate total wing area for ratio calculations
        total_wing_area = np.sum(wing_mask)
        
        for region in regions:
            if region.label == 0:
                continue
            
            # Calculate region properties
            wing_ratio = region.area / total_wing_area
            aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
            
            # More lenient criteria for small regions that could be regions 1 or 2
            min_area = self.config.auto_intervein_min_area
            max_area = self.config.auto_intervein_max_area
            
            # Special handling for potential regions 1 and 2 (small, elongated, marginal)
            is_potential_marginal = (
                wing_ratio < 0.15 and  # Small relative to wing
                aspect_ratio > 4.0 and  # Elongated
                self.is_marginal_position(region, wing_mask)
            )
            
            if is_potential_marginal:
                min_area = min_area // 4  # Much more lenient for marginal regions
                min_solidity = 0.5  # Very lenient solidity
                logger.info(f"Potential marginal region {region.label}: "
                           f"area={region.area}, AR={aspect_ratio:.1f}, "
                           f"wing_ratio={wing_ratio:.3f}")
            else:
                min_solidity = self.config.min_intervein_solidity
            
            # Apply size filter
            if min_area <= region.area <= max_area:
                # Apply shape filter
                if region.solidity >= min_solidity:
                    # Additional quality check: region should have reasonable intervein probability
                    region_mask = combined_regions == region.label
                    avg_prob = np.mean(enhanced_prob[region_mask])
                    
                    if avg_prob > 0.25:  # Reasonable intervein probability
                        filtered_mask[region_mask] = new_label
                        new_label += 1
                        
                        region_type = "marginal" if is_potential_marginal else "standard"
                        logger.info(f"Kept {region_type} region {new_label-1}: "
                                   f"area={region.area}, AR={aspect_ratio:.1f}, "
                                   f"sol={region.solidity:.3f}, prob={avg_prob:.3f}")
                    else:
                        logger.info(f"Excluded region {region.label}: low intervein probability ({avg_prob:.3f})")
                else:
                    logger.info(f"Excluded region {region.label}: low solidity ({region.solidity:.3f})")
            else:
                logger.info(f"Excluded region {region.label}: size ({region.area}) outside range [{min_area}, {max_area}]")
        
        logger.info(f"Enhanced filtering: kept {new_label-1} regions")
        return filtered_mask
    
    def is_marginal_position(self, region, wing_mask):
        """Check if region is in marginal (edge) position of wing."""
        
        # Get wing boundaries
        wing_coords = np.column_stack(np.where(wing_mask))
        if len(wing_coords) == 0:
            return False
        
        min_y, min_x = wing_coords.min(axis=0)
        max_y, max_x = wing_coords.max(axis=0)
        
        # Get region centroid
        cy, cx = region.centroid
        
        # Normalize position within wing
        if max_y > min_y and max_x > min_x:
            norm_y = (cy - min_y) / (max_y - min_y)
            norm_x = (cx - min_x) / (max_x - min_x)
            
            # Marginal regions are typically at wing edges
            is_edge_x = norm_x < 0.2 or norm_x > 0.8
            is_edge_y = norm_y < 0.3 or norm_y > 0.9
            
            return is_edge_x or is_edge_y
        
        return False
    
    def template_matching_with_completion(self, filtered_regions, wing_mask, enhanced_prob):
        """Template matching with attempts to complete missing regions."""
        
        logger.info("Template matching with region completion...")
        
        # First, try normal template matching
        initial_result = self.template_matcher.wing_ratio_based_matching(
            filtered_regions, wing_mask
        )
        
        # Check what regions are missing
        detected_labels = set(np.unique(initial_result))
        detected_labels.discard(0)  # Remove background
        
        expected_labels = set(range(1, 6))
        missing_labels = expected_labels - detected_labels
        
        if missing_labels:
            logger.info(f"Missing regions: {missing_labels}")
            logger.info("Attempting to complete missing regions...")
            
            # Try to find missing regions using template-guided search
            completed_result = self.search_for_missing_regions(
                initial_result, missing_labels, wing_mask, enhanced_prob
            )
            
            return completed_result
        else:
            logger.info("All 5 regions successfully detected!")
            return initial_result
    
    def search_for_missing_regions(self, current_result, missing_labels, wing_mask, enhanced_prob):
        """Search for missing regions using template guidance."""
        
        logger.info(f"Searching for missing regions: {missing_labels}")
        
        # Create mask of currently assigned areas
        assigned_mask = current_result > 0
        
        # Create search areas for each missing region type
        search_results = current_result.copy()
        
        for missing_label in missing_labels:
            logger.info(f"Searching for region {missing_label}...")
            
            # Get expected properties for this region
            if missing_label in self.template_matcher.target_regions:
                template = self.template_matcher.target_regions[missing_label]
                expected_ratio = template['wing_area_ratio']
                expected_ar_range = template['aspect_ratio_range']
                
                # Search in unassigned areas
                search_area = wing_mask & (~assigned_mask)
                
                # Use lower threshold for missing regions
                candidate_mask = (enhanced_prob > 0.2) & search_area
                
                if np.sum(candidate_mask) > 0:
                    # Label connected components
                    labeled_candidates = label(candidate_mask)
                    regions = regionprops(labeled_candidates)
                    
                    # Find best candidate based on template
                    best_candidate = None
                    best_score = float('inf')
                    
                    total_wing_area = np.sum(wing_mask)
                    
                    for region in regions:
                        if region.area < 1000:  # Too small
                            continue
                        
                        wing_ratio = region.area / total_wing_area
                        aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
                        
                        # Score based on how well it matches template
                        ratio_error = abs(wing_ratio - expected_ratio)
                        ar_mid = (expected_ar_range[0] + expected_ar_range[1]) / 2
                        ar_error = abs(aspect_ratio - ar_mid) / ar_mid
                        
                        score = ratio_error * 100 + ar_error * 50
                        
                        # Bonus for marginal regions if they're actually marginal
                        if missing_label in [1, 2] and self.is_marginal_position(region, wing_mask):
                            score -= 20  # Bonus
                        
                        if score < best_score and score < 80:  # Reasonable threshold
                            best_score = score
                            best_candidate = region
                    
                    if best_candidate is not None:
                        # Add this region
                        candidate_mask = labeled_candidates == best_candidate.label
                        search_results[candidate_mask] = missing_label
                        assigned_mask |= candidate_mask
                        
                        logger.info(f"Found region {missing_label}: "
                                   f"area={best_candidate.area}, "
                                   f"AR={best_candidate.major_axis_length/best_candidate.minor_axis_length:.1f}, "
                                   f"score={best_score:.1f}")
                    else:
                        logger.warning(f"Could not find suitable candidate for region {missing_label}")
                else:
                    logger.warning(f"No search area available for region {missing_label}")
        
        return search_results

# REPLACE: Add this complete function after your existing processing functions

def enhanced_intervein_processing_with_hybrid_wing_detection(prob_map, raw_img, cfg, output_dir, basename, 
                                                           force_pentagone=False):
    """Enhanced processing using hybrid wing detection (probability + trichome validation)."""
    
    logger.info("Starting enhanced processing with hybrid wing detection...")
    
    # Step 1: Detect trichomes FIRST
    tri_prob = prob_map[..., 0] if prob_map.shape[-1] >= 1 else prob_map
    peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
    
    logger.info(f"Detected {len(peaks)} trichomes for analysis")
    
    # Step 2: Detect veins (can be done in parallel)
    # vein_detector = ImprovedWingVeinDetector(cfg)
    # vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    vein_mask = prob_map[..., 2] > 0.5 if prob_map.shape[-1] >= 3 else np.zeros(prob_map.shape[:2], dtype=bool)
    skeleton = morphology.skeletonize(vein_mask) 
    
    # Step 3: Use hybrid wing detection
    hybrid_detector = HybridWingDetector(cfg)
    wing_mask = hybrid_detector.detect_wing_boundary_hybrid(prob_map, raw_img, peaks)
    
    # Step 4: Create comparison visualization for debugging
    prob_only_mask = hybrid_detector._detect_from_probability_enhanced(prob_map, raw_img)
    comparison_path = os.path.join(output_dir, f"{basename}_wing_detection_comparison.png")
    hybrid_detector.create_detection_comparison_visualization(
        prob_map, raw_img, peaks, prob_only_mask, wing_mask, comparison_path
    )
    
    # Step 5: Validate wing
    if not WingBorderChecker.is_valid_wing(wing_mask, cfg.min_wing_area, cfg.border_buffer):
        logger.warning(f"Skipping {basename} - wing invalid after hybrid detection")
        return None, None, None, None, None
    
    # Step 6: Enhanced string removal adapted for wing sparsity
    trichome_density = len(peaks) / prob_map.size if prob_map.size > 0 else 0
    enhanced_string_filter = EnhancedStringRemovalFilter(cfg)
    #filtered_peaks = enhanced_string_filter.adaptive_string_removal(
        #peaks, prob_map.shape[:2], trichome_density
    #)
    filtered_peaks = peaks
    logger.info(f"After adaptive string removal: {len(filtered_peaks)} trichomes kept")
    
    # Step 7: Continue with enhanced segmentation
    intervein_detector = EnhancedInterveinDetectorWithPentagone(cfg)
    virtual_veins = intervein_detector.create_virtual_boundary_veins(wing_mask, vein_mask)
    combined_vein_mask = vein_mask | virtual_veins
    barrier_mask = morphology.binary_dilation(combined_vein_mask, 
                                             morphology.disk(cfg.vein_width_estimate * 2))
    
    if prob_map.shape[-1] >= 4:
        intervein_prob = prob_map[..., 3]
        intervein_mask = (intervein_prob > cfg.intervein_threshold) & wing_mask & (~barrier_mask)
    else:
        intervein_mask = wing_mask & (~barrier_mask)
    
    intervein_mask = morphology.remove_small_objects(intervein_mask, min_size=5000)
    intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=5000)
    labeled_regions = label(intervein_mask)
    filtered_labeled_mask = intervein_detector._filter_intervein_regions(labeled_regions, wing_mask)
    logger.info("Creating pre-labeling debug visualization...")
    create_pre_labeling_debug_visualization(
        filtered_labeled_mask, wing_mask, prob_map, raw_img,
        output_dir, basename,
        mode_name="merge" if cfg.force_merge_regions_4_5 else "normal"
    )
    
    # Step 8: Apply anatomical labeling with pentagone detection
    if force_pentagone:
        logger.info("FORCED PENTAGONE MODE")
        final_labeled_mask = intervein_detector.pentagone_handler._apply_four_region_labeling(
            filtered_labeled_mask, wing_mask)
        is_pentagone_mode = True
    else:
        # Automatic detection
        final_labeled_mask, is_pentagone = intervein_detector.pentagone_handler.apply_pentagone_labeling(
            filtered_labeled_mask, wing_mask)
        is_pentagone_mode = is_pentagone
    
    # Step 9: Get final regions
    final_regions = regionprops(final_labeled_mask)
    
    # Step 10: Create enhanced visualization showing hybrid approach
    create_hybrid_detection_visualization(
        raw_img, prob_map, peaks, filtered_peaks, wing_mask, vein_mask, 
        virtual_veins, final_labeled_mask, output_dir, basename, is_pentagone_mode, trichome_density
    )
    
    mode_str = "pentagone (4-region)" if is_pentagone_mode else "normal (5-region)"
    density_str = "sparse" if trichome_density < 0.1 else "dense"
    logger.info(f"Hybrid detection found {len(final_regions)} regions using {mode_str} labeling on {density_str} wing")
    is_merged_mode = intervein_detector.is_merged_mode if hasattr(intervein_detector, 'is_merged_mode') else False
    
    return final_labeled_mask, final_regions, combined_vein_mask, skeleton, is_pentagone_mode, is_merged_mode


def create_hybrid_detection_visualization(raw_img, prob_map, all_peaks, filtered_peaks, wing_mask, 
                                        vein_mask, virtual_veins, intervein_regions, output_dir, 
                                        basename, is_pentagone_mode, trichome_density):
    """Create comprehensive visualization showing hybrid detection process."""
    
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    
    bg_img = raw_img if raw_img is not None else prob_map[..., 0]
    density_str = "SPARSE" if trichome_density < 0.1 else "DENSE"
    
    # Row 1: Input data and detection approach
    axes[0, 0].imshow(bg_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Show probability channels
    if prob_map.shape[-1] >= 4:
        # Show the most relevant probability channel
        intervein_prob = prob_map[..., 3]
        axes[0, 1].imshow(intervein_prob, cmap='viridis')
        axes[0, 1].set_title('Intervein Probability')
    else:
        axes[0, 1].imshow(prob_map[..., 0], cmap='viridis')
        axes[0, 1].set_title('Trichome Probability')
    axes[0, 1].axis('off')
    
    # Show trichomes with density info
    axes[0, 2].imshow(bg_img, cmap='gray')
    if len(all_peaks) > 0:
        axes[0, 2].scatter(all_peaks[:, 1], all_peaks[:, 0], c='red', s=2, alpha=0.7)
    axes[0, 2].set_title(f'Trichomes - {density_str} WING\n(n={len(all_peaks)}, density={trichome_density:.6f})')
    axes[0, 2].axis('off')
    
    # Row 2: Wing detection process
    axes[1, 0].imshow(wing_mask, cmap='Blues')
    axes[1, 0].set_title(f'Hybrid Wing Mask\n({np.sum(wing_mask)} pixels)')
    axes[1, 0].axis('off')
    
    # Show string filtering results
    axes[1, 1].imshow(bg_img, cmap='gray')
    if len(all_peaks) > 0:
        axes[1, 1].scatter(all_peaks[:, 1], all_peaks[:, 0], c='red', s=1, alpha=0.4, label='Original')
    if len(filtered_peaks) > 0:
        axes[1, 1].scatter(filtered_peaks[:, 1], filtered_peaks[:, 0], c='blue', s=2, alpha=0.8, label='Kept')
    
    removed_count = len(all_peaks) - len(filtered_peaks)
    axes[1, 1].set_title(f'Adaptive String Removal\nRemoved: {removed_count}, Kept: {len(filtered_peaks)}')
    if len(all_peaks) > 0 and len(filtered_peaks) > 0:
        axes[1, 1].legend()
    axes[1, 1].axis('off')
    
    # Show vein detection
    axes[1, 2].imshow(bg_img, cmap='gray')
    axes[1, 2].imshow(vein_mask, cmap='Reds', alpha=0.7)
    axes[1, 2].imshow(virtual_veins, cmap='Blues', alpha=0.6)
    axes[1, 2].set_title('Vein Network\n(Red: Real, Blue: Virtual)')
    axes[1, 2].axis('off')
    
    # Row 3: Final results
    if intervein_regions is not None:
        max_label = 4 if is_pentagone_mode else 5
        axes[2, 0].imshow(intervein_regions, cmap='nipy_spectral', vmin=0, vmax=max_label)
        
        # Add region labels
        regions = regionprops(intervein_regions)
        if is_pentagone_mode:
            pentagone_handler = PentagoneMutantHandler()
            target_regions = pentagone_handler.pentagone_target_regions
        else:
            template_matcher = AnatomicalTemplateMatcher()
            target_regions = template_matcher.target_regions
        
        for region in regions:
            if region.label > 0:
                region_name = target_regions.get(region.label, {}).get('name', f'Region-{region.label}')
                cy, cx = region.centroid
                axes[2, 0].text(cx, cy, f"{region.label}\n{region_name}", 
                               color='white', fontsize=8, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
        
        mode_str = "Pentagone (4-region)" if is_pentagone_mode else "Normal (5-region)"
        axes[2, 0].set_title(f'Final Regions - {mode_str}\n(n={len(regions)})')
    else:
        axes[2, 0].text(0.5, 0.5, 'No regions detected', ha='center', va='center')
        axes[2, 0].set_title('Final Regions')
    axes[2, 0].axis('off')
    
    # Complete overlay showing method used
    axes[2, 1].imshow(bg_img, cmap='gray')
    if wing_mask is not None:
        axes[2, 1].imshow(wing_mask, cmap='Blues', alpha=0.3)
    axes[2, 1].imshow(vein_mask, cmap='Reds', alpha=0.5)
    if intervein_regions is not None:
        axes[2, 1].imshow(intervein_regions > 0, cmap='Greens', alpha=0.4)
    if len(filtered_peaks) > 0:
        sample_peaks = filtered_peaks[::max(1, len(filtered_peaks)//300)]
        axes[2, 1].scatter(sample_peaks[:, 1], sample_peaks[:, 0], c='yellow', s=1, alpha=0.8)
    
    detection_method = "Probability-Dominant" if trichome_density < 0.1 else "Hybrid (Prob + Trichome)"
    axes[2, 1].set_title(f'Complete Analysis\nMethod: {detection_method}')
    axes[2, 1].axis('off')
    
    # Summary statistics
    wing_area = np.sum(wing_mask) if wing_mask is not None else 0
    vein_coverage = np.sum(vein_mask) / wing_area * 100 if wing_area > 0 else 0
    virtual_coverage = np.sum(virtual_veins) / wing_area * 100 if wing_area > 0 else 0
    
    stats_text = f"""HYBRID DETECTION SUMMARY

Wing Classification: {density_str} WING
Detection Method: {detection_method}
Analysis Mode: {mode_str if intervein_regions is not None else 'Failed'}

TRICHOME ANALYSIS:
- Original peaks: {len(all_peaks)}
- After string removal: {len(filtered_peaks)}
- Density: {trichome_density:.6f} per pixel
- Removal rate: {(len(all_peaks) - len(filtered_peaks))/max(len(all_peaks), 1)*100:.1f}%

WING DETECTION:
- Wing area: {wing_area:,} pixels
- Method used: {'Probability-dominant (sparse)' if trichome_density < 0.1 else 'Hybrid validation'}

VEIN ANALYSIS:
- Real vein coverage: {vein_coverage:.2f}%
- Virtual vein coverage: {virtual_coverage:.2f}%
- Total vein pixels: {np.sum(vein_mask | virtual_veins):,}

REGION ANALYSIS:
- Regions detected: {len(regions) if intervein_regions is not None else 0}
- Expected regions: {4 if is_pentagone_mode else 5}
- Success rate: {len(regions)/(4 if is_pentagone_mode else 5)*100:.1f}% if intervein_regions is not None else 0%

ADAPTIVE FEATURES:
- String removal: {'Conservative (sparse)' if trichome_density < 0.1 else 'Standard (dense)'}
- Wing validation: {'Probability-only' if trichome_density < 0.05 else 'Trichome-validated'}
"""
    
    axes[2, 2].text(0.05, 0.95, stats_text, transform=axes[2, 2].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=9)
    axes[2, 2].set_title('Analysis Summary')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{basename}_hybrid_detection_analysis.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved hybrid detection visualization to {save_path}")
# Modified main processing function that uses improved detection
def enhanced_intervein_processing_with_improved_detection(prob_map, raw_img, cfg, output_dir, basename, 
                                                        force_pentagone=False):
    """Complete processing with improved small region detection."""
    
    logger.info("Starting enhanced processing with improved small region detection...")
    
    # Step 1: Detect veins
    # vein_detector = ImprovedWingVeinDetector(cfg)
    # vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    vein_mask = prob_map[..., 2] > 0.5 if prob_map.shape[-1] >= 3 else np.zeros(prob_map.shape[:2], dtype=bool)
    skeleton = morphology.skeletonize(vein_mask) 
    
    # Step 2: Detect wing boundary
    if raw_img is not None:
        # Use existing method
        intervein_detector = EnhancedInterveinDetectorWithPentagone(cfg)
        wing_mask = intervein_detector.detect_wing_boundary(prob_map, raw_img)
    else:
        # Fallback method
        if prob_map.shape[-1] >= 4:
            intervein_prob = prob_map[..., 3]
            wing_mask = intervein_prob > 0.3
            wing_mask = morphology.remove_small_objects(wing_mask, min_size=50000)
        else:
            raise ValueError("Need raw image or 4-channel probability map")
    
    # Check if wing is valid
    if not WingBorderChecker.is_valid_wing(wing_mask, cfg.min_wing_area, cfg.border_buffer):
        logger.warning(f"Skipping {basename} - wing touching border or too small")
        return None, None, None, None, None
    
    # Step 3: Use improved region detector
    improved_detector = ImprovedRegionDetector(cfg)
    intervein_regions, enhanced_prob, filled_mask = improved_detector.segment_with_marginal_region_recovery(
        prob_map, vein_mask, wing_mask, raw_img
    )
    
    # Step 4: Apply pentagone detection if requested
    if force_pentagone:
        logger.info("Applying forced pentagone mode to improved detection results...")
        pentagone_handler = PentagoneMutantHandler()
        intervein_regions = pentagone_handler._apply_four_region_labeling(intervein_regions, wing_mask)
        is_pentagone_mode = True
    else:
        # Check if this looks like a pentagone wing
        pentagone_handler = PentagoneMutantHandler()
        is_pentagone, reason = pentagone_handler.detect_pentagone_pattern(intervein_regions, wing_mask)
        
        if is_pentagone:
            logger.info(f"Pentagone pattern detected: {reason}")
            intervein_regions = pentagone_handler._apply_four_region_labeling(intervein_regions, wing_mask)
            is_pentagone_mode = True
        else:
            is_pentagone_mode = False
    
    # Step 5: Get final regions
    final_regions = regionprops(intervein_regions)
    
    # Step 6: Create visualization
    create_improved_detection_visualization(
        raw_img, prob_map, enhanced_prob, vein_mask, intervein_regions, 
        output_dir, basename, is_pentagone_mode
    )
    
    logger.info(f"Improved detection completed: {len(final_regions)} regions found "
               f"({'pentagone' if is_pentagone_mode else 'normal'} mode)")
    
    return intervein_regions, final_regions, vein_mask, skeleton, is_pentagone_mode


def create_improved_detection_visualization(raw_img, prob_map, enhanced_prob, vein_mask, 
                                          intervein_regions, output_dir, basename, is_pentagone_mode):
    """Create visualization showing the improved detection process."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Choose background
    bg_img = raw_img if raw_img is not None else prob_map[..., 0]
    
    # Top row: Original, enhanced probability, and detected regions
    axes[0, 0].imshow(bg_img, cmap='gray')
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(prob_map[..., 3] if prob_map.shape[-1] >= 4 else bg_img, cmap='viridis')
    axes[0, 1].set_title('Original Intervein Probability')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(enhanced_prob, cmap='viridis')
    axes[0, 2].set_title('Enhanced Probability (Holes Filled)')
    axes[0, 2].axis('off')
    
    # Bottom row: Veins, final regions, and overlay
    axes[1, 0].imshow(bg_img, cmap='gray')
    axes[1, 0].imshow(vein_mask, cmap='Reds', alpha=0.7)
    axes[1, 0].set_title('Detected Veins')
    axes[1, 0].axis('off')
    
    # Final regions with labels
    max_label = 6 if not is_pentagone_mode else 4
    axes[1, 1].imshow(intervein_regions, cmap='nipy_spectral', vmin=0, vmax=max_label)
    
    # Add region labels
    regions = regionprops(intervein_regions)
    template_matcher = AnatomicalTemplateMatcher()
    
    for region in regions:
        if region.label > 0:
            if is_pentagone_mode and region.label == 4:
                region_name = "Region-4+5-Combined"
            else:
                region_name = template_matcher.target_regions.get(region.label, {}).get('name', f'Region-{region.label}')
            
            cy, cx = region.centroid
            axes[1, 1].text(cx, cy, f"{region.label}\n{region_name}", 
                           color='white', fontsize=8, ha='center', va='center',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    mode_str = "Pentagone (4-region)" if is_pentagone_mode else "Normal (5-region)"
    axes[1, 1].set_title(f'Final Regions - {mode_str} (n={len(regions)})')
    axes[1, 1].axis('off')
    
    # Complete overlay
    axes[1, 2].imshow(bg_img, cmap='gray')
    axes[1, 2].imshow(vein_mask, cmap='Reds', alpha=0.5)
    axes[1, 2].imshow(intervein_regions > 0, cmap='Blues', alpha=0.4)
    axes[1, 2].set_title('Complete Overlay')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    mode_suffix = "_pentagone" if is_pentagone_mode else "_normal"
    save_path = os.path.join(output_dir, f"{basename}_improved_detection{mode_suffix}.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved improved detection visualization to {save_path}")
    plt.close()


# Updated main function using improved detection
def main_with_improved_detection(directory: str, cfg: TrichomeDetectionConfig = CONFIG, 
                                output_directory: Optional[str] = None, 
                                force_pentagone_mode: bool = False):
    """Main function with improved small region detection."""
    
    cfg.validate()
    
    if output_directory is None:
        output_directory = directory
    
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_directory, "improved_detection.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info("IMPROVED TRICHOME DETECTION WITH ENHANCED SMALL REGION RECOVERY")
    logger.info("="*60)
    
    mapping = find_associated_files(directory)
    if not mapping:
        logger.error("No probability files found")
        return
    
    total_files = len(mapping)
    processed_files = 0
    failed_files = 0
    skipped_files = 0
    five_region_wings = 0
    pentagone_wings = 0
    total_peaks = 0
    total_regions = 0
    
    for basename, files in mapping.items():
        logger.info(f"\nProcessing: {basename}")
        logger.info("="*40)
        
        try:
            # Load data
            prob_file = files["probabilities"]
            raw_file = files.get("raw")
            
            tri_prob, inter_prob, full_prob = load_probability_map(prob_file)
            raw_img = load_raw_image(raw_file) if raw_file else None
            
            # Peak detection
            peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
            total_peaks += len(peaks)
            
            # Save peak results
            save_peak_coordinates_enhanced(peaks, basename, output_directory, metrics, tri_prob)
            create_detection_visualization(tri_prob, peaks, basename, output_directory, raw_img)
            
            # Enhanced region detection
            if full_prob.shape[-1] >= 4 and inter_prob is not None:
                result = enhanced_intervein_processing_with_improved_detection(
                    full_prob, raw_img, cfg, output_directory, basename, force_pentagone_mode
                )
                
                if result[0] is None:
                    logger.warning(f"Skipped {basename} - wing invalid")
                    skipped_files += 1
                    continue
                
                labeled_mask, valid_regions, vein_mask, skeleton, is_pentagone_mode, is_merged_mode = result
                
                # Track statistics
                if is_pentagone_mode:
                    pentagone_wings += 1
                else:
                    five_region_wings += 1
                
                total_regions += len(valid_regions)
                
                # Assign peaks and analyze
                region_peaks_dict = assign_peaks_to_regions(peaks, labeled_mask, valid_regions)
                
                # Voronoi analysis
                voronoi_results = {}
                for region in valid_regions:
                    label_val = region.label
                    region_peaks = region_peaks_dict.get(label_val, np.empty((0, 2)))
                    
                    if region_peaks.shape[0] >= 2:
                        stats = voronoi_average_cell_stats(region, region_peaks, cfg)
                        voronoi_results[label_val] = stats
                    else:
                        voronoi_results[label_val] = None
                
                # Save results
                save_voronoi_results_with_mode(voronoi_results, basename, output_directory, is_pentagone_mode)
                save_anatomical_region_info_with_pentagone(valid_regions, basename, output_directory, is_pentagone_mode)
                
                # Enhanced visualization
                if valid_regions:
                    background_img = raw_img if raw_img is not None else tri_prob
                    plot_voronoi_with_pentagone_info(
                        valid_regions, region_peaks_dict, background_img, 
                        vein_mask, skeleton, output_directory, basename, cfg, is_pentagone_mode
                    )
                
                logger.info(f"Successfully processed {basename} in {'pentagone' if is_pentagone_mode else 'normal'} mode")
                logger.info(f"  Detected {len(valid_regions)} regions")
                
                # Log region details
                for region in valid_regions:
                    region_name = "Combined-4+5" if (is_pentagone_mode and region.label == 4) else f"Region-{region.label}"
                    wing_area = np.sum(labeled_mask > 0)
                    region_ratio = region.area / wing_area if wing_area > 0 else 0
                    aspect_ratio = region.major_axis_length / (region.minor_axis_length + 1e-8)
                    
                    logger.info(f"    {region_name}: area={region.area} ({region_ratio:.3f} of wing), "
                               f"AR={aspect_ratio:.1f}, solidity={region.solidity:.3f}")
            
            else:
                logger.info("Skipping intervein analysis (4th channel missing)")
            
            processed_files += 1
            
        except Exception as e:
            failed_files += 1
            logger.error(f"Error processing {basename}: {str(e)}", exc_info=True)
            continue
    
    # Final summary
    logger.info("="*60)
    logger.info("DETECTION ANALYSIS COMPLETE")
    logger.info("="*60)
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"  - Complete 5-region wings: {five_region_wings}")
    logger.info(f"  - Pentagone wings: {pentagone_wings}")
    logger.info(f"Skipped (invalid): {skipped_files}")
    logger.info(f"Failed: {failed_files}")
    logger.info(f"Total regions detected: {total_regions}")
    logger.info(f"Average regions per wing: {total_regions/max(processed_files, 1):.1f}")
    
    if five_region_wings > 0:
        logger.info(f"5-region detection success rate: {five_region_wings}/{processed_files} = {five_region_wings/processed_files*100:.1f}%")
    
    # Create summary report
    summary_path = os.path.join(output_directory, "improved_detection_summary.txt")
    with open(summary_path, "w") as f:
        f.write("IMPROVED TRICHOME DETECTION - ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {directory}\n")
        f.write(f"Output Directory: {output_directory}\n")
        f.write("\nIMPROVEMENTS APPLIED:\n")
        f.write("- Trichome hole filling in intervein probability\n")
        f.write("- Multi-threshold segmentation for different region types\n")
        f.write("- Marginal region recovery (regions 1 & 2)\n")
        f.write("- Enhanced filtering for small elongated regions\n")
        f.write("- Template-guided missing region search\n")
        f.write("\nPROCESSING SUMMARY:\n")
        f.write(f"Total files: {total_files}\n")
        f.write(f"Processed: {processed_files}\n")
        f.write(f"Complete 5-region wings: {five_region_wings}\n")
        f.write(f"Pentagone wings: {pentagone_wings}\n")
        f.write(f"Failed: {failed_files}\n")
        f.write(f"Skipped: {skipped_files}\n")
        f.write(f"\nDETECTION PERFORMANCE:\n")
        f.write(f"Total regions: {total_regions}\n")
        f.write(f"Average per wing: {total_regions/max(processed_files, 1):.1f}\n")
        f.write(f"5-region success rate: {five_region_wings/max(processed_files, 1)*100:.1f}%\n")
    
    logger.info(f"Summary saved to {summary_path}")


# Configuration adjustments for better small region detection
def configure_for_small_regions(config):
    """Adjust configuration parameters for better small region detection."""
    
    # Make filtering more lenient for small regions
    config.auto_intervein_min_area = 200  # Reduced from 8000
    config.min_intervein_solidity = 0.5    # Reduced from 0.6
    config.intervein_threshold = 0.35      # Reduced from 0.4
    
    # Adjust vein detection to be less aggressive
    config.vein_width_estimate = 2      # Reduced from 10
    
    logger.info("Applied configuration adjustments for small region detection")
    return config



# Quick fix function to apply to existing EnhancedInterveinDetector
def apply_small_region_improvements(existing_detector):
    """Apply improvements to existing detector instance."""
    
    # Add the improved region detector
    existing_detector.improved_detector = ImprovedRegionDetector(existing_detector.config)
    
    # Modify the segmentation method
    original_segment = existing_detector.segment_intervein_regions_enhanced
    
    def enhanced_segment_with_improvements(prob_map, vein_mask, raw_img=None, force_pentagone=False):
        """Enhanced segmentation that uses improved detection for better small region recovery."""
        
        logger.info("Using enhanced segmentation with small region improvements...")
        
        # Detect wing boundary (same as before)
        wing_mask = existing_detector.detect_wing_boundary(prob_map, raw_img)
        
        if not WingBorderChecker.is_valid_wing(wing_mask, existing_detector.config.min_wing_area, 
                                               existing_detector.config.border_buffer):
            return None, None, None
        
        # Use improved detection
        intervein_regions, enhanced_prob, filled_mask = existing_detector.improved_detector.segment_with_marginal_region_recovery(
            prob_map, vein_mask, wing_mask, raw_img
        )
        
        # Create virtual veins for visualization (same as before)
        virtual_veins = existing_detector.create_virtual_boundary_veins(wing_mask, vein_mask)
        combined_vein_mask = vein_mask | virtual_veins
        
        # Apply pentagone detection if needed
        if force_pentagone:
            pentagone_handler = PentagoneMutantHandler()
            intervein_regions = pentagone_handler._apply_four_region_labeling(intervein_regions, wing_mask)
            existing_detector.is_pentagone_mode = True
        else:
            pentagone_handler = PentagoneMutantHandler()
            is_pentagone, reason = pentagone_handler.detect_pentagone_pattern(intervein_regions, wing_mask)
            
            if is_pentagone:
                logger.info(f"Pentagone detected: {reason}")
                intervein_regions = pentagone_handler._apply_four_region_labeling(intervein_regions, wing_mask)
                existing_detector.is_pentagone_mode = True
            else:
                existing_detector.is_pentagone_mode = False
        
        return intervein_regions, combined_vein_mask, virtual_veins
    

    existing_detector.segment_intervein_regions_enhanced = enhanced_segment_with_improvements
    
    logger.info("Applied small region improvements to existing detector")
    return existing_detector
def build_local_threshold_mask(proc: np.ndarray, cfg: TrichomeDetectionConfig) -> np.ndarray:
    """Build adaptive threshold mask with edge handling."""
    # Ensure block size is appropriate for image size
    block_size = min(cfg.local_block_size, min(proc.shape) // 3)
    if block_size % 2 == 0:
        block_size += 1
    
    if block_size != cfg.local_block_size:
        logger.warning(f"Adjusted block_size from {cfg.local_block_size} to {block_size}")
    
    thresh_img = threshold_local(
        proc, 
        block_size=block_size, 
        offset=cfg.local_offset, 
        method="gaussian"
    )
    
    mask = proc > thresh_img
    logger.info(f"Local threshold mask: {np.sum(mask)} pixels above threshold")
    
    return mask

def multi_scale_peak_detection(proc: np.ndarray, cfg: TrichomeDetectionConfig) -> np.ndarray:
    """Multi-scale peak detection for enhanced reliability."""
    all_peaks = []
    weights = []
    
    for i, scale in enumerate(cfg.scales):
        if scale != 1.0:
            # Scale the image
            scaled = gaussian_filter(proc, sigma=scale)
        else:
            scaled = proc
        
        # Detect peaks at this scale
        peaks = peak_local_max(
            scaled,
            min_distance=max(1, int(cfg.min_distance / scale)),
            threshold_abs=cfg.high_thresh_abs,
            exclude_border=cfg.edge_exclusion_buffer
        )
        
        if peaks.size > 0:
            all_peaks.append(peaks)
            # Weight peaks by scale (center scales get higher weight)
            weight = cfg.scale_weight_decay ** abs(i - len(cfg.scales) // 2)
            weights.extend([weight] * len(peaks))
            logger.info(f"Scale {scale}: found {len(peaks)} peaks")
    
    if not all_peaks:
        logger.warning("No peaks found at any scale")
        return np.empty((0, 2), dtype=int)
    
    # Combine all peaks
    combined_peaks = np.vstack(all_peaks)
    weights = np.array(weights)
    
    # Remove duplicates using clustering
    if len(combined_peaks) > 1:
        clustering = DBSCAN(eps=cfg.min_distance, min_samples=1).fit(combined_peaks)
        final_peaks = []
        
        for label in np.unique(clustering.labels_):
            cluster_mask = clustering.labels_ == label
            cluster_peaks = combined_peaks[cluster_mask]
            cluster_weights = weights[cluster_mask]
            
            # Keep the highest weighted peak in each cluster
            best_idx = np.argmax(cluster_weights)
            final_peaks.append(cluster_peaks[best_idx])
        
        return np.array(final_peaks, dtype=int)
    
    return combined_peaks

def detect_trichome_peaks(prob: np.ndarray, cfg: TrichomeDetectionConfig) -> Tuple[np.ndarray, PeakDetectionMetrics]:
    """Enhanced peak detection with comprehensive quality control."""
    metrics = PeakDetectionMetrics()
    start_time = time.time()
    
    try:
        # Preprocessing
        proc = preprocess_probability_map(prob, cfg)
        
        # Build mask if requested
        mask = build_local_threshold_mask(proc, cfg) if cfg.use_local_threshold else None
        
        # Multi-scale detection
        metrics.scales_used = cfg.scales
        peaks = multi_scale_peak_detection(proc, cfg)
        metrics.raw_peaks = len(peaks)
        
        if peaks.size == 0:
            logger.warning("No peaks detected in initial pass")
            metrics.processing_time = time.time() - start_time
            return peaks, metrics
        
        # Apply mask filter
        if mask is not None:
            valid_peaks = mask[peaks[:, 0], peaks[:, 1]]
            peaks = peaks[valid_peaks]
            metrics.filtered_peaks = len(peaks)

        # Intensity filter
        intensities = proc[peaks[:, 0], peaks[:, 1]]
        intensity_mask = intensities >= cfg.min_peak_intensity
        peaks = peaks[intensity_mask]
        metrics.intensity_filtered = metrics.filtered_peaks - len(peaks)
        
        # Edge exclusion (additional safety)
        h, w = prob.shape
        edge_mask = (
            (peaks[:, 0] >= cfg.edge_exclusion_buffer) &
            (peaks[:, 0] < h - cfg.edge_exclusion_buffer) &
            (peaks[:, 1] >= cfg.edge_exclusion_buffer) &
            (peaks[:, 1] < w - cfg.edge_exclusion_buffer)
        )
        peaks = peaks[edge_mask]
        metrics.edge_excluded = np.sum(~edge_mask)
        
        # Density check
        density = len(peaks) / (h * w)
        if density > cfg.max_peak_density:
            logger.warning(f"Peak density {density:.6f} exceeds maximum {cfg.max_peak_density}")
        
        # Final clustering and filtering
        if len(peaks) > 1:
            peaks = _cluster_and_filter(peaks, proc, cfg)
            metrics.cluster_merged = metrics.filtered_peaks - len(peaks)
        
        metrics.final_peaks = len(peaks)
        metrics.processing_time = time.time() - start_time
        
        logger.info(f"Peak detection completed: {metrics.final_peaks} peaks in {metrics.processing_time:.2f}s")
        
        return peaks, metrics
        
    except Exception as e:
        logger.error(f"Error in peak detection: {e}")
        metrics.processing_time = time.time() - start_time
        return np.empty((0, 2), dtype=int), metrics

def _cluster_and_filter(peaks: np.ndarray, proc: np.ndarray, cfg: TrichomeDetectionConfig) -> np.ndarray:
    """Enhanced clustering with better valley detection."""
    if peaks.shape[0] < 2:
        return peaks

    clustering = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(peaks)
    final_peaks = []

    for label in np.unique(clustering.labels_):
        cluster_peaks = peaks[clustering.labels_ == label]
        if len(cluster_peaks) == 1:
            final_peaks.append(cluster_peaks[0])
            continue

        # Sort by intensity (descending)
        intensities = [proc[tuple(p)] for p in cluster_peaks]
        sorted_indices = np.argsort(intensities)[::-1]
        cluster_peaks = cluster_peaks[sorted_indices]
        
        # Keep the strongest peak
        kept_peaks = [cluster_peaks[0]]
        
        # Check remaining peaks against kept ones
        for candidate in cluster_peaks[1:]:
            should_keep = True
            for kept_peak in kept_peaks:
                valley_intensity = _intensity_valley(proc, candidate, kept_peak, samples=20)
                min_peak_intensity = min(proc[tuple(candidate)], proc[tuple(kept_peak)])
                
                # More stringent valley criterion
                if valley_intensity > min_peak_intensity * (1 - cfg.valley_drop):
                    should_keep = False
                    break
            
            if should_keep:
                kept_peaks.append(candidate)
        
        final_peaks.extend(kept_peaks)

    return np.array(final_peaks, dtype=int)

def _intensity_valley(proc: np.ndarray, p1: np.ndarray, p2: np.ndarray, samples: int = 20) -> float:
    """Improved valley detection with more samples."""
    if np.array_equal(p1, p2):
        return proc[tuple(p1)]
    
    # Create line between points
    y_coords = np.linspace(p1[0], p2[0], samples)
    x_coords = np.linspace(p1[1], p2[1], samples)
    
    # Ensure coordinates are within bounds
    y_coords = np.clip(y_coords, 0, proc.shape[0] - 1).astype(int)
    x_coords = np.clip(x_coords, 0, proc.shape[1] - 1).astype(int)
    
    # Get intensities along the line
    line_intensities = proc[y_coords, x_coords]
    
    return np.min(line_intensities)

def save_peak_coordinates_enhanced(peaks: np.ndarray, basename: str, output_dir: str,
                                 metrics: PeakDetectionMetrics, prob_map: np.ndarray) -> None:
    """Save peak coordinates with additional metadata and quality metrics."""
    output_path = os.path.join(output_dir, f"{basename}_peak_coordinates.csv")

    peaks = np.asarray(peaks)
    if peaks.size == 0:
        coord_peaks = np.empty((0, 2), dtype=int)
    else:
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)
        coord_peaks = np.asarray(peaks[:, :2], dtype=int)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Write header with metadata
        writer.writerow(["# Trichome Peak Detection Results"])
        writer.writerow([f"# File: {basename}"])
        writer.writerow([f"# Total peaks detected: {len(peaks)}"])
        writer.writerow([f"# Processing time: {metrics.processing_time:.2f}s"])
        writer.writerow([f"# Detection efficiency: {(metrics.final_peaks/max(metrics.raw_peaks,1)*100):.1f}%"])
        writer.writerow(["# "])
        
        # Data header
        writer.writerow(["row", "col", "intensity", "local_max_score"])

        # Write peak data with additional metrics
        for r, c in coord_peaks:
            intensity = prob_map[r, c]

            # Calculate local maximum score (how much higher than neighborhood)
            neighborhood = prob_map[max(0, r-3):r+4, max(0, c-3):c+4]
            local_max_score = intensity - np.median(neighborhood) if neighborhood.size > 1 else 0

            writer.writerow([r, c, f"{intensity:.4f}", f"{local_max_score:.4f}"])

    # Also save the metrics report
    metrics_path = os.path.join(output_dir, f"{basename}_detection_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(metrics.report())
    
    logger.info(f"Saved peak coordinates to {output_path}")
    logger.info(f"Saved detection metrics to {metrics_path}")

def create_detection_visualization(prob_map: np.ndarray, peaks: np.ndarray, basename: str,
                                 output_dir: str, raw_img: Optional[np.ndarray] = None) -> None:
    """Create comprehensive visualization of detection results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Background image
    bg_img = raw_img if raw_img is not None else prob_map

    peaks = np.asarray(peaks)
    if peaks.size == 0:
        coord_peaks = np.empty((0, 2), dtype=int)
    else:
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)
        coord_peaks = np.asarray(peaks[:, :2], dtype=int)

    # Top-left: Original probability map
    axes[0, 0].imshow(prob_map, cmap='viridis')
    axes[0, 0].set_title('Trichome Probability Map')
    axes[0, 0].axis('off')

    # Top-right: Detected peaks overlay
    axes[0, 1].imshow(bg_img, cmap='gray')
    if coord_peaks.size > 0:
        axes[0, 1].scatter(coord_peaks[:, 1], coord_peaks[:, 0], c='red', s=20, alpha=0.7)
    axes[0, 1].set_title(f'Detected Peaks (n={len(coord_peaks)})')
    axes[0, 1].axis('off')

    # Bottom-left: Peak intensity histogram
    if coord_peaks.size > 0:
        intensities = prob_map[coord_peaks[:, 0], coord_peaks[:, 1]]
        axes[1, 0].hist(intensities, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(np.mean(intensities), color='red', linestyle='--',
                          label=f'Mean: {np.mean(intensities):.3f}')
        axes[1, 0].set_xlabel('Peak Intensity')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Peak Intensity Distribution')
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No peaks detected', ha='center', va='center')
        axes[1, 0].set_title('Peak Intensity Distribution')

    # Bottom-right: Spatial distribution
    axes[1, 1].imshow(bg_img, cmap='gray', alpha=0.5)
    if coord_peaks.size > 0:
        # Create density map
        from scipy.stats import gaussian_kde
        if len(coord_peaks) > 1:
            try:
                kde = gaussian_kde(coord_peaks[:, :2].T)
                y, x = np.mgrid[0:prob_map.shape[0]:50j, 0:prob_map.shape[1]:50j]
                positions = np.vstack([x.ravel(), y.ravel()])
                density = kde(positions).reshape(x.shape)

                contour = axes[1, 1].contour(x, y, density, levels=5, alpha=0.7, cmap='Reds')
                axes[1, 1].clabel(contour, inline=True, fontsize=8)
            except:
                pass  # Skip density plot if it fails

        axes[1, 1].scatter(coord_peaks[:, 1], coord_peaks[:, 0], c='red', s=10, alpha=0.8)

    axes[1, 1].set_title('Spatial Distribution with Density Contours')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f"{basename}_detection_visualization.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved detection visualization to {save_path}")

def assign_peaks_to_regions(peaks, labeled_mask, valid_regions):
    """Assign detected peaks to their corresponding region with improved accuracy."""
    from matplotlib.path import Path

    peaks = np.asarray(peaks)
    if peaks.size == 0:
        coord_peaks = np.empty((0, 2), dtype=int)
    else:
        if peaks.ndim == 1:
            peaks = peaks.reshape(1, -1)
        coord_peaks = np.asarray(peaks[:, :2], dtype=int)

    region_polygons = {}
    for region in valid_regions:
        label_val = region.label
        points = region.coords
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_points = points[hull.vertices]
                region_polygons[label_val] = Path(
                    np.column_stack((hull_points[:, 1], hull_points[:, 0]))
                )
            except:
                region_polygons[label_val] = None
        else:
            region_polygons[label_val] = None
    
    region_peaks_dict = {region.label: [] for region in valid_regions}
    unassigned_peaks = []
    
    for peak in coord_peaks:
        r, c = map(int, peak[:2])
        # First, try direct mask lookup
        region_label = labeled_mask[r, c]
        if region_label != 0 and region_label in region_peaks_dict:
            region_peaks_dict[region_label].append(np.array([r, c], dtype=int))
        else:
            # Fall back to polygon containment check
            assigned = False
            for label_val, path in region_polygons.items():
                if path is not None and path.contains_point((c, r)):
                    region_peaks_dict[label_val].append(np.array([r, c], dtype=int))
                    assigned = True
                    break
            if not assigned:
                unassigned_peaks.append(np.array([r, c], dtype=int))
    
    if unassigned_peaks:
        logger.warning(f"{len(unassigned_peaks)} peaks could not be assigned to regions")
    
    # Convert lists to arrays
    for label_val in region_peaks_dict:
        if region_peaks_dict[label_val]:
            region_peaks_dict[label_val] = np.array(region_peaks_dict[label_val], dtype=int)
        else:
            region_peaks_dict[label_val] = np.empty((0, 2), dtype=int)

    return region_peaks_dict

def voronoi_average_cell_stats(region, region_peaks, cfg=CONFIG):
    """Enhanced Voronoi analysis with better error handling and validation."""
    
    # Basic sanity checks
    region_coords = np.asarray(region.coords)
    if region_coords.shape[0] < 3 or region_peaks.shape[0] < 2:
        logger.warning(f"Region {region.label}: insufficient data for Voronoi analysis")
        return None

    try:
        # Convex hull of region (for clipping)
        hull = ConvexHull(region_coords)
        hull_pts = region_coords[hull.vertices]
        region_poly = Polygon([(p[1], p[0]) for p in hull_pts]).buffer(0)
    except Exception as e:
        logger.error(f"Convex-hull error in region {region.label}: {e}")
        return None

    # Voronoi tessellation of the peak coordinates
    points_vor = np.asarray([(p[1], p[0]) for p in region_peaks])
    
    try:
        vor = Voronoi(points_vor)
        regions_vor, vertices = voronoi_finite_polygons_2d(vor)
    except Exception as e:
        logger.error(f"Voronoi tessellation error in region {region.label}: {e}")
        return None

    # Keep polygons and areas aligned with point index
    n_pts = len(points_vor)
    cell_polys = [None] * n_pts
    cell_areas = np.full(n_pts, np.nan, dtype=float)

    for i, reg in enumerate(regions_vor):
        try:
            poly = Polygon(vertices[reg]).buffer(0).intersection(region_poly)
            if not poly.is_empty and poly.area > 0:
                cell_polys[i] = poly
                cell_areas[i] = poly.area
        except Exception as e:
            logger.debug(f"Error processing Voronoi cell {i} in region {region.label}: {e}")
            continue

    valid_idx = ~np.isnan(cell_areas)
    if not valid_idx.any():
        logger.warning(f"Region {region.label}: no valid Voronoi cells")
        return None

    # Neighbour-consistency filter
    adj = _voronoi_adjacency(vor)
    keep_idx = np.zeros(n_pts, dtype=bool)

    for i, area in enumerate(cell_areas):
        if np.isnan(area):
            continue
        nbrs = [j for j in adj[i] if not np.isnan(cell_areas[j])]
        if len(nbrs) < cfg.min_neighbours:
            continue
        med = np.median(cell_areas[nbrs])
        if med > 0 and abs(area - med) < cfg.neighbour_tolerance * med:
            keep_idx[i] = True

    kept_areas = cell_areas[keep_idx]
    if kept_areas.size == 0:
        logger.warning(f"Region {region.label}: all areas removed by neighbour filtering")
        return None

    # IQR outlier removal on the kept set
    Q1, Q3 = np.percentile(kept_areas, (25, 75))
    IQR = Q3 - Q1
    if IQR > 0:
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        final = kept_areas[(kept_areas >= lower) & (kept_areas <= upper)]
    else:
        final = kept_areas
    
    if final.size == 0:  # fallback if IQR removed everything
        final = kept_areas
        logger.warning(f"Region {region.label}: IQR filtering removed all cells, using all kept cells")

    # Calculate statistics
    region_area = region.area
    
    average_cell_area = float(np.mean(final))
    std_cell_area = float(np.std(final))
    percentage_measured = 100.0 * final.size / valid_idx.sum()
    
    # Additional quality metrics
    cv_cell_area = std_cell_area / average_cell_area if average_cell_area > 0 else 0
    
    # Get region name from template
    template_matcher = AnatomicalTemplateMatcher()
    region_name = template_matcher.target_regions.get(region.label, {}).get('name', f'Region_{region.label}')
    
    logger.info(f"Region {region.label} ({region_name}): {final.size} cells, avg area {average_cell_area:.1f}, CV {cv_cell_area:.3f}")

    return {
        "region_area": region_area,
        "average_cell_area": average_cell_area,
        "std_cell_area": std_cell_area,
        "cv_cell_area": cv_cell_area,
        "percentage_measured": percentage_measured,
        "n_cells_measured": int(final.size),
        "n_cells_total": int(valid_idx.sum()),
        "region_name": region_name
    }

def _voronoi_adjacency(vor: Voronoi) -> Dict[int, set]:
    """Return adjacency dictionary from Voronoi ridges."""
    adj = {i: set() for i in range(len(vor.points))}
    for p, q in vor.ridge_points:
        adj[p].add(q)
        adj[q].add(p)
    return adj








def voronoi_finite_polygons_2d(vor, radius=None):
    """Reconstruct infinite Voronoi regions in a 2D diagram to finite regions."""
    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")
        
    new_regions = []
    new_vertices = vor.vertices.tolist()
    center = vor.points.mean(axis=0)
    if radius is None:
        radius = np.ptp(vor.points, axis=0).max() * 2

    # Map ridge vertices to each point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))
        
    for p1, region_index in enumerate(vor.point_region):
        vertices_indices = vor.regions[region_index]
        if all(v >= 0 for v in vertices_indices):
            new_regions.append(vertices_indices)
            continue
        ridges = all_ridges[p1]
        new_region = [v for v in vertices_indices if v >= 0]
        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0 and v2 >= 0:
                continue
            t = vor.points[p2] - vor.points[p1]
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])
            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius
            new_vertices.append(far_point.tolist())
            new_region.append(len(new_vertices) - 1)
        new_regions.append(new_region)
    return new_regions, np.asarray(new_vertices)

def save_voronoi_results(voronoi_results, basename, output_dir):
    """Save enhanced Voronoi results with anatomical region names."""
    output_path = os.path.join(output_dir, f"{basename}_voronoi_average_cell_area.csv")
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Enhanced header with region names
        writer.writerow([
            "region_label", "region_name", "region_area", "average_cell_area", "std_cell_area",
            "cv_cell_area", "percentage_measured", "n_cells_measured", "n_cells_total"
        ])
        
        # Sort by region label for consistent ordering
        sorted_items = sorted(voronoi_results.items(), key=lambda x: x[0])
        
        for region_label, stats in sorted_items:
            if stats is not None:
                writer.writerow([
                    region_label,
                    stats["region_name"],
                    stats["region_area"],
                    stats["average_cell_area"],
                    stats["std_cell_area"],
                    stats["cv_cell_area"],
                    stats["percentage_measured"],
                    stats["n_cells_measured"],
                    stats["n_cells_total"]
                ])
            else:
                writer.writerow([region_label, f"Region_{region_label}", None, None, None, None, None, None, None])
    
    logger.info(f"Saved enhanced Voronoi results to {output_path}")

def save_anatomical_region_info(valid_regions, basename, output_dir):
    """Save anatomical region information with names and properties."""
    output_path = os.path.join(output_dir, f"{basename}_anatomical_regions.csv")
    
    # Create template matcher to get region names
    template_matcher = AnatomicalTemplateMatcher()
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Header
        writer.writerow([
            "anatomical_id", "region_name", "centroid_y", "centroid_x", 
            "area", "solidity", "eccentricity", "major_axis_length", 
            "minor_axis_length", "orientation"
        ])
        
        # Sort regions by label
        sorted_regions = sorted(valid_regions, key=lambda r: r.label)
        
        for region in sorted_regions:
            anatomical_id = region.label
            
            # Get region name from template
            if anatomical_id in template_matcher.target_regions:
                region_name = template_matcher.target_regions[anatomical_id]['name']
            else:
                region_name = f"Unknown_{anatomical_id}"
            
            writer.writerow([
                anatomical_id,
                region_name,
                region.centroid[0],
                region.centroid[1],
                region.area,
                region.solidity,
                region.eccentricity,
                region.major_axis_length,
                region.minor_axis_length,
                region.orientation
            ])
    
    logger.info(f"Saved anatomical region information to {output_path}")


def find_associated_files(directory):
    """Windows-compatible file finder with improved pattern matching."""
    mapping = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return mapping
    
    # More comprehensive patterns for Windows (case-insensitive)
    prob_patterns = [
        "*[Pp]robabilities.h5", "*[Pp]robabilities.H5",
        "*[Pp]robabilities.hdf5", "*[Pp]robabilities.HDF5",
        "*_prob.h5", "*_prob.H5", "*_prob.hdf5", "*_prob.HDF5",
        "*_.h5", "*_.H5", "*_.hdf5", "*_.HDF5"
    ]
    
    prob_files = []
    for pattern in prob_patterns:
        try:
            # Use case-insensitive matching
            matches = list(directory_path.glob(pattern))
            # Also try rglob for recursive search in case files are in subdirectories
            matches.extend(list(directory_path.rglob(pattern)))
            prob_files.extend(matches)
        except Exception as e:
            logger.warning(f"Error with pattern {pattern}: {e}")
            continue
    
    # Remove duplicates
    prob_files = list(set(prob_files))
    
    logger.info(f"Found {len(prob_files)} potential probability files")
    for pf in prob_files:
        logger.info(f"  - {pf.name}")
    
    for prob_path in prob_files:
        filename = prob_path.name.lower()  # Convert to lowercase for matching
        basename = None
        
        # More robust basename extraction
        extraction_patterns = [
            ("_probabilities.h5", ""), ("_probabilities.hdf5", ""),
            ("_probabilities.H5", ""), ("_probabilities.HDF5", ""),
            ("probabilities.h5", ""), ("probabilities.hdf5", ""),
            ("probabilities.H5", ""), ("probabilities.HDF5", ""),
            ("_prob.h5", ""), ("_prob.hdf5", ""),
            ("_prob.H5", ""), ("_prob.HDF5", ""),
            (".h5", ""), (".H5", ""), (".hdf5", ""), (".HDF5", "")
        ]
        
        for suffix, replacement in extraction_patterns:
            if filename.endswith(suffix.lower()):
                basename = filename.replace(suffix.lower(), replacement)
                break
        
        if basename is None:
            # Fallback: use stem
            basename = prob_path.stem
            if basename.lower().endswith('_probabilities'):
                basename = basename[:-14]  # Remove '_probabilities'
            elif basename.lower().endswith('_prob'):
                basename = basename[:-5]   # Remove '_prob'
        
        # Clean basename
        basename = basename.strip('_').strip()
        
        if not basename:
            logger.warning(f"Could not extract basename from {prob_path.name}")
            continue
            
        mapping[basename] = {'probabilities': str(prob_path)}
        
        # Look for corresponding raw files with more patterns
        raw_patterns = [
            f"{basename}_[Rr]aw.h5", f"{basename}_[Rr]aw.H5",
            f"{basename}_[Rr]aw.hdf5", f"{basename}_[Rr]aw.HDF5",
            f"{basename}_[Rr]aw.tiff", f"{basename}_[Rr]aw.tif", 
            f"{basename}_[Rr]aw.TIFF", f"{basename}_[Rr]aw.TIF",
            f"{basename}_[Rr]aw.png", f"{basename}_[Rr]aw.PNG",
            f"{basename}_[Rr]aw.jpg", f"{basename}_[Rr]aw.jpeg",
            f"{basename}_[Rr]aw.JPG", f"{basename}_[Rr]aw.JPEG",
            f"{basename}.tiff", f"{basename}.tif", 
            f"{basename}.TIFF", f"{basename}.TIF",
            f"{basename}.png", f"{basename}.PNG",
            f"{basename}.jpg", f"{basename}.jpeg",
            f"{basename}.JPG", f"{basename}.JPEG"
        ]
        
        raw_candidates = []
        for pattern in raw_patterns:
            try:
                candidates = list(directory_path.glob(pattern))
                raw_candidates.extend([str(c) for c in candidates])
            except Exception as e:
                continue
        
        # Remove duplicates and pick first
        raw_candidates = list(set(raw_candidates))
        mapping[basename]['raw'] = raw_candidates[0] if raw_candidates else None
        
        if mapping[basename]['raw']:
            logger.info(f"Windows: Found pair: {basename} (prob + raw)")
        else:
            logger.info(f"Windows: Found: {basename} (prob only)")
    
    logger.info(f"Final mapping contains {len(mapping)} entries")
    return mapping




def enhanced_intervein_processing(prob_map, raw_img, cfg, output_dir, basename):
    """Complete automated intervein processing pipeline with improved boundary handling."""
    
    logger.info("Starting enhanced automated vein and intervein detection...")
    
    # Step 1: Detect veins using the improved detector
    # vein_detector = ImprovedWingVeinDetector(cfg)
    # vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    vein_mask = prob_map[..., 2] > 0.5 if prob_map.shape[-1] >= 3 else np.zeros(prob_map.shape[:2], dtype=bool)
    skeleton = morphology.skeletonize(vein_mask) 
    
    # Step 2: Use enhanced intervein detector for better boundary handling
    intervein_detector = EnhancedInterveinDetector(cfg)
    result = intervein_detector.segment_intervein_regions_enhanced(
        prob_map, vein_mask, raw_img
    )
    
    # Check if wing was valid
    if result[0] is None:
        logger.warning(f"Skipping {basename} - wing touching border or too small")
        return None, None, None, None
    
    intervein_regions, combined_veins, virtual_veins = result
    
    # Step 3: Save enhanced visualization showing virtual boundaries
    vis_path = os.path.join(output_dir, f"{basename}_enhanced_segmentation.png")
    intervein_detector.visualize_enhanced_segmentation(
        raw_img, vein_mask, virtual_veins, intervein_regions, vis_path
    )
    
    # Also save the original vein detection visualization
    orig_vis_path = os.path.join(output_dir, f"{basename}_vein_detection.png")
    # vein_detector.visualize_detection_results(raw_img, vein_mask, skeleton, 
    #                                         intervein_regions, orig_vis_path)
    
    # Step 4: Get final regions for analysis
    final_regions = regionprops(intervein_regions)
    
    logger.info(f"Enhanced processing found {len(final_regions)} intervein regions")
    
    # Log statistics about boundary regions
    n_boundary_regions = 0
    n_bottom_regions = 0
    
    for region in final_regions:
        # Check if touches boundary
        region_mask = intervein_regions == region.label
        wing_mask = intervein_regions > 0
        wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
        
        if np.any(region_mask & wing_boundary):
            n_boundary_regions += 1
            
            # Check if in bottom third
            if region.centroid[0] > intervein_regions.shape[0] * 0.67:
                n_bottom_regions += 1
    
    logger.info(f"Found {n_boundary_regions} boundary regions, {n_bottom_regions} in bottom third of wing")
    
    # Return combined veins (including virtual) for visualization
    return intervein_regions, final_regions, combined_veins, skeleton

def plot_voronoi_with_veins_enhanced(valid_regions, region_peaks_dict, background_img, 
                                    vein_mask, skeleton, output_directory, basename, cfg):
    """Enhanced Voronoi visualization that shows both real and virtual veins."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Top-left: Original with all veins (real + virtual)
    axes[0, 0].imshow(background_img, cmap="gray")
    axes[0, 0].imshow(vein_mask, cmap="Reds", alpha=0.6)
    axes[0, 0].set_title("Complete Vein Network (Real + Virtual)")
    axes[0, 0].axis('off')
    
    # Top-right: Skeleton with intervein regions
    axes[0, 1].imshow(background_img, cmap="gray")
    axes[0, 1].imshow(skeleton, cmap="Reds", alpha=0.8)
    
    # Color intervein regions
    labeled_regions = np.zeros_like(background_img)
    for i, region in enumerate(valid_regions, 1):
        labeled_regions[region.coords[:, 0], region.coords[:, 1]] = region.label
    
    axes[0, 1].imshow(labeled_regions, cmap="nipy_spectral", alpha=0.4, vmin=0, vmax=6)
    
    # Add anatomical labels
    template_matcher = AnatomicalTemplateMatcher()
    for region in valid_regions:
        if region.label in template_matcher.target_regions:
            region_name = template_matcher.target_regions[region.label]['name']
            cy, cx = region.centroid
            axes[0, 1].text(cx, cy, f"{region.label}\n{region_name}", 
                          color='white', fontsize=8, ha='center', va='center',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    axes[0, 1].set_title(f"Anatomically Labeled Intervein Regions (n={len(valid_regions)})")
    axes[0, 1].axis('off')
    
    # Bottom-left: Voronoi cells with quality indicators
    axes[1, 0].imshow(background_img, cmap="gray")
    
    total_kept = 0
    total_excluded_nc = 0
    total_excluded_iqr = 0
    boundary_regions = 0
    bottom_regions = 0

    for region in valid_regions:
        label_val = region.label
        region_peaks = region_peaks_dict.get(label_val, np.empty((0, 2)))
        
        # Check if boundary/bottom region
        region_mask = labeled_regions == label_val
        wing_boundary = morphology.binary_erosion(labeled_regions > 0) ^ (labeled_regions > 0)
        is_boundary = np.any(region_mask & wing_boundary)
        is_bottom = region.centroid[0] > background_img.shape[0] * 0.67
        
        if is_boundary:
            boundary_regions += 1
        if is_bottom:
            bottom_regions += 1
        
        if region_peaks.shape[0] < 2:
            continue

        # Region polygon (convex hull)
        region_coords = np.asarray(region.coords)
        if region_coords.shape[0] < 3:
            continue
        
        try:
            hull_pts = region_coords[ConvexHull(region_coords).vertices]
            region_poly = Polygon([(p[1], p[0]) for p in hull_pts]).buffer(0)
        except Exception as e:
            logger.warning(f"Convex-hull error in region {label_val}: {e}")
            continue

        # Voronoi tessellation
        points_vor = np.asarray([(p[1], p[0]) for p in region_peaks])
        try:
            vor = Voronoi(points_vor)
            regs, verts = voronoi_finite_polygons_2d(vor)
        except Exception as e:
            logger.warning(f"Voronoi error in region {label_val}: {e}")
            continue

        n_pts = len(points_vor)
        cell_polys = [None] * n_pts
        cell_areas = np.full(n_pts, np.nan, float)

        for i, reg in enumerate(regs):
            try:
                poly = Polygon(verts[reg]).buffer(0).intersection(region_poly)
                if not poly.is_empty:
                    cell_polys[i] = poly
                    cell_areas[i] = poly.area
            except:
                continue

        valid_idx = ~np.isnan(cell_areas)
        if not valid_idx.any():
            continue

        # Apply same filtering as in voronoi_average_cell_stats
        adj = _voronoi_adjacency(vor)
        keep_nc = np.zeros(n_pts, bool)

        for i, area in enumerate(cell_areas):
            if np.isnan(area):
                continue
            nbrs = [j for j in adj[i] if not np.isnan(cell_areas[j])]
            if len(nbrs) < cfg.min_neighbours:
                continue
            med = np.median(cell_areas[nbrs])
            if med > 0 and abs(area - med) < cfg.neighbour_tolerance * med:
                keep_nc[i] = True

        kept_areas = cell_areas[keep_nc]
        if kept_areas.size > 0:
            Q1, Q3 = np.percentile(kept_areas, (25, 75))
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                keep_final = keep_nc & (cell_areas >= lower) & (cell_areas <= upper)
            else:
                keep_final = keep_nc
        else:
            keep_final = np.zeros(n_pts, bool)

        # Count cells in each category
        excl_nc = valid_idx & (~keep_nc)
        excl_iqr = keep_nc & (~keep_final)
        
        region_kept = np.sum(keep_final)
        region_excl_nc = np.sum(excl_nc)
        region_excl_iqr = np.sum(excl_iqr)
        
        total_kept += region_kept
        total_excluded_nc += region_excl_nc
        total_excluded_iqr += region_excl_iqr

        # Plot polygons with color coding
        for idx, poly in enumerate(cell_polys):
            if poly is None:
                continue
            
            if keep_final[idx]:
                color, alpha, linewidth = "blue", 0.8, 1.5
            elif excl_iqr[idx]:
                color, alpha, linewidth = "red", 0.6, 1.0
            elif excl_nc[idx]:
                color, alpha, linewidth = "orange", 0.6, 1.0
            else:
                color, alpha, linewidth = "gray", 0.3, 0.5

            if poly.geom_type == "Polygon":
                xs, ys = poly.exterior.xy
                axes[1, 0].plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)
            else:
                for sub in getattr(poly, 'geoms', [poly]):
                    if hasattr(sub, 'exterior'):
                        xs, ys = sub.exterior.xy
                        axes[1, 0].plot(xs, ys, color=color, alpha=alpha, linewidth=linewidth)

        # Label region with anatomical name
        cx, cy = region.centroid[1], region.centroid[0]
        region_name = template_matcher.target_regions.get(label_val, {}).get('name', f'R{label_val}')
        label_text = f"{label_val}: {region_name}\n({region_kept} cells)"
        if is_bottom:
            label_text += "\n[BTM]"
        elif is_boundary:
            label_text += "\n[BDY]"
            
        axes[1, 0].text(cx, cy, label_text, 
                       color="yellow", fontsize=9, weight="bold", 
                       ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))

    # Enhanced title with statistics
    axes[1, 0].set_title(
        f"Voronoi Cell Classification\n"
        f"Blue: Kept ({total_kept}) | Orange: Neighbor-fail ({total_excluded_nc}) | "
        f"Red: IQR-outlier ({total_excluded_iqr})",
        fontsize=14, pad=20
    )
    axes[1, 0].axis('off')
    
    # Bottom-right: Summary statistics
    vein_coverage = np.sum(vein_mask) / vein_mask.size * 100
    intervein_coverage = np.sum(labeled_regions > 0) / labeled_regions.size * 100
    
    # Count missing regions
    detected_labels = set(region.label for region in valid_regions)
    expected_labels = set(range(1,6))
    missing_labels = expected_labels - detected_labels
    missing_names = [template_matcher.get_region_name(label) for label in missing_labels]
    
    stats_text = f"""Enhanced Analysis Summary:

Vein Detection:
- Vein coverage: {vein_coverage:.1f}%
- Skeleton length: {np.sum(skeleton)} pixels

Anatomical Region Analysis:
- Detected regions: {len(valid_regions)}/6
- Missing regions: {', '.join(missing_names) if missing_names else 'None'}
- Boundary regions: {boundary_regions}
- Bottom regions: {bottom_regions}
- Intervein coverage: {intervein_coverage:.1f}%

Trichome Analysis:
- Total trichomes: {sum(len(peaks) for peaks in region_peaks_dict.values())}
- Cells kept for analysis: {total_kept}
- Average trichomes/region: {sum(len(peaks) for peaks in region_peaks_dict.values())/len(valid_regions):.1f}

Quality Metrics:
- Avg region area: {np.mean([r.area for r in valid_regions]):.0f} px²
- Avg region solidity: {np.mean([r.solidity for r in valid_regions]):.3f}

[BTM] = Bottom third of wing
[BDY] = Boundary region
"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_title("Analysis Summary")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    save_path = os.path.join(output_directory, f"{basename}_complete_analysis_enhanced.png")
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved enhanced complete analysis visualization to {save_path}")
    plt.close()

def main(directory: str, cfg: TrichomeDetectionConfig = CONFIG, 
         output_directory: Optional[str] = None):
    """Fully automated main function with comprehensive error handling and reporting."""
    
    # Validate configuration
    cfg.validate()
    
    if output_directory is None:
        output_directory = directory
    
    # Ensure output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(output_directory, "trichome_detection_automated.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info("="*50)
    logger.info("Starting FULLY AUTOMATED trichome detection analysis")
    logger.info("with template-based anatomical region labeling")
    logger.info(f"Input directory: {directory}")
    logger.info(f"Output directory: {output_directory}")
    logger.info("="*50)
    
    mapping = find_associated_files(directory)
    if not mapping:
        logger.error("No probability files found in the directory")
        return

    logger.info(f"Found {len(mapping)} file pairs to process")
    
    # Summary statistics
    total_files = len(mapping)
    processed_files = 0
    failed_files = 0
    skipped_files = 0
    total_peaks = 0
    total_regions = 0

    for basename, files in mapping.items():
        logger.info("="*30)
        logger.info(f"Processing: {basename}")
        logger.info("="*30)
        
        try:
            prob_file = files["probabilities"]
            raw_file = files.get("raw")
            
            # Load data
            tri_prob, inter_prob, full_prob = load_probability_map(prob_file)
            raw_img = load_raw_image(raw_file) if raw_file else None
            
            # Enhanced peak detection
            peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
            total_peaks += len(peaks)
            
            #logger.info(metrics.report())
            
            # Save enhanced results
            save_peak_coordinates_enhanced(peaks, basename, output_directory, metrics, tri_prob)
            
            # Create detection visualization
            create_detection_visualization(tri_prob, peaks, basename, output_directory, raw_img)
            
            # Process intervein regions if available - FULLY AUTOMATED
            if full_prob.shape[-1] >= 4 and inter_prob is not None:
                logger.info("Processing intervein segmentation with IMPROVED AUTOMATED vein detection...")
                
                # === AUTOMATED VEIN DETECTION AND INTERVEIN PROCESSING ===
                result = enhanced_intervein_processing(
                    full_prob, raw_img, cfg, output_directory, basename
                )
                
                if result[0] is None:
                    logger.warning(f"Skipped {basename} - wing touching border or invalid")
                    skipped_files += 1
                    continue
                
                labeled_mask, valid_regions, vein_mask, skeleton = result
                
                total_regions += len(valid_regions)
                logger.info(f"Found {len(valid_regions)} valid anatomically labeled intervein regions")
                
                # NO MANUAL VERIFICATION - fully automated
                logger.info("Automated segmentation completed successfully - proceeding with analysis")
                
                # Assign peaks to regions
                region_peaks_dict = assign_peaks_to_regions(peaks, labeled_mask, valid_regions)
                
                # Voronoi analysis
                voronoi_results = {}
                successful_regions = 0
                for region in valid_regions:
                    label_val = region.label
                    region_peaks = region_peaks_dict.get(label_val, np.empty((0, 2)))
                    
                    if region_peaks.shape[0] < 2:
                        logger.warning(f"Region {label_val}: insufficient peaks for Voronoi analysis")
                        voronoi_results[label_val] = None
                    else:
                        stats = voronoi_average_cell_stats(region, region_peaks, cfg)
                        voronoi_results[label_val] = stats
                        if stats is not None:
                            successful_regions += 1
                
                # Save results
                save_voronoi_results(voronoi_results, basename, output_directory)
                save_anatomical_region_info(valid_regions, basename, output_directory)
                
                # Print summary
                logger.info(f"Voronoi analysis summary for {basename}:")
                for region_label, stats in voronoi_results.items():
                    if stats is not None:
                        logger.info(f"  Region {region_label} ({stats['region_name']}): "
                                  f"{stats['n_cells_measured']} cells, "
                                  f"avg area {stats['average_cell_area']:.1f} px², "
                                  f"CV {stats['cv_cell_area']:.3f}")
                    else:
                        region_name = AnatomicalTemplateMatcher().target_regions.get(
                            region_label, {}).get('name', f'Region_{region_label}')
                        logger.warning(f"  Region {region_label} ({region_name}): Analysis failed")
                
                logger.info(f"Successfully analyzed {successful_regions}/{len(valid_regions)} regions")
                
                # Enhanced visualization with vein information
                if valid_regions and any(region_peaks_dict[label].shape[0] >= 2 
                                       for label in region_peaks_dict):
                    background_img = raw_img if raw_img is not None else tri_prob
                    plot_voronoi_with_veins_enhanced(
                        valid_regions, region_peaks_dict, background_img, 
                        vein_mask, skeleton, output_directory, basename, cfg
                    )
            else:
                logger.info("Skipping intervein analysis (4th channel missing)")
            
            processed_files += 1
            logger.info(f"Successfully processed {basename}")
            
        except Exception as e:
            failed_files += 1
            logger.error(f"Error processing {basename}: {str(e)}", exc_info=True)
            continue

    # Final summary
    logger.info("="*50)
    logger.info("AUTOMATED ANALYSIS COMPLETE")
    logger.info("="*50)
    logger.info(f"Total files found: {total_files}")
    logger.info(f"Successfully processed: {processed_files}")
    logger.info(f"Skipped (border/invalid): {skipped_files}")
    logger.info(f"Failed: {failed_files}")
    logger.info(f"Total trichomes detected: {total_peaks}")
    logger.info(f"Total regions analyzed: {total_regions}")
    logger.info(f"Results saved to: {output_directory}")
    logger.info("="*50)
    
    # Create summary report
    summary_path = os.path.join(output_directory, "analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("TRICHOME DETECTION ANALYSIS SUMMARY\n")
        f.write("="*50 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {directory}\n")
        f.write(f"Output Directory: {output_directory}\n")
        f.write("\nPROCESSING SUMMARY:\n")
        f.write(f"Total files found: {total_files}\n")
        f.write(f"Successfully processed: {processed_files}\n")
        f.write(f"Skipped (border/invalid): {skipped_files}\n")
        f.write(f"Failed: {failed_files}\n")
        f.write("\nDETECTION SUMMARY:\n")
        f.write(f"Total trichomes detected: {total_peaks}\n")
        f.write(f"Total regions analyzed: {total_regions}\n")
        f.write(f"Average trichomes per wing: {total_peaks/max(processed_files, 1):.1f}\n")
        f.write(f"Average regions per wing: {total_regions/max(processed_files, 1):.1f}\n")
    
    logger.info(f"Summary report saved to {summary_path}")

# REPLACE: Replace your entire __main__ section at the very end of your script with this

if __name__ == "__main__":
    # Ensure Windows compatibility
    # ensure_windows_compatibility()
    
    # Check if GUI should be launched
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Command line mode with hybrid detection
        root = tk.Tk()
        root.withdraw()
        
        # Configure paths - UPDATE THESE FOR YOUR SYSTEM
        input_directory = r"/path/to/your/wing/data"
        output_directory = input_directory
        
        if not input_directory or not os.path.exists(input_directory):
            print("Invalid input directory. Please check the path.")
        else:
            logger.info("Starting hybrid wing detection analysis...")
            main_with_hybrid_wing_detection(
                directory=input_directory,
                cfg=CONFIG,  # Uses the enhanced config
                output_directory=output_directory,
                force_pentagone_mode=False,
                auto_detect_pentagone=True
            )
    else:
        # Launch GUI with hybrid detection support
        run_gui()
