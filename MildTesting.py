#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

HomerWings for public beta. IJFM!



"""

from __future__ import annotations


import sys
import os
from pathlib import Path


def get_resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    
    return os.path.join(base_path, relative_path)


if sys.platform.startswith('win'):
    # Fix for Windows multiprocessing in PyInstaller
    import multiprocessing
    multiprocessing.freeze_support()
    
    # Ensure proper DLL loading on Windows
    if hasattr(sys, '_MEIPASS'):
        os.environ['PATH'] = sys._MEIPASS + ';' + os.environ.get('PATH', '')




import os
import csv
import math
import glob
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
import numpy as np
import h5py
import imageio

from scipy.spatial import ConvexHull, Voronoi, distance_matrix
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import json
import base64
from pathlib import Path
from dataclasses import asdict
import time
from skimage import exposure, morphology, filters, feature, measure, segmentation
from skimage.morphology import white_tophat, disk, remove_small_objects, binary_dilation, binary_erosion, skeletonize
from skimage.filters import threshold_local, gaussian, frangi, sato
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import h_maxima, remove_small_holes
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy import ndimage
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib
import drosophila_gif
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
PIXELS_PER_MICRON = 1.4464

@dataclass
class TrichomeDetectionConfig:
    """Enhanced configuration with automated vein detection parameters."""
    
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
    intervein_threshold: float = 0.4
    min_region_area: int = 60000
    max_region_area: int = 1500000
    max_hole_size: int = 10000
    
    # IMPROVED VEIN DETECTION PARAMETERS
    vein_width_estimate: int = 7
    min_vein_length: int = 100
    vein_detection_sensitivity: float = 0.9
    use_template_matching: bool = True
    use_vesselness_filters: bool = True
    use_morphological_detection: bool = True
    
    # Wing-specific parameters
    wing_orientation_correction: bool = True
    expected_wing_aspect_ratio: float = 2.5
    min_wing_area: int = 100000  # Minimum area for valid wing
    border_buffer: int = 20  # Buffer from image edge for valid wings
    
    # Intervein region refinement
    auto_intervein_min_area: int = 800
    auto_intervein_max_area: int = 800000
    intervein_shape_filter: bool = True
    min_intervein_solidity: float = 0.4
    
    def validate(self) -> None:
        """Validate configuration parameters."""
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
        logger.info("Configuration validation passed")

CONFIG = TrichomeDetectionConfig()







class StringRemovalTrichomeFilter:
    """Remove long strings of trichomes (bubble artifacts) using morphological-like operations."""
    
    def __init__(self, config):
        self.config = config
        # Parameters for string detection and removal
        self.connection_distance = 25     # Max distance to consider trichomes "connected"
        self.min_string_length = 8       # Min number of trichomes to be considered a "string"
        self.max_string_width = 50       # Max width of valid string (bubbles are very thin)
        self.linearity_threshold = 0.7   # How linear a string must be to be removed (0-1)
        
    def remove_trichome_strings(self, peaks, image_shape):
        """Remove long, thin strings of trichomes that represent bubble artifacts."""
        
        if len(peaks) < 20:
            print("Too few trichomes for string filtering")
            return peaks
            
        print(f"Filtering strings from {len(peaks)} trichomes...")
        
        # Step 1: Build connectivity graph between nearby trichomes
        adjacency_graph = self._build_trichome_graph(peaks)
        
        # Step 2: Find connected components (chains/strings)
        components = self._find_connected_components(adjacency_graph)
        
        # Step 3: Identify which components are "strings" vs "blobs"
        string_components = self._identify_string_components(peaks, components)
        
        # Step 4: Remove trichomes that belong to string components
        filtered_peaks = self._remove_string_trichomes(peaks, string_components)
        
        removed_count = len(peaks) - len(filtered_peaks)
        print(f"  Removed {removed_count} trichomes from {len(string_components)} string artifacts")
        print(f"  Remaining: {len(filtered_peaks)} trichomes")
        
        return filtered_peaks
    
    def _build_trichome_graph(self, peaks):
        """Build graph of connected trichomes based on distance."""
        n_peaks = len(peaks)
        
        # Calculate pairwise distances
        distances = squareform(pdist(peaks))
        
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
    
    def _identify_string_components(self, peaks, components):
        """Identify which components are long strings vs compact blobs."""
        string_components = []
        
        for i, component in enumerate(components):
            if len(component) < self.min_string_length:
                continue  # Too short to be a problematic string
            
            component_peaks = peaks[component]
            
            # Calculate component geometry
            is_string = self._is_linear_string(component_peaks)
            
            if is_string:
                string_components.append(component)
                print(f"    String {len(string_components)}: {len(component)} trichomes")
            else:
                print(f"    Blob {i}: {len(component)} trichomes (kept)")
        
        return string_components
    
    def _is_linear_string(self, component_peaks):
        """Check if a component is a linear string (bubble artifact)."""
        
        if len(component_peaks) < 3:
            return False
        
        # Method 1: Check aspect ratio of bounding box
        min_coords = np.min(component_peaks, axis=0)
        max_coords = np.max(component_peaks, axis=0)
        bbox_dims = max_coords - min_coords
        
        if bbox_dims[0] > 0 and bbox_dims[1] > 0:
            aspect_ratio = max(bbox_dims) / min(bbox_dims)
            
            # Very elongated = likely string
            if aspect_ratio > 8.0:
                return True
        
        # Method 2: Check linearity using PCA
        try:
            # Center the points
            centered = component_peaks - np.mean(component_peaks, axis=0)
            
            # Compute covariance matrix
            cov_matrix = np.cov(centered.T)
            
            # Get eigenvalues
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues = np.sort(eigenvalues)[::-1]  # Sort descending
            
            # Linearity measure: ratio of largest to smallest eigenvalue
            if eigenvalues[1] > 0:
                linearity = eigenvalues[0] / eigenvalues[1]
                
                # High linearity = string-like
                if linearity > 15.0:
                    return True
        except:
            pass  # Skip if PCA fails
        
        # Method 3: Check "width" of the string
        if len(component_peaks) >= 4:
            # Fit a line through the points and measure perpendicular distances
            try:
                # Simple line fitting using first and last points
                start_point = component_peaks[0]
                end_point = component_peaks[-1]
                
                # Vector along the line
                line_vector = end_point - start_point
                line_length = np.linalg.norm(line_vector)
                
                if line_length > 0:
                    line_unit = line_vector / line_length
                    
                    # Calculate perpendicular distances
                    perp_distances = []
                    for point in component_peaks:
                        to_point = point - start_point
                        # Project onto line
                        projection_length = np.dot(to_point, line_unit)
                        projection = start_point + projection_length * line_unit
                        # Perpendicular distance
                        perp_dist = np.linalg.norm(point - projection)
                        perp_distances.append(perp_dist)
                    
                    # If most points are very close to the line = string
                    max_width = np.max(perp_distances)
                    
                    if max_width < self.max_string_width and line_length > 100:
                        return True
            except:
                pass
        
        return False
    
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
        
        if len(filtered_peaks) < 10:
            print("Too few filtered trichomes")
            return None
        
        print(f"Creating wing mask from {len(filtered_peaks)} filtered trichomes...")
        
        # Method: Dense region growing
        density_map = np.zeros(image_shape, dtype=np.float32)
        
        # Add gaussian blob at each trichome
        sigma = 12  # Smoothing radius
        for peak in filtered_peaks:
            y, x = peak
            
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
        
        print(f"  Wing mask: {np.sum(wing_mask)} pixels")
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
        
        print(f"Saved string removal visualization to {output_path}")



class TrichomeAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("HomerWings V2 (Windows Version)")
        
        # Windows-specific fixes
        if sys.platform.startswith('win'):
            self.root.lift()
            self.root.focus_force()
            self.root.state('normal')
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
        
        # CRITICAL FIX: Initialize GUI variables dictionary FIRST
        self.gui_vars = {}  # This will hold all our GUI variables
        
        # Setup GUI components
        logger.info("Setting up GUI components...")
        self.setup_gui()
        
        # SIMPLIFIED: Load configuration with verification
        logger.info("Loading initial configuration...")
        self.root.after(500, self._load_config_safely)
        
        # Start progress monitoring
        self.root.after(1000, self.check_progress)

    def _set_default_values(self):
        """Set default values using the same system."""
        defaults = {
            'min_distance': '8',
            'high_thresh_abs': '0.30',
            'low_thresh_abs': '0.20',
            'scales': "0.8,1.0,1.2",
            'scale_weight_decay': '0.8',
            'use_clahe': False,
            'clahe_clip_limit': '0.01',
            'use_white_tophat': True,
            'tophat_radius': '2',
            'use_local_threshold': True,
            'local_block_size': '71',
            'min_peak_intensity': '0.25',
            'max_peak_density': '0.002',
            'edge_exclusion_buffer': '10',
            'dbscan_eps': '10.0',
            'dbscan_min_samples': '1',
            'valley_drop': '0.25',
            'neighbour_tolerance': '0.20',
            'min_neighbours': '7',
            'intervein_threshold': '0.4',
            'min_region_area': '60000',
            'max_region_area': '1500000',
            'vein_width_estimate': '7',
            'min_vein_length': '100',
            'vein_detection_sensitivity': '0.9',
            'min_wing_area': '100000',
            'border_buffer': '20'
        }
        
        for attr_name, default_value in defaults.items():
            if attr_name in self.gui_vars:
                try:
                    var = self.gui_vars[attr_name]['var']
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(default_value))
                    else:
                        var.set(str(default_value))
                except Exception as e:
                    logger.warning(f"Could not set default for {attr_name}: {e}")

    # def _load_config_safely(self):
    #     """Safely load configuration with proper verification."""
    #     try:
    #         # Verify all GUI variables exist first
    #         required_vars = [
    #             'min_distance', 'high_thresh_abs', 'low_thresh_abs', 'scales',
    #             'use_clahe', 'use_white_tophat', 'use_local_threshold', 'local_block_size'
    #         ]
            
    #         missing_vars = [var for var in required_vars if var not in self.gui_vars]
            
    #         if missing_vars:
    #             logger.warning(f"GUI not ready, missing: {missing_vars}")
    #             # Try again
    #             self.root.after(500, self._load_config_safely)
    #             return
            
    #         # All variables exist, proceed with loading
    #         self._apply_config_to_gui()
    #         logger.info("Configuration loaded successfully")
            
    #     except Exception as e:
    #         logger.error(f"Error in safe config load: {e}")
    #         self._set_default_values()
    def setup_config_tab_simple(self):
        """Simplified config tab without scrolling - to test if that's the issue."""
        logger.info("Creating SIMPLIFIED config tab...")
        
        # NO CANVAS/SCROLLING - direct children of config_frame
        
        # Test frame 1
        test_frame = ttk.LabelFrame(self.config_frame, text="TEST SECTION", padding=10)
        test_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Add some simple widgets directly
        tk.Label(test_frame, text="If you can see this text, basic widgets work", 
                 bg='yellow', fg='black').pack(pady=5)
        
        # Test our create_config_entry method
        logger.info("Creating test config entries...")
        
        try:
            self.create_config_entry(test_frame, "min_distance", "Min Distance:", 8, int)
            self.create_config_entry(test_frame, "high_thresh_abs", "High Threshold:", 0.3, float)
            self.create_config_checkbox(test_frame, "use_clahe", "Use CLAHE", False)
            
            logger.info(f"After creating test entries: gui_vars has {len(self.gui_vars)} items")
            
            # Manually set values to something obvious
            if 'min_distance' in self.gui_vars:
                self.gui_vars['min_distance']['var'].set("VISIBLE TEST")
                logger.info("Set min_distance to 'VISIBLE TEST'")
                
            if 'high_thresh_abs' in self.gui_vars:
                self.gui_vars['high_thresh_abs']['var'].set("0.999")
                logger.info("Set high_thresh_abs to '0.999'")
            
        except Exception as e:
            logger.error(f"Error creating test entries: {e}")
            import traceback
            traceback.print_exc()
        
        # Add a button to test interaction
        def test_button_clicked():
            logger.info("Test button clicked!")
            if 'min_distance' in self.gui_vars:
                current = self.gui_vars['min_distance']['var'].get()
                new_value = "BUTTON CLICKED!"
                self.gui_vars['min_distance']['var'].set(new_value)
                logger.info(f"Changed min_distance from '{current}' to '{new_value}'")
            
            # Force update
            self.config_frame.update_idletasks()
            self.config_frame.update()
        
        tk.Button(test_frame, text="TEST BUTTON - CLICK ME", 
                  command=test_button_clicked, bg='red', fg='white').pack(pady=10)
        
        # Force immediate update
        self.config_frame.update_idletasks()
        self.config_frame.update()
        
        logger.info("Simplified config tab created")
    def _load_config_safely(self):
        """Safely load configuration with testing."""
        try:
            logger.info("Attempting to load configuration...")
            
            # Debug what we have
            self.debug_gui_variables()
            
            if not hasattr(self, 'gui_vars') or not self.gui_vars:
                logger.error("gui_vars is empty or missing!")
                self.root.after(500, self._load_config_safely)
                return
            
            # Check required vars
            required_vars = [
                'min_distance', 'high_thresh_abs', 'low_thresh_abs', 'scales',
                'use_clahe', 'use_white_tophat', 'use_local_threshold', 'local_block_size'
            ]
            
            missing_vars = [var for var in required_vars if var not in self.gui_vars]
            
            if missing_vars:
                logger.warning(f"GUI not ready, missing: {missing_vars}")
                if not hasattr(self, '_config_load_attempts'):
                    self._config_load_attempts = 0
                
                self._config_load_attempts += 1
                if self._config_load_attempts < 5:
                    delay = 500 * self._config_load_attempts
                    logger.info(f"Attempt {self._config_load_attempts}/5, retrying in {delay}ms")
                    self.root.after(delay, self._load_config_safely)
                    return
                else:
                    logger.error("Max attempts reached. Using defaults.")
                    self._set_default_values()
                    return
            
            # Apply configuration
            logger.info("All required GUI variables found. Applying configuration...")
            self._apply_config_to_gui()
            
            # TEST: Check if widgets are actually connected
            self.test_gui_widget_connection()
            
            # Force a complete refresh
            self.root.after(100, self.force_refresh_all_widgets)
            
            logger.info("Configuration loaded successfully")
            
            # Reset attempt counter on success
            self._config_load_attempts = 0
            
        except Exception as e:
            logger.error(f"Error in safe config load: {e}")
            self._set_default_values()
    def create_config_entry(self, parent, attr_name, label, default_value, data_type):
        """Simplified config entry creation using dictionary storage."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
        
        # Create variable and store in dictionary
        var = tk.StringVar(value=str(default_value))
        entry = ttk.Entry(frame, textvariable=var, width=20)
        entry.pack(side=tk.LEFT, padx=(5,0))
        
        # Store in our dictionary (much more reliable than setattr)
        self.gui_vars[attr_name] = {
            'var': var,
            'type': data_type,
            'widget': entry
        }
        
        logger.debug(f"Created config entry: {attr_name}")
    def create_config_checkbox(self, parent, attr_name, label, default_value):
        """Simplified checkbox creation using dictionary storage."""
        var = tk.BooleanVar(value=default_value)
        checkbox = ttk.Checkbutton(parent, text=label, variable=var)
        checkbox.pack(anchor=tk.W, pady=2)
        
        # Store in our dictionary
        self.gui_vars[attr_name] = {
            'var': var,
            'type': bool,
            'widget': checkbox
        }
        
        logger.debug(f"Created config checkbox: {attr_name}")

    def _apply_config_to_gui(self):
        """Apply configuration values to GUI - simplified and reliable."""
        config_values = {
            'min_distance': self.config.min_distance,
            'high_thresh_abs': self.config.high_thresh_abs,
            'low_thresh_abs': self.config.low_thresh_abs,
            'scales': ",".join(map(str, self.config.scales)),
            'scale_weight_decay': self.config.scale_weight_decay,
            'use_clahe': self.config.use_clahe,
            'clahe_clip_limit': self.config.clahe_clip_limit,
            'use_white_tophat': self.config.use_white_tophat,
            'tophat_radius': self.config.tophat_radius,
            'use_local_threshold': self.config.use_local_threshold,
            'local_block_size': self.config.local_block_size,
            'min_peak_intensity': self.config.min_peak_intensity,
            'max_peak_density': self.config.max_peak_density,
            'edge_exclusion_buffer': self.config.edge_exclusion_buffer,
            'dbscan_eps': self.config.dbscan_eps,
            'dbscan_min_samples': self.config.dbscan_min_samples,
            'valley_drop': self.config.valley_drop,
            'neighbour_tolerance': self.config.neighbour_tolerance,
            'min_neighbours': self.config.min_neighbours,
            'intervein_threshold': self.config.intervein_threshold,
            'min_region_area': self.config.min_region_area,
            'max_region_area': self.config.max_region_area,
            'vein_width_estimate': self.config.vein_width_estimate,
            'min_vein_length': self.config.min_vein_length,
            'vein_detection_sensitivity': self.config.vein_detection_sensitivity,
            'min_wing_area': self.config.min_wing_area,
            'border_buffer': self.config.border_buffer
        }
        
        for attr_name, value in config_values.items():
            if attr_name in self.gui_vars:
                try:
                    var = self.gui_vars[attr_name]['var']
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(value))
                    else:
                        var.set(str(value))
                    logger.debug(f"Set {attr_name} = {value}")
                except Exception as e:
                    logger.warning(f"Could not set {attr_name}: {e}")
            else:
                logger.warning(f"GUI variable {attr_name} not found")

    def load_initial_config(self):
        """Load initial configuration values into GUI - Windows safe with better error handling."""
        try:
            # Use a more robust approach for Windows
            self._update_gui_safely()
            logger.info("Initial configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading initial config: {e}")
            self.set_default_gui_values()
    
    def _update_gui_safely(self):
        """Safely update GUI from config with Windows-specific handling"""
        try:
            config_dict = {
                'min_distance': self.config.min_distance,
                'high_thresh_abs': self.config.high_thresh_abs,
                'low_thresh_abs': self.config.low_thresh_abs,
                'scales': ",".join(map(str, self.config.scales)),
                'scale_weight_decay': self.config.scale_weight_decay,
                'use_clahe': self.config.use_clahe,
                'use_white_tophat': self.config.use_white_tophat,
                'use_local_threshold': self.config.use_local_threshold,
                'clahe_clip_limit': self.config.clahe_clip_limit,
                'tophat_radius': self.config.tophat_radius,
                'local_block_size': self.config.local_block_size,
                'local_offset': self.config.local_offset,
                'min_peak_intensity': self.config.min_peak_intensity,
                'max_peak_density': self.config.max_peak_density,
                'edge_exclusion_buffer': self.config.edge_exclusion_buffer,
                'dbscan_eps': self.config.dbscan_eps,
                'dbscan_min_samples': self.config.dbscan_min_samples,
                'valley_drop': self.config.valley_drop,
                'neighbour_tolerance': self.config.neighbour_tolerance,
                'min_neighbours': self.config.min_neighbours,
                'intervein_threshold': self.config.intervein_threshold,
                'min_region_area': self.config.min_region_area,
                'max_region_area': self.config.max_region_area,
                'vein_width_estimate': self.config.vein_width_estimate,
                'min_vein_length': self.config.min_vein_length,
                'vein_detection_sensitivity': self.config.vein_detection_sensitivity,
                'min_wing_area': self.config.min_wing_area,
                'border_buffer': self.config.border_buffer
            }
            
            for attr_name, value in config_dict.items():
                var_name = f"{attr_name}_var"
                if hasattr(self, var_name):
                    try:
                        var = getattr(self, var_name)
                        if isinstance(var, tk.BooleanVar):
                            # Schedule boolean updates on main thread
                            self.root.after_idle(lambda v=var, val=value: v.set(bool(val)))
                        else:
                            # Schedule string/number updates on main thread
                            self.root.after_idle(lambda v=var, val=str(value): v.set(val))
                    except Exception as e:
                        logger.warning(f"Could not set GUI variable {attr_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in safe GUI update: {e}")
            raise
    
    def browse_master_folder(self):
        """Fixed Windows folder browser with better error handling"""
        try:
            # Clear any existing dialogs and ensure main window has focus
            self.root.focus_set()
            self.root.update_idletasks()
            
            # Force any existing grab to be released
            try:
                current_grab = self.root.grab_current()
                if current_grab:
                    current_grab.grab_release()
            except:
                pass
            
            # Wait a moment for any previous operations to complete
            self.root.after(100, self._do_folder_browse)
            
        except Exception as e:
            logger.error(f"Error preparing folder browser: {e}")
            messagebox.showerror("Error", f"Could not open folder browser: {e}")

    def _do_folder_browse(self):
        """Actually perform the folder browsing after ensuring clean state"""
        try:
            folder = filedialog.askdirectory(
                parent=self.root,
                title="Select Master Folder (containing subfolders with wing data)",
                mustexist=True
            )
            
            if folder:
                # Normalize path for Windows
                folder = os.path.normpath(folder)
                
                # FIXED: Ensure GUI update happens on main thread
                def update_folder_gui():
                    try:
                        self.master_folder_var.set(folder)
                        logger.info(f"Set master folder to: {folder}")
                        
                        # Force the entry widget to update its display
                        self.root.update_idletasks()
                        
                        # Auto-set output folder if not already set
                        if not self.output_folder_var.get().strip():
                            parent_dir = os.path.dirname(folder)
                            folder_name = os.path.basename(folder)
                            output_folder = os.path.join(parent_dir, f"{folder_name}_Results")
                            output_folder = os.path.normpath(output_folder)
                            self.output_folder_var.set(output_folder)
                            logger.info(f"Auto-set output folder to: {output_folder}")
                        
                        # Schedule refresh for next event loop cycle
                        self.root.after(300, self.refresh_folder_list)
                        
                    except Exception as e:
                        logger.error(f"Error updating folder GUI: {e}")
                
                # Schedule GUI update for next event loop cycle
                self.root.after_idle(update_folder_gui)
                    
        except Exception as e:
            logger.error(f"Error in folder browsing: {e}")
            messagebox.showerror("Error", f"Could not browse for folder: {e}")
    
    def refresh_folder_list(self):
        """Windows-safe folder list refresh with better threading"""
        try:
            # Clear existing list immediately
            self.folder_listbox.delete(0, tk.END)
            
            master_folder = self.master_folder_var.get().strip()
            if not master_folder:
                return
            
            if not os.path.exists(master_folder):
                self.folder_listbox.insert(tk.END, "⚠ Folder path does not exist")
                return
            
            # Show scanning message
            self.folder_listbox.insert(tk.END, "Scanning folders...")
            self.root.update()
            
            # Use thread-safe folder scanning
            try:
                subfolders = self._scan_folders_safe(master_folder)
                
                # Clear and populate with results
                self.folder_listbox.delete(0, tk.END)
                
                if subfolders:
                    # Sort and add to listbox
                    subfolders.sort()
                    for folder_info in subfolders:
                        self.folder_listbox.insert(tk.END, folder_info)
                else:
                    self.folder_listbox.insert(tk.END, "No subfolders with valid files found")
                    
            except PermissionError:
                self.folder_listbox.delete(0, tk.END)
                self.folder_listbox.insert(tk.END, "⚠ Permission denied - check folder access")
                logger.error(f"Permission denied accessing {master_folder}")
            except Exception as e:
                self.folder_listbox.delete(0, tk.END)
                self.folder_listbox.insert(tk.END, f"⚠ Error scanning: {str(e)[:50]}...")
                logger.error(f"Error scanning master folder: {e}")
                
        except Exception as e:
            logger.error(f"Critical error in refresh_folder_list: {e}")
    
    def _scan_folders_safe(self, master_folder):
        """Thread-safe folder scanning for Windows"""
        subfolders = []
        
        try:
            # Use pathlib for better Windows compatibility
            master_path = Path(master_folder)
            
            for entry in master_path.iterdir():
                if entry.is_dir():
                    try:
                        # Quick check for valid files
                        mapping = find_associated_files(str(entry))
                        if mapping:
                            file_count = len(mapping)
                            subfolders.append(f"✓ {entry.name} ({file_count} file pairs)")
                        else:
                            subfolders.append(f"✗ {entry.name} (no valid files)")
                    except Exception as e:
                        subfolders.append(f"⚠ {entry.name} (scan error: {str(e)[:20]}...)")
                        logger.warning(f"Error scanning {entry.name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error in safe folder scan: {e}")
            raise
            
        return subfolders
    
    def create_drosophila_loading_overlay(self):
        """Fixed loading overlay with Windows-specific grab handling"""
        if hasattr(self, 'loading_window'):
            try:
                if self.loading_window.winfo_exists():
                    return
            except tk.TclError:
                # Window was destroyed, continue
                pass
        
        # Create overlay window
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("HomerWings Processing")
        self.loading_window.geometry("600x450")
        self.loading_window.resizable(False, False)
        
        # Windows-specific window setup
        self.loading_window.transient(self.root)
        
        # Position relative to main window with error handling
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 300
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 225
            self.loading_window.geometry(f"500x450+{x}+{y}")
        except tk.TclError:
            # Fallback positioning
            self.loading_window.geometry("500x450+100+100")
        
        self.loading_window.configure(bg='#1a1a1a')
        
        # FIXED: Handle grab_set with Windows-specific error handling
        def safe_grab():
            try:
                # Wait a moment for window to be fully created
                self.root.after(100, lambda: self._attempt_grab())
            except Exception as e:
                logger.warning(f"Could not set window grab: {e}")
                # Continue without grab - window will still work
        
        safe_grab()
        
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
        
        # Load embedded GIF
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
        else:
            # Start the GIF animation
            self.animate_gif()
        
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
        
        # Progress indicator
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
           bg='#dc3545',
           fg='white',
           activebackground='#c82333',
           activeforeground='white',
           font=('Arial', 11, 'bold'),
           relief=tk.RAISED,
           bd=2,
           padx=25,
           pady=10,
           cursor='hand2'
           )
        cancel_button.pack()
       
        # Start fact cycling
        self.cycle_wing_facts()
       
        # Prevent closing with X button
        self.loading_window.protocol("WM_DELETE_WINDOW", self.on_loading_window_close)
    
    def _attempt_grab(self):
        """Attempt to set window grab with Windows error handling"""
        try:
            if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
                # Try to grab - if it fails, just continue without it
                self.loading_window.grab_set()
        except tk.TclError as e:
            logger.warning(f"Could not grab window (this is often normal on Windows): {e}")
            # Don't raise the error - the window will still work fine
        except Exception as e:
            logger.warning(f"Unexpected error setting grab: {e}")
    
    def close_loading_overlay(self):
        """Windows-safe loading overlay cleanup"""
        self.animation_running = False
        
        # Clear GIF references to prevent memory leaks
        if hasattr(self, 'gif_frames'):
            self.gif_frames = None
        if hasattr(self, 'gif_durations'):
            self.gif_durations = None
        
        if hasattr(self, 'loading_window'):
            try:
                if self.loading_window.winfo_exists():
                    # Windows-safe grab release
                    try:
                        self.loading_window.grab_release()
                    except tk.TclError:
                        pass  # Grab might not have been set
                    
                    self.loading_window.destroy()
            except tk.TclError:
                pass  # Window might already be destroyed
            except Exception as e:
                logger.warning(f"Error closing loading overlay: {e}")

    def start_dapi_pipeline(self):
        """Run the DAPI nuclei pipeline from the GUI context."""
        input_root = self.master_folder_var.get().strip()
        output_root = self.output_folder_var.get().strip() or input_root

        if not input_root:
            messagebox.showerror("Error", "Please select a master folder for DAPI processing")
            return

        disable_large_output = self.disable_large_output_var.get()
        self.is_processing = True
        self.dapi_button.config(state=tk.DISABLED)
        self.progress_var.set("Running DAPI nuclei pipeline...")

        def worker():
            try:
                from dapi_nuclei_pipeline import DapiNucleiPipelineConfig, run_dapi_nuclei_pipeline

                cfg = DapiNucleiPipelineConfig(
                    input_root=Path(input_root),
                    output_root=Path(output_root),
                    large_output_enabled=not disable_large_output,
                )
                results = run_dapi_nuclei_pipeline(cfg)
                message = f"DAPI pipeline complete for {len(results)} experiments (outputs in {output_root})"
            except Exception as exc:
                logger.error("DAPI pipeline failed: %s", exc, exc_info=True)
                message = f"DAPI pipeline failed: {exc}"

            def finalize():
                self.progress_var.set(message)
                self.dapi_button.config(state=tk.NORMAL)
                self.is_processing = False

            self.root.after(0, finalize)

        threading.Thread(target=worker, daemon=True).start()
    
    def start_processing(self):
        """Fixed start processing with better Windows error handling"""
        if not self.master_folder_var.get() or not self.output_folder_var.get():
            messagebox.showerror("Error", "Please select both master folder and output folder")
            return
        
        if not self.update_config_from_gui():
            return
        
        # Disable buttons first
        self.is_processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Create loading overlay with error handling
        try:
            self.create_drosophila_loading_overlay()
        except Exception as e:
            logger.error(f"Error creating loading overlay: {e}")
            # Continue without overlay
            pass
        
        # Start processing in separate thread
        self.processing_thread = threading.Thread(target=self.process_folders, daemon=True)
        self.processing_thread.start()
    
    def check_progress(self):
        """Improved Windows progress checker with better error handling"""
        try:
            messages_processed = 0
            max_messages = 15  # Process fewer messages per check on Windows
            
            while messages_processed < max_messages:
                try:
                    msg_type, message = self.progress_queue.get_nowait()
                    messages_processed += 1
                    
                    # Use after_idle for all GUI updates to avoid conflicts
                    if msg_type == "progress":
                        self.root.after_idle(lambda msg=message: self._safe_set_progress(msg))
                    elif msg_type == "current_folder":
                        self.root.after_idle(lambda msg=message: self._safe_set_current_folder(msg))
                    elif msg_type == "progress_percent":
                        self.root.after_idle(lambda pct=message: self._safe_set_progress_percent(pct))
                    elif msg_type == "log":
                        self.root.after_idle(lambda msg=message: self._safe_log_message(msg))
                    elif msg_type == "error":
                        self.root.after_idle(lambda msg=message: self._safe_log_message(f"ERROR: {msg}"))
                    elif msg_type == "summary":
                        self.root.after_idle(lambda msg=message: self._update_summary(msg))
                    elif msg_type == "finished":
                        self.root.after_idle(self._processing_finished)
                        return  # Stop checking
                        
                except queue.Empty:
                    break
                except Exception as e:
                    logger.error(f"Error processing progress message: {e}")
                    break
           
        except Exception as e:
            logger.error(f"Progress check error: {e}")
        
        # Schedule next check if still processing
        if self.is_processing:
            self.root.after(200, self.check_progress)  # Longer interval for Windows
    
    def _safe_set_progress(self, message):
        """Thread-safe progress update with error handling"""
        try:
            self.progress_var.set(message)
            self.update_loading_status(message)
        except Exception as e:
            logger.debug(f"Error setting progress: {e}")
    
    def _safe_set_current_folder(self, message):
        """Thread-safe current folder update with error handling"""
        try:
            self.current_folder_var.set(message)
            self.update_loading_status(message)
        except Exception as e:
            logger.debug(f"Error setting current folder: {e}")
    
    def _safe_set_progress_percent(self, percent):
        """Thread-safe progress bar update with error handling"""
        try:
            self.progress_bar['value'] = percent
        except Exception as e:
            logger.debug(f"Error setting progress bar: {e}")
    
    def _safe_log_message(self, message):
        """Thread-safe log message with error handling"""
        try:
            self.log_message(message)
        except Exception as e:
            logger.debug(f"Error logging message: {e}")
    
    def _processing_finished(self):
        """Handle processing completion - Windows safe with better error handling"""
        try:
            self.is_processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            
            # Close overlay safely
            self.close_loading_overlay()
            
            # Force GUI update
            self.root.update_idletasks()
            
            # Show completion message
            self.progress_var.set("Processing completed!")
            
        except Exception as e:
            logger.error(f"Error in processing finished handler: {e}")
    
    # Rest of the methods remain the same...
    # def setup_gui(self):
    #     # Create main notebook for tabs
    #     self.notebook = ttk.Notebook(self.root)
    #     self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    #     # Tab 1: File Selection and Processing
    #     self.processing_frame = ttk.Frame(self.notebook)
    #     self.notebook.add(self.processing_frame, text="Processing")
        
    #     # Tab 2: Configuration
    #     self.config_frame = ttk.Frame(self.notebook)
    #     self.notebook.add(self.config_frame, text="Configuration")
        
    #     # Tab 3: Advanced Settings
    #     self.advanced_frame = ttk.Frame(self.notebook)
    #     self.notebook.add(self.advanced_frame, text="Advanced")
        
    #     # Tab 4: Results/Log
    #     self.results_frame = ttk.Frame(self.notebook)
    #     self.notebook.add(self.results_frame, text="Results & Log")
        
    #     self.setup_processing_tab()
    #     self.setup_config_tab()
    #     self.setup_advanced_tab()
    #     self.setup_results_tab()
    def debug_gui_variables(self):
        """Debug method to check GUI variable creation."""
        logger.info("=== GUI VARIABLES DEBUG ===")
        logger.info(f"self.gui_vars exists: {hasattr(self, 'gui_vars')}")
        
        if hasattr(self, 'gui_vars'):
            logger.info(f"gui_vars type: {type(self.gui_vars)}")
            logger.info(f"gui_vars length: {len(self.gui_vars)}")
            
            if self.gui_vars:
                logger.info("GUI Variables found:")
                for name, info in self.gui_vars.items():
                    try:
                        var = info['var']
                        current_value = var.get()
                        var_type = type(var).__name__
                        logger.info(f"  {name}: {var_type} = {current_value}")
                    except Exception as e:
                        logger.error(f"  {name}: ERROR reading value - {e}")
            else:
                logger.error("gui_vars dictionary is empty!")
        else:
            logger.error("self.gui_vars does not exist!")
        
        logger.info("=== END DEBUG ===")

    def setup_gui(self):
        """Modified setup_gui with debugging."""
        logger.info("Starting setup_gui...")
        
        # Ensure gui_vars exists
        if not hasattr(self, 'gui_vars'):
            self.gui_vars = {}
            logger.info("Created gui_vars dictionary")
        
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
        
        logger.info("Setting up processing tab...")
        self.setup_processing_tab()
        
        logger.info("Setting up config tab...")
        self.setup_config_tab()
        
        logger.info("Setting up advanced tab...")
        self.setup_advanced_tab()
        
        logger.info("Setting up results tab...")
        self.setup_results_tab()
        
        # Debug: Check what we created
        self.debug_gui_variables()
        
        logger.info("setup_gui completed")
    
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

        self.disable_large_output_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            options_frame,
            text="Disable large DAPI outputs (modal slice only)",
            variable=self.disable_large_output_var,
        ).pack(anchor=tk.W, pady=2)
        
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

        self.dapi_button = ttk.Button(
            control_frame,
            text="Run DAPI Nuclei Pipeline",
            command=self.start_dapi_pipeline,
        )
        self.dapi_button.pack(side=tk.LEFT, padx=(10, 0))

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
        """Debug version to see what's happening."""
        logger.info("=== SETUP_CONFIG_TAB STARTED ===")
        logger.info(f"gui_vars exists: {hasattr(self, 'gui_vars')}")
        logger.info(f"gui_vars length before: {len(self.gui_vars) if hasattr(self, 'gui_vars') else 'N/A'}")
        
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
        
        logger.info("Creating config entries...")
        
        try:
            logger.info("Creating min_distance entry...")
            self.create_config_entry(core_frame, "min_distance", "Minimum distance between peaks:", self.config.min_distance, int)
            logger.info(f"gui_vars after min_distance: {len(self.gui_vars)}")
            
            logger.info("Creating high_thresh_abs entry...")
            self.create_config_entry(core_frame, "high_thresh_abs", "High threshold (absolute):", self.config.high_thresh_abs, float)
            logger.info(f"gui_vars after high_thresh_abs: {len(self.gui_vars)}")
            
            logger.info("Creating low_thresh_abs entry...")
            self.create_config_entry(core_frame, "low_thresh_abs", "Low threshold (absolute):", self.config.low_thresh_abs, float)
            logger.info(f"gui_vars after low_thresh_abs: {len(self.gui_vars)}")
            
        except Exception as e:
            logger.error(f"Error creating config entries: {e}")
            import traceback
            traceback.print_exc()
        
        # Multi-scale Detection
        scale_frame = ttk.LabelFrame(scrollable_frame, text="Multi-scale Detection", padding=10)
        scale_frame.pack(fill=tk.X, padx=10, pady=5)
        
        try:
            self.create_config_entry(scale_frame, "scales", "Detection scales (comma-separated):", 
                                    ",".join(map(str, self.config.scales)), str)
            self.create_config_entry(scale_frame, "scale_weight_decay", "Scale weight decay:", self.config.scale_weight_decay, float)
        except Exception as e:
            logger.error(f"Error creating scale entries: {e}")
        
        # Pre-processing
        preproc_frame = ttk.LabelFrame(scrollable_frame, text="Pre-processing", padding=10)
        preproc_frame.pack(fill=tk.X, padx=10, pady=5)
        
        try:
            self.create_config_checkbox(preproc_frame, "use_clahe", "Use CLAHE enhancement", self.config.use_clahe)
            self.create_config_entry(preproc_frame, "clahe_clip_limit", "CLAHE clip limit:", self.config.clahe_clip_limit, float)
            
            self.create_config_checkbox(preproc_frame, "use_white_tophat", "Use white top-hat", self.config.use_white_tophat)
            self.create_config_entry(preproc_frame, "tophat_radius", "Top-hat radius:", self.config.tophat_radius, int)
            
            self.create_config_checkbox(preproc_frame, "use_local_threshold", "Use local thresholding", self.config.use_local_threshold)
            self.create_config_entry(preproc_frame, "local_block_size", "Local block size:", self.config.local_block_size, int)
        except Exception as e:
            logger.error(f"Error creating preprocessing entries: {e}")
        
        # Peak Detection
        peak_frame = ttk.LabelFrame(scrollable_frame, text="Peak Detection", padding=10)
        peak_frame.pack(fill=tk.X, padx=10, pady=5)
        
        try:
            self.create_config_entry(peak_frame, "min_peak_intensity", "Minimum peak intensity:", self.config.min_peak_intensity, float)
            self.create_config_entry(peak_frame, "max_peak_density", "Maximum peak density:", self.config.max_peak_density, float)
            self.create_config_entry(peak_frame, "edge_exclusion_buffer", "Edge exclusion buffer:", self.config.edge_exclusion_buffer, int)
        except Exception as e:
            logger.error(f"Error creating peak detection entries: {e}")
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        logger.info(f"=== SETUP_CONFIG_TAB FINISHED - gui_vars length: {len(self.gui_vars)} ===")
    
    def setup_advanced_tab(self):
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
        
        # Clustering and Filtering
        cluster_frame = ttk.LabelFrame(scrollable_frame, text="Clustering and Filtering", padding=10)
        cluster_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.create_config_entry(cluster_frame, "dbscan_eps", "DBSCAN epsilon:", self.config.dbscan_eps, float)
        self.create_config_entry(cluster_frame, "dbscan_min_samples", "DBSCAN min samples:", self.config.dbscan_min_samples, int)
        self.create_config_entry(cluster_frame, "valley_drop", "Valley drop threshold:", self.config.valley_drop, float)
        self.create_config_entry(cluster_frame, "neighbour_tolerance", "Neighbor tolerance:", self.config.neighbour_tolerance, float)
        self.create_config_entry(cluster_frame, "min_neighbours", "Minimum neighbors:", self.config.min_neighbours, int)
        
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
        
        # DEBUG: Log total variables after advanced tab
        logger.info(f"setup_advanced_tab completed. Total GUI variables: {len(self.gui_vars)}")
    
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
        """Debug version of create_config_entry."""
        logger.info(f"create_config_entry called: {attr_name}")
        
        try:
            frame = ttk.Frame(parent)
            frame.pack(fill=tk.X, pady=2)
            
            ttk.Label(frame, text=label, width=25).pack(side=tk.LEFT)
            
            # Create variable and store in dictionary
            var = tk.StringVar(value=str(default_value))
            entry = ttk.Entry(frame, textvariable=var, width=20)
            entry.pack(side=tk.LEFT, padx=(5,0))
            
            # Check if gui_vars exists
            if not hasattr(self, 'gui_vars'):
                logger.error("gui_vars does not exist! Creating it...")
                self.gui_vars = {}
            
            # Store in our dictionary
            self.gui_vars[attr_name] = {
                'var': var,
                'type': data_type,
                'widget': entry
            }
            
            logger.info(f"Successfully created and stored {attr_name}. Total vars: {len(self.gui_vars)}")
            
        except Exception as e:
            logger.error(f"Error in create_config_entry for {attr_name}: {e}")
            import traceback
            traceback.print_exc()
    def force_refresh_all_widgets(self):
        """Force refresh of all GUI widgets."""
        logger.info("Forcing refresh of all GUI widgets...")
        
        try:
            # Method 1: Update all widgets in gui_vars
            for var_name, gui_info in self.gui_vars.items():
                try:
                    widget = gui_info['widget']
                    widget.update_idletasks()
                except Exception as e:
                    logger.debug(f"Could not update widget {var_name}: {e}")
            
            # Method 2: Update entire window
            self.root.update_idletasks()
            self.root.update()
            
            # Method 3: Force redraw by temporarily changing focus
            current_focus = self.root.focus_get()
            self.root.focus_set()
            if current_focus:
                current_focus.focus_set()
            
            logger.info("Forced refresh completed")
            
        except Exception as e:
            logger.error(f"Error during forced refresh: {e}")
    def test_gui_widget_connection(self):
        """Test if GUI widgets are properly connected to variables."""
        logger.info("=== TESTING GUI WIDGET CONNECTION ===")
        
        if not hasattr(self, 'gui_vars') or not self.gui_vars:
            logger.error("No GUI variables to test!")
            return
        
        # Test a few key variables
        test_vars = ['min_distance', 'high_thresh_abs', 'use_clahe', 'scales']
        
        for var_name in test_vars:
            if var_name in self.gui_vars:
                try:
                    gui_info = self.gui_vars[var_name]
                    var = gui_info['var']
                    widget = gui_info['widget']
                    
                    # Get current value
                    current_value = var.get()
                    logger.info(f"{var_name}: variable value = {current_value}")
                    
                    # Try to change the value
                    test_value = "999" if not isinstance(var, tk.BooleanVar) else not current_value
                    
                    logger.info(f"Setting {var_name} to test value: {test_value}")
                    var.set(test_value)
                    
                    # Check if it changed
                    new_value = var.get()
                    logger.info(f"{var_name}: after setting = {new_value}")
                    
                    # Force widget update
                    try:
                        widget.update_idletasks()
                        widget.update()
                        logger.info(f"Forced update for {var_name} widget")
                    except Exception as e:
                        logger.error(f"Could not force update {var_name} widget: {e}")
                    
                    # Reset to original
                    var.set(current_value)
                    
                    # Check widget state
                    try:
                        widget_state = widget.cget('state')
                        logger.info(f"{var_name} widget state: {widget_state}")
                    except Exception as e:
                        logger.warning(f"Could not get {var_name} widget state: {e}")
                    
                except Exception as e:
                    logger.error(f"Error testing {var_name}: {e}")
        
        logger.info("=== END GUI CONNECTION TEST ===")

    def create_config_checkbox(self, parent, attr_name, label, default_value):
        """Debug version of create_config_checkbox."""
        logger.info(f"create_config_checkbox called: {attr_name}")
        
        try:
            var = tk.BooleanVar(value=default_value)
            checkbox = ttk.Checkbutton(parent, text=label, variable=var)
            checkbox.pack(anchor=tk.W, pady=2)
            
            # Check if gui_vars exists
            if not hasattr(self, 'gui_vars'):
                logger.error("gui_vars does not exist! Creating it...")
                self.gui_vars = {}
            
            # Store in our dictionary
            self.gui_vars[attr_name] = {
                'var': var,
                'type': bool,
                'widget': checkbox
            }
            
            logger.info(f"Successfully created and stored checkbox {attr_name}. Total vars: {len(self.gui_vars)}")
            
        except Exception as e:
            logger.error(f"Error in create_config_checkbox for {attr_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def load_initial_config(self):
        """Load initial configuration values into GUI - improved with better error handling."""
        logger.info("Starting initial configuration load...")
        
        # Check if GUI is fully initialized first
        required_vars = [
            'min_distance_var', 'high_thresh_abs_var', 'low_thresh_abs_var',
            'scales_var', 'use_clahe_var', 'use_white_tophat_var'
        ]
        
        missing_vars = [var for var in required_vars if not hasattr(self, var)]
        if missing_vars:
            logger.warning(f"GUI not fully initialized. Missing variables: {missing_vars}")
            # Try again after a short delay
            self.root.after(500, self.load_initial_config)
            return
        
        try:
            self.update_gui_from_config()
            logger.info("Initial configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error in load_initial_config: {e}")
            self.set_default_gui_values()

    def set_default_gui_values(self):
        """Set default values with Windows-safe GUI updates and better error handling."""
        logger.info("Setting default GUI values...")
        
        defaults = {
            'min_distance_var': '8',
            'high_thresh_abs_var': '0.30',
            'low_thresh_abs_var': '0.20',
            'scales_var': "0.8,1.0,1.2",
            'scale_weight_decay_var': '0.8',
            'use_clahe_var': False,
            'clahe_clip_limit_var': '0.01',
            'use_white_tophat_var': True,
            'tophat_radius_var': '2',
            'use_local_threshold_var': True,
            'local_block_size_var': '71',

            'min_peak_intensity_var': '0.25',
            'max_peak_density_var': '0.002',
            'edge_exclusion_buffer_var': '10',
            'dbscan_eps_var': '10.0',
            'dbscan_min_samples_var': '1',
            'valley_drop_var': '0.25',
            'neighbour_tolerance_var': '0.20',
            'min_neighbours_var': '7',
            'intervein_threshold_var': '0.4',
            'min_region_area_var': '60000',
            'max_region_area_var': '1500000',
            'vein_width_estimate_var': '7',
            'min_vein_length_var': '100',
            'vein_detection_sensitivity_var': '0.9',
            'min_wing_area_var': '100000',
            'border_buffer_var': '20'
        }
        
        def set_var_safe(var_name, default_value):
            """Safely set a GUI variable with error handling"""
            if hasattr(self, var_name):
                try:
                    var = getattr(self, var_name)
                    if isinstance(var, tk.BooleanVar):
                        var.set(bool(default_value))
                    else:
                        var.set(str(default_value))
                    logger.debug(f"Set {var_name} to {default_value}")
                except Exception as e:
                    logger.warning(f"Could not set default for {var_name}: {e}")
            else:
                logger.warning(f"Variable {var_name} not found for default setting")
        
        # Set defaults using safe method
        for var_name, default_value in defaults.items():
            # Schedule each update for the next event loop cycle
            self.root.after_idle(set_var_safe, var_name, default_value)
        
        logger.info("Default values set successfully")

    def update_gui_from_config(self):
        """Windows-safe GUI update from config object with better error handling."""
        try:
            # Dictionary of config values to set
            config_mappings = {
                'min_distance_var': str(self.config.min_distance),
                'high_thresh_abs_var': str(self.config.high_thresh_abs),
                'low_thresh_abs_var': str(self.config.low_thresh_abs),
                'scales_var': ",".join(map(str, self.config.scales)),
                'scale_weight_decay_var': str(self.config.scale_weight_decay),
                'use_clahe_var': self.config.use_clahe,  # Boolean
                'clahe_clip_limit_var': str(self.config.clahe_clip_limit),
                'use_white_tophat_var': self.config.use_white_tophat,  # Boolean
                'tophat_radius_var': str(self.config.tophat_radius),
                'use_local_threshold_var': self.config.use_local_threshold,  # Boolean
                'local_block_size_var': str(self.config.local_block_size),

                'min_peak_intensity_var': str(self.config.min_peak_intensity),
                'max_peak_density_var': str(self.config.max_peak_density),
                'edge_exclusion_buffer_var': str(self.config.edge_exclusion_buffer),
                'dbscan_eps_var': str(self.config.dbscan_eps),
                'dbscan_min_samples_var': str(self.config.dbscan_min_samples),
                'valley_drop_var': str(self.config.valley_drop),
                'neighbour_tolerance_var': str(self.config.neighbour_tolerance),
                'min_neighbours_var': str(self.config.min_neighbours),
                'intervein_threshold_var': str(self.config.intervein_threshold),
                'min_region_area_var': str(self.config.min_region_area),
                'max_region_area_var': str(self.config.max_region_area),
                'vein_width_estimate_var': str(self.config.vein_width_estimate),
                'min_vein_length_var': str(self.config.min_vein_length),
                'vein_detection_sensitivity_var': str(self.config.vein_detection_sensitivity),
                'min_wing_area_var': str(self.config.min_wing_area),
                'border_buffer_var': str(self.config.border_buffer)
            }
            
            # Set each variable safely
            for var_name, value in config_mappings.items():
                if hasattr(self, var_name):
                    try:
                        var = getattr(self, var_name)
                        
                        # Schedule the update for the next event loop cycle
                        def set_var(v=var, val=value):
                            try:
                                if isinstance(v, tk.BooleanVar):
                                    v.set(bool(val))
                                else:
                                    v.set(str(val))
                            except Exception as e:
                                logger.warning(f"Could not set {var_name}: {e}")
                        
                        self.root.after_idle(set_var)
                        
                    except Exception as e:
                        logger.warning(f"Error setting {var_name}: {e}")
                else:
                    logger.warning(f"GUI variable {var_name} not found")
            
            logger.info("GUI updated from configuration successfully")
            
        except Exception as e:
            logger.error(f"Critical error updating GUI from config: {e}")
            # Fall back to defaults
            self.set_default_gui_values()

    def browse_master_folder(self):
        """Windows-optimized folder browser."""
        try:
            # Use askdirectory with Windows-specific options
            folder = filedialog.askdirectory(
                title="Select Master Folder (containing subfolders with wing data)",
                mustexist=True
            )
            
            if folder:
                # Normalize path for Windows
                folder = os.path.normpath(folder)
                self.master_folder_var.set(folder)
                
                # Auto-set output folder if not already set
                if not self.output_folder_var.get().strip():
                    parent_dir = os.path.dirname(folder)
                    folder_name = os.path.basename(folder)
                    output_folder = os.path.join(parent_dir, f"{folder_name}_Results")
                    output_folder = os.path.normpath(output_folder)
                    self.output_folder_var.set(output_folder)
                
                # Force refresh of folder list
                self.root.after(100, self.refresh_folder_list)
                
        except Exception as e:
            logger.error(f"Error browsing for master folder: {e}")
            messagebox.showerror("Error", f"Could not browse for folder: {e}")

    def browse_output_folder(self):
        """Windows-optimized output folder browser with better error handling."""
        try:
            # Ensure focus is on main window
            self.root.focus_set()
            self.root.update()
            
            folder = filedialog.askdirectory(
                parent=self.root,
                title="Select Output Folder",
                mustexist=False  # Allow creating new folders
            )
            
            if folder:
                folder = os.path.normpath(folder)
                self.output_folder_var.set(folder)
                
        except Exception as e:
            logger.error(f"Error browsing for output folder: {e}")
            messagebox.showerror("Error", f"Could not browse for folder: {e}")

    def refresh_folder_list(self):
        """Windows-safe folder list refresh."""
        # Clear existing list
        self.folder_listbox.delete(0, tk.END)
        
        master_folder = self.master_folder_var.get().strip()
        if not master_folder:
            return
        
        if not os.path.exists(master_folder):
            self.folder_listbox.insert(tk.END, "⚠ Folder path does not exist")
            return
        
        try:
            subfolders = []
            
            # Use os.scandir for better Windows performance
            with os.scandir(master_folder) as entries:
                for entry in entries:
                    if entry.is_dir():
                        try:
                            # Check if folder contains valid files
                            mapping = find_associated_files(entry.path)
                            if mapping:
                                file_count = len(mapping)
                                subfolders.append(f"✓ {entry.name} ({file_count} file pairs)")
                            else:
                                subfolders.append(f"✗ {entry.name} (no valid files)")
                        except Exception as e:
                            subfolders.append(f"⚠ {entry.name} (scan error)")
                            logger.warning(f"Error scanning {entry.name}: {e}")
            
            # Sort and add to listbox
            subfolders.sort()
            for folder_info in subfolders:
                self.folder_listbox.insert(tk.END, folder_info)
            
            if not subfolders:
                self.folder_listbox.insert(tk.END, "No subfolders found")
                
        except PermissionError:
            self.folder_listbox.insert(tk.END, "⚠ Permission denied - check folder access")
            logger.error(f"Permission denied accessing {master_folder}")
        except Exception as e:
            self.folder_listbox.insert(tk.END, f"⚠ Error scanning folder: {str(e)}")
            logger.error(f"Error scanning master folder: {e}")
    
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

    def create_drosophila_loading_overlay(self):
        """Fixed loading overlay with proper Windows grab handling"""
        if hasattr(self, 'loading_window'):
            try:
                if self.loading_window.winfo_exists():
                    return
            except tk.TclError:
                # Window was destroyed, continue
                pass
        
        # Create overlay window
        self.loading_window = tk.Toplevel(self.root)
        self.loading_window.title("HomerWings Processing")
        self.loading_window.geometry("600x450")
        self.loading_window.resizable(False, False)
        
        # Windows-specific window setup
        self.loading_window.transient(self.root)
        
        # Position relative to main window with error handling
        try:
            x = self.root.winfo_x() + (self.root.winfo_width() // 2) - 300
            y = self.root.winfo_y() + (self.root.winfo_height() // 2) - 225
            self.loading_window.geometry(f"500x450+{x}+{y}")
        except tk.TclError:
            # Fallback positioning
            self.loading_window.geometry("500x450+100+100")
        
        self.loading_window.configure(bg='#1a1a1a')
        
        # FIXED: Safer grab handling with proper timing
        def safe_grab():
            try:
                # Make sure window is fully created and visible first
                self.loading_window.update_idletasks()
                self.loading_window.focus_set()
                
                # Try to grab after a short delay to ensure window is ready
                def attempt_grab():
                    try:
                        # Only grab if window still exists and is ready
                        if hasattr(self, 'loading_window') and self.loading_window.winfo_exists():
                            # Check if any other window has grab first
                            current_grab = self.root.grab_current()
                            if current_grab is None:
                                self.loading_window.grab_set()
                                logger.info("Successfully set window grab")
                            else:
                                logger.warning(f"Another window has grab: {current_grab}")
                                # Still functional without grab
                    except tk.TclError as e:
                        logger.warning(f"Could not set window grab (window still functional): {e}")
                    except Exception as e:
                        logger.warning(f"Unexpected error setting grab: {e}")
                
                # Schedule grab attempt after window is fully ready
                self.root.after(200, attempt_grab)
                
            except Exception as e:
                logger.warning(f"Error in safe_grab setup: {e}")
                # Continue without grab - window will still work
        
        # Schedule grab setup for after window creation
        self.root.after_idle(safe_grab)
        
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
        
        # Load embedded GIF
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
        else:
            # Start the GIF animation
            self.animate_gif()
        
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
        
        # Progress indicator
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
           bg='#dc3545',
           fg='white',
           activebackground='#c82333',
           activeforeground='white',
           font=('Arial', 11, 'bold'),
           relief=tk.RAISED,
           bd=2,
           padx=25,
           pady=10,
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
        """Update configuration from GUI - simplified and reliable."""
        try:
            errors = []
            
            # Process each GUI variable
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
                logger.info("Configuration updated from GUI successfully")
                return True
            except Exception as e:
                messagebox.showerror("Validation Error", f"Configuration validation failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Critical error updating config: {e}")
            messagebox.showerror("Critical Error", f"Failed to update configuration: {e}")
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
                   
                   result = main_with_pentagone_support(
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
       """Windows-optimized progress checker with better error handling."""
       try:
           messages_processed = 0
           max_messages = 20  # Process more messages per check on Windows
           
           while messages_processed < max_messages:
               try:
                   msg_type, message = self.progress_queue.get_nowait()
                   messages_processed += 1
                   
                   # Use after_idle for thread-safe GUI updates
                   if msg_type == "progress":
                       self.root.after_idle(self._safe_set_progress, message)
                   elif msg_type == "current_folder":
                       self.root.after_idle(self._safe_set_current_folder, message)
                   elif msg_type == "progress_percent":
                       self.root.after_idle(self._safe_set_progress_percent, message)
                   elif msg_type == "log":
                       self.root.after_idle(self._safe_log_message, message)
                   elif msg_type == "error":
                       self.root.after_idle(self._safe_log_message, f"ERROR: {message}")
                   elif msg_type == "summary":
                       self.root.after_idle(self._update_summary, message)
                   elif msg_type == "finished":
                       self.root.after_idle(self._processing_finished)
                       return  # Stop checking
                       
               except queue.Empty:
                   break
               except Exception as e:
                   logger.error(f"Error processing progress message: {e}")
                   break
           
       except Exception as e:
           logger.error(f"Progress check error: {e}")
       
       # Schedule next check if still processing
       if self.is_processing:
           self.root.after(150, self.check_progress)  # Slightly longer interval for Windows

    def _safe_set_progress(self, message):
       """Thread-safe progress update."""
       try:
           self.progress_var.set(message)
           self.update_loading_status(message)
       except:
           pass

    def _safe_set_current_folder(self, message):
       """Thread-safe current folder update."""
       try:
           self.current_folder_var.set(message)
           self.update_loading_status(message)
       except:
           pass

    def _safe_set_progress_percent(self, percent):
       """Thread-safe progress bar update."""
       try:
           self.progress_bar['value'] = percent
       except:
           pass

    def _safe_log_message(self, message):
       """Thread-safe log message."""
       try:
           self.log_message(message)
       except:
           pass

    def _processing_finished(self):
       """Handle processing completion - Windows safe."""
       try:
           self.is_processing = False
           self.start_button.config(state=tk.NORMAL)
           self.stop_button.config(state=tk.DISABLED)
           self.close_loading_overlay()
           
           # Force GUI update
           self.root.update_idletasks()
           
           # Show completion message
           self.progress_var.set("Processing completed!")
           
       except Exception as e:
           logger.error(f"Error in processing finished handler: {e}")

    def _update_summary(self, summary_text):
       """Update summary text safely."""
       try:
           self.summary_text.delete(1.0, tk.END)
           self.summary_text.insert(1.0, summary_text)
           self.summary_text.see(1.0)
       except Exception as e:
           logger.error(f"Error updating summary: {e}")

    def log_message(self, message):
       timestamp = time.strftime('%H:%M:%S')
       try:
           self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
           self.log_text.see(tk.END)
       except:
           pass
   
    def clear_log(self):
       try:
           self.log_text.delete(1.0, tk.END)
       except:
           pass
   
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
        """Reset configuration to defaults and update GUI."""
        self.config = TrichomeDetectionConfig()
        self._apply_config_to_gui()
        messagebox.showinfo("Reset", "Configuration reset to defaults")
        
                 


def run_gui():
    """Windows-optimized GUI launcher with comprehensive error handling."""
    
    # Windows-specific initialization
    if sys.platform.startswith('win'):
        try:
            # Set DPI awareness for better display on high-DPI screens
            import ctypes
            try:
                # Try the newer method first
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # Per monitor DPI aware
            except:
                try:
                    # Fallback to older method
                    ctypes.windll.user32.SetProcessDPIAware()
                except:
                    pass  # If both fail, continue without DPI awareness
        except Exception as e:
            logger.warning(f"Could not set DPI awareness: {e}")
    
    # Create root window with error handling
    try:
        root = tk.Tk()
    except Exception as e:
        logger.error(f"Failed to create Tk root window: {e}")
        print(f"Error: Could not initialize GUI: {e}")
        return
    
    # Windows-specific window setup
    if sys.platform.startswith('win'):
        try:
            # Ensure window is properly displayed
            root.lift()
            root.wm_attributes("-topmost", True)
            root.after(2000, lambda: root.wm_attributes("-topmost", False))  # Longer delay for Windows
            
            # Set Windows-specific options
            root.wm_state('normal')
            root.focus_force()
        except Exception as e:
            logger.warning(f"Windows-specific setup failed: {e}")
    
    # Configure window
    root.title("HomerWings V2 (Windows)")
    
    # Set window geometry with Windows compatibility
    try:
        # Get screen dimensions safely
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        
        # Set reasonable window size (slightly smaller for Windows)
        window_width = min(1200, int(screen_width * 0.75))  # Reduced from 0.8
        window_height = min(900, int(screen_height * 0.75))  # Reduced from 0.8
        
        # Center window
        pos_x = max(0, (screen_width - window_width) // 2)
        pos_y = max(0, (screen_height - window_height) // 2)
        
        # Apply geometry
        root.geometry(f"{window_width}x{window_height}+{pos_x}+{pos_y}")
        
        logger.info(f"Set window geometry: {window_width}x{window_height}+{pos_x}+{pos_y}")
        
    except Exception as e:
        logger.warning(f"Could not set optimal window geometry: {e}")
        # Fallback to fixed size
        root.geometry("1000x800+100+100")
    
    # Set application icon if available
    try:
        icon_path = get_resource_path('icon.ico')
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
            logger.info(f"Set application icon: {icon_path}")
    except Exception as e:
        logger.warning(f"Could not set application icon: {e}")
    
    # Create application with comprehensive error handling
    try:
        app = TrichomeAnalysisGUI(root)
        logger.info("TrichomeAnalysisGUI created successfully")
        
        # Enhanced Windows error handler
        def handle_error(exc, val, tb):
            error_msg = f"GUI Error: {exc.__name__}: {val}"
            logger.error(error_msg)
            
            # Don't show GUI error dialogs for common Windows issues
            common_windows_errors = [
                "grab failed: another application has grab",
                "can't invoke \"update\" command",
                "invalid command name",
                "application has been destroyed"
            ]
            
            show_error_dialog = True
            for common_error in common_windows_errors:
                if common_error in str(val).lower():
                    show_error_dialog = False
                    break
            
            if show_error_dialog:
                try:
                    messagebox.showerror("Application Error", 
                                       f"An error occurred: {val}\n\nCheck the log for details.")
                except:
                    pass  # If messagebox fails, just log
            
            # Print traceback to console for debugging
            try:
                import traceback
                traceback.print_exception(exc, val, tb)
            except:
                pass
            
            return True  # Prevent default error handling
        
        root.report_callback_exception = handle_error
        
        # Enhanced shutdown handler
        def on_closing():
            try:
                logger.info("Application closing requested")
                
                if hasattr(app, 'is_processing') and app.is_processing:
                    result = messagebox.askyesnocancel(
                        "Quit", 
                        "Processing is still running. Do you want to stop and quit?"
                    )
                    if result is True:  # Yes - stop and quit
                        logger.info("User chose to stop processing and quit")
                        app.is_processing = False
                        if hasattr(app, 'close_loading_overlay'):
                            app.close_loading_overlay()
                        root.after(1000, lambda: safe_destroy(root))
                    elif result is False:  # No - quit anyway
                        logger.info("User chose to quit without stopping")
                        safe_destroy(root)
                    # Cancel - do nothing
                else:
                    safe_destroy(root)
                    
            except Exception as e:
                logger.error(f"Error during shutdown: {e}")
                safe_destroy(root)
        
        def safe_destroy(window):
            """Safely destroy the window"""
            try:
                window.quit()
                window.destroy()
            except Exception as e:
                logger.error(f"Error destroying window: {e}")
                try:
                    import sys
                    sys.exit(0)
                except:
                    pass
        
        root.protocol("WM_DELETE_WINDOW", on_closing)
        
        # Ensure window is visible and properly focused
        try:
            root.deiconify()
            root.lift()
            root.focus_force()
            
            # Windows-specific: Additional focus handling
            if sys.platform.startswith('win'):
                root.after(1000, lambda: root.focus_set())
                
        except Exception as e:
            logger.warning(f"Could not properly focus window: {e}")
        
        # Start main loop with comprehensive error handling
        logger.info("Starting Windows GUI main loop...")
        try:
            root.mainloop()
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            try:
                messagebox.showerror("Critical Error", 
                                   f"A critical error occurred: {e}\n\nThe application will close.")
            except:
                pass
        finally:
            # Cleanup
            try:
                logger.info("Performing cleanup...")
                if hasattr(app, 'close_loading_overlay'):
                    app.close_loading_overlay()
                root.quit()
            except Exception as e:
                logger.error(f"Error during cleanup: {e}")
            finally:
                try:
                    root.destroy()
                except:
                    pass
                
    except Exception as e:
        logger.error(f"Failed to create GUI application: {e}")
        try:
            messagebox.showerror("Startup Error", 
                               f"Failed to start application: {e}\n\nPlease check the log for details.")
        except:
            print(f"Critical Error: Failed to start application: {e}")
        finally:
            try:
                root.destroy()
            except:
                pass

# Additional Windows-specific helper functions
def ensure_windows_compatibility():
    """Ensure Windows compatibility for file operations"""
    if sys.platform.startswith('win'):
        # Set proper encoding for Windows
        import locale
        try:
            locale.setlocale(locale.LC_ALL, '')
        except:
            pass
        
        # Ensure matplotlib works on Windows
        try:
            import matplotlib
            matplotlib.use('TkAgg')
        except:
            pass
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






logger = logging.getLogger(__name__)





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
            if wing_ratio < 0.05 or wing_ratio > 0.35:
                logger.info(f"  REJECT Region {region.label}: wing_ratio={wing_ratio:.4f} outside range [0.05, 0.35]")
                continue
            
            # Basic shape filter - target regions have AR between 2-11
            if aspect_ratio < 1.5 or aspect_ratio > 12.0:
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
    vein_detector = ImprovedWingVeinDetector(cfg)
    vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    
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
            score += 20 + excess_error * 200
        
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
            cv.morphologyEx(intervein_prob, cv.MORPH_CLOSE, (10,10))
            wing_mask = intervein_prob < 0.4
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
        dilated_veins = morphology.binary_dilation(actual_vein_mask, morphology.disk(vein_width))
        virtual_veins = thick_boundary & (~dilated_veins)
        virtual_veins = morphology.remove_small_objects(virtual_veins, min_size=100)
        return virtual_veins
    
    def _filter_intervein_regions(self, labeled_mask, wing_mask):
        """Filter intervein regions with criteria suitable for boundary regions."""
        filtered_mask = np.zeros_like(labeled_mask)
        new_label = 1
        
        regions = regionprops(labeled_mask)
        regions.sort(key=lambda r: r.area, reverse=True)
        
        for region in regions:
            if region.label == 0:
                continue
            
            min_area = self.config.auto_intervein_min_area
            max_area = self.config.auto_intervein_max_area
            
            region_mask = labeled_mask == region.label
            wing_boundary = morphology.binary_erosion(wing_mask) ^ wing_mask
            boundary_overlap = np.sum(region_mask & wing_boundary)
            boundary_ratio = boundary_overlap / region.area if region.area > 0 else 0
            
            if boundary_ratio > 0.1:
                min_area = min_area // 2
                if region.centroid[0] > labeled_mask.shape[0] * 0.7:
                    min_area = min_area // 2
            
            if min_area <= region.area <= max_area:
                if self.config.intervein_shape_filter:
                    min_solidity = self.config.min_intervein_solidity
                    if boundary_ratio > 0.1:
                        min_solidity *= 0.7
                    
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
    
    def segment_intervein_regions_enhanced(self, prob_map, vein_mask, raw_img=None, 
                                         force_pentagone=False):
        """Enhanced segmentation with automatic pentagone detection."""
        
        logger.info("Starting enhanced intervein segmentation with pentagone support...")
        
        # Step 1-6: Same as before (detect wing boundary, create barriers, etc.)
        wing_mask = self.detect_wing_boundary(prob_map, raw_img)
        
        # Import the border checker from your existing code
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
        intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=5000)
        labeled_regions = label(intervein_mask)
        filtered_labeled_mask = self._filter_intervein_regions(labeled_regions, wing_mask)
        
        # Step 7: Apply anatomical labeling with pentagone detection
        if force_pentagone:
            logger.info("FORCED PENTAGONE MODE")
            final_labeled_mask = self.pentagone_handler._apply_four_region_labeling(
                filtered_labeled_mask, wing_mask)
            self.is_pentagone_mode = True
        else:
            # Automatic detection
            final_labeled_mask, is_pentagone = self.pentagone_handler.apply_pentagone_labeling(
                filtered_labeled_mask, wing_mask)
            self.is_pentagone_mode = is_pentagone
        
        # Count regions
        unique_labels = np.unique(final_labeled_mask)
        n_regions = len(unique_labels[unique_labels > 0])
        mode_str = "pentagone (4-region)" if self.is_pentagone_mode else "normal (5-region)"
        logger.info(f"Enhanced detection found {n_regions} regions using {mode_str} labeling")
        
        return final_labeled_mask, combined_vein_mask, virtual_veins
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
        """Enhanced segmentation with automatic pentagone detection."""
        
        logger.info("Starting enhanced intervein segmentation with pentagone support...")
        
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
        intervein_mask = morphology.remove_small_holes(intervein_mask, area_threshold=5000)
        labeled_regions = label(intervein_mask)
        filtered_labeled_mask = self._filter_intervein_regions(labeled_regions, wing_mask)
        
        # Step 7: Apply anatomical labeling with pentagone detection
        if force_pentagone:
            logger.info("FORCED PENTAGONE MODE")
            final_labeled_mask = self.pentagone_handler._apply_four_region_labeling(
                filtered_labeled_mask, wing_mask)
            self.is_pentagone_mode = True
        else:
            # Automatic detection
            final_labeled_mask, is_pentagone = self.pentagone_handler.apply_pentagone_labeling(
                filtered_labeled_mask, wing_mask)
            self.is_pentagone_mode = is_pentagone
        
        # Count regions
        unique_labels = np.unique(final_labeled_mask)
        n_regions = len(unique_labels[unique_labels > 0])
        mode_str = "pentagone (4-region)" if self.is_pentagone_mode else "normal (5-region)"
        logger.info(f"Enhanced detection found {n_regions} regions using {mode_str} labeling")
        
        return final_labeled_mask, combined_vein_mask, virtual_veins


# Modified save functions for pentagone support
def save_anatomical_region_info_with_pentagone(valid_regions, basename, output_dir, is_pentagone_mode=False):
    """Save anatomical region information with pentagone support."""
    output_path = os.path.join(output_dir, f"{basename}_anatomical_regions.csv")
    
    # Choose appropriate template matcher
    if is_pentagone_mode:
        pentagone_handler = PentagoneMutantHandler()
        target_regions = pentagone_handler.pentagone_target_regions
        mode_suffix = "_pentagone"
    else:
        template_matcher = AnatomicalTemplateMatcher()
        target_regions = template_matcher.target_regions
        mode_suffix = "_normal"
    
    # Add mode info to filename
    base, ext = os.path.splitext(output_path)
    output_path = f"{base}{mode_suffix}{ext}"
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Header with mode info
        writer.writerow(["# Analysis Mode: " + ("Pentagone (4-region)" if is_pentagone_mode else "Normal (5-region)")])
        writer.writerow([
            "anatomical_id", "region_name", "centroid_y", "centroid_x", 
            "area", "solidity", "eccentricity", "major_axis_length", 
            "minor_axis_length", "orientation"
        ])
        
        sorted_regions = sorted(valid_regions, key=lambda r: r.label)
        
        for region in sorted_regions:
            anatomical_id = region.label
            
            if anatomical_id in target_regions:
                region_name = target_regions[anatomical_id]['name']
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
    
    logger.info(f"Saved anatomical region information ({mode_suffix[1:]}) to {output_path}")


# Modified main processing function
def enhanced_intervein_processing_with_pentagone(prob_map, raw_img, cfg, output_dir, basename, 
                                               force_pentagone=False):
    """Complete automated intervein processing with pentagone support."""
    
    logger.info("Starting enhanced automated processing with pentagone support...")
    
    # Step 1: Detect veins
    vein_detector = ImprovedWingVeinDetector(cfg)
    vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    
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


def main_with_pentagone_support(directory: str, cfg: TrichomeDetectionConfig = CONFIG, 
                               output_directory: Optional[str] = None, 
                               force_pentagone_mode: bool = False,
                               auto_detect_pentagone: bool = True,
                               progress_callback=None):
    """Main function with pentagone mutant support and GUI progress callbacks."""
    
    def log_progress(message, level="INFO"):
        """Helper function to log progress"""
        logger.info(message)
        if progress_callback:
            try:
                progress_callback.put(("log", f"{level}: {message}"))
            except:
                pass  # Ignore callback errors
    
    # Validate configuration
    cfg.validate()
    
    if output_directory is None:
        output_directory = directory
    
    # Ensure output directory exists
    Path(output_directory).mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    log_file = os.path.join(output_directory, "trichome_detection_pentagone.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    mode_str = "FORCED PENTAGONE" if force_pentagone_mode else "AUTO-DETECT PENTAGONE" if auto_detect_pentagone else "NORMAL"
    
    log_progress("="*50)
    log_progress(f"Starting AUTOMATED TRICHOME DETECTION with PENTAGONE SUPPORT ({mode_str})")
    log_progress(f"Input directory: {directory}")
    log_progress(f"Output directory: {output_directory}")
    log_progress("="*50)
    
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
        
        log_progress("="*30)
        log_progress(f"Processing: {basename} ({i+1}/{total_files})")
        log_progress("="*30)
        
        try:
            prob_file = files["probabilities"]
            raw_file = files.get("raw")
            
            # Load data
            tri_prob, inter_prob, full_prob = load_probability_map(prob_file)
            raw_img = load_raw_image(raw_file) if raw_file else None
            
            # Enhanced peak detection
            peaks, metrics = detect_trichome_peaks(tri_prob, cfg)
            total_peaks += len(peaks)
            
            #log_progress(metrics.report())s
            
            # Save enhanced results
            save_peak_coordinates_enhanced(peaks, basename, output_directory, metrics, tri_prob)
            create_detection_visualization(tri_prob, peaks, basename, output_directory, raw_img)
            
            # Process intervein regions with pentagone support
            if full_prob.shape[-1] >= 4 and inter_prob is not None:
                log_progress("Processing intervein segmentation with pentagone support...")
                
                # Use force_pentagone_mode if specified
                result = enhanced_intervein_processing_with_trichome_wing(
                    full_prob, raw_img, cfg, output_directory, basename, 
                    force_pentagone=force_pentagone_mode
                )
                if result[0] is None:
                    log_progress(f"Skipped {basename} - wing touching border or invalid", "WARNING")
                    skipped_files += 1
                    continue
                
                labeled_mask, valid_regions, vein_mask, skeleton, is_pentagone_mode = result
                
                # Track mode statistics
                if is_pentagone_mode:
                    pentagone_wings += 1
                else:
                    normal_wings += 1
                
                total_regions += len(valid_regions)
                log_progress(f"Found {len(valid_regions)} valid regions in {'pentagone' if is_pentagone_mode else 'normal'} mode")
                
                # Assign peaks to regions
                region_peaks_dict = assign_peaks_to_regions(peaks, labeled_mask, valid_regions)
                
                # Voronoi analysis with mode-aware saving
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
                
                # Save results with mode information
                save_voronoi_results_with_mode(voronoi_results, basename, output_directory, is_pentagone_mode)
                save_anatomical_region_info_with_pentagone(valid_regions, basename, output_directory, is_pentagone_mode)
                
                # Enhanced visualization
                if valid_regions and any(region_peaks_dict[label].shape[0] >= 2 
                                       for label in region_peaks_dict):
                    background_img = raw_img if raw_img is not None else tri_prob
                    plot_voronoi_with_pentagone_info(
                        valid_regions, region_peaks_dict, background_img, 
                        vein_mask, skeleton, output_directory, basename, cfg, is_pentagone_mode
                    )
                
                # Print summary with mode info
                mode_str = "pentagone (4-region)" if is_pentagone_mode else "normal (5-region)"
                log_progress(f"Analysis summary for {basename} ({mode_str}):")
                for region_label, stats in voronoi_results.items():
                    if stats is not None:
                        log_progress(f"  Region {region_label} ({stats['region_name']}): "
                                  f"{stats['n_cells_measured']} cells, "
                                  f"avg area {stats['average_cell_area']:.1f} px², "
                                  f"CV {stats['cv_cell_area']:.3f}")
                    else:
                        # Get region name based on mode
                        if is_pentagone_mode:
                            pentagone_handler = PentagoneMutantHandler()
                            region_name = pentagone_handler.pentagone_target_regions.get(
                                region_label, {}).get('name', f'Region_{region_label}')
                        else:
                            template_matcher = AnatomicalTemplateMatcher()
                            region_name = template_matcher.target_regions.get(
                                region_label, {}).get('name', f'Region_{region_label}')
                        log_progress(f"  Region {region_label} ({region_name}): Analysis failed", "WARNING")
                
                log_progress(f"Successfully analyzed {successful_regions}/{len(valid_regions)} regions")
            else:
                log_progress("Skipping intervein analysis (4th channel missing)")
            
            processed_files += 1
            log_progress(f"Successfully processed {basename}")
            
        except Exception as e:
            failed_files += 1
            log_progress(f"Error processing {basename}: {str(e)}", "ERROR")
            continue
    
    # Final summary with pentagone statistics
    log_progress("="*50)
    log_progress("PENTAGONE-AWARE ANALYSIS COMPLETE")
    log_progress("="*50)
    log_progress(f"Total files found: {total_files}")
    log_progress(f"Successfully processed: {processed_files}")
    log_progress(f"  - Normal wings: {normal_wings}")
    log_progress(f"  - Pentagone wings: {pentagone_wings}")
    log_progress(f"Skipped (border/invalid): {skipped_files}")
    log_progress(f"Failed: {failed_files}")
    log_progress(f"Total trichomes detected: {total_peaks}")
    log_progress(f"Total regions analyzed: {total_regions}")
    log_progress(f"Results saved to: {output_directory}")
    log_progress("="*50)
    
    # Create enhanced summary report
    summary_path = os.path.join(output_directory, "pentagone_analysis_summary.txt")
    with open(summary_path, "w") as f:
        f.write("TRICHOME DETECTION WITH PENTAGONE SUPPORT - ANALYSIS SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input Directory: {directory}\n")
        f.write(f"Output Directory: {output_directory}\n")
        f.write(f"Mode: {mode_str}\n")
        f.write("\nPROCESSING SUMMARY:\n")
        f.write(f"Total files found: {total_files}\n")
        f.write(f"Successfully processed: {processed_files}\n")
        f.write(f"  - Normal wings (5 regions): {normal_wings}\n")
        f.write(f"  - Pentagone wings (4 regions): {pentagone_wings}\n")
        f.write(f"Skipped (border/invalid): {skipped_files}\n")
        f.write(f"Failed: {failed_files}\n")
        f.write("\nDETECTION SUMMARY:\n")
        f.write(f"Total trichomes detected: {total_peaks}\n")
        f.write(f"Total regions analyzed: {total_regions}\n")
        f.write(f"Average trichomes per wing: {total_peaks/max(processed_files, 1):.1f}\n")
        f.write(f"Average regions per wing: {total_regions/max(processed_files, 1):.1f}\n")
        f.write("\nPENTAGONE DETECTION:\n")
        f.write(f"Pentagone detection rate: {pentagone_wings/max(processed_files, 1)*100:.1f}%\n")
        f.write(f"Normal wing detection rate: {normal_wings/max(processed_files, 1)*100:.1f}%\n")
    
    log_progress(f"Enhanced summary report saved to {summary_path}")
    
    # Return summary statistics for GUI
    return {
        'total_files': total_files,
        'processed_files': processed_files,
        'failed_files': failed_files,
        'skipped_files': skipped_files,
        'normal_wings': normal_wings,
        'pentagone_wings': pentagone_wings,
        'total_peaks': total_peaks,
        'total_regions': total_regions
    }
def save_voronoi_results_with_mode(voronoi_results, basename, output_dir, is_pentagone_mode):
    """Save Voronoi results with mode information."""
    mode_suffix = "_pentagone" if is_pentagone_mode else "_normal"
    output_path = os.path.join(output_dir, f"{basename}_voronoi_average_cell_area{mode_suffix}.csv")
    
    with open(output_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        
        # Enhanced header with mode info
        writer.writerow(["# Analysis Mode: " + ("Pentagone (4-region combined)" if is_pentagone_mode else "Normal (5-region)")])
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
                # Get appropriate region name based on mode
                if is_pentagone_mode:
                    pentagone_handler = PentagoneMutantHandler()
                    region_name = pentagone_handler.pentagone_target_regions.get(
                        region_label, {}).get('name', f'Region_{region_label}')
                else:
                    template_matcher = AnatomicalTemplateMatcher()
                    region_name = template_matcher.target_regions.get(
                        region_label, {}).get('name', f'Region_{region_label}')
                
                writer.writerow([region_label, region_name, None, None, None, None, None, None, None])
    
    logger.info(f"Saved Voronoi results ({mode_suffix[1:]}) to {output_path}")


def plot_voronoi_with_pentagone_info(valid_regions, region_peaks_dict, background_img, 
                                    vein_mask, skeleton, output_directory, basename, cfg, 
                                    is_pentagone_mode):
    """Enhanced Voronoi visualization with pentagone mode information."""
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 20))
    
    # Choose appropriate template matcher based on mode
    if is_pentagone_mode:
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

{"Special Note: Region 4 combines normal regions 4+5" if is_pentagone_mode else ""}

Trichome Analysis:
- Total trichomes: {sum(len(peaks) for peaks in region_peaks_dict.values())}
- Cells kept for analysis: {total_kept}
- Average trichomes/region: {sum(len(peaks) for peaks in region_peaks_dict.values())/len(valid_regions):.1f}

Quality Metrics:
- Avg region area: {np.mean([r.area for r in valid_regions]):.0f} px²
- Avg region solidity: {np.mean([r.solidity for r in valid_regions]):.3f}

Mode: {"Pentagone mutant (automated detection)" if is_pentagone_mode else "Normal wing"}
"""
    
    axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
                    verticalalignment='top', fontfamily='monospace', fontsize=10)
    axes[1, 1].set_title(f"Analysis Summary - {mode_title}")
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    mode_suffix = "_pentagone" if is_pentagone_mode else "_normal"
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
            prob_support = intervein_prob > 0.2
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
            intervein_prob = prob_map[..., 1]
            wing_mask = intervein_prob < 0.4
            cv.morphologyEx(wing_mask, cv.MORPH_CLOSE, (10,10))
            # More conservative threshold
            wing_mask = intervein_prob < 0.4
        
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
    
    def create_virtual_boundary_veins(self, wing_mask, actual_vein_mask, vein_width=10):
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
                vein_prob = self._intensity_based_vein_detection(raw_img)
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

def load_probability_map(h5_path: str) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """Load Ilastik probability map with enhanced error handling."""
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
        closed_intervein = morphology.binary_closing(
            intervein_prob > self.config.intervein_threshold,
            morphology.disk(3)
        )
        
        # Step 3: Fill holes that are surrounded by intervein tissue
        filled_intervein = morphology.remove_small_holes(closed_intervein, area_threshold=500)
        
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


# Modified main processing function that uses improved detection
def enhanced_intervein_processing_with_improved_detection(prob_map, raw_img, cfg, output_dir, basename, 
                                                        force_pentagone=False):
    """Complete processing with improved small region detection."""
    
    logger.info("Starting enhanced processing with improved small region detection...")
    
    # Step 1: Detect veins
    vein_detector = ImprovedWingVeinDetector(cfg)
    vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    
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
                
                labeled_mask, valid_regions, vein_mask, skeleton, is_pentagone_mode = result
                
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
    config.auto_intervein_min_area = 5000  # Reduced from 8000
    config.min_intervein_solidity = 0.5    # Reduced from 0.6
    config.intervein_threshold = 0.35      # Reduced from 0.4
    
    # Adjust vein detection to be less aggressive
    config.vein_width_estimate = 10       # Reduced from 10
    
    logger.info("Applied configuration adjustments for small region detection")
    return config


# Example usage function with improved detection
def run_improved_detection_example():
    """Example of how to run the improved detection."""
    
    # Configure paths
    input_directory = r"/path/to/your/wings"
    output_directory = r"/path/to/output"
    
    # Adjust configuration for small regions
    improved_config = configure_for_small_regions(CONFIG)
    
    # Run improved detection
    main_with_improved_detection(
        directory=input_directory,
        cfg=improved_config,
        output_directory=output_directory,
        force_pentagone_mode=False  # Let it auto-detect
    )


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
    
    # Replace the method
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
        for peak in peaks:
            r, c = peak
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
    
    # Top-left: Original probability map
    axes[0, 0].imshow(prob_map, cmap='viridis')
    axes[0, 0].set_title('Trichome Probability Map')
    axes[0, 0].axis('off')
    
    # Top-right: Detected peaks overlay
    axes[0, 1].imshow(bg_img, cmap='gray')
    if peaks.size > 0:
        axes[0, 1].scatter(peaks[:, 1], peaks[:, 0], c='red', s=20, alpha=0.7)
    axes[0, 1].set_title(f'Detected Peaks (n={len(peaks)})')
    axes[0, 1].axis('off')
    
    # Bottom-left: Peak intensity histogram
    if peaks.size > 0:
        intensities = prob_map[peaks[:, 0], peaks[:, 1]]
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
    if peaks.size > 0:
        # Create density map
        from scipy.stats import gaussian_kde
        if len(peaks) > 1:
            try:
                kde = gaussian_kde(peaks.T)
                y, x = np.mgrid[0:prob_map.shape[0]:50j, 0:prob_map.shape[1]:50j]
                positions = np.vstack([x.ravel(), y.ravel()])
                density = kde(positions).reshape(x.shape)
                
                contour = axes[1, 1].contour(x, y, density, levels=5, alpha=0.7, cmap='Reds')
                axes[1, 1].clabel(contour, inline=True, fontsize=8)
            except:
                pass  # Skip density plot if it fails
        
        axes[1, 1].scatter(peaks[:, 1], peaks[:, 0], c='red', s=10, alpha=0.8)
    
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
    
    for peak in peaks:
        r, c = peak
        # First, try direct mask lookup
        region_label = labeled_mask[r, c]
        if region_label != 0 and region_label in region_peaks_dict:
            region_peaks_dict[region_label].append(peak)
        else:
            # Fall back to polygon containment check
            assigned = False
            for label_val, path in region_polygons.items():
                if path is not None and path.contains_point((c, r)):
                    region_peaks_dict[label_val].append(peak)
                    assigned = True
                    break
            if not assigned:
                unassigned_peaks.append(peak)
    
    if unassigned_peaks:
        logger.warning(f"{len(unassigned_peaks)} peaks could not be assigned to regions")
    
    # Convert lists to arrays
    for label_val in region_peaks_dict:
        region_peaks_dict[label_val] = np.array(region_peaks_dict[label_val])
    
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
    """Windows-optimized file finder with improved compatibility."""
    mapping = {}
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return mapping
    
    logger.info(f"Scanning directory: {directory}")
    
    # Use more robust file matching for Windows
    prob_files = []
    
    try:
        for file_path in directory_path.iterdir():
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            filename_lower = filename.lower()
            
            # Check for probability files with case-insensitive matching
            prob_patterns = [
                'probabilities.h5', 'probabilities.hdf5',
                'prob.h5', 'prob.hdf5',
                '.h5', '.hdf5'
            ]
            
            is_prob_file = False
            for pattern in prob_patterns:
                if filename_lower.endswith(pattern.lower()):
                    is_prob_file = True
                    break
            
            if is_prob_file:
                prob_files.append(file_path)
                
    except PermissionError as e:
        logger.error(f"Permission denied accessing directory: {e}")
        return mapping
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")
        return mapping
    
    logger.info(f"Found {len(prob_files)} potential probability files")
    
    for prob_path in prob_files:
        try:
            filename = prob_path.name
            filename_lower = filename.lower()
            
            # Extract basename more robustly
            basename = filename
            
            # Remove common suffixes
            suffixes_to_remove = [
                "_probabilities.h5", "_probabilities.hdf5",
                "_prob.h5", "_prob.hdf5",
                "_probabilities.H5", "_probabilities.HDF5",
                "_prob.H5", "_prob.HDF5",
                ".h5", ".hdf5", ".H5", ".HDF5"
            ]
            
            for suffix in suffixes_to_remove:
                if filename_lower.endswith(suffix.lower()):
                    # Use original case for extraction
                    basename = filename[:-len(suffix)]
                    break
            
            # Clean up basename
            basename = basename.strip('_').strip()
            
            if not basename:
                logger.warning(f"Could not extract basename from {filename}")
                continue
            
            mapping[basename] = {'probabilities': str(prob_path)}
            
            # Look for raw files
            raw_file = find_raw_file(directory_path, basename)
            mapping[basename]['raw'] = raw_file
            
            if raw_file:
                logger.info(f"✓ Found pair: {basename}")
            else:
                logger.info(f"✓ Found: {basename} (prob only)")
                
        except Exception as e:
            logger.error(f"Error processing {prob_path.name}: {e}")
            continue
    
    logger.info(f"Successfully mapped {len(mapping)} file sets")
    return mapping


def find_raw_file(directory_path, basename):
    """Find corresponding raw file for a basename."""
    raw_extensions = ['.tiff', '.tif', '.png', '.jpg', '.jpeg', '.h5', '.hdf5']
    raw_prefixes = [f"{basename}_raw", f"{basename}_Raw", f"{basename}"]
    
    for prefix in raw_prefixes:
        for ext in raw_extensions:
            # Try both cases
            for ext_variant in [ext.lower(), ext.upper()]:
                candidate = directory_path / f"{prefix}{ext_variant}"
                if candidate.exists() and candidate.is_file():
                    return str(candidate)
    
    return None

# def find_associated_files(directory):
#     """Locate probability and raw image files with better pattern matching."""
#     mapping = {}
#     directory_path = Path(directory)
    
#     # More flexible pattern matching
#     prob_patterns = ["*_Probabilities.h5", "*_probabilities.h5", "*_prob.h5","*_.h5"]
#     prob_files = []
#     for pattern in prob_patterns:
#         prob_files.extend(directory_path.glob(pattern))
    
#     for prob_path in prob_files:
#         filename = prob_path.name
#         # Extract basename more robustly
#         for suffix in ["_Probabilities.h5", "_probabilities.h5", "_prob.h5"]:
#             if filename.endswith(suffix):
#                 basename = filename.replace(suffix, "")
#                 break
#         else:
#             continue
            
#         mapping[basename] = {'probabilities': str(prob_path)}
        
#         # Look for corresponding raw files
#         raw_patterns = [
#             f"{basename}_Raw.h5", f"{basename}_raw.h5", f"{basename}_Raw.tiff",
#             f"{basename}_Raw.tif", f"{basename}_raw.tiff", f"{basename}_raw.tif",
#             f"{basename}_Raw.png", f"{basename}_raw.png",
#             f"{basename}_Raw.jpg", f"{basename}_raw.jpg",
#             f"{basename}.tiff", f"{basename}.tif", f"{basename}.png", f"{basename}.jpg"
#         ]
        
#         raw_candidates = []
#         for pattern in raw_patterns:
#             candidates = list(directory_path.glob(pattern))
#             raw_candidates.extend([str(c) for c in candidates])
        
#         mapping[basename]['raw'] = raw_candidates[0] if raw_candidates else None
        
#         if mapping[basename]['raw']:
#             logger.info(f"Found pair: {basename} (prob + raw)")
#         else:
#             logger.info(f"Found: {basename} (prob only)")
    
#     return mapping

def enhanced_intervein_processing(prob_map, raw_img, cfg, output_dir, basename):
    """Complete automated intervein processing pipeline with improved boundary handling."""
    
    logger.info("Starting enhanced automated vein and intervein detection...")
    
    # Step 1: Detect veins using the improved detector
    vein_detector = ImprovedWingVeinDetector(cfg)
    vein_mask, skeleton, _ = vein_detector.detect_veins_multi_approach(prob_map, raw_img)
    
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
    vein_detector.visualize_detection_results(raw_img, vein_mask, skeleton, 
                                            intervein_regions, orig_vis_path)
    
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

if __name__ == "__main__":
    # Ensure Windows compatibility
    ensure_windows_compatibility()

    # Check if GUI should be launched
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--dapi-pipeline":
        from dapi_nuclei_pipeline import run_dapi_pipeline_cli

        run_dapi_pipeline_cli(sys.argv[2:])
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "--cli":
        # Command line mode for backwards compatibility
        root = tk.Tk()
        root.withdraw()
        
        # Configure paths - UPDATE THESE FOR YOUR SYSTEM
        input_directory = r"/Users/rfa/Desktop/PhD/Organised Data/High Resolution Adult Wings/Data_Dir/New Set/110137_SxlRNai_25_Female"
        output_directory = input_directory  # Or set a different output directory
        
        if not input_directory or not os.path.exists(input_directory):
            print("Invalid input directory. Please check the path.")
        else:
            logger.info("Starting fully automated trichome detection with template-based labeling...")
            main_with_pentagone_support(
                directory=input_directory,
                output_directory=output_directory,
                force_pentagone_mode=False,      # Let algorithm decide
                auto_detect_pentagone=True       # Enable auto-detection
            )
    else:
        # Launch GUI by default
        run_gui()