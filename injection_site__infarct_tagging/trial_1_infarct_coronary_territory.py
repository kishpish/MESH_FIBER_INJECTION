#!/usr/bin/env python3
"""
ANATOMICALLY-CORRECT INFARCT ASSIGNMENT
Based on AHA 17-Segment Model and Coronary Artery Territories

This script assigns infarct regions that follow realistic coronary artery 
distributions.

CORONARY ARTERY TERRITORIES:

LAD (Left Anterior Descending):
  - Anterior wall
  - Anterior septum  
  - Apex
  - ~40-50% of LV mass
  - Most common infarct territory

RCA (Right Coronary Artery):
  - Inferior wall
  - Inferior septum (proximal)
  - ~30-40% of LV mass

LCx (Left Circumflex):
  - Lateral wall
  - Posterior wall
  - ~15-25% of LV mass

AHA 17-SEGMENT MODEL:

Basal (segments 1-6):
  1. Basal anterior
  2. Basal anteroseptal
  3. Basal inferoseptal
  4. Basal inferior
  5. Basal inferolateral
  6. Basal anterolateral

Mid (segments 7-12):
  7. Mid anterior
  8. Mid anteroseptal
  9. Mid inferoseptal
  10. Mid inferior
  11. Mid inferolateral
  12. Mid anterolateral

Apical (segments 13-16):
  13. Apical anterior
  14. Apical septal
  15. Apical inferior
  16. Apical lateral

Apex (segment 17):
  17. Apex

INFARCT PATTERNS:
- LAD: Segments 1, 2, 7, 8, 13, 14, 17 (anterior + septal + apex)
- RCA: Segments 3, 4, 9, 10, 15 (inferior + inferoseptal)
- LCx: Segments 5, 6, 11, 12, 16 (lateral)

"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
import os
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
PATIENT_IDS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]

BASE_DIR = "/home/shadeform/SCD_MODELS"
OUTPUT_DIR = "/home/shadeform/SCD_MODELS/infarct_coronary_territory"

# Tissue tags
TAG_HEALTHY = 1
TAG_BORDER = 2
TAG_INFARCT = 3

# AHA 17-SEGMENT MODEL DEFINITION
class AHA17Segments:
    """
    AHA 17-segment model for left ventricle.
    
    Circumferential angles (viewed from apex, RV on left):
    - 0° = Anterior (12 o'clock)
    - 60° = Anteroseptal
    - 120° = Inferoseptal
    - 180° = Inferior (6 o'clock)
    - 240° = Inferolateral
    - 300° = Anterolateral
    
    Longitudinal levels:
    - Basal: 0-33% from base
    - Mid: 33-67% from base
    - Apical: 67-95% from base
    - Apex: 95-100% (tip)
    """
    
    # Segment names
    NAMES = {
        1: "Basal anterior",
        2: "Basal anteroseptal",
        3: "Basal inferoseptal",
        4: "Basal inferior",
        5: "Basal inferolateral",
        6: "Basal anterolateral",
        7: "Mid anterior",
        8: "Mid anteroseptal",
        9: "Mid inferoseptal",
        10: "Mid inferior",
        11: "Mid inferolateral",
        12: "Mid anterolateral",
        13: "Apical anterior",
        14: "Apical septal",
        15: "Apical inferior",
        16: "Apical lateral",
        17: "Apex"
    }
    
    # Coronary artery territories (typical distribution)
    # Note: There's anatomical variation, but this is the standard mapping
    LAD_SEGMENTS = [1, 2, 7, 8, 13, 14, 17]  # Anterior + anteroseptal + apex
    RCA_SEGMENTS = [3, 4, 9, 10, 15]          # Inferior + inferoseptal
    LCX_SEGMENTS = [5, 6, 11, 12, 16]         # Lateral
    
    # Extended LAD (when LAD wraps around apex) - common variant
    LAD_EXTENDED = [1, 2, 7, 8, 13, 14, 15, 17]
    
    # Circumferential angle boundaries (degrees, from anterior = 0°)
    # For basal and mid levels (6 segments each)
    CIRC_BOUNDARIES_6 = [0, 60, 120, 180, 240, 300, 360]
    
    # For apical level (4 segments)
    CIRC_BOUNDARIES_4 = [0, 90, 180, 270, 360]
    
    # Longitudinal boundaries (fraction from base)
    LONG_BOUNDARIES = {
        'basal': (0.0, 0.33),
        'mid': (0.33, 0.67),
        'apical': (0.67, 0.95),
        'apex': (0.95, 1.0)
    }


# INFARCT PATTERNS (Realistic distributions)
class InfarctPatterns:
    """
    Predefined infarct patterns based on clinical presentations.
    Each pattern specifies which AHA segments are affected.
    """
    
    PATTERNS = {
        # LAD territory infarcts (most common, ~40-50% of MIs)
        'LAD_anterior': {
            'name': 'LAD - Anterior MI',
            'core_segments': [1, 7, 13],  # Anterior wall
            'border_segments': [2, 6, 8, 12, 14, 16],
            'description': 'Proximal LAD occlusion - anterior wall'
        },
        'LAD_anteroseptal': {
            'name': 'LAD - Anteroseptal MI',
            'core_segments': [2, 8, 14],  # Anteroseptal
            'border_segments': [1, 3, 7, 9, 13, 15, 17],
            'description': 'LAD with septal branches - anteroseptal'
        },
        'LAD_extensive': {
            'name': 'LAD - Extensive Anterior MI',
            'core_segments': [1, 2, 7, 8, 13, 14, 17],  # Full LAD territory
            'border_segments': [3, 6, 9, 12, 15, 16],
            'description': 'Proximal LAD - extensive anterior + apex'
        },
        'LAD_apical': {
            'name': 'LAD - Apical MI',
            'core_segments': [13, 14, 17],  # Apical
            'border_segments': [7, 8, 15, 16],
            'description': 'Distal LAD - apical'
        },
        
        # RCA territory infarcts (~30-40% of MIs)
        'RCA_inferior': {
            'name': 'RCA - Inferior MI',
            'core_segments': [4, 10, 15],  # Inferior wall
            'border_segments': [3, 5, 9, 11, 14, 16],
            'description': 'RCA occlusion - inferior wall'
        },
        'RCA_inferoseptal': {
            'name': 'RCA - Inferoseptal MI',
            'core_segments': [3, 4, 9, 10],  # Inferior + inferoseptal
            'border_segments': [2, 5, 8, 11, 14, 15],
            'description': 'Proximal RCA - inferoseptal'
        },
        
        # LCx territory infarcts (~15-20% of MIs)
        'LCX_lateral': {
            'name': 'LCx - Lateral MI',
            'core_segments': [5, 11, 16],  # Lateral wall
            'border_segments': [4, 6, 10, 12, 15],
            'description': 'LCx occlusion - lateral wall'
        },
        'LCX_posterolateral': {
            'name': 'LCx - Posterolateral MI',
            'core_segments': [5, 6, 11, 12, 16],  # Lateral + anterolateral
            'border_segments': [1, 4, 7, 10, 13, 15],
            'description': 'Proximal LCx - posterolateral'
        },
    }
    
    # Probability weights for random selection (based on clinical frequency)
    PATTERN_WEIGHTS = {
        'LAD_anterior': 0.15,
        'LAD_anteroseptal': 0.15,
        'LAD_extensive': 0.10,
        'LAD_apical': 0.10,
        'RCA_inferior': 0.20,
        'RCA_inferoseptal': 0.10,
        'LCX_lateral': 0.12,
        'LCX_posterolateral': 0.08,
    }


# MESH LOADING
def load_mesh(patient_id, base_dir):
    """Load tetrahedral mesh"""
    pts_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.pts"
    elem_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.elem"
    
    with open(pts_file, 'r') as f:
        n_nodes = int(f.readline().strip())
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            coords[i] = [float(x) for x in f.readline().split()[:3]]
    
    with open(elem_file, 'r') as f:
        n_elems = int(f.readline().strip())
        elements = np.zeros((n_elems, 4), dtype=np.int32)
        for i in range(n_elems):
            parts = f.readline().split()
            elements[i] = [int(x) for x in parts[1:5]]
    
    return coords, elements


def load_fibers(patient_id, base_dir):
    """Load fiber orientations"""
    lon_file = f"{base_dir}/fibers/{patient_id}/{patient_id}.lon"
    
    with open(lon_file, 'r') as f:
        _ = f.readline()
        lines = f.readlines()
        fibers = np.zeros((len(lines), 3), dtype=np.float64)
        for i, line in enumerate(lines):
            fibers[i] = [float(x) for x in line.split()[:3]]
    
    norms = np.linalg.norm(fibers, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return fibers / norms


# ANATOMICAL COORDINATE SYSTEM
def compute_anatomical_coordinates(coords, elements):
    """
    Compute anatomical coordinate system for the LV.
    
    Returns:
    - long_axis: Unit vector from apex to base
    - apex_point: Coordinates of apex
    - base_center: Center of base
    - centroids: Element centroids
    - longitudinal: Fractional position along long axis (0=base, 1=apex)
    - circumferential: Angle around long axis (0=anterior, 180=inferior)
    """
    n_elems = len(elements)
    centroids = np.mean(coords[elements], axis=1)
    
    # Find long axis using PCA
    center = np.mean(centroids, axis=0)
    centered = centroids - center
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Long axis = direction of maximum variance
    long_axis = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Project all points onto long axis
    projections = centered @ long_axis
    
    # Apex = most negative projection, Base = most positive
    # (Convention: apex is at bottom/negative end)
    if np.mean(projections[projections < np.median(projections)]) > 0:
        long_axis = -long_axis
        projections = -projections
    
    # Normalize longitudinal coordinate: 0 = base, 1 = apex
    proj_min, proj_max = projections.min(), projections.max()
    longitudinal = (projections - proj_min) / (proj_max - proj_min)
    
    # Find apex and base points
    apex_idx = np.argmax(longitudinal)
    base_indices = np.where(longitudinal < 0.1)[0]
    
    apex_point = centroids[apex_idx]
    base_center = np.mean(centroids[base_indices], axis=0)
    
    # Compute circumferential angle
    # First, establish a reference direction (perpendicular to long axis)
    # We'll use the direction to the RV (typically +x in standard orientation)
    
    # Find a perpendicular reference direction
    arbitrary = np.array([1, 0, 0])
    if abs(np.dot(long_axis, arbitrary)) > 0.9:
        arbitrary = np.array([0, 1, 0])
    
    ref_dir = arbitrary - np.dot(arbitrary, long_axis) * long_axis
    ref_dir = ref_dir / np.linalg.norm(ref_dir)
    
    # Second perpendicular direction
    perp_dir = np.cross(long_axis, ref_dir)
    
    # Compute circumferential angle for each element
    circumferential = np.zeros(n_elems)
    
    for i in range(n_elems):
        # Vector from axis to centroid (perpendicular to long axis)
        to_centroid = centered[i] - projections[i] * long_axis
        
        if np.linalg.norm(to_centroid) < 1e-10:
            circumferential[i] = 0
            continue
        
        to_centroid = to_centroid / np.linalg.norm(to_centroid)
        
        # Angle from reference direction
        cos_angle = np.dot(to_centroid, ref_dir)
        sin_angle = np.dot(to_centroid, perp_dir)
        
        angle = np.arctan2(sin_angle, cos_angle)
        circumferential[i] = np.degrees(angle)
        
        # Normalize to [0, 360)
        if circumferential[i] < 0:
            circumferential[i] += 360
    
    return {
        'long_axis': long_axis,
        'apex_point': apex_point,
        'base_center': base_center,
        'centroids': centroids,
        'longitudinal': longitudinal,  # 0=base, 1=apex
        'circumferential': circumferential,  # 0-360 degrees
        'ref_dir': ref_dir,
        'perp_dir': perp_dir
    }


def assign_aha_segments(anat_coords, n_elems):
    """
    Assign each element to an AHA segment based on anatomical coordinates.
    
    Circumferential mapping (viewed from apex):
    - Anterior: 330-30° (segment x.1)
    - Anteroseptal: 30-90° (segment x.2)
    - Inferoseptal: 90-150° (segment x.3)
    - Inferior: 150-210° (segment x.4)
    - Inferolateral: 210-270° (segment x.5)
    - Anterolateral: 270-330° (segment x.6)
    
    For apical level (4 segments):
    - Anterior: 315-45° (segment 13)
    - Septal: 45-135° (segment 14)
    - Inferior: 135-225° (segment 15)
    - Lateral: 225-315° (segment 16)
    """
    longitudinal = anat_coords['longitudinal']
    circumferential = anat_coords['circumferential']
    
    segments = np.zeros(n_elems, dtype=np.int32)
    
    for i in range(n_elems):
        long_pos = longitudinal[i]  # 0=base, 1=apex
        circ_pos = circumferential[i]  # 0-360 degrees
        
        # Determine longitudinal level
        if long_pos < 0.33:
            level = 'basal'
            base_segment = 0
        elif long_pos < 0.67:
            level = 'mid'
            base_segment = 6
        elif long_pos < 0.95:
            level = 'apical'
            base_segment = 12
        else:
            # Apex (segment 17)
            segments[i] = 17
            continue
        
        # Determine circumferential position
        if level in ['basal', 'mid']:
            # 6 segments per level
            # Shift angle so anterior (segment 1) is centered at 0°
            shifted_angle = (circ_pos + 30) % 360
            
            if shifted_angle < 60:
                circ_segment = 1  # Anterior
            elif shifted_angle < 120:
                circ_segment = 2  # Anteroseptal
            elif shifted_angle < 180:
                circ_segment = 3  # Inferoseptal
            elif shifted_angle < 240:
                circ_segment = 4  # Inferior
            elif shifted_angle < 300:
                circ_segment = 5  # Inferolateral
            else:
                circ_segment = 6  # Anterolateral
                
        else:  # apical level - 4 segments
            shifted_angle = (circ_pos + 45) % 360
            
            if shifted_angle < 90:
                circ_segment = 1  # Anterior (13)
            elif shifted_angle < 180:
                circ_segment = 2  # Septal (14)
            elif shifted_angle < 270:
                circ_segment = 3  # Inferior (15)
            else:
                circ_segment = 4  # Lateral (16)
        
        segments[i] = base_segment + circ_segment
    
    return segments


# INFARCT ASSIGNMENT
def select_infarct_pattern(patient_id, seed=None):
    """
    Select an infarct pattern for a patient.
    Uses patient ID as seed for reproducibility but varied results.
    """
    if seed is None:
        # Use patient ID to generate consistent but varied patterns
        seed = int(''.join(filter(str.isdigit, patient_id))) % 1000
    
    np.random.seed(seed)
    
    patterns = list(InfarctPatterns.PATTERNS.keys())
    weights = [InfarctPatterns.PATTERN_WEIGHTS[p] for p in patterns]
    weights = np.array(weights) / sum(weights)
    
    selected = np.random.choice(patterns, p=weights)
    
    return selected, InfarctPatterns.PATTERNS[selected]


def compute_segment_distances(segments, anat_coords, target_segments):
    """
    Compute distance from each element to the nearest target segment.
    Used for creating smooth transitions at borders.
    """
    n_elems = len(segments)
    centroids = anat_coords['centroids']
    
    # Find elements in target segments
    target_mask = np.isin(segments, target_segments)
    target_centroids = centroids[target_mask]
    
    if len(target_centroids) == 0:
        return np.ones(n_elems) * 1000  # No targets
    
    # Build KD-tree for fast nearest neighbor
    tree = cKDTree(target_centroids)
    
    # Query distance for all elements
    distances, _ = tree.query(centroids)
    
    return distances


def assign_infarct_territory(segments, anat_coords, pattern_info, elements,
                             infarct_fraction=0.08, border_fraction=0.12):
    """
    Assign infarct and border zones based on coronary territory pattern.
    
    Parameters:
    - segments: AHA segment assignment for each element
    - anat_coords: Anatomical coordinate system
    - pattern_info: Dictionary with core_segments and border_segments
    - elements: Mesh element connectivity (for adjacency)
    - infarct_fraction: Target fraction of LV for dense scar (default 8%)
    - border_fraction: Target fraction for border zone (default 12%)
    
    Returns:
    - classification: Array with 1=healthy, 2=border, 3=infarct
    """
    n_elems = len(segments)
    longitudinal = anat_coords['longitudinal']
    centroids = anat_coords['centroids']
    
    core_segments = pattern_info['core_segments']
    border_segments = pattern_info['border_segments']
    
    # Initialize all as healthy
    classification = np.ones(n_elems, dtype=np.int32)
    
    # Step 1: Mark core infarct segments
    core_mask = np.isin(segments, core_segments)
    
    # Step 2: Add transmural gradient (infarct more likely subendocardially)
    # We don't have exact endo/epi info, so use a probability gradient
    # Elements closer to apex in core segments are more likely to be infarct
    
    # Compute distance to infarct center for probability weighting
    core_centroids = centroids[core_mask]
    if len(core_centroids) > 0:
        infarct_center = np.mean(core_centroids, axis=0)
        distances_to_center = np.linalg.norm(centroids - infarct_center, axis=1)
        max_dist = np.percentile(distances_to_center[core_mask], 95)
        
        # Probability decreases with distance from center
        prob_infarct = np.clip(1 - distances_to_center / (max_dist * 1.5), 0, 1)
    else:
        prob_infarct = np.zeros(n_elems)
    
    # Step 3: Assign infarct based on segment + probability
    # Elements in core segments with high probability become infarct
    
    # Calculate how many elements we need for target fraction
    target_infarct_count = int(n_elems * infarct_fraction)
    target_border_count = int(n_elems * border_fraction)
    
    # Score for infarct: high if in core segment and high probability
    infarct_score = np.zeros(n_elems)
    infarct_score[core_mask] = prob_infarct[core_mask] + 1.0  # Boost for core segments
    
    # Add slight randomness for natural appearance
    np.random.seed(42)
    infarct_score += np.random.uniform(0, 0.3, n_elems)
    
    # Select top elements as infarct
    infarct_threshold = np.percentile(infarct_score, 100 * (1 - infarct_fraction))
    infarct_mask = infarct_score >= infarct_threshold
    
    # Ensure infarcts are primarily in core segments
    infarct_mask = infarct_mask & (core_mask | np.isin(segments, border_segments))
    
    classification[infarct_mask] = TAG_INFARCT
    
    # Step 4: Create border zone around infarct
    # Border = adjacent to infarct OR in border segments near infarct
    
    # Build adjacency
    adjacency = build_adjacency(elements)
    
    # Find elements adjacent to infarct
    adjacent_to_infarct = set()
    for i in np.where(infarct_mask)[0]:
        for j in adjacency.get(i, []):
            if not infarct_mask[j]:
                adjacent_to_infarct.add(j)
    
    # Also include border segments that are near the infarct
    border_segment_mask = np.isin(segments, border_segments)
    
    # Distance from each element to nearest infarct element
    infarct_centroids = centroids[infarct_mask]
    if len(infarct_centroids) > 0:
        tree = cKDTree(infarct_centroids)
        dist_to_infarct, _ = tree.query(centroids)
    else:
        dist_to_infarct = np.ones(n_elems) * 1000
    
    # Border zone: adjacent to infarct OR (in border segment AND close to infarct)
    median_dist = np.median(dist_to_infarct[list(adjacent_to_infarct)]) if adjacent_to_infarct else 10
    close_threshold = median_dist * 3
    
    border_mask = np.zeros(n_elems, dtype=bool)
    border_mask[list(adjacent_to_infarct)] = True
    border_mask |= (border_segment_mask & (dist_to_infarct < close_threshold))
    border_mask &= ~infarct_mask  # Don't overwrite infarct
    
    # Expand border slightly using adjacency
    for _ in range(2):  # 2 layers
        new_border = set()
        for i in np.where(border_mask)[0]:
            for j in adjacency.get(i, []):
                if not infarct_mask[j] and not border_mask[j]:
                    if dist_to_infarct[j] < close_threshold * 1.5:
                        new_border.add(j)
        border_mask[list(new_border)] = True
    
    classification[border_mask] = TAG_BORDER
    
    return classification


def build_adjacency(elements):
    """Build element adjacency via shared faces"""
    face_to_elem = defaultdict(list)
    for i, nodes in enumerate(elements):
        for face in [
            tuple(sorted([nodes[0], nodes[1], nodes[2]])),
            tuple(sorted([nodes[0], nodes[1], nodes[3]])),
            tuple(sorted([nodes[0], nodes[2], nodes[3]])),
            tuple(sorted([nodes[1], nodes[2], nodes[3]]))
        ]:
            face_to_elem[face].append(i)
    
    adjacency = defaultdict(list)
    for face, elems in face_to_elem.items():
        if len(elems) == 2:
            adjacency[elems[0]].append(elems[1])
            adjacency[elems[1]].append(elems[0])
    return adjacency


# SMOOTHING AND REFINEMENT
def smooth_classification(classification, adjacency, iterations=3):
    """
    Smooth classification boundaries to remove isolated elements.
    Uses majority voting in local neighborhood.
    """
    n_elems = len(classification)
    
    for _ in range(iterations):
        new_classification = classification.copy()
        
        for i in range(n_elems):
            neighbors = adjacency.get(i, [])
            if len(neighbors) < 2:
                continue
            
            # Count neighbor types
            neighbor_types = [classification[j] for j in neighbors]
            neighbor_types.append(classification[i])  # Include self
            
            # Majority vote
            counts = {TAG_HEALTHY: 0, TAG_BORDER: 0, TAG_INFARCT: 0}
            for t in neighbor_types:
                counts[t] += 1
            
            # Only change if strong majority
            max_count = max(counts.values())
            if max_count >= len(neighbor_types) * 0.6:
                majority_type = max(counts, key=counts.get)
                
                # Don't change infarct to healthy directly
                if classification[i] == TAG_INFARCT and majority_type == TAG_HEALTHY:
                    new_classification[i] = TAG_BORDER
                # Don't change healthy to infarct directly
                elif classification[i] == TAG_HEALTHY and majority_type == TAG_INFARCT:
                    new_classification[i] = TAG_BORDER
                else:
                    new_classification[i] = majority_type
        
        classification = new_classification
    
    return classification


def ensure_border_continuity(classification, adjacency):
    """
    Ensure border zone forms a continuous layer around infarct.
    Any healthy element adjacent to infarct becomes border.
    """
    n_elems = len(classification)
    
    for i in range(n_elems):
        if classification[i] == TAG_HEALTHY:
            # Check if adjacent to infarct
            for j in adjacency.get(i, []):
                if classification[j] == TAG_INFARCT:
                    classification[i] = TAG_BORDER
                    break
    
    return classification


# OUTPUT FUNCTIONS
def write_vtk(filepath, coords, elements, classification, segments, anat_coords):
    """Save VTK with classification and anatomical data"""
    n_nodes, n_elems = len(coords), len(elements)
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Coronary Territory Infarct Classification\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {n_nodes} float\n")
        for c in coords:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        f.write(f"\nCELLS {n_elems} {n_elems * 5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"\nCELL_TYPES {n_elems}\n")
        f.write("10\n" * n_elems)
        
        # Cell data
        f.write(f"\nCELL_DATA {n_elems}\n")
        
        f.write("SCALARS TissueType int 1\nLOOKUP_TABLE default\n")
        for c in classification:
            f.write(f"{c}\n")
        
        f.write("\nSCALARS AHA_Segment int 1\nLOOKUP_TABLE default\n")
        for s in segments:
            f.write(f"{s}\n")
        
        f.write("\nSCALARS Longitudinal float 1\nLOOKUP_TABLE default\n")
        for l in anat_coords['longitudinal']:
            f.write(f"{l:.6f}\n")
        
        f.write("\nSCALARS Circumferential float 1\nLOOKUP_TABLE default\n")
        for c in anat_coords['circumferential']:
            f.write(f"{c:.6f}\n")


def write_region_vtk(filepath, coords, elements, classification, region_code):
    """Save VTK for single region"""
    mask = classification == region_code
    region_elems = np.where(mask)[0]
    
    if len(region_elems) == 0:
        return 0
    
    region_elements = elements[region_elems]
    unique_nodes = np.unique(region_elements.flatten())
    node_map = {old: new for new, old in enumerate(unique_nodes)}
    remapped = np.array([[node_map[n] for n in elem] for elem in region_elements])
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nRegion\nASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(unique_nodes)} float\n")
        for n in unique_nodes:
            c = coords[n]
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        f.write(f"\nCELLS {len(region_elems)} {len(region_elems) * 5}\n")
        for e in remapped:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"\nCELL_TYPES {len(region_elems)}\n")
        f.write("10\n" * len(region_elems))
    
    return len(region_elems)


def write_tagged_elem(filepath, elements, classification):
    """Save OpenCARP format"""
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n")
        for i, e in enumerate(elements):
            f.write(f"Tt {e[0]} {e[1]} {e[2]} {e[3]} {classification[i]}\n")


# MAIN PIPELINE
def process_patient(patient_id, pattern_override=None):
    """
    Process a single patient with coronary territory-based infarct assignment.
    """
    print(f"PROCESSING: {patient_id}")
    
    # Load mesh
    print("\n  Loading mesh...")
    coords, elements = load_mesh(patient_id, BASE_DIR)
    n_elems = len(elements)
    print(f"      {len(coords):,} nodes, {n_elems:,} elements")
    
    # Make elements globally available for adjacency building
    global elements_global
    elements_global = elements
    
    # Compute anatomical coordinates
    print("\n  Computing anatomical coordinate system...")
    anat_coords = compute_anatomical_coordinates(coords, elements)
    print(f"      Long axis: [{anat_coords['long_axis'][0]:.3f}, "
          f"{anat_coords['long_axis'][1]:.3f}, {anat_coords['long_axis'][2]:.3f}]")
    
    # Assign AHA segments
    print("\n  Assigning AHA 17-segment model...")
    segments = assign_aha_segments(anat_coords, n_elems)
    
    segment_counts = {}
    for s in range(1, 18):
        count = np.sum(segments == s)
        if count > 0:
            segment_counts[s] = count
    print(f"      Segments assigned: {len(segment_counts)} of 17")
    
    # Select infarct pattern
    print("\n  Selecting infarct pattern...")
    if pattern_override:
        pattern_name = pattern_override
        pattern_info = InfarctPatterns.PATTERNS[pattern_name]
    else:
        pattern_name, pattern_info = select_infarct_pattern(patient_id)
    
    print(f"      Pattern: {pattern_info['name']}")
    print(f"      {pattern_info['description']}")
    print(f"      Core segments: {pattern_info['core_segments']}")
    print(f"      Border segments: {pattern_info['border_segments']}")
    
    # Assign infarct territory
    print("\n  Assigning infarct territory...")
    
    # Build adjacency for this mesh
    adjacency = build_adjacency(elements)
    
    # Vary infarct size slightly per patient (6-12%)
    np.random.seed(int(''.join(filter(str.isdigit, patient_id))) % 1000 + 1)
    infarct_fraction = np.random.uniform(0.06, 0.12)
    border_fraction = np.random.uniform(0.10, 0.18)
    
    classification = assign_infarct_territory(
        segments, anat_coords, pattern_info, elements,
        infarct_fraction=infarct_fraction,
        border_fraction=border_fraction
    )
    
    # Smooth and refine
    print("\n  Smoothing boundaries...")
    classification = smooth_classification(classification, adjacency)
    classification = ensure_border_continuity(classification, adjacency)
    
    # Statistics
    n_healthy = np.sum(classification == TAG_HEALTHY)
    n_border = np.sum(classification == TAG_BORDER)
    n_infarct = np.sum(classification == TAG_INFARCT)
    
    print(f"\n  FINAL CLASSIFICATION:")
    print(f"      Healthy: {n_healthy:,} ({100*n_healthy/n_elems:.1f}%)")
    print(f"      Border:  {n_border:,} ({100*n_border/n_elems:.1f}%)")
    print(f"      Infarct: {n_infarct:,} ({100*n_infarct/n_elems:.1f}%)")
    
    # Save outputs
    print("\n  Saving outputs...")
    patient_output = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    write_vtk(
        os.path.join(patient_output, f"{patient_id}_classified.vtk"),
        coords, elements, classification, segments, anat_coords
    )
    
    write_region_vtk(
        os.path.join(patient_output, f"{patient_id}_INFARCT.vtk"),
        coords, elements, classification, TAG_INFARCT
    )
    
    write_region_vtk(
        os.path.join(patient_output, f"{patient_id}_BORDER.vtk"),
        coords, elements, classification, TAG_BORDER
    )
    
    write_tagged_elem(
        os.path.join(patient_output, f"{patient_id}_tagged.elem"),
        elements, classification
    )
    
    # Summary JSON
    summary = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'method': 'Coronary Territory-Based Infarct Assignment',
        'pattern': {
            'name': pattern_info['name'],
            'description': pattern_info['description'],
            'core_segments': pattern_info['core_segments'],
            'border_segments': pattern_info['border_segments']
        },
        'statistics': {
            'n_elements': int(n_elems),
            'n_healthy': int(n_healthy),
            'n_border': int(n_border),
            'n_infarct': int(n_infarct),
            'pct_healthy': round(100*n_healthy/n_elems, 2),
            'pct_border': round(100*n_border/n_elems, 2),
            'pct_infarct': round(100*n_infarct/n_elems, 2),
        },
        'aha_segments': {str(k): int(v) for k, v in segment_counts.items()}
    }
    
    with open(os.path.join(patient_output, f"{patient_id}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n  Outputs saved to: {patient_output}")
    
    return summary


def main():
    """Main entry point"""
    print("CORONARY TERRITORY-BASED INFARCT ASSIGNMENT")
    
    print("\nMETHOD:")
    print("  - AHA 17-segment model for anatomical localization")
    print("  - Coronary artery territory mapping (LAD, RCA, LCx)")
    print("  - Realistic infarct patterns based on clinical presentations")
    print("  - Smooth transitions with proper border zones")
    
    print("\nPATTERN DISTRIBUTION:")
    for name, prob in InfarctPatterns.PATTERN_WEIGHTS.items():
        pattern = InfarctPatterns.PATTERNS[name]
        print(f"  {prob*100:4.0f}% - {pattern['name']}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    all_results = []
    
    # Process each patient
    for patient_id in PATIENT_IDS:
        try:
            result = process_patient(patient_id)
            all_results.append(result)
        except Exception as e:
            print(f"\n  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'patient_id': patient_id,
                'status': 'FAILED',
                'error': str(e)
            })
    
    # Summary table
    print("SUMMARY")
    
    print(f"\n{'Patient':<15} {'Pattern':<30} {'Healthy%':>10} {'Border%':>10} {'Infarct%':>10}")
    
    for result in all_results:
        if 'status' not in result:
            stats = result['statistics']
            pattern = result['pattern']['name'][:28]
            print(f"{result['patient_id']:<15} {pattern:<30} "
                  f"{stats['pct_healthy']:>10.1f} {stats['pct_border']:>10.1f} "
                  f"{stats['pct_infarct']:>10.1f}")
    
    # Combined summary
    successful = [r for r in all_results if 'status' not in r]
    if successful:
        avg_healthy = np.mean([r['statistics']['pct_healthy'] for r in successful])
        avg_border = np.mean([r['statistics']['pct_border'] for r in successful])
        avg_infarct = np.mean([r['statistics']['pct_infarct'] for r in successful])
        std_infarct = np.std([r['statistics']['pct_infarct'] for r in successful])
        
        print(f"{'AVERAGE':<15} {'':<30} {avg_healthy:>10.1f} {avg_border:>10.1f} {avg_infarct:>10.1f}")
        print(f"{'STD DEV':<15} {'':<30} {'':<10} {'':<10} {std_infarct:>10.1f}")
    
    # Save combined summary
    with open(os.path.join(OUTPUT_DIR, "all_patients_summary.json"), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'method': 'Coronary Territory-Based Infarct Assignment',
            'patients': all_results
        }, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()