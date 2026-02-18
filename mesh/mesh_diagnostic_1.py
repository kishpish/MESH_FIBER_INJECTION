#!/usr/bin/env python3
"""
MESH QUALITY ANALYSIS

This script analyzes ALL quality metrics for tetrahedral meshes and 
identifies exactly what's preventing FEBio/OpenCarp readiness.

Metrics analyzed:
1. Aspect Ratio (AR)
2. Scaled Jacobian (J)
3. Dihedral Angles (min/max)
4. Radius-Edge Ratio (ρ)
5. Edge Length Ratio
6. Volume Distribution
7. Inverted Elements

Output:
- Detailed per-patient CSV
- Histogram data
- Identification of problematic elements
"""

import numpy as np
from pathlib import Path
import csv
from datetime import datetime
from typing import Dict, Tuple, List
from collections import defaultdict

# CONFIGURATION
MESH_DIR = "/home/nvidia/SCD_MODELS/repaired_meshes"
OUTPUT_DIR = "/home/nvidia/SCD_MODELS/quality_analysis"

ALL_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", 
    "SCD0001101", "SCD0001201"
]

# Quality thresholds
THRESHOLDS = {
    'febio': {
        'max_aspect_ratio': 50,
        'min_jacobian': 0.01,
        'min_dihedral': 10,      # degrees
        'max_dihedral': 170,     # degrees
        'max_radius_edge': 10,
    },
    'opencarp': {
        'max_aspect_ratio': 100,
        'min_jacobian': 0.001,
        'min_dihedral': 1,
        'max_dihedral': 179,
        'max_radius_edge': 50,
    },
    'ideal': {
        'max_aspect_ratio': 10,
        'min_jacobian': 0.3,
        'min_dihedral': 30,
        'max_dihedral': 140,
        'max_radius_edge': 2,
    }
}

# MESH I/O
def read_carp_mesh(mesh_dir: str, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CARP format mesh."""
    pts_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.pts"
    elem_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.elem"
    
    if not pts_path.exists():
        raise FileNotFoundError(f"Mesh not found: {pts_path}")
    
    with open(pts_path) as f:
        lines = f.readlines()
    n_verts = int(lines[0].strip())
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    for i, line in enumerate(lines[1:n_verts+1]):
        vertices[i] = [float(x) for x in line.split()]
    
    with open(elem_path) as f:
        lines = f.readlines()
    n_elems = int(lines[0].strip())
    elements = np.zeros((n_elems, 4), dtype=np.int64)
    tags = np.zeros(n_elems, dtype=np.int32)
    for i, line in enumerate(lines[1:n_elems+1]):
        parts = line.split()
        elements[i] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
        tags[i] = int(parts[5]) if len(parts) > 5 else 1
    
    return vertices, elements, tags

# QUALITY METRICS (COMPREHENSIVE)
def compute_all_quality_metrics(vertices: np.ndarray, elements: np.ndarray) -> Dict:
    """
    Compute ALL quality metrics for comprehensive analysis.
    """
    n = len(elements)
    
    # Initialize arrays
    jacobians = np.zeros(n)
    aspect_ratios = np.zeros(n)
    min_dihedrals = np.zeros(n)
    max_dihedrals = np.zeros(n)
    radius_edge_ratios = np.zeros(n)
    volumes = np.zeros(n)
    edge_ratios = np.zeros(n)
    min_edges = np.zeros(n)
    max_edges = np.zeros(n)
    
    for i, e in enumerate(elements):
        v0, v1, v2, v3 = vertices[e[0]], vertices[e[1]], vertices[e[2]], vertices[e[3]]
        
        # Edge vectors
        e01, e02, e03 = v1 - v0, v2 - v0, v3 - v0
        
        # Volume (signed)
        vol = np.dot(e01, np.cross(e02, e03)) / 6.0
        volumes[i] = vol
        
        # All 6 edges with their lengths
        edges = [
            (v1 - v0), (v2 - v0), (v3 - v0),
            (v2 - v1), (v3 - v1), (v3 - v2)
        ]
        edge_lens = np.array([np.linalg.norm(e) for e in edges])
        
        l_max = np.max(edge_lens)
        l_min = max(np.min(edge_lens), 1e-15)
        l_rms = np.sqrt(np.mean(edge_lens**2))
        
        min_edges[i] = l_min
        max_edges[i] = l_max
        edge_ratios[i] = l_max / l_min
        
        # Scaled Jacobian
        jacobians[i] = 6 * np.sqrt(2) * vol / (l_rms**3) if l_rms > 1e-15 else 0
        
        # Face areas and normals for multiple metrics
        face_verts = [
            [v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]
        ]
        face_areas = []
        face_normals = []
        
        for fv in face_verts:
            cross = np.cross(fv[1] - fv[0], fv[2] - fv[0])
            area = 0.5 * np.linalg.norm(cross)
            face_areas.append(area)
            norm = np.linalg.norm(cross)
            if norm > 1e-15:
                face_normals.append(cross / norm)
            else:
                face_normals.append(np.array([0, 0, 1]))
        
        total_face_area = sum(face_areas)
        
        # Aspect ratio (inscribed sphere method)
        r_in = 3 * abs(vol) / total_face_area if total_face_area > 1e-15 else 1e-15
        aspect_ratios[i] = l_max / (2 * r_in) if r_in > 1e-15 else 1e10
        
        # Circumradius calculation for radius-edge ratio
        # Using the formula: R = abc / (4 * area) generalized for 3D
        try:
            # Compute circumradius using Cayley-Menger determinant
            # Simplified approximation
            if abs(vol) > 1e-20:
                # Use product of edges from one vertex divided by 6V
                R_approx = (edge_lens[0] * edge_lens[1] * edge_lens[2]) / (6 * abs(vol))
                radius_edge_ratios[i] = R_approx / l_min
            else:
                radius_edge_ratios[i] = 1e10
        except:
            radius_edge_ratios[i] = 1e10
        
        # Dihedral angles - computed for all 6 edges
        # Each edge is shared by exactly 2 faces
        # Face adjacency: faces 0,1 share edge v0-v1; faces 0,2 share v0-v2; etc.
        edge_face_pairs = [
            (0, 1),  # edge v0-v1 shared by faces 0,1
            (0, 2),  # edge v0-v2 shared by faces 0,2
            (1, 2),  # edge v0-v3 shared by faces 1,2
            (0, 3),  # edge v1-v2 shared by faces 0,3
            (1, 3),  # edge v1-v3 shared by faces 1,3
            (2, 3),  # edge v2-v3 shared by faces 2,3
        ]
        
        dihedrals = []
        for f1, f2 in edge_face_pairs:
            n1, n2 = face_normals[f1], face_normals[f2]
            dot = np.clip(np.dot(n1, n2), -1, 1)
            # Dihedral angle is the supplement of the angle between outward normals
            angle = np.degrees(np.arccos(-dot))  # Use -dot for interior angle
            dihedrals.append(angle)
        
        min_dihedrals[i] = min(dihedrals)
        max_dihedrals[i] = max(dihedrals)
    
    return {
        'jacobians': jacobians,
        'aspect_ratios': aspect_ratios,
        'min_dihedrals': min_dihedrals,
        'max_dihedrals': max_dihedrals,
        'radius_edge_ratios': radius_edge_ratios,
        'volumes': volumes,
        'edge_ratios': edge_ratios,
        'min_edges': min_edges,
        'max_edges': max_edges,
    }

def analyze_quality(quality: Dict, thresholds: Dict = THRESHOLDS) -> Dict:
    """
    Comprehensive quality analysis with threshold checking.
    """
    n = len(quality['jacobians'])
    
    analysis = {
        'n_elements': n,
        'n_inverted': int(np.sum(quality['volumes'] <= 0)),
        
        # Jacobian statistics
        'jacobian_min': float(np.min(quality['jacobians'])),
        'jacobian_max': float(np.max(quality['jacobians'])),
        'jacobian_mean': float(np.mean(quality['jacobians'])),
        'jacobian_std': float(np.std(quality['jacobians'])),
        'jacobian_p01': float(np.percentile(quality['jacobians'], 1)),
        'jacobian_p05': float(np.percentile(quality['jacobians'], 5)),
        'jacobian_p50': float(np.percentile(quality['jacobians'], 50)),
        
        # Aspect ratio statistics
        'aspect_ratio_min': float(np.min(quality['aspect_ratios'])),
        'aspect_ratio_max': float(np.max(quality['aspect_ratios'])),
        'aspect_ratio_mean': float(np.mean(quality['aspect_ratios'])),
        'aspect_ratio_std': float(np.std(quality['aspect_ratios'])),
        'aspect_ratio_p95': float(np.percentile(quality['aspect_ratios'], 95)),
        'aspect_ratio_p99': float(np.percentile(quality['aspect_ratios'], 99)),
        
        # Dihedral angle statistics
        'dihedral_min': float(np.min(quality['min_dihedrals'])),
        'dihedral_max': float(np.max(quality['max_dihedrals'])),
        'dihedral_min_mean': float(np.mean(quality['min_dihedrals'])),
        'dihedral_max_mean': float(np.mean(quality['max_dihedrals'])),
        'dihedral_min_p01': float(np.percentile(quality['min_dihedrals'], 1)),
        'dihedral_max_p99': float(np.percentile(quality['max_dihedrals'], 99)),
        
        # Radius-edge ratio statistics
        'radius_edge_min': float(np.min(quality['radius_edge_ratios'])),
        'radius_edge_max': float(np.max(quality['radius_edge_ratios'])),
        'radius_edge_mean': float(np.mean(quality['radius_edge_ratios'])),
        'radius_edge_p95': float(np.percentile(quality['radius_edge_ratios'], 95)),
        'radius_edge_p99': float(np.percentile(quality['radius_edge_ratios'], 99)),
        
        # Edge statistics
        'edge_length_min': float(np.min(quality['min_edges'])),
        'edge_length_max': float(np.max(quality['max_edges'])),
        'edge_ratio_max': float(np.max(quality['edge_ratios'])),
        'edge_ratio_mean': float(np.mean(quality['edge_ratios'])),
    }
    
    # Count elements failing each threshold
    for level, thresh in thresholds.items():
        bad_jacobian = quality['jacobians'] < thresh['min_jacobian']
        bad_aspect = quality['aspect_ratios'] > thresh['max_aspect_ratio']
        bad_dihedral_min = quality['min_dihedrals'] < thresh['min_dihedral']
        bad_dihedral_max = quality['max_dihedrals'] > thresh['max_dihedral']
        bad_radius = quality['radius_edge_ratios'] > thresh['max_radius_edge']
        
        analysis[f'{level}_bad_jacobian'] = int(np.sum(bad_jacobian))
        analysis[f'{level}_bad_aspect'] = int(np.sum(bad_aspect))
        analysis[f'{level}_bad_dihedral_min'] = int(np.sum(bad_dihedral_min))
        analysis[f'{level}_bad_dihedral_max'] = int(np.sum(bad_dihedral_max))
        analysis[f'{level}_bad_radius'] = int(np.sum(bad_radius))
        analysis[f'{level}_bad_total'] = int(np.sum(
            bad_jacobian | bad_aspect | bad_dihedral_min | bad_dihedral_max | bad_radius
        ))
        
        # Check if meets threshold
        analysis[f'{level}_ready'] = (
            analysis['n_inverted'] == 0 and
            analysis[f'{level}_bad_total'] == 0
        )
    
    return analysis

def print_detailed_analysis(patient_id: str, analysis: Dict):
    """Pretty print detailed analysis."""
    print(f"QUALITY ANALYSIS: {patient_id}")
    
    print(f"\n  Elements: {analysis['n_elements']:,}")
    print(f"  Inverted: {analysis['n_inverted']}")
    
    print(f"\n  SCALED JACOBIAN (target > 0.01 FEBio, > 0.001 OpenCarp)")
    print(f"    Min: {analysis['jacobian_min']:.6f}")
    print(f"    Max: {analysis['jacobian_max']:.4f}")
    print(f"    Mean: {analysis['jacobian_mean']:.4f}")
    print(f"    P1/P5: {analysis['jacobian_p01']:.6f} / {analysis['jacobian_p05']:.6f}")
    
    print(f"\n  ASPECT RATIO (target < 50 FEBio, < 100 OpenCarp)")
    print(f"    Min: {analysis['aspect_ratio_min']:.2f}")
    print(f"    Max: {analysis['aspect_ratio_max']:.2f}")
    print(f"    Mean: {analysis['aspect_ratio_mean']:.2f}")
    print(f"    P95/P99: {analysis['aspect_ratio_p95']:.2f} / {analysis['aspect_ratio_p99']:.2f}")
    
    print(f"\n  DIHEDRAL ANGLES (target 10°-170° FEBio, 1°-179° OpenCarp)")
    print(f"    Min angle: {analysis['dihedral_min']:.2f}°")
    print(f"    Max angle: {analysis['dihedral_max']:.2f}°")
    print(f"    Mean (min/max): {analysis['dihedral_min_mean']:.2f}° / {analysis['dihedral_max_mean']:.2f}°")
    print(f"    P1 min / P99 max: {analysis['dihedral_min_p01']:.2f}° / {analysis['dihedral_max_p99']:.2f}°")
    
    print(f"\n  RADIUS-EDGE RATIO (target < 10 FEBio, < 50 OpenCarp)")
    print(f"    Min: {analysis['radius_edge_min']:.2f}")
    print(f"    Max: {analysis['radius_edge_max']:.2f}")
    print(f"    Mean: {analysis['radius_edge_mean']:.2f}")
    print(f"    P95/P99: {analysis['radius_edge_p95']:.2f} / {analysis['radius_edge_p99']:.2f}")
    
    print(f"\n  EDGE LENGTHS")
    print(f"    Min edge: {analysis['edge_length_min']:.6f}")
    print(f"    Max edge: {analysis['edge_length_max']:.6f}")
    print(f"    Max ratio: {analysis['edge_ratio_max']:.2f}")
    
    print(f"\n  THRESHOLD COMPLIANCE:")
    for level in ['febio', 'opencarp', 'ideal']:
        ready = "check" if analysis[f'{level}_ready'] else "X"
        bad = analysis[f'{level}_bad_total']
        print(f"    {level.upper():10s}: {ready} ({bad:,} bad elements)")
        if bad > 0:
            print(f"      - Jacobian: {analysis[f'{level}_bad_jacobian']:,}")
            print(f"      - Aspect ratio: {analysis[f'{level}_bad_aspect']:,}")
            print(f"      - Dihedral min: {analysis[f'{level}_bad_dihedral_min']:,}")
            print(f"      - Dihedral max: {analysis[f'{level}_bad_dihedral_max']:,}")
            print(f"      - Radius-edge: {analysis[f'{level}_bad_radius']:,}")

def identify_problem_elements(quality: Dict, thresholds: Dict) -> Dict[str, np.ndarray]:
    """Identify indices of elements failing each criterion."""
    problems = {}
    
    problems['inverted'] = np.where(quality['volumes'] <= 0)[0]
    problems['bad_jacobian'] = np.where(quality['jacobians'] < thresholds['min_jacobian'])[0]
    problems['bad_aspect'] = np.where(quality['aspect_ratios'] > thresholds['max_aspect_ratio'])[0]
    problems['bad_dihedral_min'] = np.where(quality['min_dihedrals'] < thresholds['min_dihedral'])[0]
    problems['bad_dihedral_max'] = np.where(quality['max_dihedrals'] > thresholds['max_dihedral'])[0]
    problems['bad_radius'] = np.where(quality['radius_edge_ratios'] > thresholds['max_radius_edge'])[0]
    
    # All bad elements
    all_bad = set()
    for indices in problems.values():
        all_bad.update(indices)
    problems['all_bad'] = np.array(sorted(all_bad))
    
    return problems

def save_problem_elements(output_dir: str, patient_id: str, 
                          elements: np.ndarray, quality: Dict, problems: Dict):
    """Save indices and quality values of problematic elements."""
    out_path = Path(output_dir) / patient_id
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Save problem element indices
    for problem_type, indices in problems.items():
        if len(indices) > 0:
            np.savetxt(out_path / f"problem_{problem_type}.txt", indices, fmt='%d')
    
    # Save detailed quality for bad elements
    if len(problems['all_bad']) > 0:
        with open(out_path / "problem_elements_detail.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['element_idx', 'jacobian', 'aspect_ratio', 
                           'min_dihedral', 'max_dihedral', 'radius_edge', 'volume'])
            for idx in problems['all_bad']:
                writer.writerow([
                    idx,
                    quality['jacobians'][idx],
                    quality['aspect_ratios'][idx],
                    quality['min_dihedrals'][idx],
                    quality['max_dihedrals'][idx],
                    quality['radius_edge_ratios'][idx],
                    quality['volumes'][idx],
                ])

# MAIN
def analyze_all_meshes(patients: List[str] = ALL_PATIENTS,
                       mesh_dir: str = MESH_DIR,
                       output_dir: str = OUTPUT_DIR) -> List[Dict]:
    """Analyze quality of all patient meshes."""
    print("MESH QUALITY ANALYSIS")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    all_analyses = []
    
    for i, patient_id in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}] Processing {patient_id}...")
        
        try:
            # Load mesh
            vertices, elements, tags = read_carp_mesh(mesh_dir, patient_id)
            
            # Compute quality metrics
            quality = compute_all_quality_metrics(vertices, elements)
            
            # Analyze
            analysis = analyze_quality(quality)
            analysis['patient_id'] = patient_id
            
            # Print detailed results
            print_detailed_analysis(patient_id, analysis)
            
            # Identify and save problem elements
            problems = identify_problem_elements(quality, THRESHOLDS['febio'])
            save_problem_elements(output_dir, patient_id, elements, quality, problems)
            
            all_analyses.append(analysis)
            
        except Exception as e:
            print(f"  ERROR: {e}")
            all_analyses.append({'patient_id': patient_id, 'error': str(e)})
    
    # Save summary CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"quality_analysis_{timestamp}.csv"
    
    if all_analyses:
        # Get all keys
        all_keys = set()
        for a in all_analyses:
            all_keys.update(a.keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(all_analyses)
        print(f"\n\nSummary saved to: {csv_path}")
    
    # Print overall summary
    print("OVERALL SUMMARY")
    
    n_febio = sum(1 for a in all_analyses if a.get('febio_ready', False))
    n_opencarp = sum(1 for a in all_analyses if a.get('opencarp_ready', False))
    n_ideal = sum(1 for a in all_analyses if a.get('ideal_ready', False))
    
    print(f"\n  FEBio ready:   {n_febio}/{len(all_analyses)}")
    print(f"  OpenCarp ready: {n_opencarp}/{len(all_analyses)}")
    print(f"  Ideal quality:  {n_ideal}/{len(all_analyses)}")
    
    # Identify main bottlenecks
    print(f"\n  BOTTLENECK ANALYSIS (for FEBio compliance):")
    total_bad = defaultdict(int)
    for a in all_analyses:
        if 'error' not in a:
            total_bad['jacobian'] += a.get('febio_bad_jacobian', 0)
            total_bad['aspect'] += a.get('febio_bad_aspect', 0)
            total_bad['dihedral_min'] += a.get('febio_bad_dihedral_min', 0)
            total_bad['dihedral_max'] += a.get('febio_bad_dihedral_max', 0)
            total_bad['radius'] += a.get('febio_bad_radius', 0)
    
    for metric, count in sorted(total_bad.items(), key=lambda x: -x[1]):
        print(f"    {metric:15s}: {count:,} elements across all patients")
    
    return all_analyses

if __name__ == "__main__":
    try:
        get_ipython()
        IN_JUPYTER = True
    except NameError:
        IN_JUPYTER = False
    
    if IN_JUPYTER:
        print("Running in Jupyter mode...")
        
        MESH_DIR = "/home/nvidia/SCD_MODELS/high_resolution_meshes"
        OUTPUT_DIR = "/home/nvidia/SCD_MODELS/quality_analysis"
        
        PATIENTS_TO_ANALYZE = [
            "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
            "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", 
            "SCD0001101", "SCD0001201"
        ]
        
        results = analyze_all_meshes(PATIENTS_TO_ANALYZE, MESH_DIR, OUTPUT_DIR)
        
    else:
        import argparse
        
        parser = argparse.ArgumentParser(description='Comprehensive Mesh Quality Analysis')
        parser.add_argument('--mesh-dir', type=str, default=MESH_DIR)
        parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
        parser.add_argument('--patients', type=str, nargs='+', default=ALL_PATIENTS)
        
        args = parser.parse_args()
        analyze_all_meshes(args.patients, args.mesh_dir, args.output_dir)