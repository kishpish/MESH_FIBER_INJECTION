#!/usr/bin/env python3
"""
OPTIMAL TETRAHEDRAL MESH QUALITY REPAIR

Multi-stage repair pipeline using the best available tools:

Stage 1: Surface Extraction & Repair (PyMeshLab/PyMesh)
Stage 2: MMG3D Optimization (WITH vertex insertion allowed)
Stage 3: If still bad → fTetWild complete remesh
Stage 4: Final cleanup and validation

Key insight: The previous script used `-noinsert` which prevented MMG3D
from adding vertices to fix bad elements. This script allows full optimization.

Tools used:
- MMG3D: Mesh adaptation with Delaunay-based optimization
- fTetWild: Robust tetrahedral meshing (guaranteed no inversions)
- PyMesh: Edge collapse/split, degenerate removal
- TetGen: Quality optimization mode

"""

import numpy as np
from pathlib import Path
import os
import sys
import subprocess
import tempfile
import shutil
from typing import Dict, Tuple, List, Optional
from collections import defaultdict
import time
import csv
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# CONFIGURATION
MESH_DIR = "/home/nvidia/SCD_MODELS/high_resolution_meshes"
OUTPUT_DIR = "/home/nvidia/SCD_MODELS/repaired_meshes"

ALL_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", 
    "SCD0001101", "SCD0001201"
]

# Quality targets
TARGETS = {
    'max_aspect_ratio': 50,      # FEBio requirement
    'min_jacobian': 0.01,        # FEBio requirement  
    'min_dihedral': 10,          # degrees
    'max_dihedral': 170,         # degrees
    'max_radius_edge': 10,       # FEBio requirement
}

# Relaxed targets (OpenCarp is more tolerant)
TARGETS_RELAXED = {
    'max_aspect_ratio': 100,
    'min_jacobian': 0.001,
    'min_dihedral': 1,
    'max_dihedral': 179,
    'max_radius_edge': 20,
}

# QUALITY METRICS - DETAILED EXPLANATIONS
"""
QUALITY METRICS EXPLAINED:

1. ASPECT RATIO (AR)
   - Definition: Ratio of longest edge to shortest altitude
   - Formula: AR = L_max / h_min
   - For tetrahedra: AR = L_max / (3V / A_max) where A_max is largest face area
   - Ideal (regular tet): AR = √(8/3) ≈ 1.63
   - Problem: High AR = "sliver" or "needle" element
   - Effect: Causes ill-conditioned stiffness matrices, numerical instability
   - Target: < 50 (FEBio), < 100 (OpenCarp)

2. SCALED JACOBIAN (J)
   - Definition: Measures how distorted an element is from ideal shape
   - Formula: J = 6√2 × V / L_rms³
     where V = volume, L_rms = root-mean-square edge length
   - Range: [-1, 1]
     * J = 1: Perfect regular tetrahedron
     * J = 0: Degenerate (zero volume)
     * J < 0: Inverted element (negative volume)
   - Target: > 0.01 (FEBio), > 0.0001 (OpenCarp)

3. DIHEDRAL ANGLE (θ)
   - Definition: Angle between two adjacent faces
   - Formula: θ = 180° - arccos(n₁ · n₂)
     where n₁, n₂ are face normals
   - A tetrahedron has 6 dihedral angles (one per edge)
   - Ideal (regular tet): θ ≈ 70.53°
   - Problem angles:
     * θ < 10°: Nearly coplanar faces ("sliver cap")
     * θ > 170°: Nearly flat element ("sliver")
   - Target: 10° < θ < 170° (FEBio), 0.1° < θ < 179.9° (OpenCarp)

4. RADIUS-EDGE RATIO (ρ)
   - Definition: Ratio of circumradius to shortest edge
   - Formula: ρ = R / d_min
     where R = circumradius, d_min = shortest edge
   - Ideal (regular tet): ρ = √6/4 ≈ 0.612
   - Delaunay criterion: ρ < 2 guarantees good interpolation
   - Target: < 10 (FEBio), < 20 (OpenCarp)

5. VOLUME (V)
   - Definition: Signed volume of tetrahedron
   - Formula: V = (1/6) × det([v₁-v₀, v₂-v₀, v₃-v₀])
   - Must be positive (negative = inverted)
   - Used to detect flipped elements

6. EDGE LENGTH RATIO
   - Definition: max_edge / min_edge
   - Measures edge length uniformity
   - Ideal: 1.0 (all edges equal)
   - High ratio indicates anisotropic elements

WHY SLIVERS ARE BAD:
- Condition number of element stiffness matrix ∝ AR²
- For AR = 100,000, condition number ≈ 10^10
- This causes:
  * Loss of numerical precision
  * Slow/non-convergent solvers
  * Incorrect simulation results
  * CFL condition violations (explicit solvers)
"""

# TOOL DETECTION
def detect_tools() -> Dict[str, any]:
    """Detect all available mesh processing tools."""
    tools = {}
    
    # MMG3D
    for cmd in ['mmg3d_O3', 'mmg3d', 'mmg3d_debug']:
        try:
            result = subprocess.run([cmd, '-h'], capture_output=True, timeout=5)
            tools['mmg3d'] = cmd
            print(f"  ✓ MMG3D: {cmd}")
            break
        except:
            pass
    if 'mmg3d' not in tools:
        tools['mmg3d'] = None
        print("  ✗ MMG3D: not found")
    
    # fTetWild (via wildmeshing Python package)
    try:
        import wildmeshing as wm
        tools['ftetwild'] = True
        print(f"  ✓ fTetWild (wildmeshing): available")
    except ImportError:
        tools['ftetwild'] = False
        print("  ✗ fTetWild: not found (pip install wildmeshing)")
    
    # PyMesh
    try:
        import pymesh
        tools['pymesh'] = True
        print(f"  ✓ PyMesh: available")
    except ImportError:
        tools['pymesh'] = False
        print("  ✗ PyMesh: not found")
    
    # TetGen
    try:
        result = subprocess.run(['tetgen', '-h'], capture_output=True, timeout=5)
        tools['tetgen'] = True
        print(f"  ✓ TetGen: available")
    except:
        tools['tetgen'] = False
        print("  ✗ TetGen: not found")
    
    # Gmsh
    try:
        result = subprocess.run(['gmsh', '--version'], capture_output=True, timeout=5)
        tools['gmsh'] = True
        version = result.stdout.decode().strip() or result.stderr.decode().strip()
        print(f"  ✓ Gmsh: {version}")
    except:
        tools['gmsh'] = False
        print("  ✗ Gmsh: not found")
    
    # PyMeshLab
    try:
        import pymeshlab
        tools['pymeshlab'] = True
        print(f"  ✓ PyMeshLab: available")
    except ImportError:
        tools['pymeshlab'] = False
        print("  ✗ PyMeshLab: not found")
    
    return tools

# MESH I/O
def read_carp_mesh(mesh_dir: str, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CARP format mesh (.pts, .elem)."""
    pts_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.pts"
    elem_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.elem"
    
    if not pts_path.exists():
        raise FileNotFoundError(f"Mesh not found: {pts_path}")
    
    # Read vertices
    with open(pts_path) as f:
        lines = f.readlines()
    n_verts = int(lines[0].strip())
    vertices = np.zeros((n_verts, 3), dtype=np.float64)
    for i, line in enumerate(lines[1:n_verts+1]):
        vertices[i] = [float(x) for x in line.split()]
    
    # Read elements
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

def write_carp_mesh(output_dir: str, patient_id: str, vertices: np.ndarray,
                    elements: np.ndarray, tags: np.ndarray):
    """Write CARP format mesh."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    pts_path = Path(output_dir) / f"{patient_id}_tet.pts"
    with open(pts_path, 'w') as f:
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
    
    elem_path = Path(output_dir) / f"{patient_id}_tet.elem"
    with open(elem_path, 'w') as f:
        f.write(f"{len(elements)}\n")
        for i, e in enumerate(elements):
            f.write(f"Tt {e[0]} {e[1]} {e[2]} {e[3]} {tags[i]}\n")

def write_medit_mesh(filepath: str, vertices: np.ndarray, elements: np.ndarray,
                     tags: np.ndarray = None):
    """Write Medit format (.mesh) for MMG3D."""
    with open(filepath, 'w') as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")
        
        f.write(f"Vertices\n{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} 0\n")
        
        f.write(f"\nTetrahedra\n{len(elements)}\n")
        for i, e in enumerate(elements):
            tag = tags[i] if tags is not None else 1
            f.write(f"{e[0]+1} {e[1]+1} {e[2]+1} {e[3]+1} {tag}\n")
        
        f.write("\nEnd\n")

def read_medit_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Medit format (.mesh)."""
    vertices = []
    elements = []
    tags = []
    
    with open(filepath) as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        if line == "Vertices":
            n = int(lines[i+1].strip())
            for j in range(n):
                parts = lines[i+2+j].split()
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            i += n + 2
        elif line == "Tetrahedra":
            n = int(lines[i+1].strip())
            for j in range(n):
                parts = lines[i+2+j].split()
                elements.append([int(parts[0])-1, int(parts[1])-1,
                               int(parts[2])-1, int(parts[3])-1])
                tags.append(int(parts[4]) if len(parts) > 4 else 1)
            i += n + 2
        else:
            i += 1
    
    return np.array(vertices), np.array(elements), np.array(tags)

def extract_surface(vertices: np.ndarray, elements: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Extract boundary surface triangles from tet mesh."""
    face_count = defaultdict(list)
    
    for ei, e in enumerate(elements):
        faces = [
            tuple(sorted([e[0], e[1], e[2]])),
            tuple(sorted([e[0], e[1], e[3]])),
            tuple(sorted([e[0], e[2], e[3]])),
            tuple(sorted([e[1], e[2], e[3]])),
        ]
        for f in faces:
            face_count[f].append(ei)
    
    # Boundary faces appear only once
    boundary_faces = []
    for face, elem_list in face_count.items():
        if len(elem_list) == 1:
            boundary_faces.append(list(face))
    
    return vertices, np.array(boundary_faces, dtype=np.int64)

def write_stl(filepath: str, vertices: np.ndarray, faces: np.ndarray):
    """Write binary STL file."""
    with open(filepath, 'wb') as f:
        # Header (80 bytes)
        f.write(b'\x00' * 80)
        # Number of triangles
        f.write(np.uint32(len(faces)).tobytes())
        
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Compute normal
            e1 = v1 - v0
            e2 = v2 - v0
            normal = np.cross(e1, e2)
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal /= norm
            
            # Write normal
            f.write(np.float32(normal).tobytes())
            # Write vertices
            f.write(np.float32(v0).tobytes())
            f.write(np.float32(v1).tobytes())
            f.write(np.float32(v2).tobytes())
            # Attribute byte count
            f.write(np.uint16(0).tobytes())

# QUALITY METRICS
def compute_quality_metrics(vertices: np.ndarray, elements: np.ndarray) -> Dict:
    """Compute comprehensive quality metrics for each element."""
    n = len(elements)
    
    jacobians = np.zeros(n)
    aspect_ratios = np.zeros(n)
    min_dihedrals = np.zeros(n)
    max_dihedrals = np.zeros(n)
    radius_edge_ratios = np.zeros(n)
    volumes = np.zeros(n)
    edge_ratios = np.zeros(n)
    
    for i, e in enumerate(elements):
        v0, v1, v2, v3 = vertices[e[0]], vertices[e[1]], vertices[e[2]], vertices[e[3]]
        
        # Edge vectors from v0
        e01, e02, e03 = v1 - v0, v2 - v0, v3 - v0
        
        # Volume (signed)
        vol = np.dot(e01, np.cross(e02, e03)) / 6.0
        volumes[i] = vol
        
        # All 6 edges
        edges = [
            v1 - v0, v2 - v0, v3 - v0,  # from v0
            v2 - v1, v3 - v1,            # from v1
            v3 - v2                       # from v2
        ]
        edge_lens = np.array([np.linalg.norm(e) for e in edges])
        
        l_max = np.max(edge_lens)
        l_min = np.max([np.min(edge_lens), 1e-15])
        l_rms = np.sqrt(np.mean(edge_lens**2))
        
        # Edge length ratio
        edge_ratios[i] = l_max / l_min
        
        # Scaled Jacobian: J = 6√2 × V / L_rms³
        jacobians[i] = 6 * np.sqrt(2) * vol / (l_rms**3) if l_rms > 1e-15 else 0
        
        # Aspect ratio using inscribed sphere method
        # AR = L_max × (sum of face areas) / (3 × volume)
        face_areas = []
        face_verts = [
            [v0, v1, v2], [v0, v1, v3], [v0, v2, v3], [v1, v2, v3]
        ]
        for fv in face_verts:
            area = 0.5 * np.linalg.norm(np.cross(fv[1] - fv[0], fv[2] - fv[0]))
            face_areas.append(area)
        
        total_face_area = sum(face_areas)
        # Inscribed sphere radius: r = 3V / A_total
        r_in = 3 * abs(vol) / total_face_area if total_face_area > 1e-15 else 1e-15
        aspect_ratios[i] = l_max / (2 * r_in) if r_in > 1e-15 else 1e10
        
        # Circumradius for radius-edge ratio
        # Using formula: R = |e01||e02||e03| / (6V) for specific edge config
        # Simplified: use circumsphere formula
        try:
            # Cayley-Menger determinant approach (simplified)
            d01 = np.linalg.norm(v1 - v0)
            d02 = np.linalg.norm(v2 - v0)
            d03 = np.linalg.norm(v3 - v0)
            d12 = np.linalg.norm(v2 - v1)
            d13 = np.linalg.norm(v3 - v1)
            d23 = np.linalg.norm(v3 - v2)
            
            # Approximate circumradius
            # R ≈ (a × b × c) / (4 × area) for largest face, scaled for 3D
            max_face_area = max(face_areas)
            if max_face_area > 1e-15 and abs(vol) > 1e-20:
                # Product of three edges meeting at one vertex
                R_approx = (d01 * d02 * d03) / (6 * abs(vol))
                radius_edge_ratios[i] = R_approx / l_min if l_min > 1e-15 else 1e10
            else:
                radius_edge_ratios[i] = 1e10
        except:
            radius_edge_ratios[i] = 1e10
        
        # Dihedral angles
        face_normals = []
        for fv in face_verts:
            n = np.cross(fv[1] - fv[0], fv[2] - fv[0])
            norm = np.linalg.norm(n)
            if norm > 1e-15:
                face_normals.append(n / norm)
            else:
                face_normals.append(np.array([0, 0, 1]))
        
        dihedrals = []
        # 6 edges, each shared by 2 faces
        edge_face_pairs = [
            (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)
        ]
        for f1, f2 in edge_face_pairs:
            dot = np.clip(np.dot(face_normals[f1], face_normals[f2]), -1, 1)
            angle = 180 - np.degrees(np.arccos(dot))
            dihedrals.append(angle)
        
        min_dihedrals[i] = min(dihedrals) if dihedrals else 0
        max_dihedrals[i] = max(dihedrals) if dihedrals else 180
    
    return {
        'jacobians': jacobians,
        'aspect_ratios': aspect_ratios,
        'min_dihedrals': min_dihedrals,
        'max_dihedrals': max_dihedrals,
        'radius_edge_ratios': radius_edge_ratios,
        'volumes': volumes,
        'edge_ratios': edge_ratios,
    }

def summarize_quality(quality: Dict, targets: Dict = TARGETS) -> Dict:
    """Compute summary statistics."""
    n = len(quality['jacobians'])
    
    summary = {
        'n_elements': n,
        'n_inverted': int(np.sum(quality['volumes'] <= 0)),
        
        'min_jacobian': float(np.min(quality['jacobians'])),
        'mean_jacobian': float(np.mean(quality['jacobians'])),
        'max_jacobian': float(np.max(quality['jacobians'])),
        
        'max_aspect_ratio': float(np.max(quality['aspect_ratios'])),
        'mean_aspect_ratio': float(np.mean(quality['aspect_ratios'])),
        'p99_aspect_ratio': float(np.percentile(quality['aspect_ratios'], 99)),
        
        'min_dihedral': float(np.min(quality['min_dihedrals'])),
        'max_dihedral': float(np.max(quality['max_dihedrals'])),
        
        'max_radius_edge': float(np.max(quality['radius_edge_ratios'])),
        'mean_radius_edge': float(np.mean(quality['radius_edge_ratios'])),
    }
    
    # Count bad elements
    bad_jacobian = quality['jacobians'] < targets['min_jacobian']
    bad_aspect = quality['aspect_ratios'] > targets['max_aspect_ratio']
    bad_dihedral = (quality['min_dihedrals'] < targets['min_dihedral']) | \
                   (quality['max_dihedrals'] > targets['max_dihedral'])
    bad_radius = quality['radius_edge_ratios'] > targets['max_radius_edge']
    
    summary['n_bad_jacobian'] = int(np.sum(bad_jacobian))
    summary['n_bad_aspect'] = int(np.sum(bad_aspect))
    summary['n_bad_dihedral'] = int(np.sum(bad_dihedral))
    summary['n_bad_radius'] = int(np.sum(bad_radius))
    summary['n_bad_total'] = int(np.sum(bad_jacobian | bad_aspect | bad_dihedral | bad_radius))
    
    # Check if meets targets
    summary['meets_febio'] = (
        summary['n_inverted'] == 0 and
        summary['min_jacobian'] >= targets['min_jacobian'] and
        summary['max_aspect_ratio'] <= targets['max_aspect_ratio'] and
        summary['min_dihedral'] >= targets['min_dihedral'] and
        summary['max_dihedral'] <= targets['max_dihedral']
    )
    
    summary['meets_opencarp'] = (
        summary['n_inverted'] == 0 and
        summary['min_jacobian'] >= TARGETS_RELAXED['min_jacobian'] and
        summary['max_aspect_ratio'] <= TARGETS_RELAXED['max_aspect_ratio'] and
        summary['min_dihedral'] >= TARGETS_RELAXED['min_dihedral'] and
        summary['max_dihedral'] <= TARGETS_RELAXED['max_dihedral']
    )
    
    return summary

def print_quality(summary: Dict, label: str = ""):
    """Pretty print quality summary."""
    if label:
        print(f"  {label}:")
    print(f"    Elements: {summary['n_elements']:,} | Inverted: {summary['n_inverted']}")
    print(f"    Jacobian: min={summary['min_jacobian']:.6f}, mean={summary['mean_jacobian']:.4f}")
    print(f"    Aspect Ratio: max={summary['max_aspect_ratio']:.1f}, mean={summary['mean_aspect_ratio']:.2f}, p99={summary['p99_aspect_ratio']:.1f}")
    print(f"    Dihedral: min={summary['min_dihedral']:.2f}°, max={summary['max_dihedral']:.2f}°")
    print(f"    Bad elements: {summary['n_bad_total']} (J:{summary['n_bad_jacobian']}, AR:{summary['n_bad_aspect']}, θ:{summary['n_bad_dihedral']})")
    print(f"    FEBio ready: {summary['meets_febio']} | OpenCarp ready: {summary['meets_opencarp']}")

# REPAIR METHODS
def repair_with_mmg3d_full(vertices: np.ndarray, elements: np.ndarray, 
                           tags: np.ndarray, mmg3d_cmd: str,
                           hausd: float = 0.01, hgrad: float = 1.3,
                           hmin: float = None, hmax: float = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MMG3D optimization WITH vertex insertion allowed.
    
    This is the key difference from the previous script - we allow MMG3D
    to add/remove vertices for better quality.
    
    Parameters:
    - hausd: Hausdorff distance (surface accuracy)
    - hgrad: Gradation parameter (edge length ratio between neighbors)
    - hmin/hmax: Min/max edge lengths (auto-computed if None)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.mesh"
        output_path = f"{tmpdir}/output.o.mesh"
        
        write_medit_mesh(input_path, vertices, elements, tags)
        
        # Compute edge length bounds if not provided
        if hmin is None or hmax is None:
            all_edges = []
            for e in elements[:1000]:  # Sample
                v = vertices[e]
                for i in range(4):
                    for j in range(i+1, 4):
                        all_edges.append(np.linalg.norm(v[i] - v[j]))
            mean_edge = np.mean(all_edges)
            hmin = hmin or mean_edge * 0.1
            hmax = hmax or mean_edge * 2.0
        
        # MMG3D command - NOTE: NO -noinsert flag!
        cmd = [
            mmg3d_cmd,
            input_path,
            "-out", output_path,
            "-hausd", str(hausd),
            "-hgrad", str(hgrad),
            "-hmin", str(hmin),
            "-hmax", str(hmax),
            "-optim",           # Optimization mode
            "-v", "0",          # Quiet
        ]
        
        print(f"    MMG3D: {' '.join(cmd[:4])}...")
        result = subprocess.run(cmd, capture_output=True, timeout=1200)
        
        if result.returncode != 0:
            stderr = result.stderr.decode()[:500]
            print(f"    MMG3D warning: {stderr}")
        
        # Find output file
        if os.path.exists(output_path):
            return read_medit_mesh(output_path)
        
        alt_output = input_path.replace('.mesh', '.o.mesh')
        if os.path.exists(alt_output):
            return read_medit_mesh(alt_output)
        
        raise RuntimeError("MMG3D did not produce output")

def repair_with_ftetwild(vertices: np.ndarray, elements: np.ndarray,
                         tags: np.ndarray, target_edge_length: float = None,
                         epsilon: float = 1e-3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Complete remesh using fTetWild.
    
    fTetWild guarantees:
    - No inverted elements
    - Bounded dihedral angles
    - Controllable edge length
    """
    import wildmeshing as wm
    
    # Extract surface
    surf_verts, surf_faces = extract_surface(vertices, elements)
    
    # Compute target edge length if not provided
    if target_edge_length is None:
        # Use current mean edge length
        edge_lens = []
        for e in elements[:1000]:
            v = vertices[e]
            for i in range(4):
                for j in range(i+1, 4):
                    edge_lens.append(np.linalg.norm(v[i] - v[j]))
        target_edge_length = np.mean(edge_lens)
    
    # Bounding box diagonal for relative edge length
    bbox_diag = np.linalg.norm(np.max(vertices, axis=0) - np.min(vertices, axis=0))
    
    print(f"    fTetWild: edge_length={target_edge_length:.4f}, epsilon={epsilon}")
    
    tetrahedralizer = wm.Tetrahedralizer(
        stop_quality=20,  # Quality threshold
        epsilon=epsilon,
        edge_length_r=target_edge_length / bbox_diag,
    )
    
    tetrahedralizer.set_mesh(surf_verts, surf_faces)
    tetrahedralizer.tetrahedralize()
    
    new_verts, new_elems = tetrahedralizer.get_tet_mesh()
    new_tags = np.ones(len(new_elems), dtype=np.int32)
    
    return np.array(new_verts), np.array(new_elems), new_tags

def repair_with_tetgen(vertices: np.ndarray, elements: np.ndarray,
                       tags: np.ndarray, quality: float = 1.4) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    TetGen quality optimization.
    
    Uses TetGen's -O flag for mesh improvement without remeshing.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write .node file
        node_path = f"{tmpdir}/input.node"
        with open(node_path, 'w') as f:
            f.write(f"{len(vertices)} 3 0 0\n")
            for i, v in enumerate(vertices):
                f.write(f"{i+1} {v[0]} {v[1]} {v[2]}\n")
        
        # Write .ele file
        ele_path = f"{tmpdir}/input.ele"
        with open(ele_path, 'w') as f:
            f.write(f"{len(elements)} 4 0\n")
            for i, e in enumerate(elements):
                f.write(f"{i+1} {e[0]+1} {e[1]+1} {e[2]+1} {e[3]+1}\n")
        
        # Run TetGen with optimization
        cmd = ['tetgen', '-rqO', f'-q{quality}', f"{tmpdir}/input"]
        print(f"    TetGen: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        # Read output
        out_node = f"{tmpdir}/input.1.node"
        out_ele = f"{tmpdir}/input.1.ele"
        
        if not os.path.exists(out_node):
            raise RuntimeError("TetGen did not produce output")
        
        # Read vertices
        with open(out_node) as f:
            lines = f.readlines()
        n_verts = int(lines[0].split()[0])
        new_verts = np.zeros((n_verts, 3))
        for i, line in enumerate(lines[1:n_verts+1]):
            parts = line.split()
            new_verts[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
        
        # Read elements
        with open(out_ele) as f:
            lines = f.readlines()
        n_elems = int(lines[0].split()[0])
        new_elems = np.zeros((n_elems, 4), dtype=np.int64)
        for i, line in enumerate(lines[1:n_elems+1]):
            parts = line.split()
            new_elems[i] = [int(parts[1])-1, int(parts[2])-1, 
                          int(parts[3])-1, int(parts[4])-1]
        
        new_tags = np.ones(n_elems, dtype=np.int32)
        return new_verts, new_elems, new_tags

def repair_with_pymesh(vertices: np.ndarray, elements: np.ndarray,
                       tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PyMesh-based repair using edge collapse/split.
    """
    import pymesh
    
    # Create PyMesh mesh
    mesh = pymesh.form_mesh(vertices, np.zeros((0, 3), dtype=np.int32), elements)
    
    # Remove degenerate tets
    mesh, info = pymesh.remove_degenerated_triangles(mesh)
    
    # Remove isolated vertices
    mesh, info = pymesh.remove_isolated_vertices(mesh)
    
    new_verts = mesh.vertices
    new_elems = mesh.voxels if mesh.num_voxels > 0 else mesh.faces
    new_tags = np.ones(len(new_elems), dtype=np.int32)
    
    return new_verts, new_elems, new_tags

# MAIN REPAIR PIPELINE
def repair_mesh(patient_id: str, mesh_dir: str, output_dir: str,
                tools: Dict) -> Dict:
    """
    Multi-stage mesh repair pipeline.
    
    Strategy:
    1. Try MMG3D with full optimization (vertex insertion allowed)
    2. If still bad and fTetWild available, try complete remesh
    3. Fall back to TetGen optimization if needed
    """
    print(f"REPAIRING MESH: {patient_id}")
    
    start_time = time.time()
    
    result = {
        'patient_id': patient_id,
        'status': 'FAILED',
        'method': '',
        'stages_applied': [],
        'initial_elements': 0,
        'final_elements': 0,
        'initial_max_ar': 0,
        'final_max_ar': 0,
        'initial_min_jacobian': 0,
        'final_min_jacobian': 0,
        'meets_febio': False,
        'meets_opencarp': False,
        'runtime_sec': 0,
    }
    
    try:
        # Load mesh
        print("  Loading mesh...")
        vertices, elements, tags = read_carp_mesh(mesh_dir, patient_id)
        print(f"    {len(vertices):,} vertices, {len(elements):,} elements")
        
        # Initial quality assessment
        quality = compute_quality_metrics(vertices, elements)
        initial_summary = summarize_quality(quality)
        print_quality(initial_summary, "Initial quality")
        
        result['initial_elements'] = initial_summary['n_elements']
        result['initial_max_ar'] = initial_summary['max_aspect_ratio']
        result['initial_min_jacobian'] = initial_summary['min_jacobian']
        
        # Track best result
        best_verts, best_elems, best_tags = vertices, elements, tags
        best_summary = initial_summary
        
        # STAGE 1: MMG3D with full optimization
        if tools.get('mmg3d') and initial_summary['max_aspect_ratio'] > TARGETS['max_aspect_ratio']:
            print("\n  Stage 1: MMG3D full optimization...")
            try:
                new_verts, new_elems, new_tags = repair_with_mmg3d_full(
                    vertices, elements, tags, tools['mmg3d'],
                    hausd=0.005,  # Tighter surface preservation
                    hgrad=1.2,    # Smoother gradation
                )
                
                quality = compute_quality_metrics(new_verts, new_elems)
                summary = summarize_quality(quality)
                print_quality(summary, "After MMG3D")
                
                result['stages_applied'].append('MMG3D')
                
                if summary['max_aspect_ratio'] < best_summary['max_aspect_ratio']:
                    best_verts, best_elems, best_tags = new_verts, new_elems, new_tags
                    best_summary = summary
                    
            except Exception as e:
                print(f"    MMG3D failed: {e}")
        
        # STAGE 2: fTetWild remesh (if still bad)
        if tools.get('ftetwild') and best_summary['max_aspect_ratio'] > TARGETS['max_aspect_ratio']:
            print("\n  Stage 2: fTetWild remesh...")
            try:
                new_verts, new_elems, new_tags = repair_with_ftetwild(
                    vertices, elements, tags,  # Use ORIGINAL mesh surface
                    epsilon=1e-3,
                )
                
                quality = compute_quality_metrics(new_verts, new_elems)
                summary = summarize_quality(quality)
                print_quality(summary, "After fTetWild")
                
                result['stages_applied'].append('fTetWild')
                
                if summary['max_aspect_ratio'] < best_summary['max_aspect_ratio']:
                    best_verts, best_elems, best_tags = new_verts, new_elems, new_tags
                    best_summary = summary
                    
            except Exception as e:
                print(f"    fTetWild failed: {e}")
        
        # STAGE 3: TetGen optimization (if still bad)
        if tools.get('tetgen') and best_summary['max_aspect_ratio'] > TARGETS['max_aspect_ratio']:
            print("\n  Stage 3: TetGen optimization...")
            try:
                new_verts, new_elems, new_tags = repair_with_tetgen(
                    best_verts, best_elems, best_tags,
                    quality=1.2,
                )
                
                quality = compute_quality_metrics(new_verts, new_elems)
                summary = summarize_quality(quality)
                print_quality(summary, "After TetGen")
                
                result['stages_applied'].append('TetGen')
                
                if summary['max_aspect_ratio'] < best_summary['max_aspect_ratio']:
                    best_verts, best_elems, best_tags = new_verts, new_elems, new_tags
                    best_summary = summary
                    
            except Exception as e:
                print(f"    TetGen failed: {e}")
        
        # STAGE 4: Second MMG3D pass (if improved but not meeting targets)
        if tools.get('mmg3d') and \
           TARGETS_RELAXED['max_aspect_ratio'] < best_summary['max_aspect_ratio'] <= 1000:
            print("\n  Stage 4: MMG3D refinement pass...")
            try:
                new_verts, new_elems, new_tags = repair_with_mmg3d_full(
                    best_verts, best_elems, best_tags, tools['mmg3d'],
                    hausd=0.002,  # Very tight
                    hgrad=1.1,    # Very smooth
                )
                
                quality = compute_quality_metrics(new_verts, new_elems)
                summary = summarize_quality(quality)
                print_quality(summary, "After MMG3D pass 2")
                
                result['stages_applied'].append('MMG3D-2')
                
                if summary['max_aspect_ratio'] < best_summary['max_aspect_ratio']:
                    best_verts, best_elems, best_tags = new_verts, new_elems, new_tags
                    best_summary = summary
                    
            except Exception as e:
                print(f"    MMG3D pass 2 failed: {e}")
        
        # Save result
        patient_out = Path(output_dir) / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)
        
        write_carp_mesh(str(patient_out), patient_id, best_verts, best_elems, best_tags)
        
        # Write VTK for visualization
        vtk_path = patient_out / f"{patient_id}_repaired.vtk"
        with open(vtk_path, 'w') as f:
            f.write("# vtk DataFile Version 3.0\nRepaired mesh\nASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")
            f.write(f"POINTS {len(best_verts)} double\n")
            for v in best_verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            f.write(f"CELLS {len(best_elems)} {len(best_elems)*5}\n")
            for e in best_elems:
                f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
            f.write(f"CELL_TYPES {len(best_elems)}\n")
            for _ in best_elems:
                f.write("10\n")
            
            # Add quality as cell data
            quality = compute_quality_metrics(best_verts, best_elems)
            f.write(f"\nCELL_DATA {len(best_elems)}\n")
            f.write("SCALARS aspect_ratio double 1\nLOOKUP_TABLE default\n")
            for ar in quality['aspect_ratios']:
                f.write(f"{ar}\n")
            f.write("SCALARS jacobian double 1\nLOOKUP_TABLE default\n")
            for j in quality['jacobians']:
                f.write(f"{j}\n")
        
        # Update result
        result['status'] = 'SUCCESS'
        result['method'] = '+'.join(result['stages_applied']) or 'none'
        result['final_elements'] = best_summary['n_elements']
        result['final_max_ar'] = best_summary['max_aspect_ratio']
        result['final_min_jacobian'] = best_summary['min_jacobian']
        result['meets_febio'] = best_summary['meets_febio']
        result['meets_opencarp'] = best_summary['meets_opencarp']
        
        improvement = (result['initial_max_ar'] - result['final_max_ar']) / result['initial_max_ar'] * 100
        print(f"\n  Result: AR {result['initial_max_ar']:.0f} → {result['final_max_ar']:.1f} ({improvement:.1f}% improvement)")
        print(f"  FEBio: {'✓' if result['meets_febio'] else '✗'} | OpenCarp: {'✓' if result['meets_opencarp'] else '✗'}")
        
    except Exception as e:
        import traceback
        result['error'] = str(e)
        print(f"  ERROR: {e}")
        traceback.print_exc()
    
    result['runtime_sec'] = time.time() - start_time
    return result

# MAIN
def run_all(patients: List[str] = ALL_PATIENTS,
            mesh_dir: str = MESH_DIR,
            output_dir: str = OUTPUT_DIR) -> List[Dict]:
    """Run repair pipeline on all patients."""
    print("OPTIMAL TETRAHEDRAL MESH QUALITY REPAIR")
    
    print("\nDetecting available tools...")
    tools = detect_tools()
    
    available = sum([1 for t in ['mmg3d', 'ftetwild', 'tetgen', 'pymesh'] if tools.get(t)])
    print(f"\n{available}/4 repair tools available")
    
    if available == 0:
        print("\nERROR: No repair tools available!")
        print("Install at least one of:")
        print("  - MMG3D: sudo apt install mmg3d")
        print("  - fTetWild: pip install wildmeshing")
        print("  - TetGen: sudo apt install tetgen")
        return []
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, patient_id in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}]")
        result = repair_mesh(patient_id, mesh_dir, output_dir, tools)
        results.append(result)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"repair_summary_{timestamp}.csv"
    
    if results:
        # Get all keys from all results
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSummary saved to: {csv_path}")
    
    # Print summary
    print("SUMMARY")
    
    n_febio = sum(1 for r in results if r.get('meets_febio', False))
    n_opencarp = sum(1 for r in results if r.get('meets_opencarp', False))
    n_success = sum(1 for r in results if r.get('status') == 'SUCCESS')
    
    print(f"Processed: {n_success}/{len(results)}")
    print(f"FEBio ready: {n_febio}/{len(results)}")
    print(f"OpenCarp ready: {n_opencarp}/{len(results)}")
    
    print("\nPer-patient results:")
    for r in results:
        if r['status'] == 'SUCCESS':
            fb = "✓" if r.get('meets_febio') else "✗"
            oc = "✓" if r.get('meets_opencarp') else "✗"
            print(f"  {r['patient_id']}: AR {r['initial_max_ar']:.0f} → {r['final_max_ar']:.1f} "
                  f"[FB:{fb} OC:{oc}] via {r['method']}")
        else:
            print(f"  {r['patient_id']}: FAILED - {r.get('error', 'unknown')[:50]}")
    
    return results

# ENTRY POINT
if __name__ == "__main__":
    import argparse
    import sys
    
    # Check if running in Jupyter/IPython
    try:
        get_ipython()
        in_notebook = True
    except NameError:
        in_notebook = False
    
    if in_notebook:
        # Running in Jupyter - use default parameters
        print("Running in Jupyter notebook - using default parameters")
        results = run_all(
            patients=ALL_PATIENTS,
            mesh_dir=MESH_DIR,
            output_dir=OUTPUT_DIR
        )
    else:
        # Running from command line - use argparse
        parser = argparse.ArgumentParser(description='Optimal Tetrahedral Mesh Repair')
        parser.add_argument('--mesh-dir', type=str, default=MESH_DIR)
        parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
        parser.add_argument('--patients', type=str, nargs='+', default=ALL_PATIENTS)
        
        args = parser.parse_args()
        results = run_all(args.patients, args.mesh_dir, args.output_dir)