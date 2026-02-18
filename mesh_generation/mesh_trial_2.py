#!/usr/bin/env python3
"""
TETRAHEDRAL MESH QUALITY IMPROVEMENT

This script improves the quality of existing tetrahedral meshes WITHOUT
regenerating them. It keeps element count while fixing:
- High aspect ratios
- Poor Jacobians
- Bad dihedral angles

Methods (in order of preference):
1. MMG3D optimization (best results)
2. PyMesh optimization
3. Gmsh optimization
4. Custom Laplacian + optimization-based smoothing

Usage:
    python mesh_quality_improvement.py --mesh-dir /path/to/meshes
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

# CONFIGURATION
MESH_DIR = "/home/nvidia/SCD_MODELS/high_resolution_meshes"
OUTPUT_DIR = "/home/nvidia/SCD_MODELS/improved_meshes"

ALL_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000501",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]

# Quality targets
TARGET_MAX_ASPECT_RATIO = 50
TARGET_MIN_JACOBIAN = 0.01
TARGET_MIN_DIHEDRAL = 10
TARGET_MAX_DIHEDRAL = 170

# Smoothing parameters
MAX_SMOOTHING_ITERATIONS = 100
SMOOTHING_LAMBDA = 0.3  # Step size for Laplacian smoothing
CONVERGENCE_TOL = 1e-4

# CHECK AVAILABLE TOOLS
def check_available_tools() -> Dict[str, bool]:
    """Check which mesh improvement tools are available."""
    tools = {}
    
    # MMG3D
    for cmd in ['mmg3d_O3', 'mmg3d', 'mmg3d_debug']:
        try:
            result = subprocess.run([cmd, '--help'], capture_output=True, timeout=5)
            tools['mmg3d'] = cmd
            print(f"  MMG3D: available ({cmd})")
            break
        except:
            pass
    if 'mmg3d' not in tools:
        tools['mmg3d'] = None
        print("  MMG3D: not available")
    
    # Gmsh
    try:
        result = subprocess.run(['gmsh', '--version'], capture_output=True, timeout=5)
        tools['gmsh'] = True
        version = result.stdout.decode().strip() or result.stderr.decode().strip()
        print(f"  Gmsh: available ({version})")
    except:
        tools['gmsh'] = False
        print("  Gmsh: not available")
    
    # PyMesh
    try:
        import pymesh
        tools['pymesh'] = True
        print(f"  PyMesh: available")
    except ImportError:
        tools['pymesh'] = False
        print("  PyMesh: not available")
    
    # TetGen (for optimization mode)
    try:
        result = subprocess.run(['tetgen', '-h'], capture_output=True, timeout=5)
        tools['tetgen'] = True
        print(f"  TetGen: available")
    except:
        tools['tetgen'] = False
        print("  TetGen: not available")
    
    return tools

# MESH I/O
def read_carp_mesh(mesh_dir: str, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read CARP format mesh."""
    pts_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.pts"
    elem_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.elem"
    
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
    """Write mesh in Medit format (.mesh) for MMG3D."""
    with open(filepath, 'w') as f:
        f.write("MeshVersionFormatted 2\n")
        f.write("Dimension 3\n\n")
        
        f.write(f"Vertices\n{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} 0\n")
        
        f.write(f"\nTetrahedra\n{len(elements)}\n")
        for i, e in enumerate(elements):
            tag = tags[i] if tags is not None else 1
            # Medit uses 1-based indexing
            f.write(f"{e[0]+1} {e[1]+1} {e[2]+1} {e[3]+1} {tag}\n")
        
        f.write("\nEnd\n")

def read_medit_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read mesh from Medit format (.mesh)."""
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
                # Medit uses 1-based indexing
                elements.append([int(parts[0])-1, int(parts[1])-1, 
                               int(parts[2])-1, int(parts[3])-1])
                tags.append(int(parts[4]) if len(parts) > 4 else 1)
            i += n + 2
        
        else:
            i += 1
    
    return np.array(vertices), np.array(elements), np.array(tags)

def write_vtk_mesh(filepath: str, vertices: np.ndarray, elements: np.ndarray,
                   tags: np.ndarray = None, quality: np.ndarray = None):
    """Write VTK file for visualization."""
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Improved mesh\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {len(vertices)} double\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        f.write(f"CELLS {len(elements)} {len(elements)*5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"CELL_TYPES {len(elements)}\n")
        for _ in elements:
            f.write("10\n")
        
        f.write(f"\nCELL_DATA {len(elements)}\n")
        
        if tags is not None:
            f.write("SCALARS tissue_tag int 1\nLOOKUP_TABLE default\n")
            for t in tags:
                f.write(f"{t}\n")
        
        if quality is not None:
            f.write("SCALARS element_quality double 1\nLOOKUP_TABLE default\n")
            for q in quality:
                f.write(f"{q}\n")

# QUALITY METRICS
def compute_element_quality(vertices: np.ndarray, elements: np.ndarray) -> Dict:
    """Compute per-element quality metrics."""
    n = len(elements)
    
    jacobians = np.zeros(n)
    aspect_ratios = np.zeros(n)
    min_dihedrals = np.zeros(n)
    max_dihedrals = np.zeros(n)
    volumes = np.zeros(n)
    
    for i, e in enumerate(elements):
        v0, v1, v2, v3 = vertices[e[0]], vertices[e[1]], vertices[e[2]], vertices[e[3]]
        
        # Edge vectors
        e01, e02, e03 = v1 - v0, v2 - v0, v3 - v0
        
        # Volume
        vol = np.dot(e01, np.cross(e02, e03)) / 6.0
        volumes[i] = vol
        
        # Edge lengths
        edges = [v1-v0, v2-v0, v3-v0, v2-v1, v3-v1, v3-v2]
        lens = [np.linalg.norm(e) for e in edges]
        
        # Aspect ratio (edge-based)
        l_max = max(lens)
        l_min = max(min(lens), 1e-12)
        
        # Height-based aspect ratio
        base_area = 0.5 * np.linalg.norm(np.cross(e01, e02))
        height = 3 * abs(vol) / base_area if base_area > 1e-12 else 1e-12
        aspect_ratios[i] = l_max / height if height > 1e-12 else 1e10
        
        # Scaled Jacobian
        l_rms = np.sqrt(np.mean([l**2 for l in lens]))
        jacobians[i] = 6 * np.sqrt(2) * vol / (l_rms**3) if l_rms > 1e-12 else 0
        
        # Dihedral angles
        normals = [
            np.cross(v1-v0, v2-v0),
            np.cross(v1-v0, v3-v0),
            np.cross(v2-v0, v3-v0),
            np.cross(v2-v1, v3-v1),
        ]
        normals = [n/np.linalg.norm(n) if np.linalg.norm(n) > 1e-12 else n for n in normals]
        
        dihedrals = []
        for j in range(len(normals)):
            for k in range(j+1, len(normals)):
                dot = np.clip(np.dot(normals[j], normals[k]), -1, 1)
                dihedrals.append(180 - np.degrees(np.arccos(dot)))
        
        min_dihedrals[i] = min(dihedrals) if dihedrals else 0
        max_dihedrals[i] = max(dihedrals) if dihedrals else 180
    
    return {
        'jacobians': jacobians,
        'aspect_ratios': aspect_ratios,
        'min_dihedrals': min_dihedrals,
        'max_dihedrals': max_dihedrals,
        'volumes': volumes,
    }

def compute_summary_metrics(quality: Dict) -> Dict:
    """Compute summary statistics from per-element quality."""
    metrics = {}
    
    metrics['n_elements'] = len(quality['jacobians'])
    metrics['n_inverted'] = np.sum(quality['volumes'] <= 0)
    
    metrics['min_jacobian'] = np.min(quality['jacobians'])
    metrics['mean_jacobian'] = np.mean(quality['jacobians'])
    metrics['max_jacobian'] = np.max(quality['jacobians'])
    
    metrics['max_aspect_ratio'] = np.max(quality['aspect_ratios'])
    metrics['mean_aspect_ratio'] = np.mean(quality['aspect_ratios'])
    
    metrics['min_dihedral'] = np.min(quality['min_dihedrals'])
    metrics['max_dihedral'] = np.max(quality['max_dihedrals'])
    
    # Count bad elements
    metrics['n_bad_jacobian'] = np.sum(quality['jacobians'] < TARGET_MIN_JACOBIAN)
    metrics['n_bad_aspect'] = np.sum(quality['aspect_ratios'] > TARGET_MAX_ASPECT_RATIO)
    metrics['n_bad_dihedral'] = np.sum(
        (quality['min_dihedrals'] < TARGET_MIN_DIHEDRAL) |
        (quality['max_dihedrals'] > TARGET_MAX_DIHEDRAL)
    )
    
    # Quality check
    metrics['meets_targets'] = (
        metrics['min_jacobian'] >= TARGET_MIN_JACOBIAN and
        metrics['max_aspect_ratio'] <= TARGET_MAX_ASPECT_RATIO and
        metrics['min_dihedral'] >= TARGET_MIN_DIHEDRAL and
        metrics['max_dihedral'] <= TARGET_MAX_DIHEDRAL and
        metrics['n_inverted'] == 0
    )
    
    return metrics

def identify_bad_elements(quality: Dict) -> np.ndarray:
    """Identify indices of elements that need improvement."""
    bad = (
        (quality['jacobians'] < TARGET_MIN_JACOBIAN) |
        (quality['aspect_ratios'] > TARGET_MAX_ASPECT_RATIO) |
        (quality['min_dihedrals'] < TARGET_MIN_DIHEDRAL) |
        (quality['max_dihedrals'] > TARGET_MAX_DIHEDRAL) |
        (quality['volumes'] <= 0)
    )
    return np.where(bad)[0]

# METHOD 1: MMG3D OPTIMIZATION (BEST)
def optimize_with_mmg3d(vertices: np.ndarray, elements: np.ndarray, 
                        tags: np.ndarray, mmg3d_cmd: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize mesh using MMG3D.
    
    MMG3D is excellent for mesh quality improvement. It uses:
    - Vertex relocation
    - Edge swapping
    - Local remeshing
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.mesh"
        output_path = f"{tmpdir}/output.o.mesh"
        
        # Write input mesh
        write_medit_mesh(input_path, vertices, elements, tags)
        
        # Run MMG3D with optimization settings
        # -noinsert: don't insert new vertices
        # -noswap: don't swap edges (optional, remove for more aggressive optimization)
        # -nomove: don't move vertices (we want this ON, so don't use it)
        # -hausd: Hausdorff distance (surface accuracy)
        # -hgrad: gradation control
        
        cmd = [
            mmg3d_cmd,
            input_path,
            "-out", output_path,
            "-noinsert",  # Keep vertex count similar
            "-hausd", "0.01",  # Tight surface preservation
            "-hgrad", "1.3",  # Allow some gradation
            "-optim",  # Optimization mode
            "-v", "0",  # Quiet
        ]
        
        print(f"    Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        if result.returncode != 0:
            print(f"    MMG3D warning: {result.stderr.decode()[:200]}")
        
        # Read output
        if os.path.exists(output_path):
            return read_medit_mesh(output_path)
        else:
            # Try alternate output name
            alt_output = input_path.replace('.mesh', '.o.mesh')
            if os.path.exists(alt_output):
                return read_medit_mesh(alt_output)
            raise RuntimeError("MMG3D did not produce output file")

# METHOD 2: PYMESH OPTIMIZATION
def optimize_with_pymesh(vertices: np.ndarray, elements: np.ndarray,
                         tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize mesh using PyMesh.
    """
    import pymesh
    
    # Create PyMesh mesh
    mesh = pymesh.form_mesh(vertices, elements)
    
    # Run optimization
    # PyMesh has several optimization methods
    
    # 1. Remove degenerate faces
    mesh, info = pymesh.remove_degenerated_triangles(mesh)
    
    # 2. Remove isolated vertices
    mesh, info = pymesh.remove_isolated_vertices(mesh)
    
    # 3. Collapse short edges (careful with this)
    # mesh, info = pymesh.collapse_short_edges(mesh, rel_threshold=0.1)
    
    # Extract result
    opt_vertices = mesh.vertices
    opt_elements = mesh.faces if mesh.faces.shape[1] == 4 else mesh.voxels
    
    # Recompute tags (simplified - assign based on centroid proximity)
    # In practice you'd want to preserve tags more carefully
    opt_tags = np.ones(len(opt_elements), dtype=np.int32)
    
    return opt_vertices, opt_elements, opt_tags

# METHOD 3: GMSH OPTIMIZATION
def optimize_with_gmsh(vertices: np.ndarray, elements: np.ndarray,
                       tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Optimize mesh using Gmsh.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.msh"
        output_path = f"{tmpdir}/output.msh"
        
        # Write Gmsh format
        with open(input_path, 'w') as f:
            f.write("$MeshFormat\n4.1 0 8\n$EndMeshFormat\n")
            
            f.write(f"$Nodes\n1 {len(vertices)} 1 {len(vertices)}\n")
            f.write(f"3 1 0 {len(vertices)}\n")
            for i in range(len(vertices)):
                f.write(f"{i+1}\n")
            for v in vertices:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")
            f.write("$EndNodes\n")
            
            f.write(f"$Elements\n1 {len(elements)} 1 {len(elements)}\n")
            f.write(f"3 1 4 {len(elements)}\n")  # 4 = tetrahedron
            for i, e in enumerate(elements):
                f.write(f"{i+1} {e[0]+1} {e[1]+1} {e[2]+1} {e[3]+1}\n")
            f.write("$EndElements\n")
        
        # Run Gmsh optimization
        cmd = [
            "gmsh", input_path,
            "-3",  # 3D
            "-optimize_netgen",  # Netgen optimizer
            "-o", output_path,
            "-format", "msh4",
            "-v", "0",
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        
        if not os.path.exists(output_path):
            raise RuntimeError("Gmsh did not produce output")
        
        # Read output (simplified - would need proper Gmsh reader)
        # For now, return original
        print("    Gmsh optimization completed (output parsing not implemented)")
        return vertices, elements, tags

# METHOD 4: CUSTOM SMOOTHING (FALLBACK)
def smooth_vertices_laplacian(vertices: np.ndarray, elements: np.ndarray,
                              boundary_verts: set, n_iters: int = 10,
                              lambda_: float = 0.3) -> np.ndarray:
    """
    Laplacian smoothing - moves each interior vertex toward the centroid
    of its neighbors.
    """
    # Build vertex adjacency
    neighbors = defaultdict(set)
    for e in elements:
        for i in range(4):
            for j in range(4):
                if i != j:
                    neighbors[e[i]].add(e[j])
    
    verts = vertices.copy()
    
    for iteration in range(n_iters):
        new_verts = verts.copy()
        
        for vi in range(len(verts)):
            if vi in boundary_verts:
                continue  # Don't move boundary vertices
            
            nbrs = list(neighbors[vi])
            if len(nbrs) == 0:
                continue
            
            # Compute neighbor centroid
            centroid = np.mean(verts[nbrs], axis=0)
            
            # Move toward centroid
            new_verts[vi] = verts[vi] + lambda_ * (centroid - verts[vi])
        
        verts = new_verts
    
    return verts

def smooth_vertices_optimization(vertices: np.ndarray, elements: np.ndarray,
                                 boundary_verts: set, quality: Dict,
                                 n_iters: int = 10) -> np.ndarray:
    """
    Optimization-based smoothing - moves vertices to improve element quality.
    Only moves vertices of bad elements.
    """
    # Build vertex-to-element mapping
    vert_elems = defaultdict(list)
    for ei, e in enumerate(elements):
        for vi in e:
            vert_elems[vi].append(ei)
    
    bad_elems = identify_bad_elements(quality)
    
    # Get vertices to optimize (vertices of bad elements, excluding boundary)
    verts_to_opt = set()
    for ei in bad_elems:
        for vi in elements[ei]:
            if vi not in boundary_verts:
                verts_to_opt.add(vi)
    
    print(f"    Optimizing {len(verts_to_opt)} vertices ({len(bad_elems)} bad elements)")
    
    verts = vertices.copy()
    
    for iteration in range(n_iters):
        moved = 0
        
        for vi in verts_to_opt:
            # Get elements containing this vertex
            elem_indices = vert_elems[vi]
            
            # Compute current quality
            current_worst = float('inf')
            for ei in elem_indices:
                e = elements[ei]
                v = verts[e]
                # Simple quality: min edge / max edge
                edges = [np.linalg.norm(v[i]-v[j]) for i in range(4) for j in range(i+1, 4)]
                q = min(edges) / (max(edges) + 1e-12)
                current_worst = min(current_worst, q)
            
            # Try small perturbations
            best_pos = verts[vi].copy()
            best_quality = current_worst
            
            for dx in [-0.01, 0, 0.01]:
                for dy in [-0.01, 0, 0.01]:
                    for dz in [-0.01, 0, 0.01]:
                        if dx == dy == dz == 0:
                            continue
                        
                        test_pos = verts[vi] + np.array([dx, dy, dz]) * np.linalg.norm(verts[vi])
                        verts[vi] = test_pos
                        
                        worst = float('inf')
                        valid = True
                        for ei in elem_indices:
                            e = elements[ei]
                            v = verts[e]
                            
                            # Check volume (no inversion)
                            vol = np.dot(v[1]-v[0], np.cross(v[2]-v[0], v[3]-v[0])) / 6
                            if vol <= 0:
                                valid = False
                                break
                            
                            edges = [np.linalg.norm(v[i]-v[j]) for i in range(4) for j in range(i+1, 4)]
                            q = min(edges) / (max(edges) + 1e-12)
                            worst = min(worst, q)
                        
                        if valid and worst > best_quality:
                            best_quality = worst
                            best_pos = test_pos.copy()
                        
                        verts[vi] = vertices[vi]  # Reset for next test
            
            if best_quality > current_worst:
                verts[vi] = best_pos
                moved += 1
            else:
                verts[vi] = vertices[vi]
        
        if moved == 0:
            break
    
    return verts

def get_boundary_vertices(vertices: np.ndarray, elements: np.ndarray) -> set:
    """Identify boundary (surface) vertices."""
    face_count = defaultdict(int)
    face_verts = {}
    
    for e in elements:
        faces = [
            tuple(sorted([e[0], e[1], e[2]])),
            tuple(sorted([e[0], e[1], e[3]])),
            tuple(sorted([e[0], e[2], e[3]])),
            tuple(sorted([e[1], e[2], e[3]])),
        ]
        for f in faces:
            face_count[f] += 1
            face_verts[f] = list(f)
    
    boundary = set()
    for f, count in face_count.items():
        if count == 1:
            boundary.update(face_verts[f])
    
    return boundary

def custom_mesh_improvement(vertices: np.ndarray, elements: np.ndarray,
                            tags: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Custom mesh improvement using Laplacian + optimization-based smoothing.
    """
    boundary_verts = get_boundary_vertices(vertices, elements)
    print(f"    Boundary vertices: {len(boundary_verts)}")
    
    # Compute initial quality
    quality = compute_element_quality(vertices, elements)
    metrics = compute_summary_metrics(quality)
    print(f"    Initial: AR={metrics['max_aspect_ratio']:.1f}, J={metrics['min_jacobian']:.6f}")
    
    verts = vertices.copy()
    
    for iteration in range(MAX_SMOOTHING_ITERATIONS):
        # Laplacian smoothing
        verts = smooth_vertices_laplacian(verts, elements, boundary_verts, 
                                          n_iters=5, lambda_=SMOOTHING_LAMBDA)
        
        # Optimization-based smoothing
        quality = compute_element_quality(verts, elements)
        verts = smooth_vertices_optimization(verts, elements, boundary_verts,
                                            quality, n_iters=3)
        
        # Check progress
        quality = compute_element_quality(verts, elements)
        metrics = compute_summary_metrics(quality)
        
        if iteration % 10 == 0:
            print(f"    Iter {iteration}: AR={metrics['max_aspect_ratio']:.1f}, "
                  f"J={metrics['min_jacobian']:.6f}, bad={metrics['n_bad_aspect']}")
        
        if metrics['meets_targets']:
            print(f"    Converged at iteration {iteration}")
            break
        
        # Check for convergence
        if iteration > 0 and abs(metrics['max_aspect_ratio'] - prev_ar) < CONVERGENCE_TOL:
            print(f"    Stalled at iteration {iteration}")
            break
        
        prev_ar = metrics['max_aspect_ratio']
    
    return verts, elements, tags

# MAIN PROCESSING
def improve_mesh(patient_id: str, mesh_dir: str, output_dir: str,
                 tools: Dict) -> Dict:
    """Improve mesh quality for a single patient."""
    print(f"IMPROVING MESH: {patient_id}")
    
    start_time = time.time()
    
    result = {
        'patient_id': patient_id,
        'status': 'FAILED',
        'method': '',
        'runtime_sec': 0,
    }
    
    try:
        # Load mesh
        print("  Loading mesh...")
        vertices, elements, tags = read_carp_mesh(mesh_dir, patient_id)
        print(f"    Vertices: {len(vertices)}, Elements: {len(elements)}")
        
        # Compute initial quality
        quality = compute_element_quality(vertices, elements)
        initial_metrics = compute_summary_metrics(quality)
        result['initial_max_ar'] = initial_metrics['max_aspect_ratio']
        result['initial_min_jacobian'] = initial_metrics['min_jacobian']
        result['initial_bad_elements'] = initial_metrics['n_bad_aspect']
        
        print(f"  Initial quality:")
        print(f"    Max aspect ratio: {initial_metrics['max_aspect_ratio']:.1f}")
        print(f"    Min Jacobian: {initial_metrics['min_jacobian']:.6f}")
        print(f"    Bad elements: {initial_metrics['n_bad_aspect']}")
        
        # Try optimization methods in order of preference
        opt_verts, opt_elems, opt_tags = None, None, None
        
        # Method 1: MMG3D (best)
        if tools.get('mmg3d'):
            print("  Trying MMG3D optimization...")
            try:
                opt_verts, opt_elems, opt_tags = optimize_with_mmg3d(
                    vertices, elements, tags, tools['mmg3d']
                )
                result['method'] = 'MMG3D'
            except Exception as e:
                print(f"    MMG3D failed: {e}")
        
        # Method 2: PyMesh
        if opt_verts is None and tools.get('pymesh'):
            print("  Trying PyMesh optimization...")
            try:
                opt_verts, opt_elems, opt_tags = optimize_with_pymesh(
                    vertices, elements, tags
                )
                result['method'] = 'PyMesh'
            except Exception as e:
                print(f"    PyMesh failed: {e}")
        
        # Method 3: Custom smoothing (fallback)
        if opt_verts is None:
            print("  Using custom smoothing...")
            opt_verts, opt_elems, opt_tags = custom_mesh_improvement(
                vertices, elements, tags
            )
            result['method'] = 'Custom'
        
        # Compute final quality
        quality = compute_element_quality(opt_verts, opt_elems)
        final_metrics = compute_summary_metrics(quality)
        
        result['final_max_ar'] = final_metrics['max_aspect_ratio']
        result['final_min_jacobian'] = final_metrics['min_jacobian']
        result['final_bad_elements'] = final_metrics['n_bad_aspect']
        result['n_vertices'] = len(opt_verts)
        result['n_elements'] = len(opt_elems)
        result['meets_targets'] = final_metrics['meets_targets']
        
        print(f"  Final quality:")
        print(f"    Max aspect ratio: {final_metrics['max_aspect_ratio']:.1f}")
        print(f"    Min Jacobian: {final_metrics['min_jacobian']:.6f}")
        print(f"    Bad elements: {final_metrics['n_bad_aspect']}")
        print(f"    Meets targets: {final_metrics['meets_targets']}")
        
        # Improvement
        ar_improvement = (initial_metrics['max_aspect_ratio'] - final_metrics['max_aspect_ratio']) / initial_metrics['max_aspect_ratio'] * 100
        print(f"    Aspect ratio improvement: {ar_improvement:.1f}%")
        
        # Save improved mesh
        patient_out = Path(output_dir) / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)
        
        write_carp_mesh(str(patient_out), patient_id, opt_verts, opt_elems, opt_tags)
        write_vtk_mesh(str(patient_out / f"{patient_id}_improved.vtk"),
                      opt_verts, opt_elems, opt_tags, quality['aspect_ratios'])
        
        result['status'] = 'SUCCESS'
        
    except Exception as e:
        import traceback
        result['error'] = str(e)
        print(f"  ERROR: {e}")
        traceback.print_exc()
    
    result['runtime_sec'] = time.time() - start_time
    return result

def run_all(patients: List[str] = ALL_PATIENTS,
            mesh_dir: str = MESH_DIR,
            output_dir: str = OUTPUT_DIR) -> List[Dict]:
    """Improve all patient meshes."""
    print("TETRAHEDRAL MESH QUALITY IMPROVEMENT")
    
    # Check available tools
    print("\nChecking available optimization tools...")
    tools = check_available_tools()
    
    if not any([tools.get('mmg3d'), tools.get('pymesh'), tools.get('gmsh')]):
        print("\nWARNING: No specialized tools available, using custom smoothing only.")
        print("For best results, install MMG3D:")
        print("  sudo apt-get install mmg3d")
        print("  # or")
        print("  conda install -c conda-forge mmg")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, patient_id in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}]")
        result = improve_mesh(patient_id, mesh_dir, output_dir, tools)
        results.append(result)
    
    # Save summary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"improvement_summary_{timestamp}.csv"
    
    if results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary
    print("SUMMARY")    
    for r in results:
        status = "✓" if r.get('meets_targets', False) else "✗"
        init_ar = r.get('initial_max_ar', 0)
        final_ar = r.get('final_max_ar', 0)
        method = r.get('method', 'N/A')
        print(f"  {status} {r['patient_id']}: AR {init_ar:.0f} → {final_ar:.1f} ({method})")
    
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
        # Running in Jupyter - use default parameters or set them directly
        print("Running in Jupyter notebook - using default parameters")
        results = run_all(
            patients=ALL_PATIENTS,
            mesh_dir=MESH_DIR,
            output_dir=OUTPUT_DIR
        )
    else:
        # Running from command line - use argparse
        parser = argparse.ArgumentParser(description='Improve tetrahedral mesh quality')
        parser.add_argument('--mesh-dir', type=str, default=MESH_DIR)
        parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR)
        parser.add_argument('--patients', type=str, nargs='+', default=ALL_PATIENTS)
        
        args = parser.parse_args()
        results = run_all(args.patients, args.mesh_dir, args.output_dir)