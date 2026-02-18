#!/usr/bin/env python3
"""
HIGH-RESOLUTION MESH GENERATION & TISSUE TAGGING PIPELINE


For OpenCarp Electrophysiology and FEBio Biomechanics Simulations

This script generates:
1. High-resolution tetrahedral meshes (100k+ elements)
2. Proper tissue tagging (3 regions + border zones)
3. Wall definitions (endocardium, epicardium, base)
4. Complete quality metrics for simulation readiness

Quality Metrics and Formulations:

1. SCALED JACOBIAN (J):
   J = 6*sqrt(2) * V / L_rms^3
   - V = signed tetrahedral volume
   - L_rms = root mean square of edge lengths
   - Range: [-1, 1], ideal = 1.0, must be > 0 for valid element
   - OpenCarp requirement: J > 0.0001
   - FEBio recommendation: J > 0.01

2. DIHEDRAL ANGLE (theta):
   theta = 180 - arccos(n1 . n2)
   - n1, n2 = face normal vectors
   - Range: [0, 180] degrees
   - OpenCarp requirement: 0.1 < theta < 179.9
   - FEBio recommendation: 10 < theta < 170

3. ASPECT RATIO (AR):
   AR = L_max / h_min
   - L_max = longest edge
   - h_min = minimum height (3*V / A_max)
   - Ideal = 1.0 (regular tetrahedron)
   - OpenCarp tolerance: AR < 100
   - FEBio recommendation: AR < 50

4. RADIUS-EDGE RATIO (R/d):
   R/d = circumradius / minimum_edge_length
   - Ideal (regular tet) = 0.612
   - OpenCarp tolerance: R/d < 20
   - FEBio recommendation: R/d < 10

5. ELEMENT VOLUME:
   V = (1/6) * |det([v1-v0, v2-v0, v3-v0])|
   - Must be positive (negative = inverted)
   - Uniformity: std(V)/mean(V) < 1.0


Surface Markers:

- Marker 10: Base plane
- Marker 20: Epicardial surface (outer)
- Marker 30: Endocardial surface (inner)

"""

import numpy as np
from numba import jit, prange, set_num_threads
import numba
from pathlib import Path
import subprocess
import tempfile
import struct
import os
import warnings
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Any
import time
import csv
from datetime import datetime

warnings.filterwarnings('ignore')

# CONFIG
NUM_CPUS = min(16, os.cpu_count() or 4)
set_num_threads(NUM_CPUS)

BASE_DIR = "/home/nvidia/SCD_MODELS/tetrahedral_meshes"
STL_DIR = "/home/nvidia/SCD_MODELS/processed_triangular_meshes"
OUTPUT_DIR = "/home/nvidia/SCD_MODELS/high_resolution_meshes"

# Patient list
ALL_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000501",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]

# Target element counts for different simulation types
TARGET_ELEMENTS = {
    'minimum': 50000,      # Minimum for coarse EP
    'standard': 100000,    # Standard for EP simulations
    'high': 200000,        # High resolution EP
    'mechanics': 150000,   # FEBio biomechanics
}

# Quality thresholds
QUALITY_THRESHOLDS_OPENCARP = {
    'min_jacobian': 0.0001,
    'min_dihedral': 0.1,
    'max_dihedral': 179.9,
    'max_aspect_ratio': 100.0,
    'max_radius_edge': 20.0,
}

QUALITY_THRESHOLDS_FEBIO = {
    'min_jacobian': 0.01,
    'min_dihedral': 10.0,
    'max_dihedral': 170.0,
    'max_aspect_ratio': 50.0,
    'max_radius_edge': 10.0,
}

# Tissue tagging parameters
TISSUE_TAGS = {
    'healthy': 1,
    'border_zone': 2,
    'infarct': 3,
}

SURFACE_MARKERS = {
    'base': 10,
    'epicardium': 20,
    'endocardium': 30,
}

# Check for optional dependencies
try:
    import pymeshlab
    HAS_PYMESHLAB = True
except ImportError:
    HAS_PYMESHLAB = False
    print("WARNING: PyMeshLab not available - surface repair will be limited")

try:
    import wildmeshing as wm
    HAS_FTETWILD = True
except ImportError:
    HAS_FTETWILD = False
    print("WARNING: fTetWild not available - will use TetGen only")

try:
    import meshio
    HAS_MESHIO = True
except ImportError:
    HAS_MESHIO = False
    print("WARNING: meshio not available")

print(f"System Configuration:")
print(f"  CPUs: {NUM_CPUS}")
print(f"  PyMeshLab: {'Yes' if HAS_PYMESHLAB else 'No'}")
print(f"  fTetWild: {'Yes' if HAS_FTETWILD else 'No'}")
print(f"  meshio: {'Yes' if HAS_MESHIO else 'No'}")

# NUMBA-OPTIMIZED QUALITY METRICS
@jit(nopython=True, cache=True, fastmath=True)
def cross3(a, b):
    """3D cross product."""
    return np.array([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])

@jit(nopython=True, cache=True, fastmath=True)
def dot3(a, b):
    """3D dot product."""
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

@jit(nopython=True, cache=True, fastmath=True)
def norm3(a):
    """3D vector norm."""
    return np.sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2])

@jit(nopython=True, cache=True)
def tet_signed_volume(v0, v1, v2, v3):
    """
    Compute signed volume of tetrahedron.
    
    Formula: V = (1/6) * det([v1-v0, v2-v0, v3-v0])
    
    Positive = correctly oriented
    Negative = inverted element
    """
    return dot3(v1-v0, cross3(v2-v0, v3-v0)) / 6.0

@jit(nopython=True, cache=True)
def tet_edge_lengths(v0, v1, v2, v3):
    """Compute all 6 edge lengths of tetrahedron."""
    return np.array([
        norm3(v1-v0), norm3(v2-v0), norm3(v3-v0),
        norm3(v2-v1), norm3(v3-v1), norm3(v3-v2)
    ])

@jit(nopython=True, cache=True)
def scaled_jacobian(v0, v1, v2, v3):
    """
    Scaled Jacobian quality metric.
    
    Formula: J = 6*sqrt(2) * V / L_rms^3
    
    Where:
    - V = signed tetrahedral volume
    - L_rms = sqrt(sum(L_i^2) / 6) = RMS of edge lengths
    
    Range: [-1, 1]
    - J = 1.0 for ideal regular tetrahedron
    - J < 0 indicates inverted element
    - J > 0.0001 required for OpenCarp
    - J > 0.01 recommended for FEBio
    """
    vol = tet_signed_volume(v0, v1, v2, v3)
    edges = tet_edge_lengths(v0, v1, v2, v3)
    l_sq_sum = np.sum(edges * edges)
    l_rms_cubed = (l_sq_sum / 6.0) ** 1.5
    
    if l_rms_cubed < 1e-30:
        return 0.0
    
    # 6*sqrt(2) = 8.485281374238571
    return max(-1.0, min(1.0, 8.485281374238571 * vol / l_rms_cubed))

@jit(nopython=True, cache=True)
def dihedral_angles_minmax(v0, v1, v2, v3):
    """
    Compute minimum and maximum dihedral angles in degrees.
    
    Formula: theta = 180 - arccos(n1 . n2)
    
    Where n1, n2 are outward-pointing face normals.
    
    A tetrahedron has 6 dihedral angles (one per edge).
    - Ideal regular tet: all angles = 70.53 degrees
    - OpenCarp requirement: 0.1 < theta < 179.9
    - FEBio recommendation: 10 < theta < 170
    """
    # Face normals (outward pointing)
    n0 = cross3(v2-v1, v3-v1)  # Face opposite v0
    n1 = cross3(v3-v0, v2-v0)  # Face opposite v1
    n2 = cross3(v1-v0, v3-v0)  # Face opposite v2
    n3 = cross3(v2-v0, v1-v0)  # Face opposite v3
    
    # Normalize
    mag0, mag1, mag2, mag3 = norm3(n0), norm3(n1), norm3(n2), norm3(n3)
    if mag0 > 1e-30: n0 = n0 / mag0
    if mag1 > 1e-30: n1 = n1 / mag1
    if mag2 > 1e-30: n2 = n2 / mag2
    if mag3 > 1e-30: n3 = n3 / mag3
    
    min_angle = 180.0
    max_angle = 0.0
    
    # 6 edges connect pairs of faces
    pairs = [(n2, n3), (n1, n3), (n1, n2), (n0, n3), (n0, n2), (n0, n1)]
    
    for na, nb in pairs:
        cos_a = max(-1.0, min(1.0, dot3(na, nb)))
        angle = 180.0 - np.degrees(np.arccos(cos_a))
        min_angle = min(min_angle, angle)
        max_angle = max(max_angle, angle)
    
    return min_angle, max_angle

@jit(nopython=True, cache=True)
def circumradius(v0, v1, v2, v3):
    """
    Compute circumradius of tetrahedron.
    
    The circumradius is the radius of the sphere passing through all 4 vertices.
    """
    vol = tet_signed_volume(v0, v1, v2, v3)
    if abs(vol) < 1e-30:
        return 1e30
    
    a, b, c = v1-v0, v2-v0, v3-v0
    bc = cross3(b, c)
    ca = cross3(c, a)
    ab = cross3(a, b)
    
    denom = 2.0 * dot3(a, bc)
    if abs(denom) < 1e-30:
        return 1e30
    
    cc = (dot3(a,a)*bc + dot3(b,b)*ca + dot3(c,c)*ab) / denom
    return norm3(cc)

@jit(nopython=True, cache=True)
def radius_edge_ratio(v0, v1, v2, v3):
    """
    Radius-edge ratio: R / d_min
    
    Where:
    - R = circumradius
    - d_min = minimum edge length
    
    - Ideal regular tet: R/d = 0.612
    - OpenCarp tolerance: R/d < 20
    - FEBio recommendation: R/d < 10
    """
    R = circumradius(v0, v1, v2, v3)
    edges = tet_edge_lengths(v0, v1, v2, v3)
    d_min = np.min(edges)
    
    if d_min < 1e-30:
        return 1e30
    
    return R / d_min

@jit(nopython=True, cache=True)
def tet_face_areas(v0, v1, v2, v3):
    """Compute areas of all 4 faces."""
    a0 = 0.5 * norm3(cross3(v2-v1, v3-v1))
    a1 = 0.5 * norm3(cross3(v2-v0, v3-v0))
    a2 = 0.5 * norm3(cross3(v1-v0, v3-v0))
    a3 = 0.5 * norm3(cross3(v1-v0, v2-v0))
    return np.array([a0, a1, a2, a3])

@jit(nopython=True, cache=True)
def aspect_ratio(v0, v1, v2, v3):
    """
    Aspect ratio: L_max / h_min
    
    Where:
    - L_max = longest edge length
    - h_min = minimum height = 3*V / A_max
    
    - Ideal regular tet: AR ~= 1.0
    - OpenCarp tolerance: AR < 100
    - FEBio recommendation: AR < 50
    """
    vol = abs(tet_signed_volume(v0, v1, v2, v3))
    if vol < 1e-30:
        return 1e30
    
    edges = tet_edge_lengths(v0, v1, v2, v3)
    L_max = np.max(edges)
    
    areas = tet_face_areas(v0, v1, v2, v3)
    A_max = np.max(areas)
    
    if A_max < 1e-30:
        return 1e30
    
    h_min = 3.0 * vol / A_max
    
    if h_min < 1e-30:
        return 1e30
    
    return L_max / h_min

@jit(nopython=True, parallel=True, cache=True)
def compute_all_quality_parallel(vertices, elements):
    """Compute all quality metrics for all elements using parallel loops."""
    n = len(elements)
    volumes = np.empty(n)
    jacobians = np.empty(n)
    min_dihs = np.empty(n)
    max_dihs = np.empty(n)
    rad_edge = np.empty(n)
    asp_ratios = np.empty(n)
    
    for i in prange(n):
        e = elements[i]
        v0 = vertices[e[0]]
        v1 = vertices[e[1]]
        v2 = vertices[e[2]]
        v3 = vertices[e[3]]
        
        volumes[i] = tet_signed_volume(v0, v1, v2, v3)
        jacobians[i] = scaled_jacobian(v0, v1, v2, v3)
        min_dihs[i], max_dihs[i] = dihedral_angles_minmax(v0, v1, v2, v3)
        rad_edge[i] = radius_edge_ratio(v0, v1, v2, v3)
        asp_ratios[i] = aspect_ratio(v0, v1, v2, v3)
    
    return volumes, jacobians, min_dihs, max_dihs, rad_edge, asp_ratios

@jit(nopython=True, parallel=True, cache=True)
def fix_inverted_elements(vertices, elements):
    """Fix inverted tetrahedra by swapping vertices 2 and 3."""
    n = len(elements)
    fixed = elements.copy()
    n_fixed = 0
    
    for i in prange(n):
        e = fixed[i]
        v0, v1, v2, v3 = vertices[e[0]], vertices[e[1]], vertices[e[2]], vertices[e[3]]
        vol = tet_signed_volume(v0, v1, v2, v3)
        
        if vol < 0:
            fixed[i, 2], fixed[i, 3] = fixed[i, 3], fixed[i, 2]
    
    return fixed

@jit(nopython=True, parallel=True, cache=True)
def compute_element_centroids(vertices, elements):
    """Compute centroids of all elements."""
    n = len(elements)
    centroids = np.empty((n, 3))
    
    for i in prange(n):
        e = elements[i]
        centroids[i] = (vertices[e[0]] + vertices[e[1]] + vertices[e[2]] + vertices[e[3]]) / 4.0
    
    return centroids

# JIT warmup
def warmup_jit():
    """Warm up JIT compilation."""
    v = np.random.rand(4, 3).astype(np.float64)
    _ = scaled_jacobian(v[0], v[1], v[2], v[3])
    _ = dihedral_angles_minmax(v[0], v[1], v[2], v[3])
    test_v = np.random.rand(100, 3).astype(np.float64)
    test_e = np.random.randint(0, 100, (50, 4)).astype(np.int32)
    _ = compute_all_quality_parallel(test_v, test_e)
    _ = fix_inverted_elements(test_v, test_e)
    _ = compute_element_centroids(test_v, test_e)
    print("JIT compilation complete.")


# FILE I/O FUNCTIONS
def read_stl(filepath):
    """Read STL file (binary or ASCII)."""
    with open(filepath, 'rb') as f:
        header = f.read(80)
    
    try:
        if header[:5] == b'solid':
            return _read_ascii_stl(filepath)
    except:
        pass
    return _read_binary_stl(filepath)

def _read_binary_stl(filepath):
    """Read binary STL."""
    verts, faces, vmap = [], [], {}
    with open(filepath, 'rb') as f:
        f.read(80)
        n = struct.unpack('<I', f.read(4))[0]
        for _ in range(n):
            f.read(12)
            face = []
            for _ in range(3):
                x, y, z = struct.unpack('<fff', f.read(12))
                key = (round(x, 8), round(y, 8), round(z, 8))
                if key not in vmap:
                    vmap[key] = len(verts)
                    verts.append([x, y, z])
                face.append(vmap[key])
            faces.append(face)
            f.read(2)
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)

def _read_ascii_stl(filepath):
    """Read ASCII STL."""
    verts, faces, vmap, face = [], [], {}, []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('vertex'):
                p = line.split()
                x, y, z = float(p[1]), float(p[2]), float(p[3])
                key = (round(x, 8), round(y, 8), round(z, 8))
                if key not in vmap:
                    vmap[key] = len(verts)
                    verts.append([x, y, z])
                face.append(vmap[key])
            elif line.startswith('endloop'):
                if len(face) == 3:
                    faces.append(face)
                face = []
    return np.array(verts, dtype=np.float64), np.array(faces, dtype=np.int32)

def write_smesh(filepath, verts, faces):
    """Write TetGen .smesh format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(verts)} 3 0 0\n")
        for i, v in enumerate(verts):
            f.write(f"{i+1} {v[0]:.15e} {v[1]:.15e} {v[2]:.15e}\n")
        f.write(f"{len(faces)} 0\n")
        for face in faces:
            f.write(f"3 {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("0\n0\n")

def read_tetgen_output(prefix):
    """Read TetGen output files."""
    vertices = []
    with open(f"{prefix}.1.node") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    n = int(lines[0].split()[0])
    for l in lines[1:n+1]:
        p = l.split()
        vertices.append([float(p[1]), float(p[2]), float(p[3])])
    
    elements = []
    with open(f"{prefix}.1.ele") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
    n = int(lines[0].split()[0])
    for l in lines[1:n+1]:
        p = l.split()
        elements.append([int(p[1]), int(p[2]), int(p[3]), int(p[4])])
    
    vertices = np.array(vertices, dtype=np.float64)
    elements = np.array(elements, dtype=np.int32)
    
    if np.min(elements) > 0:
        elements = elements - 1
    
    return vertices, elements

def read_msh_file(filepath):
    """Read Gmsh/fTetWild .msh file."""
    if HAS_MESHIO:
        try:
            mesh = meshio.read(filepath)
            vertices = mesh.points.astype(np.float64)
            for cell_block in mesh.cells:
                if cell_block.type == "tetra":
                    elements = cell_block.data.astype(np.int32)
                    return vertices, elements
        except Exception as e:
            print(f"    meshio read error: {e}")
    return None, None

def write_pts(filepath, vertices):
    """Write CARP .pts format."""
    with open(filepath, 'w') as f:
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.15e} {v[1]:.15e} {v[2]:.15e}\n")

def write_elem(filepath, elements, tags=None):
    """Write CARP .elem format with tissue tags."""
    if tags is None:
        tags = np.ones(len(elements), dtype=np.int32)
    with open(filepath, 'w') as f:
        f.write(f"{len(elements)}\n")
        for i, e in enumerate(elements):
            f.write(f"Tt {e[0]} {e[1]} {e[2]} {e[3]} {tags[i]}\n")

def write_vtk(filepath, vertices, elements, tags=None, scalars=None):
    """Write VTK unstructured grid format with optional scalars."""
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nTetrahedral mesh\nASCII\n")
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
        
        # Write tags as cell data
        if tags is not None:
            f.write(f"\nCELL_DATA {len(elements)}\n")
            f.write("SCALARS tissue_tag int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for t in tags:
                f.write(f"{t}\n")
        
        # Write additional scalars
        if scalars is not None:
            if tags is None:
                f.write(f"\nCELL_DATA {len(elements)}\n")
            for name, values in scalars.items():
                f.write(f"SCALARS {name} double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for v in values:
                    f.write(f"{v}\n")

def write_surface_vtk(filepath, vertices, faces, markers=None):
    """Write surface mesh as VTK."""
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\nSurface mesh\nASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {len(vertices)} double\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        f.write(f"POLYGONS {len(faces)} {len(faces)*4}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
        
        if markers is not None:
            f.write(f"\nCELL_DATA {len(faces)}\n")
            f.write("SCALARS surface_marker int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for m in markers:
                f.write(f"{m}\n")

def read_mesh_carp(patient_id, base_dir=BASE_DIR):
    """Read existing mesh in CARP format."""
    d = Path(base_dir) / patient_id
    pts = d / f"{patient_id}_tet.pts"
    elem = d / f"{patient_id}_tet.elem"
    
    if not pts.exists():
        return None, None, None
    
    with open(pts) as f:
        lines = f.readlines()
    n = int(lines[0])
    v = np.zeros((n, 3), dtype=np.float64)
    for i, l in enumerate(lines[1:n+1]):
        p = l.split()
        v[i] = [float(p[0]), float(p[1]), float(p[2])]
    
    with open(elem) as f:
        lines = f.readlines()
    n = int(lines[0])
    e = np.zeros((n, 4), dtype=np.int32)
    tags = np.zeros(n, dtype=np.int32)
    for i, l in enumerate(lines[1:n+1]):
        p = l.split()
        e[i] = [int(p[1]), int(p[2]), int(p[3]), int(p[4])]
        tags[i] = int(p[5]) if len(p) > 5 else 1
    
    return v, e, tags

# SURFACE MESH REPAIR
def repair_surface_mesh_pymeshlab(stl_input, stl_output):
    """Repair surface mesh using PyMeshLab."""
    if not HAS_PYMESHLAB:
        import shutil
        shutil.copy(stl_input, stl_output)
        return {'repaired': False}
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(stl_input)
    
    initial_v = ms.current_mesh().vertex_number()
    initial_f = ms.current_mesh().face_number()
    
    def try_filter(name, **kwargs):
        variants = {
            'remove_dup_v': ['meshing_remove_duplicate_vertices', 'remove_duplicate_vertices'],
            'remove_dup_f': ['meshing_remove_duplicate_faces', 'remove_duplicate_faces'],
            'remove_null': ['meshing_remove_null_faces', 'remove_zero_area_faces'],
            'remove_unref': ['meshing_remove_unreferenced_vertices', 'remove_unreferenced_vertices'],
            'repair_nm_e': ['meshing_repair_non_manifold_edges', 'repair_non_manifold_edges'],
            'repair_nm_v': ['meshing_repair_non_manifold_vertices', 'repair_non_manifold_vertices'],
            'close_holes': ['meshing_close_holes', 'close_holes'],
            'reorient': ['meshing_re_orient_faces_coherently', 're_orient_all_faces_coherently'],
            'invert': ['meshing_invert_face_orientation', 'invert_faces_orientation'],
        }
        for var in variants.get(name, [name]):
            try:
                method = getattr(ms, var, None)
                if method:
                    method(**kwargs)
                    return True
            except:
                continue
        return False
    
    # Cleanup iterations
    for _ in range(3):
        try_filter('remove_dup_v')
        try_filter('remove_dup_f')
        try_filter('remove_null')
        try_filter('remove_unref')
    
    # Non-manifold repair
    for _ in range(20):
        try_filter('repair_nm_e')
        try_filter('repair_nm_v')
    
    # Close holes
    for hole_size in [10, 50, 100, 500, 1000, 5000, 10000]:
        try:
            try_filter('close_holes', maxholesize=hole_size)
        except:
            pass
    
    # Reorient
    try_filter('reorient')
    
    # Check volume orientation
    m = ms.current_mesh()
    verts = m.vertex_matrix()
    faces = m.face_matrix()
    
    vol = 0.0
    for f in faces:
        v0, v1, v2 = verts[f[0]], verts[f[1]], verts[f[2]]
        vol += np.dot(v0, np.cross(v1, v2)) / 6.0
    
    if vol < 0:
        try_filter('invert')
    
    try_filter('remove_dup_v')
    try_filter('remove_unref')
    
    ms.save_current_mesh(stl_output)
    
    return {
        'repaired': True,
        'initial': (initial_v, initial_f),
        'final': (ms.current_mesh().vertex_number(), ms.current_mesh().face_number())
    }

# HIGH-RESOLUTION MESH GENERATION
def mesh_with_ftetwild(stl_path, output_msh, target_elements, bbox_diagonal):
    """
    Generate high-resolution mesh using fTetWild.
    
    Adjusts parameters to achieve target element count.
    """
    if not HAS_FTETWILD:
        return None, "fTetWild not available"
    
    # Calculate epsilon based on target elements
    # Smaller epsilon = finer mesh
    # Rough estimate: elements ~= (bbox_diag / epsilon)^3
    estimated_epsilon = bbox_diagonal / (target_elements ** (1/3))
    epsilon = max(1e-5, min(0.01, estimated_epsilon * 0.5))
    
    # Quality parameters for high-resolution mesh
    params_list = [
        {'stop_quality': 30, 'max_its': 200, 'epsilon': epsilon * 0.5},
        {'stop_quality': 25, 'max_its': 150, 'epsilon': epsilon},
        {'stop_quality': 20, 'max_its': 100, 'epsilon': epsilon * 1.5},
        {'stop_quality': 15, 'max_its': 80, 'epsilon': epsilon * 2},
    ]
    
    for params in params_list:
        try:
            success = wm.tetrahedralize(
                input=stl_path,
                output=output_msh,
                stop_quality=params['stop_quality'],
                max_its=params['max_its'],
                epsilon=params['epsilon'],
                mute_log=True,
                coarsen=False,  # Disable coarsening for high resolution
                smooth_open_boundary=True,
                max_threads=NUM_CPUS,
            )
            
            if success and os.path.exists(output_msh):
                vertices, elements = read_msh_file(output_msh)
                if vertices is not None and len(elements) >= target_elements * 0.5:
                    return (vertices, elements), None
                    
        except Exception as e:
            continue
    
    return None, "fTetWild failed to achieve target resolution"

def mesh_with_tetgen(smesh_path, target_elements, bbox_volume, timeout=1200):
    """
    Generate high-resolution mesh using TetGen.
    
    Uses volume constraint to achieve target element count.
    """
    prefix = smesh_path.rsplit('.', 1)[0]
    
    # Calculate maximum volume per element
    max_vol = bbox_volume / (target_elements * 6)
    
    # Try progressively relaxed parameters
    params_list = [
        {'quality': 1.2, 'min_dih': 18, 'opt': 10},
        {'quality': 1.4, 'min_dih': 15, 'opt': 8},
        {'quality': 1.6, 'min_dih': 12, 'opt': 6},
        {'quality': 2.0, 'min_dih': 10, 'opt': 4},
        {'quality': 2.5, 'min_dih': 8, 'opt': 3},
    ]
    
    for params in params_list:
        cmd = f"tetgen -pq{params['quality']:.1f}/{params['min_dih']}O{params['opt']}a{max_vol:.10e} {smesh_path}"
        
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True,
                                   timeout=timeout, text=True)
            
            if os.path.exists(f"{prefix}.1.node"):
                vertices, elements = read_tetgen_output(prefix)
                if len(elements) >= target_elements * 0.3:
                    return (vertices, elements), None
                    
        except subprocess.TimeoutExpired:
            continue
        except Exception as e:
            continue
    
    return None, "TetGen failed"

# TISSUE TAGGING AND WALL DETECTION
def detect_boundary_faces(elements):
    """
    Detect boundary faces using topology.
    
    A face is on the boundary if it belongs to only one element.
    """
    face_count = defaultdict(list)
    
    for ei, e in enumerate(elements):
        # Four faces of tetrahedron (sorted vertices for consistent keys)
        faces = [
            tuple(sorted([e[0], e[1], e[2]])),
            tuple(sorted([e[0], e[1], e[3]])),
            tuple(sorted([e[0], e[2], e[3]])),
            tuple(sorted([e[1], e[2], e[3]])),
        ]
        for f in faces:
            face_count[f].append(ei)
    
    boundary_faces = []
    boundary_elements = set()
    
    for face, elem_list in face_count.items():
        if len(elem_list) == 1:
            boundary_faces.append(face)
            boundary_elements.add(elem_list[0])
    
    return boundary_faces, boundary_elements

def detect_endocardium_epicardium(vertices, elements, boundary_faces):
    """
    Classify boundary nodes as endocardial (inner) or epicardial (outer).
    
    Uses centroid-based radial distance:
    - Nodes farther from centroid = epicardium
    - Nodes closer to centroid = endocardium
    """
    # Get boundary vertex indices
    boundary_vertices = set()
    for face in boundary_faces:
        for vi in face:
            boundary_vertices.add(vi)
    
    boundary_vertices = np.array(list(boundary_vertices))
    
    if len(boundary_vertices) == 0:
        return np.array([]), np.array([])
    
    # Compute centroid
    centroid = np.mean(vertices, axis=0)
    
    # Compute radial distances for boundary vertices
    radial_distances = np.linalg.norm(vertices[boundary_vertices] - centroid, axis=1)
    
    # Classify using percentiles
    # Lower 40% = endocardium, Upper 40% = epicardium
    endo_threshold = np.percentile(radial_distances, 40)
    epi_threshold = np.percentile(radial_distances, 60)
    
    endo_mask = radial_distances <= endo_threshold
    epi_mask = radial_distances >= epi_threshold
    
    endo_nodes = boundary_vertices[endo_mask]
    epi_nodes = boundary_vertices[epi_mask]
    
    return endo_nodes, epi_nodes

def compute_transmural_depth(vertices, elements, endo_nodes, epi_nodes):
    """
    Compute transmural depth for each element.
    
    Transmural depth ranges from 0 (endocardium) to 1 (epicardium).
    """
    n_elements = len(elements)
    transmural_depth = np.zeros(n_elements)
    
    if len(endo_nodes) == 0 or len(epi_nodes) == 0:
        # Fallback to radial-based estimation
        centroid = np.mean(vertices, axis=0)
        centroids = compute_element_centroids(vertices, elements)
        radial_distances = np.linalg.norm(centroids - centroid, axis=1)
        
        min_dist = np.percentile(radial_distances, 5)
        max_dist = np.percentile(radial_distances, 95)
        
        transmural_depth = (radial_distances - min_dist) / (max_dist - min_dist + 1e-10)
        transmural_depth = np.clip(transmural_depth, 0, 1)
        
        return transmural_depth
    
    # Compute mean positions
    endo_centroid = np.mean(vertices[endo_nodes], axis=0)
    epi_centroid = np.mean(vertices[epi_nodes], axis=0)
    
    # Compute element centroids
    centroids = compute_element_centroids(vertices, elements)
    
    # For each element, compute distance to endo and epi surfaces
    for i in range(n_elements):
        c = centroids[i]
        
        # Distance to endocardial surface (approximate)
        dist_endo = np.min(np.linalg.norm(vertices[endo_nodes] - c, axis=1))
        
        # Distance to epicardial surface (approximate)
        dist_epi = np.min(np.linalg.norm(vertices[epi_nodes] - c, axis=1))
        
        # Transmural depth: 0 at endo, 1 at epi
        total_dist = dist_endo + dist_epi
        if total_dist > 1e-10:
            transmural_depth[i] = dist_endo / total_dist
        else:
            transmural_depth[i] = 0.5
    
    return transmural_depth

def detect_base_plane(vertices, elements):
    """
    Detect the base plane of the LV using PCA.
    
    The base is typically at one end of the long axis.
    """
    # Compute long axis via PCA
    centroid = np.mean(vertices, axis=0)
    centered = vertices - centroid
    
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    long_axis = eigvecs[:, np.argmax(eigvals)]
    
    # Project vertices onto long axis
    projections = np.dot(centered, long_axis)
    
    # Base is at maximum projection (opposite to apex)
    base_threshold = np.percentile(projections, 90)
    
    # Get boundary faces and find those near the base
    boundary_faces, _ = detect_boundary_faces(elements)
    
    base_nodes = set()
    for vi in range(len(vertices)):
        if np.dot(vertices[vi] - centroid, long_axis) > base_threshold:
            base_nodes.add(vi)
    
    return np.array(list(base_nodes)), long_axis

def generate_tissue_tags(vertices, elements, transmural_depth):
    """
    Generate tissue tags based on transmural depth and geometry.
    
    Tags:
    - 1: Healthy myocardium (transmural depth 0-0.3 or 0.7-1.0)
    - 2: Border zone (transmural depth 0.3-0.7)
    - 3: Infarct core (optional, based on additional criteria)
    """
    n_elements = len(elements)
    tags = np.ones(n_elements, dtype=np.int32)
    
    # Border zone: mid-wall region
    border_zone_mask = (transmural_depth >= 0.3) & (transmural_depth <= 0.7)
    tags[border_zone_mask] = TISSUE_TAGS['border_zone']
    
    # For now, mark a small region as potential infarct (can be customized)
    # This is a placeholder - actual infarct detection requires additional data
    # infarct_mask = (transmural_depth >= 0.4) & (transmural_depth <= 0.6)
    # tags[infarct_mask] = TISSUE_TAGS['infarct']
    
    return tags

# QUALITY OPTIMIZATION
def laplacian_smooth(vertices, elements, n_iterations=10, lambda_factor=0.3):
    """Laplacian smoothing for internal vertices."""
    vertices = vertices.copy()
    n_verts = len(vertices)
    
    # Build vertex adjacency
    vert_neighbors = [set() for _ in range(n_verts)]
    vert_elements = [[] for _ in range(n_verts)]
    
    for ei, e in enumerate(elements):
        for vi in e:
            vert_elements[vi].append(ei)
            for vj in e:
                if vi != vj:
                    vert_neighbors[vi].add(vj)
    
    # Find boundary vertices
    boundary_faces, _ = detect_boundary_faces(elements)
    boundary_verts = set()
    for face in boundary_faces:
        for vi in face:
            boundary_verts.add(vi)
    
    # Smoothing iterations
    for _ in range(n_iterations):
        new_positions = vertices.copy()
        
        for vi in range(n_verts):
            if vi in boundary_verts:
                continue
            
            neighbors = list(vert_neighbors[vi])
            if len(neighbors) == 0:
                continue
            
            centroid = np.mean(vertices[neighbors], axis=0)
            new_pos = vertices[vi] + lambda_factor * (centroid - vertices[vi])
            
            # Check if move creates inversions
            valid = True
            for ei in vert_elements[vi]:
                e = elements[ei]
                v0 = new_pos if e[0] == vi else vertices[e[0]]
                v1 = new_pos if e[1] == vi else vertices[e[1]]
                v2 = new_pos if e[2] == vi else vertices[e[2]]
                v3 = new_pos if e[3] == vi else vertices[e[3]]
                
                if tet_signed_volume(v0, v1, v2, v3) <= 1e-20:
                    valid = False
                    break
            
            if valid:
                new_positions[vi] = new_pos
        
        vertices = new_positions
    
    return vertices

def optimize_mesh_quality(vertices, elements, max_iterations=5):
    """Optimize mesh quality through smoothing and element fixing."""
    print("    Optimizing mesh quality...")
    
    # Fix inverted elements
    vols, _, _, _, _, _ = compute_all_quality_parallel(vertices, elements)
    n_inv_before = np.sum(vols < 0)
    
    if n_inv_before > 0:
        print(f"      Fixing {n_inv_before} inverted elements...")
        elements = fix_inverted_elements(vertices, elements)
    
    # Remove degenerate elements
    vols, jacs, _, _, _, _ = compute_all_quality_parallel(vertices, elements)
    keep = (vols > 1e-20) & (np.abs(jacs) > 1e-10)
    n_removed = np.sum(~keep)
    
    if n_removed > 0:
        print(f"      Removing {n_removed} degenerate elements...")
        elements = elements[keep]
    
    # Laplacian smoothing
    vertices = laplacian_smooth(vertices, elements, n_iterations=15, lambda_factor=0.4)
    
    # Final inversion check
    elements = fix_inverted_elements(vertices, elements)
    
    return vertices, elements

def clean_unreferenced_vertices(vertices, elements):
    """Remove vertices not used by any element."""
    used = set()
    for e in elements:
        for vi in e:
            used.add(vi)
    
    old_to_new = {}
    new_verts = []
    for old_idx in sorted(used):
        old_to_new[old_idx] = len(new_verts)
        new_verts.append(vertices[old_idx])
    
    new_elems = np.zeros_like(elements)
    for i, e in enumerate(elements):
        for j in range(4):
            new_elems[i, j] = old_to_new[e[j]]
    
    return np.array(new_verts, dtype=np.float64), new_elems

# QUALITY EVALUATION
@dataclass
class MeshQualityReport:
    """Comprehensive mesh quality report."""
    patient_id: str = ""
    n_vertices: int = 0
    n_elements: int = 0
    n_inverted: int = 0
    
    # Jacobian metrics
    min_jacobian: float = 0.0
    max_jacobian: float = 0.0
    mean_jacobian: float = 0.0
    std_jacobian: float = 0.0
    pct_jacobian_below_01: float = 0.0
    
    # Dihedral angle metrics
    min_dihedral: float = 0.0
    max_dihedral: float = 0.0
    mean_min_dihedral: float = 0.0
    mean_max_dihedral: float = 0.0
    pct_dihedral_below_5: float = 0.0
    pct_dihedral_above_170: float = 0.0
    
    # Aspect ratio and radius-edge
    max_aspect_ratio: float = 0.0
    mean_aspect_ratio: float = 0.0
    max_radius_edge: float = 0.0
    mean_radius_edge: float = 0.0
    
    # Edge lengths
    min_edge_mm: float = 0.0
    max_edge_mm: float = 0.0
    mean_edge_mm: float = 0.0
    
    # Volume metrics
    total_volume_mm3: float = 0.0
    min_volume_mm3: float = 0.0
    max_volume_mm3: float = 0.0
    mean_volume_mm3: float = 0.0
    elements_per_mm3: float = 0.0
    
    # Bounding box
    bbox_x_mm: float = 0.0
    bbox_y_mm: float = 0.0
    bbox_z_mm: float = 0.0
    bbox_diagonal_mm: float = 0.0
    
    # Tissue tagging
    n_healthy: int = 0
    n_border_zone: int = 0
    n_infarct: int = 0
    
    # Surface detection
    n_endo_nodes: int = 0
    n_epi_nodes: int = 0
    n_base_nodes: int = 0
    
    # Simulation readiness
    opencarp_ready: bool = False
    febio_ready: bool = False
    
    # Method used
    method: str = ""
    runtime_sec: float = 0.0
    warnings: str = ""
    
    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

def evaluate_mesh_quality(vertices, elements, tags=None, patient_id=""):
    """Compute comprehensive quality metrics."""
    report = MeshQualityReport(patient_id=patient_id)
    
    report.n_vertices = len(vertices)
    report.n_elements = len(elements)
    
    # Compute all metrics
    vols, jacs, min_dihs, max_dihs, re, ar = compute_all_quality_parallel(vertices, elements)
    
    # Inverted elements
    report.n_inverted = int(np.sum(vols < 0))
    
    # Jacobian metrics
    report.min_jacobian = float(np.min(jacs))
    report.max_jacobian = float(np.max(jacs))
    report.mean_jacobian = float(np.mean(jacs))
    report.std_jacobian = float(np.std(jacs))
    report.pct_jacobian_below_01 = float(np.sum(jacs < 0.1) / len(jacs) * 100)
    
    # Dihedral angles
    report.min_dihedral = float(np.min(min_dihs))
    report.max_dihedral = float(np.max(max_dihs))
    report.mean_min_dihedral = float(np.mean(min_dihs))
    report.mean_max_dihedral = float(np.mean(max_dihs))
    report.pct_dihedral_below_5 = float(np.sum(min_dihs < 5) / len(min_dihs) * 100)
    report.pct_dihedral_above_170 = float(np.sum(max_dihs > 170) / len(max_dihs) * 100)
    
    # Aspect ratio and radius-edge (filter infinities)
    valid_ar = ar[np.isfinite(ar)]
    valid_re = re[np.isfinite(re)]
    
    report.max_aspect_ratio = float(np.max(valid_ar)) if len(valid_ar) > 0 else 1e10
    report.mean_aspect_ratio = float(np.mean(valid_ar)) if len(valid_ar) > 0 else 1e10
    report.max_radius_edge = float(np.max(valid_re)) if len(valid_re) > 0 else 1e10
    report.mean_radius_edge = float(np.mean(valid_re)) if len(valid_re) > 0 else 1e10
    
    # Compute edge lengths
    all_edges = []
    for e in elements[:min(10000, len(elements))]:  # Sample for speed
        v0, v1, v2, v3 = vertices[e[0]], vertices[e[1]], vertices[e[2]], vertices[e[3]]
        edges = tet_edge_lengths(v0, v1, v2, v3)
        all_edges.extend(edges)
    all_edges = np.array(all_edges)
    
    report.min_edge_mm = float(np.min(all_edges))
    report.max_edge_mm = float(np.max(all_edges))
    report.mean_edge_mm = float(np.mean(all_edges))
    
    # Volume metrics
    valid_vols = np.abs(vols)
    report.total_volume_mm3 = float(np.sum(valid_vols))
    report.min_volume_mm3 = float(np.min(valid_vols))
    report.max_volume_mm3 = float(np.max(valid_vols))
    report.mean_volume_mm3 = float(np.mean(valid_vols))
    report.elements_per_mm3 = float(len(elements) / report.total_volume_mm3) if report.total_volume_mm3 > 0 else 0
    
    # Bounding box
    bbox_min = np.min(vertices, axis=0)
    bbox_max = np.max(vertices, axis=0)
    bbox_size = bbox_max - bbox_min
    
    report.bbox_x_mm = float(bbox_size[0])
    report.bbox_y_mm = float(bbox_size[1])
    report.bbox_z_mm = float(bbox_size[2])
    report.bbox_diagonal_mm = float(np.linalg.norm(bbox_size))
    
    # Tissue tagging counts
    if tags is not None:
        report.n_healthy = int(np.sum(tags == TISSUE_TAGS['healthy']))
        report.n_border_zone = int(np.sum(tags == TISSUE_TAGS['border_zone']))
        report.n_infarct = int(np.sum(tags == TISSUE_TAGS['infarct']))
    
    # Check simulation readiness
    report.opencarp_ready = (
        report.n_inverted == 0 and
        report.min_jacobian > QUALITY_THRESHOLDS_OPENCARP['min_jacobian'] and
        report.min_dihedral > QUALITY_THRESHOLDS_OPENCARP['min_dihedral'] and
        report.max_dihedral < QUALITY_THRESHOLDS_OPENCARP['max_dihedral']
    )
    
    report.febio_ready = (
        report.n_inverted == 0 and
        report.min_jacobian > QUALITY_THRESHOLDS_FEBIO['min_jacobian'] and
        report.min_dihedral > QUALITY_THRESHOLDS_FEBIO['min_dihedral'] and
        report.max_dihedral < QUALITY_THRESHOLDS_FEBIO['max_dihedral'] and
        report.max_aspect_ratio < QUALITY_THRESHOLDS_FEBIO['max_aspect_ratio']
    )
    
    # Generate warnings
    warnings_list = []
    if report.n_elements < TARGET_ELEMENTS['minimum']:
        warnings_list.append(f"LOW RESOLUTION: {report.n_elements} elements (recommend {TARGET_ELEMENTS['minimum']}+)")
    if report.max_aspect_ratio > 50:
        warnings_list.append(f"High aspect ratios ({report.max_aspect_ratio:.1f})")
    if report.n_inverted > 0:
        warnings_list.append(f"INVERTED ELEMENTS: {report.n_inverted}")
    
    report.warnings = "; ".join(warnings_list)
    
    return report

# MAIN PROCESSING PIPELINE
def process_patient(patient_id, stl_dir=STL_DIR, output_dir=OUTPUT_DIR, 
                   target_elements=TARGET_ELEMENTS['standard']):
    """
    Process a single patient through the complete pipeline.
    
    1. Load and repair STL surface
    2. Generate high-resolution tetrahedral mesh
    3. Detect boundaries and compute transmural depth
    4. Generate tissue tags
    5. Optimize mesh quality
    6. Save all outputs
    """
    print(f"PROCESSING: {patient_id}")
    
    start_time = time.time()
    
    # Create output directory
    out_dir = Path(output_dir) / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    stl_path = f"{stl_dir}/{patient_id}.stl"
    
    if not os.path.exists(stl_path):
        print(f"  ERROR: STL not found: {stl_path}")
        return None
    
    best_result = None
    best_score = -1e20
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Stage 1: Surface repair
        print("  Stage 1: Surface mesh repair...")
        repaired_stl = f"{tmpdir}/{patient_id}_repaired.stl"
        repair_info = repair_surface_mesh_pymeshlab(stl_path, repaired_stl)
        
        # Get bounding box for parameter estimation
        if HAS_PYMESHLAB:
            ms = pymeshlab.MeshSet()
            ms.load_new_mesh(repaired_stl)
            verts = ms.current_mesh().vertex_matrix()
        else:
            verts, _ = read_stl(repaired_stl)
        
        bbox_min = np.min(verts, axis=0)
        bbox_max = np.max(verts, axis=0)
        bbox_size = bbox_max - bbox_min
        bbox_diagonal = np.linalg.norm(bbox_size)
        bbox_volume = np.prod(bbox_size)
        
        print(f"    Bounding box: {bbox_size[0]:.2f} x {bbox_size[1]:.2f} x {bbox_size[2]:.2f} mm")
        print(f"    Target elements: {target_elements}")
        
        # Stage 2: High-resolution mesh generation
        print("  Stage 2: High-resolution mesh generation...")
        
        # Try fTetWild first
        if HAS_FTETWILD and HAS_MESHIO:
            print("    Trying fTetWild...")
            output_msh = f"{tmpdir}/{patient_id}_ftet.msh"
            
            result, error = mesh_with_ftetwild(repaired_stl, output_msh, target_elements, bbox_diagonal)
            
            if result is not None:
                vertices, elements = result
                print(f"    fTetWild: {len(elements)} elements generated")
                
                # Optimize quality
                vertices, elements = optimize_mesh_quality(vertices, elements)
                vertices, elements = clean_unreferenced_vertices(vertices, elements)
                
                # Evaluate
                vols, jacs, _, _, _, _ = compute_all_quality_parallel(vertices, elements)
                n_inv = np.sum(vols < 0)
                min_jac = np.min(jacs)
                
                score = len(elements) * 10 + min_jac * 10000 - n_inv * 1000000
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'vertices': vertices.copy(),
                        'elements': elements.copy(),
                        'method': 'fTetWild'
                    }
        
        # Try TetGen if needed
        if best_result is None or len(best_result['elements']) < target_elements * 0.5:
            print("    Trying TetGen...")
            
            # Prepare smesh file
            if HAS_PYMESHLAB:
                ms = pymeshlab.MeshSet()
                ms.load_new_mesh(repaired_stl)
                verts = ms.current_mesh().vertex_matrix().astype(np.float64)
                faces = ms.current_mesh().face_matrix().astype(np.int32)
            else:
                verts, faces = read_stl(repaired_stl)
            
            smesh_path = f"{tmpdir}/{patient_id}.smesh"
            write_smesh(smesh_path, verts, faces)
            
            result, error = mesh_with_tetgen(smesh_path, target_elements, bbox_volume)
            
            if result is not None:
                vertices, elements = result
                print(f"    TetGen: {len(elements)} elements generated")
                
                # Optimize quality
                vertices, elements = optimize_mesh_quality(vertices, elements)
                vertices, elements = clean_unreferenced_vertices(vertices, elements)
                
                # Evaluate
                vols, jacs, _, _, _, _ = compute_all_quality_parallel(vertices, elements)
                n_inv = np.sum(vols < 0)
                min_jac = np.min(jacs)
                
                score = len(elements) * 10 + min_jac * 10000 - n_inv * 1000000
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        'vertices': vertices.copy(),
                        'elements': elements.copy(),
                        'method': 'TetGen'
                    }
    
    if best_result is None:
        print("  FAILED: Could not generate mesh")
        return None
    
    vertices = best_result['vertices']
    elements = best_result['elements']
    method = best_result['method']
    
    print(f"  Best method: {method} with {len(elements)} elements")
    
    # Stage 3: Boundary detection and transmural depth
    print("  Stage 3: Boundary detection and tissue tagging...")
    
    boundary_faces, boundary_elements = detect_boundary_faces(elements)
    endo_nodes, epi_nodes = detect_endocardium_epicardium(vertices, elements, boundary_faces)
    base_nodes, long_axis = detect_base_plane(vertices, elements)
    
    print(f"    Endocardial nodes: {len(endo_nodes)}")
    print(f"    Epicardial nodes: {len(epi_nodes)}")
    print(f"    Base nodes: {len(base_nodes)}")
    
    # Compute transmural depth
    transmural_depth = compute_transmural_depth(vertices, elements, endo_nodes, epi_nodes)
    
    # Generate tissue tags
    tags = generate_tissue_tags(vertices, elements, transmural_depth)
    
    n_healthy = np.sum(tags == TISSUE_TAGS['healthy'])
    n_border = np.sum(tags == TISSUE_TAGS['border_zone'])
    n_infarct = np.sum(tags == TISSUE_TAGS['infarct'])
    
    print(f"    Healthy elements: {n_healthy}")
    print(f"    Border zone elements: {n_border}")
    print(f"    Infarct elements: {n_infarct}")
    
    # Stage 4: Quality evaluation
    print("  Stage 4: Quality evaluation...")
    
    report = evaluate_mesh_quality(vertices, elements, tags, patient_id)
    report.method = method
    report.runtime_sec = time.time() - start_time
    report.n_endo_nodes = len(endo_nodes)
    report.n_epi_nodes = len(epi_nodes)
    report.n_base_nodes = len(base_nodes)
    
    print(f"    Elements: {report.n_elements}")
    print(f"    Inverted: {report.n_inverted}")
    print(f"    Min Jacobian: {report.min_jacobian:.6f}")
    print(f"    Dihedral range: [{report.min_dihedral:.2f}, {report.max_dihedral:.2f}] deg")
    print(f"    Max Aspect Ratio: {report.max_aspect_ratio:.2f}")
    print(f"    OpenCarp Ready: {'YES' if report.opencarp_ready else 'NO'}")
    print(f"    FEBio Ready: {'YES' if report.febio_ready else 'NO'}")
    
    # Stage 5: Save outputs
    print("  Stage 5: Saving outputs...")
    
    # CARP format
    write_pts(str(out_dir / f"{patient_id}_tet.pts"), vertices)
    write_elem(str(out_dir / f"{patient_id}_tet.elem"), elements, tags)
    
    # VTK with quality scalars
    scalars = {
        'transmural_depth': transmural_depth,
    }
    write_vtk(str(out_dir / f"{patient_id}_tet.vtk"), vertices, elements, tags, scalars)
    
    # Save surface markers
    # Create surface mesh from boundary faces
    boundary_face_array = np.array(list(boundary_faces))
    surface_markers = np.ones(len(boundary_face_array), dtype=np.int32) * SURFACE_MARKERS['epicardium']
    
    # Mark endocardial faces
    endo_set = set(endo_nodes)
    for i, face in enumerate(boundary_face_array):
        if all(vi in endo_set for vi in face):
            surface_markers[i] = SURFACE_MARKERS['endocardium']
    
    # Mark base faces
    base_set = set(base_nodes)
    for i, face in enumerate(boundary_face_array):
        if all(vi in base_set for vi in face):
            surface_markers[i] = SURFACE_MARKERS['base']
    
    write_surface_vtk(str(out_dir / f"{patient_id}_surface.vtk"), vertices, boundary_face_array, surface_markers)
    
    # Save node lists for OpenCarp
    np.savetxt(str(out_dir / f"{patient_id}_endo.vtx"), endo_nodes, fmt='%d', 
               header=f"{len(endo_nodes)}\nintra", comments='')
    np.savetxt(str(out_dir / f"{patient_id}_epi.vtx"), epi_nodes, fmt='%d',
               header=f"{len(epi_nodes)}\nintra", comments='')
    np.savetxt(str(out_dir / f"{patient_id}_base.vtx"), base_nodes, fmt='%d',
               header=f"{len(base_nodes)}\nintra", comments='')
    
    # Save transmural depth
    np.savetxt(str(out_dir / f"{patient_id}_transmural.dat"), transmural_depth, fmt='%.10f')
    
    # Save long axis
    np.savetxt(str(out_dir / f"{patient_id}_long_axis.dat"), long_axis, fmt='%.10f')
    
    print(f"  Completed in {report.runtime_sec:.2f} seconds")
    
    return report

# MAIN EXECUTION

def run_all_patients(patients=ALL_PATIENTS, output_dir=OUTPUT_DIR, target_elements=TARGET_ELEMENTS['standard']):
    """Process all patients sequentially."""
    print("COMPREHENSIVE HIGH-RESOLUTION MESH GENERATION PIPELINE")
    print(f"Patients: {len(patients)}")
    print(f"Target elements: {target_elements}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Warmup JIT
    warmup_jit()
    
    results = []
    total_start = time.time()
    
    for i, patient_id in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}]", end="")
        report = process_patient(patient_id, output_dir=output_dir, target_elements=target_elements)
        if report:
            results.append(report)
    
    total_time = time.time() - total_start
    
    # Save summary CSV
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = f"{output_dir}/mesh_quality_summary_{timestamp}.csv"
    
    if results:
        with open(csv_path, 'w', newline='') as f:
            fieldnames = list(results[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_dict())
        
        print(f"\nSummary saved to: {csv_path}")
    
    # Print final summary
    print("FINAL RESULTS")
    print(f"{'Patient':<14} {'Elements':<12} {'Inv':<6} {'MinJac':<10} {'MinDih':<10} "
          f"{'MaxDih':<10} {'MaxAR':<10} {'OC':<6} {'FB':<6} {'Method':<10}")
    
    n_oc_ready = 0
    n_fb_ready = 0
    
    for r in results:
        oc = "YES" if r.opencarp_ready else "NO"
        fb = "YES" if r.febio_ready else "NO"
        
        if r.opencarp_ready:
            n_oc_ready += 1
        if r.febio_ready:
            n_fb_ready += 1
        
        print(f"{r.patient_id:<14} {r.n_elements:<12} {r.n_inverted:<6} "
              f"{r.min_jacobian:<10.4f} {r.min_dihedral:<10.2f} "
              f"{r.max_dihedral:<10.2f} {r.max_aspect_ratio:<10.2f} "
              f"{oc:<6} {fb:<6} {r.method:<10}")
    
    print(f"OpenCarp Ready: {n_oc_ready}/{len(results)}")
    print(f"FEBio Ready: {n_fb_ready}/{len(results)}")
    print(f"Total time: {total_time:.1f}s ({total_time/len(patients):.1f}s per patient)")
    
    return results

if __name__ == "__main__":
    import sys
    
    try:
        get_ipython()
        IN_JUPYTER = True
    except NameError:
        IN_JUPYTER = False
    
    if IN_JUPYTER:
        # Running in Jupyter - use defaults or modify these directly
        print("Running in Jupyter notebook mode")
        print("To customize parameters, edit the values below:\n")
        
        # CUSTOMIZE THESE PARAMETERS FOR JUPYTER EXECUTION:
        jupyter_stl_dir = STL_DIR  # Use default from config
        jupyter_output_dir = OUTPUT_DIR  # Use default from config
        jupyter_target_elements = TARGET_ELEMENTS['standard']  # 100,000 elements
        jupyter_patients = ["SCD0000501", "SCD0000601", "SCD0000701", "SCD0000801", 
                        "SCD0001001", "SCD0001101", "SCD0001201"]  # ← CHANGED - processing only the remaining patients after previous generation attempt cut off due to CPU usage.
        
     
        
        print(f"STL Directory: {jupyter_stl_dir}")
        print(f"Output Directory: {jupyter_output_dir}")
        print(f"Target Elements: {jupyter_target_elements}")
        print(f"Patients: {len(jupyter_patients)} patients")
        print("\nStarting processing...\n")
        
        results = run_all_patients(
            patients=jupyter_patients,
            output_dir=jupyter_output_dir,
            target_elements=jupyter_target_elements
        )
        
    else:
        # Running from command line - use argparse
        import argparse
        
        parser = argparse.ArgumentParser(description='High-Resolution Mesh Generation Pipeline')
        parser.add_argument('--stl-dir', type=str, default=STL_DIR, help='Directory containing STL files')
        parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR, help='Output directory')
        parser.add_argument('--target-elements', type=int, default=TARGET_ELEMENTS['standard'], 
                           help='Target number of elements')
        parser.add_argument('--patients', type=str, nargs='+', default=ALL_PATIENTS,
                           help='Patient IDs to process')
        
        args = parser.parse_args()
        
        results = run_all_patients(
            patients=args.patients,
            output_dir=args.output_dir,
            target_elements=args.target_elements
        )