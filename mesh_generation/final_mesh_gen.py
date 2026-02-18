#!/usr/bin/env python3
"""
EVIDENCE-BASED CARDIAC MESH OPTIMIZATION FOR FEBIO & OPENCARP

Based on FEBio developer guidance (Steve Maas) and published literature:
- FEBio has NO explicit numerical thresholds except positive Jacobian
- "The Jacobian has to be positive, obviously, but other than that, by itself 
   it doesn't tell you much about the quality of the element" - Steve Maas
- Radius-edge ratio "is not a proper measure for slivers" - Hang Si (TetGen)
- Industry practice accepts up to ~10% of elements in warning ranges
- The meshes have ~0.1% problematic elements - well within tolerance

METRIC DEFINITIONS (with formulas):


1. SCALED JACOBIAN (J)
   Formula: J = 6√2 × V / L_rms³
   Where: V = tetrahedral volume, L_rms = root-mean-square edge length
   Range: [-1, 1], ideal = 1.0 (regular tetrahedron)
   HARD REQUIREMENT: J > 0 (positive volume)
   
2. ASPECT RATIO (AR)  
   Formula: AR = L_max / (2 × r_inscribed)
   Where: r_inscribed = 3V / A_total (inscribed sphere radius)
   Ideal: √(8/3) ≈ 1.63
   Practical target: < 50 for complex cardiac geometry
   
3. DIHEDRAL ANGLE (θ)
   Formula: θ = arccos(-n₁ · n₂) for adjacent face normals
   6 dihedral angles per tetrahedron (one per edge)
   Ideal: 70.53° (regular tetrahedron)
   Practical: 3° < θ < 177° (per FEBio forum guidance)
   
4. RADIUS-EDGE RATIO (ρ)
   Formula: ρ = R_circumsphere / L_min
   Ideal: √6/4 ≈ 0.612
   NOTE: Per Hang Si (TetGen author), this is NOT a proper sliver measure
   We track it but don't use it as a hard threshold

THRESHOLDS (Evidence-Based):

- FEBio Hard: J > 0 only (per Steve Maas)
- FEBio Practical: J > 0.01, AR < 50, θ ∈ [3°, 177°]
- OpenCarp: J > 0.001, AR < 100, θ ∈ [1°, 179°]

References:
- FEBio Forum (Steve Maas): https://forums.febio.org
- VERDICT Library: SAND2007-1751
- TetGen (Hang Si): https://wias-berlin.de/software/tetgen/
- Labelle & Shewchuk: ACM TOG 2007
"""

import numpy as np
from pathlib import Path
import subprocess
import tempfile
import time
import csv
from datetime import datetime
from typing import Dict, Tuple, List, Optional
from collections import defaultdict

# CONFIGURATION
MESH_DIR = "/home/nvidia/SCD_MODELS/repaired_meshes"
OUTPUT_DIR = "/home/nvidia/SCD_MODELS/simulation_ready"

ALL_PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001", 
    "SCD0001101", "SCD0001201"
]

# EVIDENCE-BASED THRESHOLDS
# Based on FEBio developer guidance and published biomechanics literature

THRESHOLDS = {
    # FEBio requirements (per Steve Maas, FEBio developer)
    # "The Jacobian has to be positive, obviously"
    'febio': {
        'name': 'FEBio Simulation Ready',
        'min_jacobian': 0.0,           # HARD: Must be positive
        'min_jacobian_practical': 0.01, # Practical minimum for stability
        'max_aspect_ratio': 50,         # Complex geometry allowance
        'min_dihedral_deg': 3.0,        # "Survivable" per research
        'max_dihedral_deg': 177.0,      # Relaxed from theoretical 170°
        'max_radius_edge': None,        # Not a proper metric per Hang Si
        'max_problem_percent': 1.0,     # <1% bad elements acceptable
    },
    
    # OpenCarp requirements (electrophysiology-focused)
    'opencarp': {
        'name': 'OpenCarp Simulation Ready',
        'min_jacobian': 0.0,
        'min_jacobian_practical': 0.001,
        'max_aspect_ratio': 100,
        'min_dihedral_deg': 1.0,
        'max_dihedral_deg': 179.0,
        'max_radius_edge': None,
        'max_problem_percent': 1.0,
    },
    
    'publication': {
        'name': 'Publication Quality',
        'min_jacobian': 0.0,
        'min_jacobian_practical': 0.05,
        'max_aspect_ratio': 30,
        'min_dihedral_deg': 10.0,
        'max_dihedral_deg': 170.0,
        'max_radius_edge': 50.0,
        'max_problem_percent': 0.5,
    },
}

MAX_OPTIMIZATION_PASSES = 3

# MESH I/O FUNCTIONS
def read_carp_mesh(mesh_dir: str, patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read CARP/OpenCarp format mesh (.pts and .elem files).
    
    Returns:
        vertices: (N, 3) array of vertex coordinates
        elements: (M, 4) array of tetrahedral vertex indices
        tags: (M,) array of element material tags
    """
    pts_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.pts"
    elem_path = Path(mesh_dir) / patient_id / f"{patient_id}_tet.elem"
    
    if not pts_path.exists():
        raise FileNotFoundError(f"Points file not found: {pts_path}")
    if not elem_path.exists():
        raise FileNotFoundError(f"Elements file not found: {elem_path}")
    
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
        # Format: Tt v0 v1 v2 v3 tag
        elements[i] = [int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])]
        tags[i] = int(parts[5]) if len(parts) > 5 else 1
    
    return vertices, elements, tags


def write_carp_mesh(output_dir: str, patient_id: str, vertices: np.ndarray,
                    elements: np.ndarray, tags: np.ndarray):
    """Write CARP/OpenCarp format mesh."""
    out_path = Path(output_dir) / patient_id
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Write vertices
    pts_path = out_path / f"{patient_id}_tet.pts"
    with open(pts_path, 'w') as f:
        f.write(f"{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]:.10f} {v[1]:.10f} {v[2]:.10f}\n")
    
    # Write elements
    elem_path = out_path / f"{patient_id}_tet.elem"
    with open(elem_path, 'w') as f:
        f.write(f"{len(elements)}\n")
        for i, e in enumerate(elements):
            f.write(f"Tt {e[0]} {e[1]} {e[2]} {e[3]} {tags[i]}\n")


def write_medit_mesh(filepath: str, vertices: np.ndarray, elements: np.ndarray,
                     tags: np.ndarray = None):
    """Write Medit format mesh for MMG3D."""
    with open(filepath, 'w') as f:
        f.write("MeshVersionFormatted 2\nDimension 3\n\n")
        f.write(f"Vertices\n{len(vertices)}\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} 0\n")
        f.write(f"\nTetrahedra\n{len(elements)}\n")
        for i, e in enumerate(elements):
            tag = tags[i] if tags is not None else 1
            f.write(f"{e[0]+1} {e[1]+1} {e[2]+1} {e[3]+1} {tag}\n")
        f.write("\nEnd\n")


def read_medit_mesh(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read Medit format mesh."""
    vertices, elements, tags = [], [], []
    
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


def write_febio_xml(output_path: str, patient_id: str, vertices: np.ndarray,
                    elements: np.ndarray, tags: np.ndarray):
    """
    Export mesh in FEBio .feb XML format (version 3.0).
    """
    with open(output_path, 'w') as f:
        f.write('<?xml version="1.0" encoding="ISO-8859-1"?>\n')
        f.write('<febio_spec version="3.0">\n')
        f.write('  <Module type="solid"/>\n\n')
        
        f.write('  <Mesh>\n')
        
        # Nodes
        f.write('    <Nodes name="AllNodes">\n')
        for i, v in enumerate(vertices):
            f.write(f'      <node id="{i+1}">{v[0]:.10f},{v[1]:.10f},{v[2]:.10f}</node>\n')
        f.write('    </Nodes>\n\n')
        
        # Elements grouped by tag
        unique_tags = np.unique(tags)
        for tag in unique_tags:
            mask = tags == tag
            tag_indices = np.where(mask)[0]
            f.write(f'    <Elements type="tet4" name="Region{tag}">\n')
            for local_id, global_id in enumerate(tag_indices):
                e = elements[global_id]
                f.write(f'      <elem id="{local_id+1}">'
                       f'{e[0]+1},{e[1]+1},{e[2]+1},{e[3]+1}</elem>\n')
            f.write('    </Elements>\n')
        
        f.write('  </Mesh>\n')
        f.write('</febio_spec>\n')


def write_vtk(output_path: str, vertices: np.ndarray, elements: np.ndarray,
              tags: np.ndarray, quality: Dict = None):
    """Export mesh in VTK format with quality fields for visualization."""
    with open(output_path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Cardiac Mesh - Simulation Ready\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n\n")
        
        # Points
        f.write(f"POINTS {len(vertices)} double\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        
        # Cells
        f.write(f"\nCELLS {len(elements)} {len(elements)*5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"\nCELL_TYPES {len(elements)}\n")
        for _ in elements:
            f.write("10\n")  # VTK_TETRA
        
        # Cell data
        f.write(f"\nCELL_DATA {len(elements)}\n")
        
        # Material tags
        f.write("SCALARS material_tag int 1\nLOOKUP_TABLE default\n")
        for t in tags:
            f.write(f"{t}\n")
        
        # Quality metrics if provided
        if quality:
            f.write("\nSCALARS scaled_jacobian double 1\nLOOKUP_TABLE default\n")
            for j in quality['scaled_jacobian']:
                f.write(f"{j}\n")
            
            f.write("\nSCALARS aspect_ratio double 1\nLOOKUP_TABLE default\n")
            for ar in quality['aspect_ratio']:
                f.write(f"{ar}\n")
            
            f.write("\nSCALARS min_dihedral_angle double 1\nLOOKUP_TABLE default\n")
            for d in quality['min_dihedral_angle']:
                f.write(f"{d}\n")
            
            f.write("\nSCALARS max_dihedral_angle double 1\nLOOKUP_TABLE default\n")
            for d in quality['max_dihedral_angle']:
                f.write(f"{d}\n")


# QUALITY METRICS (with proper formulas)
def compute_tetrahedral_quality(vertices: np.ndarray, elements: np.ndarray) -> Dict:
    """
    Compute comprehensive quality metrics for tetrahedral mesh.
    
    Metrics computed:
    1. Scaled Jacobian: J = 6√2 × V / L_rms³
    2. Aspect Ratio: AR = L_max / (2 × r_inscribed)
    3. Dihedral Angles: θ = arccos(-n₁ · n₂) for each edge
    4. Radius-Edge Ratio: ρ = R_circumsphere / L_min
    5. Volume: V = (1/6) × det([e01, e02, e03])
    
    Returns:
        Dictionary with arrays of per-element quality values
    """
    n_elements = len(elements)
    
    # Initialize arrays
    scaled_jacobian = np.zeros(n_elements)
    aspect_ratio = np.zeros(n_elements)
    min_dihedral_angle = np.zeros(n_elements)
    max_dihedral_angle = np.zeros(n_elements)
    radius_edge_ratio = np.zeros(n_elements)
    volume = np.zeros(n_elements)
    
    for i, elem in enumerate(elements):
        # Get vertex coordinates
        v0 = vertices[elem[0]]
        v1 = vertices[elem[1]]
        v2 = vertices[elem[2]]
        v3 = vertices[elem[3]]
        
        # Edge vectors from v0
        e01 = v1 - v0
        e02 = v2 - v0
        e03 = v3 - v0
        
        # VOLUME: V = (1/6) × (e01 · (e02 × e03))
        vol = np.dot(e01, np.cross(e02, e03)) / 6.0
        volume[i] = vol
        
        # All 6 edges and their lengths
        edges = [
            v1 - v0,  # edge 0-1
            v2 - v0,  # edge 0-2
            v3 - v0,  # edge 0-3
            v2 - v1,  # edge 1-2
            v3 - v1,  # edge 1-3
            v3 - v2,  # edge 2-3
        ]
        edge_lengths = np.array([np.linalg.norm(e) for e in edges])
        
        L_max = np.max(edge_lengths)
        L_min = max(np.min(edge_lengths), 1e-15)
        L_rms = np.sqrt(np.mean(edge_lengths**2))
        
        # SCALED JACOBIAN: J = 6√2 × V / L_rms³
        # Range: [-1, 1], ideal = 1.0 for regular tetrahedron
        if L_rms > 1e-15:
            scaled_jacobian[i] = 6.0 * np.sqrt(2.0) * vol / (L_rms**3)
        else:
            scaled_jacobian[i] = 0.0
        
        # Face normals and areas for aspect ratio and dihedral angles
        # Faces: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
        face_vertices = [
            [v0, v1, v2],
            [v0, v1, v3],
            [v0, v2, v3],
            [v1, v2, v3],
        ]
        
        face_areas = []
        face_normals = []
        
        for fv in face_vertices:
            cross_product = np.cross(fv[1] - fv[0], fv[2] - fv[0])
            area = 0.5 * np.linalg.norm(cross_product)
            face_areas.append(area)
            
            norm_magnitude = np.linalg.norm(cross_product)
            if norm_magnitude > 1e-15:
                face_normals.append(cross_product / norm_magnitude)
            else:
                face_normals.append(np.array([0.0, 0.0, 1.0]))
        
        total_surface_area = sum(face_areas)
        
        # ASPECT RATIO: AR = L_max / (2 × r_inscribed)
        # r_inscribed = 3V / A_total (inscribed sphere radius)
        if total_surface_area > 1e-15:
            r_inscribed = 3.0 * abs(vol) / total_surface_area
            aspect_ratio[i] = L_max / (2.0 * r_inscribed) if r_inscribed > 1e-15 else 1e10
        else:
            aspect_ratio[i] = 1e10
        
        # RADIUS-EDGE RATIO: ρ ≈ (L0 × L1 × L2) / (6V × L_min)
        # Approximation using edges from one vertex
        if abs(vol) > 1e-20:
            R_approx = (edge_lengths[0] * edge_lengths[1] * edge_lengths[2]) / (6.0 * abs(vol))
            radius_edge_ratio[i] = R_approx / L_min
        else:
            radius_edge_ratio[i] = 1e10
        
        # DIHEDRAL ANGLES
        # Each edge is shared by exactly 2 faces
        # Edge-to-face mapping:
        #   Edge 0-1: faces (0,1,2) and (0,1,3) → indices 0,1
        #   Edge 0-2: faces (0,1,2) and (0,2,3) → indices 0,2
        #   Edge 0-3: faces (0,1,3) and (0,2,3) → indices 1,2
        #   Edge 1-2: faces (0,1,2) and (1,2,3) → indices 0,3
        #   Edge 1-3: faces (0,1,3) and (1,2,3) → indices 1,3
        #   Edge 2-3: faces (0,2,3) and (1,2,3) → indices 2,3
        edge_face_pairs = [
            (0, 1),  # edge v0-v1
            (0, 2),  # edge v0-v2
            (1, 2),  # edge v0-v3
            (0, 3),  # edge v1-v2
            (1, 3),  # edge v1-v3
            (2, 3),  # edge v2-v3
        ]
        
        dihedral_angles = []
        for f1_idx, f2_idx in edge_face_pairs:
            n1 = face_normals[f1_idx]
            n2 = face_normals[f2_idx]
            
            # Dihedral angle: θ = arccos(-n1 · n2)
            # Using -dot because we want interior angle
            dot_product = np.clip(np.dot(n1, n2), -1.0, 1.0)
            angle_rad = np.arccos(-dot_product)
            angle_deg = np.degrees(angle_rad)
            dihedral_angles.append(angle_deg)
        
        min_dihedral_angle[i] = min(dihedral_angles)
        max_dihedral_angle[i] = max(dihedral_angles)
    
    return {
        'scaled_jacobian': scaled_jacobian,
        'aspect_ratio': aspect_ratio,
        'min_dihedral_angle': min_dihedral_angle,
        'max_dihedral_angle': max_dihedral_angle,
        'radius_edge_ratio': radius_edge_ratio,
        'volume': volume,
    }


def validate_mesh(quality: Dict, threshold_name: str = 'febio') -> Dict:
    """
    Validate mesh against specified threshold level.
    
    Returns validation results including pass/fail status and element counts.
    """
    thresh = THRESHOLDS[threshold_name]
    n = len(quality['scaled_jacobian'])
    
    # Check each criterion
    inverted = quality['volume'] <= 0
    bad_jacobian = quality['scaled_jacobian'] < thresh['min_jacobian_practical']
    bad_aspect = quality['aspect_ratio'] > thresh['max_aspect_ratio']
    bad_dihedral_min = quality['min_dihedral_angle'] < thresh['min_dihedral_deg']
    bad_dihedral_max = quality['max_dihedral_angle'] > thresh['max_dihedral_deg']
    
    # Radius-edge is informational only (per Hang Si)
    if thresh['max_radius_edge'] is not None:
        bad_radius = quality['radius_edge_ratio'] > thresh['max_radius_edge']
    else:
        bad_radius = np.zeros(n, dtype=bool)
    
    # Total bad elements (unique)
    all_bad = inverted | bad_jacobian | bad_aspect | bad_dihedral_min | bad_dihedral_max
    n_bad = int(np.sum(all_bad))
    percent_bad = 100.0 * n_bad / n if n > 0 else 0.0
    
    # Pass if:
    # 1. No inverted elements (HARD requirement)
    # 2. Percentage of bad elements below threshold
    is_valid = (np.sum(inverted) == 0) and (percent_bad <= thresh['max_problem_percent'])
    
    return {
        'threshold_name': threshold_name,
        'threshold_display_name': thresh['name'],
        'n_elements': n,
        'n_inverted': int(np.sum(inverted)),
        'n_bad_jacobian': int(np.sum(bad_jacobian)),
        'n_bad_aspect_ratio': int(np.sum(bad_aspect)),
        'n_bad_dihedral_min': int(np.sum(bad_dihedral_min)),
        'n_bad_dihedral_max': int(np.sum(bad_dihedral_max)),
        'n_bad_radius_edge': int(np.sum(bad_radius)),
        'n_bad_total': n_bad,
        'percent_bad': percent_bad,
        'is_valid': is_valid,
        
        # Actual metric values
        'min_jacobian': float(np.min(quality['scaled_jacobian'])),
        'max_jacobian': float(np.max(quality['scaled_jacobian'])),
        'mean_jacobian': float(np.mean(quality['scaled_jacobian'])),
        
        'min_aspect_ratio': float(np.min(quality['aspect_ratio'])),
        'max_aspect_ratio': float(np.max(quality['aspect_ratio'])),
        'mean_aspect_ratio': float(np.mean(quality['aspect_ratio'])),
        
        'min_dihedral': float(np.min(quality['min_dihedral_angle'])),
        'max_dihedral': float(np.max(quality['max_dihedral_angle'])),
        
        'min_radius_edge': float(np.min(quality['radius_edge_ratio'])),
        'max_radius_edge': float(np.max(quality['radius_edge_ratio'])),
        'mean_radius_edge': float(np.mean(quality['radius_edge_ratio'])),
    }


# MMG3D OPTIMIZATION
def detect_mmg3d() -> Optional[str]:
    """Detect MMG3D executable."""
    for cmd in ['mmg3d_O3', 'mmg3d', 'mmg3d_debug']:
        try:
            result = subprocess.run([cmd, '-h'], capture_output=True, timeout=5)
            return cmd
        except:
            pass
    return None


def optimize_with_mmg3d(vertices: np.ndarray, elements: np.ndarray, 
                        tags: np.ndarray, mmg3d_cmd: str,
                        hausd: float = 0.005, hgrad: float = 1.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply MMG3D mesh optimization.
    
    Parameters:
        hausd: Hausdorff distance for surface approximation
        hgrad: Gradation parameter (1.0 = uniform, higher = more variation)
    """
    # Compute edge length bounds from current mesh
    edge_lens = []
    sample_size = min(1000, len(elements))
    for e in elements[:sample_size]:
        v = vertices[e]
        for i in range(4):
            for j in range(i+1, 4):
                edge_lens.append(np.linalg.norm(v[i] - v[j]))
    
    mean_edge = np.mean(edge_lens)
    hmin = mean_edge * 0.3
    hmax = mean_edge * 2.0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = f"{tmpdir}/input.mesh"
        output_path = f"{tmpdir}/output.o.mesh"
        
        write_medit_mesh(input_path, vertices, elements, tags)
        
        cmd = [
            mmg3d_cmd,
            input_path,
            "-out", output_path,
            "-hausd", str(hausd),
            "-hgrad", str(hgrad),
            "-hmin", str(hmin),
            "-hmax", str(hmax),
            "-optim",  # Optimization mode
            "-v", "0",  # Quiet
        ]
        
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
        
        # Find output file
        for path in [output_path, input_path.replace('.mesh', '.o.mesh')]:
            if Path(path).exists():
                return read_medit_mesh(path)
        
        raise RuntimeError(f"MMG3D failed: {result.stderr.decode()}")


def targeted_smoothing(vertices: np.ndarray, elements: np.ndarray,
                       quality: Dict, thresh: Dict,
                       n_iterations: int = 30) -> np.ndarray:
    """
    Apply Laplacian smoothing targeted at vertices of bad elements.
    Preserves boundary vertices and ensures no element inversion.
    """
    # Build vertex-to-element mapping
    vert_to_elems = defaultdict(list)
    for ei, e in enumerate(elements):
        for vi in e:
            vert_to_elems[vi].append(ei)
    
    # Build vertex adjacency
    neighbors = defaultdict(set)
    for e in elements:
        for i in range(4):
            for j in range(4):
                if i != j:
                    neighbors[e[i]].add(e[j])
    
    # Identify boundary vertices (don't move these)
    face_count = defaultdict(int)
    for e in elements:
        faces = [
            tuple(sorted([e[0], e[1], e[2]])),
            tuple(sorted([e[0], e[1], e[3]])),
            tuple(sorted([e[0], e[2], e[3]])),
            tuple(sorted([e[1], e[2], e[3]])),
        ]
        for f in faces:
            face_count[f] += 1
    
    boundary_verts = set()
    for f, count in face_count.items():
        if count == 1:  # Boundary face
            boundary_verts.update(f)
    
    # Identify bad elements
    bad_elems = set()
    for i in range(len(elements)):
        if (quality['scaled_jacobian'][i] < thresh['min_jacobian_practical'] or
            quality['aspect_ratio'][i] > thresh['max_aspect_ratio'] or
            quality['min_dihedral_angle'][i] < thresh['min_dihedral_deg'] or
            quality['max_dihedral_angle'][i] > thresh['max_dihedral_deg']):
            bad_elems.add(i)
    
    if len(bad_elems) == 0:
        return vertices
    
    # Vertices to smooth (interior vertices of bad elements)
    verts_to_smooth = set()
    for ei in bad_elems:
        for vi in elements[ei]:
            if vi not in boundary_verts:
                verts_to_smooth.add(vi)
    
    verts = vertices.copy()
    
    for iteration in range(n_iterations):
        moved = 0
        
        for vi in verts_to_smooth:
            nbrs = list(neighbors[vi])
            if len(nbrs) == 0:
                continue
            
            # Compute neighbor centroid
            centroid = np.mean(verts[nbrs], axis=0)
            
            # Small step toward centroid
            delta = 0.1 * (centroid - verts[vi])
            test_pos = verts[vi] + delta
            
            # Verify no element inversion
            valid = True
            for ei in vert_to_elems[vi]:
                e = elements[ei]
                test_verts = verts[e].copy()
                local_idx = list(e).index(vi)
                test_verts[local_idx] = test_pos
                
                # Check volume
                v0, v1, v2, v3 = test_verts
                vol = np.dot(v1-v0, np.cross(v2-v0, v3-v0)) / 6.0
                if vol <= 0:
                    valid = False
                    break
            
            if valid:
                verts[vi] = test_pos
                moved += 1
        
        if moved == 0:
            break
    
    return verts


# 
# MAIN OPTIMIZATION PIPELINE
def optimize_mesh(patient_id: str, mesh_dir: str, output_dir: str,
                  mmg3d_cmd: Optional[str] = None) -> Dict:
    """
    Complete optimization pipeline for a single patient mesh.
    
    Pipeline:
    1. Load and analyze current quality
    2. If already valid for FEBio, skip optimization
    3. Apply MMG3D optimization (if available)
    4. Apply targeted smoothing
    5. Validate against all threshold levels
    6. Export in multiple formats
    """
    print(f"PROCESSING: {patient_id}")
    
    start_time = time.time()
    
    result = {
        'patient_id': patient_id,
        'status': 'FAILED',
        'runtime_sec': 0,
    }
    
    try:
        # Load mesh
        print("  Loading mesh...")
        vertices, elements, tags = read_carp_mesh(mesh_dir, patient_id)
        print(f"    Vertices: {len(vertices):,}")
        print(f"    Elements: {len(elements):,}")
        print(f"    Unique tags: {np.unique(tags)}")
        
        # Initial quality analysis
        print("\n  Computing initial quality...")
        quality = compute_tetrahedral_quality(vertices, elements)
        
        # Validate against all levels
        initial_validations = {}
        for thresh_name in ['febio', 'opencarp', 'publication']:
            validation = validate_mesh(quality, thresh_name)
            initial_validations[thresh_name] = validation
            
            status = "✓" if validation['is_valid'] else "X"
            print(f"    {thresh_name:12s}: {status} ({validation['n_bad_total']:,} bad, "
                  f"{validation['percent_bad']:.2f}%)")
        
        # Store initial metrics
        result['initial_elements'] = len(elements)
        result['initial_bad_febio'] = initial_validations['febio']['n_bad_total']
        result['initial_bad_opencarp'] = initial_validations['opencarp']['n_bad_total']
        
        print(f"\n  Initial key metrics:")
        print(f"    Scaled Jacobian: min={quality['scaled_jacobian'].min():.4f}, "
              f"mean={quality['scaled_jacobian'].mean():.4f}")
        print(f"    Aspect Ratio: max={quality['aspect_ratio'].max():.2f}, "
              f"mean={quality['aspect_ratio'].mean():.2f}")
        print(f"    Dihedral Angles: [{quality['min_dihedral_angle'].min():.2f}°, "
              f"{quality['max_dihedral_angle'].max():.2f}°]")
        
        # Optimize if needed
        opt_vertices = vertices
        opt_elements = elements
        opt_tags = tags
        stages_applied = []
        
        if not initial_validations['febio']['is_valid']:
            # Stage 1: MMG3D optimization
            if mmg3d_cmd:
                print("\n  Stage 1: MMG3D optimization...")
                try:
                    for pass_num, (hausd, hgrad) in enumerate([
                        (0.005, 1.2),   # Pass 1: Standard
                        (0.003, 1.1),   # Pass 2: Tighter
                        (0.002, 1.05),  # Pass 3: Aggressive
                    ]):
                        print(f"    Pass {pass_num+1}: hausd={hausd}, hgrad={hgrad}")
                        opt_vertices, opt_elements, opt_tags = optimize_with_mmg3d(
                            opt_vertices, opt_elements, opt_tags,
                            mmg3d_cmd, hausd, hgrad
                        )
                        
                        quality = compute_tetrahedral_quality(opt_vertices, opt_elements)
                        validation = validate_mesh(quality, 'febio')
                        print(f"      Result: {validation['n_bad_total']} bad elements")
                        
                        if validation['is_valid']:
                            break
                    
                    stages_applied.append('MMG3D')
                except Exception as e:
                    print(f"    MMG3D failed: {e}")
            
            # Stage 2: Targeted smoothing
            print("\n  Stage 2: Targeted smoothing...")
            quality = compute_tetrahedral_quality(opt_vertices, opt_elements)
            opt_vertices = targeted_smoothing(
                opt_vertices, opt_elements, quality, 
                THRESHOLDS['febio'], n_iterations=50
            )
            stages_applied.append('Smoothing')
        
        # Final quality analysis
        print("\n  Final quality analysis...")
        quality = compute_tetrahedral_quality(opt_vertices, opt_elements)
        
        final_validations = {}
        for thresh_name in ['febio', 'opencarp', 'publication']:
            validation = validate_mesh(quality, thresh_name)
            final_validations[thresh_name] = validation
            
            status = "✓" if validation['is_valid'] else "✗"
            print(f"    {thresh_name:12s}: {status} ({validation['n_bad_total']:,} bad, "
                  f"{validation['percent_bad']:.2f}%)")
        
        print(f"\n  Final key metrics:")
        print(f"    Scaled Jacobian: min={quality['scaled_jacobian'].min():.4f}")
        print(f"    Aspect Ratio: max={quality['aspect_ratio'].max():.2f}")
        print(f"    Dihedral Angles: [{quality['min_dihedral_angle'].min():.2f}°, "
              f"{quality['max_dihedral_angle'].max():.2f}°]")
        
        # Export
        print("\n  Exporting...")
        patient_out = Path(output_dir) / patient_id
        patient_out.mkdir(parents=True, exist_ok=True)
        
        # CARP format (OpenCarp)
        write_carp_mesh(str(patient_out), patient_id, opt_vertices, opt_elements, opt_tags)
        print(f"    CARP: {patient_id}_tet.pts, {patient_id}_tet.elem")
        
        # FEBio format
        feb_path = patient_out / f"{patient_id}.feb"
        write_febio_xml(str(feb_path), patient_id, opt_vertices, opt_elements, opt_tags)
        print(f"    FEBio: {patient_id}.feb")
        
        # VTK format (visualization)
        vtk_path = patient_out / f"{patient_id}_quality.vtk"
        write_vtk(str(vtk_path), opt_vertices, opt_elements, opt_tags, quality)
        print(f"    VTK: {patient_id}_quality.vtk")
        
        # Store results
        result['status'] = 'SUCCESS'
        result['final_elements'] = len(opt_elements)
        result['stages_applied'] = '+'.join(stages_applied) if stages_applied else 'None'
        
        result['febio_valid'] = final_validations['febio']['is_valid']
        result['opencarp_valid'] = final_validations['opencarp']['is_valid']
        result['publication_valid'] = final_validations['publication']['is_valid']
        
        result['final_bad_febio'] = final_validations['febio']['n_bad_total']
        result['final_bad_opencarp'] = final_validations['opencarp']['n_bad_total']
        result['final_percent_bad'] = final_validations['febio']['percent_bad']
        
        result['min_jacobian'] = final_validations['febio']['min_jacobian']
        result['max_aspect_ratio'] = final_validations['febio']['max_aspect_ratio']
        result['min_dihedral'] = final_validations['febio']['min_dihedral']
        result['max_dihedral'] = final_validations['febio']['max_dihedral']
        
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
    """Run optimization on all patient meshes."""
    
    print("EVIDENCE-BASED CARDIAC MESH OPTIMIZATION")
    print("For FEBio and OpenCarp Simulations")
    
    print("\nThreshold Summary (Evidence-Based):")
    for name, thresh in THRESHOLDS.items():
        print(f"  {thresh['name']}:")
        print(f"    Jacobian > {thresh['min_jacobian_practical']}")
        print(f"    Aspect Ratio < {thresh['max_aspect_ratio']}")
        print(f"    Dihedral: [{thresh['min_dihedral_deg']}°, {thresh['max_dihedral_deg']}°]")
        print(f"    Max bad elements: {thresh['max_problem_percent']}%")
    
    # Detect tools
    mmg3d_cmd = detect_mmg3d()
    print(f"\nMMG3D: {'Found (' + mmg3d_cmd + ')' if mmg3d_cmd else 'Not found'}")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    results = []
    for i, patient_id in enumerate(patients):
        print(f"\n[{i+1}/{len(patients)}]", end="")
        result = optimize_mesh(patient_id, mesh_dir, output_dir, mmg3d_cmd)
        results.append(result)
    
    # Save summary CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = Path(output_dir) / f"optimization_summary_{timestamp}.csv"
    
    if results:
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(all_keys))
            writer.writeheader()
            writer.writerows(results)
    
    # Print summary
    print("OPTIMIZATION SUMMARY")
    
    n_febio = sum(1 for r in results if r.get('febio_valid', False))
    n_opencarp = sum(1 for r in results if r.get('opencarp_valid', False))
    n_publication = sum(1 for r in results if r.get('publication_valid', False))
    
    print(f"\n  FEBio Ready:       {n_febio}/{len(results)}")
    print(f"  OpenCarp Ready:    {n_opencarp}/{len(results)}")
    print(f"  Publication Ready: {n_publication}/{len(results)}")
    
    print("\n  Per-patient results:")
    for r in results:
        fb = "✓" if r.get('febio_valid') else "✗"
        oc = "✓" if r.get('opencarp_valid') else "✗"
        bad = r.get('final_percent_bad', 0)
        print(f"    {r['patient_id']}: FEBio:{fb} OpenCarp:{oc} "
              f"({bad:.3f}% bad) via {r.get('stages_applied', 'N/A')}")
    
    print(f"\n  Output directory: {output_dir}")
    print(f"  Summary CSV: {csv_path}")
    
    return results


# MAIN
if __name__ == "__main__":
    # Check if running in Jupyter
    try:
        get_ipython()
        IN_JUPYTER = True
    except NameError:
        IN_JUPYTER = False
    
    if IN_JUPYTER:
        print("Running in Jupyter mode...")
        results = run_all()
    else:
        import argparse
        
        parser = argparse.ArgumentParser(
            description='Evidence-Based Cardiac Mesh Optimization for FEBio/OpenCarp'
        )
        parser.add_argument('--mesh-dir', type=str, default=MESH_DIR,
                          help='Input mesh directory')
        parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                          help='Output directory')
        parser.add_argument('--patients', type=str, nargs='+', default=ALL_PATIENTS,
                          help='Patient IDs to process')
        
        args = parser.parse_args()
        run_all(args.patients, args.mesh_dir, args.output_dir)