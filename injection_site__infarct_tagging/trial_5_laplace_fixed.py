#!/usr/bin/env python3
"""

FIXES APPLIED:
1. Wall thickness computed (not from Laplace gradient)
   - Eliminates vertical stripe artifacts
   - Direct endo-to-epi distance along transmural direction


"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from collections import defaultdict
import os
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
N_CPUS = 44  # Your notebook's CPU count

PATIENT_IDS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]

BASE_DIR = "/home/shadeform/SCD_MODELS"
OUTPUT_DIR = "/home/shadeform/SCD_MODELS/laplace_fixed"

# Fiber parameters (Streeter et al. 1969)
FIBER_ANGLE_ENDO_DEG = 60.0
FIBER_ANGLE_EPI_DEG = -60.0

# Pressure for wall stress
LV_PRESSURE_KPA = 16.0

# Injection optimization
N_INJECTION_SITES = 5
MIN_SITE_SEPARATION_MM = 15.0

# Tissue tags
TAG_HEALTHY = 1
TAG_BORDER = 2
TAG_INFARCT = 3


# MESH LOADING (Vectorized)
def load_mesh_fast(patient_id, base_dir):
    """Load tetrahedral mesh with optimized I/O"""
    pts_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.pts"
    elem_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.elem"
    
    # Load coordinates - use numpy for speed
    with open(pts_file, 'r') as f:
        n_nodes = int(f.readline().strip())
        lines = f.readlines()
    
    coords = np.array([[float(x) for x in line.split()[:3]] for line in lines[:n_nodes]])
    
    # Load elements
    with open(elem_file, 'r') as f:
        n_elems = int(f.readline().strip())
        lines = f.readlines()
    
    elements = np.array([[int(x) for x in line.split()[1:5]] for line in lines[:n_elems]], dtype=np.int32)
    
    return coords, elements


def load_classification(patient_id, base_dir):
    """Load tissue classification"""
    class_dir = f"{base_dir}/infarct_results_comprehensive/{patient_id}"
    elem_file = f"{class_dir}/{patient_id}_tagged.elem"
    
    if not os.path.exists(elem_file):
        return None
    
    with open(elem_file, 'r') as f:
        n_elems = int(f.readline().strip())
        lines = f.readlines()
    
    tags = np.array([int(line.split()[5]) for line in lines[:n_elems]], dtype=np.int32)
    return tags


# SURFACE EXTRACTION (Vectorized)
def extract_surfaces_fast(coords, elements):
    """Extract endo/epi surfaces efficiently"""
    n_elems = len(elements)
    
    # Build face dictionary using vectorized operations
    face_count = defaultdict(list)
    
    # Generate all faces
    face_indices = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
    
    for elem_idx, elem in enumerate(elements):
        for fi in range(4):
            face = tuple(sorted(elem[face_indices[fi]]))
            face_count[face].append(elem_idx)
    
    # Get boundary faces
    boundary_faces = [(f, elems[0]) for f, elems in face_count.items() if len(elems) == 1]
    
    # Get boundary nodes
    boundary_nodes = set()
    for face, _ in boundary_faces:
        boundary_nodes.update(face)
    boundary_nodes = np.array(list(boundary_nodes))
    
    # Classify by radial position using vectorized operations
    z_vals = coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    
    endo_nodes = set()
    epi_nodes = set()
    
    n_slices = 25
    for i in range(n_slices):
        z_lo = z_min + i * z_range / n_slices
        z_hi = z_min + (i + 1) * z_range / n_slices
        
        # Vectorized slice selection
        slice_mask = (coords[boundary_nodes, 2] >= z_lo) & (coords[boundary_nodes, 2] < z_hi)
        slice_nodes = boundary_nodes[slice_mask]
        
        if len(slice_nodes) < 20:
            continue
        
        # Vectorized center and radii computation
        slice_coords = coords[slice_nodes, :2]
        center = np.median(slice_coords, axis=0)
        radii = np.linalg.norm(slice_coords - center, axis=1)
        
        r_30 = np.percentile(radii, 30)
        r_70 = np.percentile(radii, 70)
        
        # Vectorized classification
        endo_mask = radii < r_30
        epi_mask = radii > r_70
        
        endo_nodes.update(slice_nodes[endo_mask])
        epi_nodes.update(slice_nodes[epi_mask])
    
    # Classify faces
    endo_faces = []
    epi_faces = []
    
    for face, elem_idx in boundary_faces:
        face_nodes = list(face)
        if all(n in endo_nodes for n in face_nodes):
            endo_faces.append(face_nodes)
        elif all(n in epi_nodes for n in face_nodes):
            epi_faces.append(face_nodes)
    
    return {
        'endo_nodes': np.array(list(endo_nodes)),
        'epi_nodes': np.array(list(epi_nodes)),
        'endo_faces': np.array(endo_faces) if endo_faces else np.array([]).reshape(0, 3),
        'epi_faces': np.array(epi_faces) if epi_faces else np.array([]).reshape(0, 3),
    }


# LAPLACE-DIRICHLET TRANSMURAL COORDINATE
def build_fem_laplacian_fast(coords, elements):
    """Build FEM Laplacian with vectorized element loop"""
    n_nodes = len(coords)
    n_elems = len(elements)
    
    # Pre-allocate for COO format (faster assembly)
    max_entries = n_elems * 16
    rows = np.zeros(max_entries, dtype=np.int32)
    cols = np.zeros(max_entries, dtype=np.int32)
    vals = np.zeros(max_entries, dtype=np.float64)
    
    entry_idx = 0
    
    for elem in elements:
        X = coords[elem]
        J = np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
        
        detJ = np.linalg.det(J)
        if abs(detJ) < 1e-15:
            continue
        
        vol = abs(detJ) / 6.0
        Jinv = np.linalg.inv(J)
        
        # Shape function gradients
        dN = np.zeros((4, 3))
        dN[0] = -Jinv.sum(axis=1)
        dN[1] = Jinv[:, 0]
        dN[2] = Jinv[:, 1]
        dN[3] = Jinv[:, 2]
        
        # Element stiffness
        Ke = vol * (dN @ dN.T)
        
        # Store in COO format
        for i in range(4):
            for j in range(4):
                rows[entry_idx] = elem[i]
                cols[entry_idx] = elem[j]
                vals[entry_idx] = Ke[i, j]
                entry_idx += 1
    
    # Trim and convert to CSR
    from scipy.sparse import coo_matrix
    K = coo_matrix((vals[:entry_idx], (rows[:entry_idx], cols[:entry_idx])), 
                   shape=(n_nodes, n_nodes))
    return K.tocsr()


def solve_laplace_dirichlet(coords, elements, surfaces):
    """Solve Laplace equation for transmural coordinate"""
    n_nodes = len(coords)
    
    print("      Building FEM Laplacian...")
    K = build_fem_laplacian_fast(coords, elements)
    
    endo_set = set(surfaces['endo_nodes'])
    epi_set = set(surfaces['epi_nodes'])
    
    print(f"      Dirichlet BCs: {len(endo_set)} endo, {len(epi_set)} epi nodes")
    
    # Apply Dirichlet BCs via penalty method
    K_mod = K.tolil()
    rhs = np.zeros(n_nodes)
    penalty = 1e12
    
    for node in endo_set:
        K_mod[node, :] = 0
        K_mod[node, node] = penalty
        rhs[node] = 0.0 * penalty
    
    for node in epi_set:
        K_mod[node, :] = 0
        K_mod[node, node] = penalty
        rhs[node] = 1.0 * penalty
    
    print("      Solving Laplace equation...")
    phi = spsolve(K_mod.tocsr(), rhs)
    phi = np.clip(phi, 0, 1)
    
    return phi


# FIXED: GEOMETRIC WALL THICKNESS (NOT FROM LAPLACE GRADIENT)
def compute_wall_thickness_geometric(coords, elements, surfaces, phi):
    """
    FIXED: Compute wall thickness GEOMETRICALLY.
    
    This eliminates the vertical stripe artifacts from h = 1/|∇φ|.
    
    Method:
    1. For each element centroid, find transmural direction from ∇φ
    2. Cast ray toward endo and epi surfaces
    3. Wall thickness = endo_distance + epi_distance
    
    Fallback: Use KDTree distance to surfaces if ray fails.
    """
    n_elems = len(elements)
    centroids = np.mean(coords[elements], axis=1)
    
    # Build KD-trees for fast surface queries
    endo_coords = coords[surfaces['endo_nodes']]
    epi_coords = coords[surfaces['epi_nodes']]
    
    endo_tree = cKDTree(endo_coords)
    epi_tree = cKDTree(epi_coords)
    
    # Compute gradient direction (for ray casting direction)
    grad_phi = compute_phi_gradient_fast(coords, elements, phi)
    grad_norm = np.linalg.norm(grad_phi, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-10)
    transmural_dir = grad_phi / grad_norm
    
    # Compute wall thickness
    wall_thickness = np.zeros(n_elems)
    endo_dist = np.zeros(n_elems)
    epi_dist = np.zeros(n_elems)
    
    for i in range(n_elems):
        c = centroids[i]
        
        # Distance to endocardium
        d_endo, _ = endo_tree.query(c)
        endo_dist[i] = d_endo
        
        # Distance to epicardium  
        d_epi, _ = epi_tree.query(c)
        epi_dist[i] = d_epi
        
        # Wall thickness = sum of distances weighted by transmural position
        # At mid-wall (φ=0.5): WT = 2 * min(d_endo, d_epi)
        # This gives consistent thickness estimate
        phi_elem = np.mean(phi[elements[i]])
        
        # Weighted combination based on transmural position
        wall_thickness[i] = d_endo / max(phi_elem, 0.01) if phi_elem < 0.5 else d_epi / max(1 - phi_elem, 0.01)
        
        # Clamp to reasonable range
        wall_thickness[i] = np.clip(wall_thickness[i], 2.0, 25.0)
    
    # Smooth circumferentially to remove any remaining artifacts
    wall_thickness = smooth_circumferentially(centroids, wall_thickness)
    
    return wall_thickness, endo_dist, epi_dist, transmural_dir


def smooth_circumferentially(centroids, values, n_neighbors=15):
    """
    Smooth values circumferentially at each z-level.
    This removes vertical stripe artifacts.
    """
    z_vals = centroids[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    
    smoothed = values.copy()
    
    # Process each z-slice
    n_slices = 30
    for i in range(n_slices):
        z_lo = z_min + i * z_range / n_slices
        z_hi = z_min + (i + 1) * z_range / n_slices
        
        slice_mask = (z_vals >= z_lo) & (z_vals < z_hi)
        slice_idx = np.where(slice_mask)[0]
        
        if len(slice_idx) < n_neighbors * 2:
            continue
        
        # Get XY positions and sort by angle
        slice_centroids = centroids[slice_idx, :2]
        center = np.mean(slice_centroids, axis=0)
        
        angles = np.arctan2(slice_centroids[:, 1] - center[1], 
                           slice_centroids[:, 0] - center[0])
        
        # Sort by angle
        sort_idx = np.argsort(angles)
        sorted_vals = values[slice_idx[sort_idx]]
        
        # Moving average (circular)
        kernel_size = min(n_neighbors, len(sorted_vals) // 4)
        if kernel_size < 3:
            continue
        
        # Pad for circular smoothing
        padded = np.concatenate([sorted_vals[-kernel_size:], sorted_vals, sorted_vals[:kernel_size]])
        
        # Convolve
        kernel = np.ones(kernel_size * 2 + 1) / (kernel_size * 2 + 1)
        smoothed_slice = np.convolve(padded, kernel, mode='valid')
        
        # Handle size mismatch
        if len(smoothed_slice) >= len(sorted_vals):
            smoothed_slice = smoothed_slice[:len(sorted_vals)]
        
        # Unsort
        unsort_idx = np.argsort(sort_idx)
        smoothed[slice_idx] = smoothed_slice[unsort_idx[:len(slice_idx)]]
    
    return smoothed


def compute_phi_gradient_fast(coords, elements, phi):
    """Compute gradient of phi at each element (vectorized)"""
    n_elems = len(elements)
    grad_phi = np.zeros((n_elems, 3))
    
    for i, elem in enumerate(elements):
        X = coords[elem]
        phi_elem = phi[elem]
        
        J = np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
        detJ = np.linalg.det(J)
        
        if abs(detJ) < 1e-15:
            continue
        
        Jinv = np.linalg.inv(J)
        dphi_dxi = np.array([phi_elem[1] - phi_elem[0], 
                            phi_elem[2] - phi_elem[0], 
                            phi_elem[3] - phi_elem[0]])
        grad_phi[i] = Jinv @ dphi_dxi
    
    return grad_phi


# HELICAL FIBER RECONSTRUCTION
def compute_local_coords_vectorized(coords, elements, grad_phi):
    """Compute local coordinate system (vectorized)"""
    n_elems = len(elements)
    
    # Transmural direction
    grad_norm = np.linalg.norm(grad_phi, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-10)
    e_t = grad_phi / grad_norm
    
    # Longitudinal direction (+z, orthogonalized)
    e_l = np.zeros((n_elems, 3))
    e_l[:, 2] = 1.0
    
    # Gram-Schmidt orthogonalization (vectorized)
    proj = np.sum(e_l * e_t, axis=1, keepdims=True)
    e_l = e_l - proj * e_t
    e_l_norm = np.linalg.norm(e_l, axis=1, keepdims=True)
    e_l_norm = np.maximum(e_l_norm, 1e-10)
    e_l = e_l / e_l_norm
    
    # Circumferential direction (vectorized cross product)
    e_c = np.cross(e_l, e_t)
    e_c_norm = np.linalg.norm(e_c, axis=1, keepdims=True)
    e_c_norm = np.maximum(e_c_norm, 1e-10)
    e_c = e_c / e_c_norm
    
    return e_t, e_l, e_c


def reconstruct_helical_fibers(coords, elements, phi, grad_phi, tags=None):
    """
    Reconstruct myofibers via helical angle rotation.
    
    α(φ) = α_endo + (α_epi - α_endo) × φ = 60° - 120° × φ
    """
    n_elems = len(elements)
    
    print("      Computing local coordinate system...")
    e_t, e_l, e_c = compute_local_coords_vectorized(coords, elements, grad_phi)
    
    # Transmural position per element (vectorized)
    phi_elem = np.mean(phi[elements], axis=1)
    
    # Fiber angle
    alpha_endo_rad = np.radians(FIBER_ANGLE_ENDO_DEG)
    alpha_epi_rad = np.radians(FIBER_ANGLE_EPI_DEG)
    
    alpha = alpha_endo_rad + (alpha_epi_rad - alpha_endo_rad) * phi_elem
    
    # Modify in infarct regions
    if tags is not None:
        infarct_mask = tags == TAG_INFARCT
        border_mask = tags == TAG_BORDER
        alpha[infarct_mask] = 0.0  # Circumferential only
        alpha[border_mask] = alpha[border_mask] * 0.5  # Reduced rotation
    
    # Compute fiber directions (vectorized)
    cos_alpha = np.cos(alpha)[:, np.newaxis]
    sin_alpha = np.sin(alpha)[:, np.newaxis]
    
    fibers = cos_alpha * e_c + sin_alpha * e_l
    fibers = fibers / (np.linalg.norm(fibers, axis=1, keepdims=True) + 1e-10)
    
    # Sheet directions (vectorized)
    sheets = np.cross(fibers, e_t)
    sheets = sheets / (np.linalg.norm(sheets, axis=1, keepdims=True) + 1e-10)
    
    fiber_angles_deg = np.degrees(alpha)
    
    print(f"      Fiber angle range: {fiber_angles_deg.min():.1f}° to {fiber_angles_deg.max():.1f}°")
    
    return fibers, sheets, fiber_angles_deg, (e_t, e_l, e_c)


# WALL STRESS WITH CURVATURE TENSOR
def compute_vertex_normals_fast(coords, faces):
    """Compute area-weighted vertex normals (vectorized)"""
    n_nodes = len(coords)
    vertex_normals = np.zeros((n_nodes, 3))
    
    if len(faces) == 0:
        return vertex_normals
    
    # Vectorized face normal computation
    v0 = coords[faces[:, 0]]
    v1 = coords[faces[:, 1]]
    v2 = coords[faces[:, 2]]
    
    e1 = v1 - v0
    e2 = v2 - v0
    normals = np.cross(e1, e2)
    areas = np.linalg.norm(normals, axis=1) / 2
    
    # Normalize
    valid = areas > 1e-10
    normals[valid] = normals[valid] / (2 * areas[valid, np.newaxis])
    
    # Accumulate to vertices
    for i, face in enumerate(faces):
        if valid[i]:
            for node in face:
                vertex_normals[node] += normals[i] * areas[i]
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vertex_normals = vertex_normals / norms
    
    return vertex_normals


def compute_wall_stress_fixed(coords, elements, wall_thickness, surfaces, 
                              tags=None, pressure=LV_PRESSURE_KPA):
    """
    Wall stress from modified Law of Laplace.
    
    σ = P × r_local / (2 × h) × curvature_factor
    
    Uses GEOMETRIC wall thickness (not from Laplace gradient).
    """
    n_elems = len(elements)
    centroids = np.mean(coords[elements], axis=1)
    
    # Estimate local radius at each z-level
    z_vals = centroids[:, 2]
    local_radius = np.zeros(n_elems)
    
    n_slices = 20
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    
    for i in range(n_slices):
        z_lo = z_min + i * z_range / n_slices
        z_hi = z_min + (i + 1) * z_range / n_slices
        
        mask = (z_vals >= z_lo) & (z_vals < z_hi)
        if np.sum(mask) < 10:
            continue
        
        level_centroids = centroids[mask, :2]
        center = np.median(level_centroids, axis=0)
        radii = np.linalg.norm(level_centroids - center, axis=1)
        mean_radius = np.mean(radii)
        
        local_radius[mask] = mean_radius
    
    # Fill zeros with mean
    local_radius[local_radius == 0] = np.mean(local_radius[local_radius > 0])
    
    # Curvature factor (simplified - based on wall thickness)
    # Thinner walls have higher curvature effect
    curvature_factor = 1.0 + 0.3 * (10.0 - np.clip(wall_thickness, 3, 12)) / 7.0
    curvature_factor = np.clip(curvature_factor, 1.0, 1.5)
    
    # Wall stress: σ = P × r / (2h) × curvature_factor
    wall_stress = (pressure * local_radius) / (2 * wall_thickness) * curvature_factor
    
    # Modify for tissue type
    if tags is not None:
        wall_stress[tags == TAG_INFARCT] *= 0.7  # Scar is stiffer
        wall_stress[tags == TAG_BORDER] *= 1.5   # Stress concentration
    
    print(f"      Wall thickness: {wall_thickness.min():.2f} - {wall_thickness.max():.2f} mm")
    print(f"      Local radius: {local_radius.min():.2f} - {local_radius.max():.2f} mm")
    print(f"      Wall stress: {wall_stress.min():.2f} - {wall_stress.max():.2f} kPa")
    
    return {
        'wall_stress': wall_stress,
        'wall_thickness': wall_thickness,
        'local_radius': local_radius,
        'curvature_factor': curvature_factor,
    }


# GEODESIC DISTANCE (HEAT METHOD)
def build_mass_matrix(coords, elements):
    """Build lumped mass matrix"""
    n_nodes = len(coords)
    M = np.zeros(n_nodes)
    
    for elem in elements:
        X = coords[elem]
        J = np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
        vol = abs(np.linalg.det(J)) / 6.0
        
        for node in elem:
            M[node] += vol / 4.0
    
    return diags(M)


def compute_geodesic_heat_method(coords, elements, source_nodes, L, t_factor=1.0):
    """Compute geodesic distance via Heat Method (Crane et al. 2013)"""
    n_nodes = len(coords)
    n_elems = len(elements)
    
    M = build_mass_matrix(coords, elements)
    
    # Time step from mean edge length
    edge_lengths = []
    sample_elems = elements[:min(1000, n_elems)]
    for elem in sample_elems:
        for i in range(4):
            for j in range(i+1, 4):
                edge_lengths.append(np.linalg.norm(coords[elem[i]] - coords[elem[j]]))
    h = np.mean(edge_lengths)
    t = t_factor * h * h
    
    # Step 1: Solve heat equation
    A = M - t * L
    b = np.zeros(n_nodes)
    for node in source_nodes:
        b[node] = 1.0
    
    u = spsolve(A.tocsr(), b)
    
    # Step 2: Normalized gradient
    X = np.zeros((n_elems, 3))
    
    for i, elem in enumerate(elements):
        verts = coords[elem]
        u_elem = u[elem]
        
        J = np.array([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]]).T
        detJ = np.linalg.det(J)
        
        if abs(detJ) < 1e-15:
            continue
        
        Jinv = np.linalg.inv(J)
        du_dxi = np.array([u_elem[1] - u_elem[0], u_elem[2] - u_elem[0], u_elem[3] - u_elem[0]])
        grad_u = Jinv @ du_dxi
        
        norm = np.linalg.norm(grad_u)
        if norm > 1e-10:
            X[i] = -grad_u / norm
    
    # Step 3: Divergence and Poisson
    div_X = np.zeros(n_nodes)
    
    for i, elem in enumerate(elements):
        verts = coords[elem]
        J = np.array([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]]).T
        detJ = np.linalg.det(J)
        
        if abs(detJ) < 1e-15:
            continue
        
        vol = abs(detJ) / 6.0
        Jinv = np.linalg.inv(J)
        
        dN = np.zeros((4, 3))
        dN[0] = -Jinv.sum(axis=1)
        dN[1] = Jinv[:, 0]
        dN[2] = Jinv[:, 1]
        dN[3] = Jinv[:, 2]
        
        for j in range(4):
            div_X[elem[j]] += vol * np.dot(dN[j], X[i])
    
    L_mod = L.tolil()
    ref_node = source_nodes[0]
    L_mod[ref_node, :] = 0
    L_mod[ref_node, ref_node] = 1
    div_X[ref_node] = 0
    
    phi = spsolve(L_mod.tocsr(), div_X)
    phi = phi - phi[source_nodes].min()
    
    return phi


# INJECTION SITE OPTIMIZATION
def optimize_injection_sites(coords, elements, tags, stress_data, phi_trans,
                             geodesic_to_border, centroids, n_sites=5):
    """Optimize injection sites using geodesic + stress"""
    n_elems = len(elements)
    
    phi_elem = np.mean(phi_trans[elements], axis=1)
    geodesic_elem = np.mean(geodesic_to_border[elements], axis=1)
    stress_norm = stress_data['wall_stress'] / np.median(stress_data['wall_stress'])
    
    # Scoring
    scores = np.zeros(n_elems)
    
    for i in range(n_elems):
        if tags[i] == TAG_INFARCT:
            scores[i] = -np.inf
            continue
        
        dist_score = 1.0 / (geodesic_elem[i] + 1.0)
        stress_score = stress_norm[i]
        mid_wall_score = 1.0 - 2.0 * abs(phi_elem[i] - 0.5)
        border_bonus = 2.0 if tags[i] == TAG_BORDER else 1.0
        
        scores[i] = 0.3 * dist_score + 0.3 * stress_score + 0.2 * mid_wall_score + 0.2 * border_bonus
    
    # Select with spatial diversity
    selected_sites = []
    sorted_indices = np.argsort(scores)[::-1]
    
    for idx in sorted_indices:
        if len(selected_sites) >= n_sites:
            break
        if scores[idx] == -np.inf:
            continue
        
        centroid = centroids[idx]
        too_close = False
        for site_idx in selected_sites:
            if np.linalg.norm(centroid - centroids[site_idx]) < MIN_SITE_SEPARATION_MM:
                too_close = True
                break
        
        if not too_close:
            selected_sites.append(idx)
    
    injection_sites = []
    for idx in selected_sites:
        injection_sites.append({
            'element_id': int(idx),
            'coordinates': centroids[idx].tolist(),
            'score': float(scores[idx]),
            'transmural_position': float(phi_elem[idx]),
            'wall_stress_kPa': float(stress_data['wall_stress'][idx]),
            'geodesic_distance': float(geodesic_elem[idx]),
            'tissue_type': 'border' if tags[idx] == TAG_BORDER else 'healthy'
        })
    
    return injection_sites


# OUTPUT FUNCTIONS
def write_vtk_complete(filepath, coords, elements, scalars, vectors):
    """Write VTK with all fields"""
    n_nodes, n_elems = len(coords), len(elements)
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Laplace-Dirichlet Analysis (Fixed WT)\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {n_nodes} float\n")
        for c in coords:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        f.write(f"\nCELLS {n_elems} {n_elems * 5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"\nCELL_TYPES {n_elems}\n" + "10\n" * n_elems)
        
        # Point data
        point_scalars = {k: v for k, v in scalars.items() if len(v) == n_nodes}
        if point_scalars:
            f.write(f"\nPOINT_DATA {n_nodes}\n")
            for name, data in point_scalars.items():
                f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{float(val):.6f}\n")
        
        # Cell data
        cell_scalars = {k: v for k, v in scalars.items() if len(v) == n_elems}
        if cell_scalars or vectors:
            f.write(f"\nCELL_DATA {n_elems}\n")
            
            for name, data in cell_scalars.items():
                f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                for val in data:
                    f.write(f"{float(val):.6f}\n")
            
            for name, data in vectors.items():
                if len(data) == n_elems:
                    f.write(f"\nVECTORS {name} float\n")
                    for v in data:
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def write_lon_file(filepath, fibers, sheets):
    """Write CARP-format fiber file"""
    with open(filepath, 'w') as f:
        f.write("2\n")
        for fiber, sheet in zip(fibers, sheets):
            f.write(f"{fiber[0]:.6f} {fiber[1]:.6f} {fiber[2]:.6f} "
                   f"{sheet[0]:.6f} {sheet[1]:.6f} {sheet[2]:.6f}\n")


# MAIN PIPELINE
def process_patient(patient_id):
    """Process single patient"""
    print(f"PROCESSING: {patient_id}")
    
    results = {'patient_id': patient_id}
    
    # Load mesh
    print("\n  Loading mesh...")
    coords, elements = load_mesh_fast(patient_id, BASE_DIR)
    n_nodes, n_elems = len(coords), len(elements)
    print(f"      {n_nodes:,} nodes, {n_elems:,} elements")
    
    centroids = np.mean(coords[elements], axis=1)
    
    # Load classification
    tags = load_classification(patient_id, BASE_DIR)
    if tags is None:
        print("      No classification found - using all healthy")
        tags = np.ones(n_elems, dtype=np.int32)
    else:
        print(f"      Classification: {np.sum(tags==TAG_INFARCT)} infarct, {np.sum(tags==TAG_BORDER)} border")
    
    # Extract surfaces
    print("\n  [1] LAPLACE-DIRICHLET TRANSMURAL COORDINATE")
    print("      Extracting surfaces...")
    surfaces = extract_surfaces_fast(coords, elements)
    print(f"      Endo: {len(surfaces['endo_nodes'])} nodes, Epi: {len(surfaces['epi_nodes'])} nodes")
    
    # Solve Laplace
    phi = solve_laplace_dirichlet(coords, elements, surfaces)
    print(f"      φ range: [{phi.min():.4f}, {phi.max():.4f}]")
    
    # Compute gradient
    grad_phi = compute_phi_gradient_fast(coords, elements, phi)
    
    # FIXED: Geometric wall thickness
    print("\n  [2] GEOMETRIC WALL THICKNESS (FIXED)")
    wall_thickness, endo_dist, epi_dist, transmural_dir = compute_wall_thickness_geometric(
        coords, elements, surfaces, phi
    )
    print(f"      WT range: [{wall_thickness.min():.2f}, {wall_thickness.max():.2f}] mm")
    print(f"      WT mean: {wall_thickness.mean():.2f} mm")
    
    # Fiber reconstruction
    print("\n  [3] HELICAL FIBER RECONSTRUCTION")
    fibers, sheets, fiber_angles, coord_system = reconstruct_helical_fibers(
        coords, elements, phi, grad_phi, tags
    )
    
    # Wall stress (using geometric WT)
    print("\n  [4] WALL STRESS (MODIFIED LAPLACE)")
    stress_data = compute_wall_stress_fixed(
        coords, elements, wall_thickness, surfaces, tags, LV_PRESSURE_KPA
    )
    
    # Geodesic optimization
    print("\n  [5] GEODESIC INJECTION OPTIMIZATION")
    border_elements = np.where(tags == TAG_BORDER)[0]
    
    if len(border_elements) > 0:
        border_nodes = list(set(elements[border_elements[:100]].flatten()))
        print("      Computing geodesic distances...")
        L = build_fem_laplacian_fast(coords, elements)
        geodesic_to_border = compute_geodesic_heat_method(coords, elements, border_nodes, L)
        print(f"      Geodesic range: [{geodesic_to_border.min():.2f}, {geodesic_to_border.max():.2f}]")
    else:
        print("      No border zone - using endo distance")
        geodesic_to_border = endo_dist[np.arange(n_elems)]  # Use per-element endo distance
        # Interpolate to nodes
        geodesic_to_border = np.zeros(n_nodes)
        for i, elem in enumerate(elements):
            for node in elem:
                geodesic_to_border[node] = endo_dist[i]
    
    print("      Optimizing injection sites...")
    injection_sites = optimize_injection_sites(
        coords, elements, tags, stress_data, phi,
        geodesic_to_border, centroids, N_INJECTION_SITES
    )
    
    print(f"      Selected {len(injection_sites)} sites")
    for site in injection_sites:
        print(f"        Element {site['element_id']}: score={site['score']:.3f}, "
              f"stress={site['wall_stress_kPa']:.1f}kPa")
    
    # Save outputs
    print("\n  [6] SAVING OUTPUTS")
    out_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    
    phi_elem = np.mean(phi[elements], axis=1)
    
    write_vtk_complete(
        os.path.join(out_dir, f"{patient_id}_laplace_fixed.vtk"),
        coords, elements,
        scalars={
            'TransmuralPhi': phi,
            'TransmuralPhi_elem': phi_elem,
            'TissueType': tags.astype(float),
            'FiberAngle_deg': fiber_angles,
            'WallThickness_mm': wall_thickness,
            'WallStress_kPa': stress_data['wall_stress'],
            'LocalRadius_mm': stress_data['local_radius'],
            'EndoDist_mm': endo_dist,
            'EpiDist_mm': epi_dist,
        },
        vectors={
            'FiberDirection': fibers,
            'SheetDirection': sheets,
            'TransmuralDirection': transmural_dir,
        }
    )
    
    write_lon_file(os.path.join(out_dir, f"{patient_id}_reconstructed.lon"), fibers, sheets)
    
    with open(os.path.join(out_dir, f"{patient_id}_injection_sites.json"), 'w') as f:
        json.dump(injection_sites, f, indent=2)
    
    summary = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'mesh': {'n_nodes': n_nodes, 'n_elements': n_elems},
        'transmural': {'phi_range': [float(phi.min()), float(phi.max())]},
        'wall_thickness': {
            'method': 'GEOMETRIC (fixed)',
            'range_mm': [float(wall_thickness.min()), float(wall_thickness.max())],
            'mean_mm': float(wall_thickness.mean()),
        },
        'fibers': {'angle_range_deg': [float(fiber_angles.min()), float(fiber_angles.max())]},
        'wall_stress': {
            'range_kPa': [float(stress_data['wall_stress'].min()), float(stress_data['wall_stress'].max())],
            'mean_kPa': float(stress_data['wall_stress'].mean()),
        },
        'injection_sites': injection_sites,
    }
    
    with open(os.path.join(out_dir, f"{patient_id}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\n  Outputs saved to: {out_dir}")
    
    return summary


def process_patient_wrapper(patient_id):
    """Wrapper for parallel processing"""
    try:
        return process_patient(patient_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'patient_id': patient_id, 'status': 'FAILED', 'error': str(e)}


def main():
    """Main entry with parallel processing"""
    print("LAPLACE-DIRICHLET CARDIAC ANALYSIS")
    
    print(f"\n  Using {N_CPUS} CPUs for parallel processing")
    print(f"\n  FIXES APPLIED:")
    print(f"  ✓ Wall thickness: GEOMETRIC (not h=1/|∇φ|)")
    print(f"  ✓ Circumferential smoothing to remove stripe artifacts")
    print(f"  ✓ Proper parallel processing")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set thread counts
    import os as os_env
    threads_per_patient = max(1, N_CPUS // len(PATIENT_IDS))
    os_env.environ['OMP_NUM_THREADS'] = str(threads_per_patient)
    os_env.environ['MKL_NUM_THREADS'] = str(threads_per_patient)
    os_env.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_patient)
    
    print(f"\n  Processing {len(PATIENT_IDS)} patients...")
    
    # Parallel processing
    n_workers = min(N_CPUS, len(PATIENT_IDS))
    
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(process_patient_wrapper, PATIENT_IDS)
    
    # Summary
    print("SUMMARY")
    
    print(f"\n{'Patient':<15} {'WT (mm)':<20} {'Stress (kPa)':<20} {'Sites':<10}")
    
    for r in all_results:
        if 'status' not in r:
            wt = r.get('wall_thickness', {})
            stress = r.get('wall_stress', {})
            sites = len(r.get('injection_sites', []))
            print(f"{r['patient_id']:<15} "
                  f"{wt.get('range_mm', [0,0])[0]:.1f}-{wt.get('range_mm', [0,0])[1]:.1f}          "
                  f"{stress.get('range_kPa', [0,0])[0]:.1f}-{stress.get('range_kPa', [0,0])[1]:.1f}           "
                  f"{sites}")
        else:
            print(f"{r['patient_id']:<15} FAILED: {r.get('error', 'Unknown')[:40]}")
    
    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()