#!/usr/bin/env python3
"""
COMPLETE LAPLACE-DIRICHLET CARDIAC ANALYSIS FRAMEWORK

COMPONENT 1: LAPLACE-DIRICHLET TRANSMURAL COORDINATE
    - Solve ∇²φ = 0 with Dirichlet BCs
    - φ ∈ [0,1] defines transmural position
    
COMPONENT 2: HELICAL FIBER RECONSTRUCTION VIA φ
    - α(φ) = α_endo + (α_epi - α_endo) × φ
    - Fiber: f = cos(α)·e_c + sin(α)·e_l
    - Modified in infarct regions
    
COMPONENT 3: WALL STRESS FROM MODIFIED LAPLACE LAW WITH CURVATURE TENSOR
    - Classical: σ = Pr/(2h)
    - Modified: σ = Pr/(2h) × f(κ₁, κ₂)
    - κ₁, κ₂ = principal curvatures from shape operator
    
COMPONENT 4: LAPLACIAN-GEODESIC INJECTION SITE OPTIMIZATION
    - Geodesic distance via Heat Method (Crane et al. 2013)
    - Optimization: min(geodesic_to_border) + max(stress_reduction)
    - Constraints: avoid core scar, prefer mid-wall

"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve, eigsh
from scipy.spatial import cKDTree
from collections import defaultdict
import os
import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings
warnings.filterwarnings('ignore')

# Try to import joblib for parallel loops (more notebook-friendly)
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False
    print("Note: joblib not available, using sequential processing for loops")

# PARALLEL PROCESSING CONFIGURATION
N_CPUS = 44  # Number of CPUs to use
PARALLEL_PATIENTS = True  # Process patients in parallel
PARALLEL_ELEMENTS = True  # Parallelize element loops within each patient

# CONFIGURATION
PATIENT_IDS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]

BASE_DIR = "/home/shadeform/SCD_MODELS"
OUTPUT_DIR = "/home/shadeform/SCD_MODELS/laplace_complete"

# Fiber parameters (Streeter et al. 1969)
FIBER_ANGLE_ENDO_DEG = 60.0    # Subendocardial: right-handed helix
FIBER_ANGLE_EPI_DEG = -60.0    # Subepicardial: left-handed helix

# Pressure for wall stress
LV_PRESSURE_KPA = 16.0         # Peak systolic

# Injection optimization
N_INJECTION_SITES = 5
MIN_SITE_SEPARATION_MM = 15.0

# Tissue tags
TAG_HEALTHY = 1
TAG_BORDER = 2
TAG_INFARCT = 3


# MESH LOADING
def load_mesh(patient_id, base_dir):
    """Load tetrahedral mesh"""
    pts_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.pts"
    elem_file = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.elem"
    
    # Load coordinates
    with open(pts_file, 'r') as f:
        n_nodes = int(f.readline().strip())
        coords = np.zeros((n_nodes, 3), dtype=np.float64)
        for i in range(n_nodes):
            coords[i] = [float(x) for x in f.readline().split()[:3]]
    
    # Load elements
    with open(elem_file, 'r') as f:
        n_elems = int(f.readline().strip())
        elements = np.zeros((n_elems, 4), dtype=np.int32)
        for i in range(n_elems):
            parts = f.readline().split()
            elements[i] = [int(x) for x in parts[1:5]]
    
    return coords, elements


def load_classification(patient_id, base_dir):
    """Load tissue classification from coronary territory-based detection"""
    # Try new coronary territory results first, then fall back to comprehensive
    class_dir = f"{base_dir}/infarct_coronary_territory/{patient_id}"
    elem_file = f"{class_dir}/{patient_id}_tagged.elem"
    
    if not os.path.exists(elem_file):
        # Fallback to old location
        class_dir = f"{base_dir}/infarct_results_comprehensive/{patient_id}"
        elem_file = f"{class_dir}/{patient_id}_tagged.elem"
    
    if not os.path.exists(elem_file):
        return None
    
    with open(elem_file, 'r') as f:
        n_elems = int(f.readline().strip())
        tags = np.zeros(n_elems, dtype=np.int32)
        for i in range(n_elems):
            parts = f.readline().split()
            tags[i] = int(parts[5])
    
    return tags


# COMPONENT 1: LAPLACE-DIRICHLET TRANSMURAL COORDINATE
def extract_surfaces(coords, elements):
    """Extract endocardial and epicardial surfaces"""
    # Find boundary faces
    face_count = defaultdict(list)
    for elem_idx, elem in enumerate(elements):
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]
        for face in faces:
            face_count[face].append(elem_idx)
    
    boundary_faces = [(f, elems[0]) for f, elems in face_count.items() if len(elems) == 1]
    
    # Get boundary nodes
    boundary_nodes = set()
    for face, _ in boundary_faces:
        boundary_nodes.update(face)
    boundary_nodes = np.array(list(boundary_nodes))
    
    # Classify by radial position
    z_vals = coords[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min
    
    endo_nodes = set()
    epi_nodes = set()
    endo_faces = []
    epi_faces = []
    
    for i in range(20):
        z_lo = z_min + i * z_range / 20
        z_hi = z_min + (i + 1) * z_range / 20
        
        slice_mask = (coords[boundary_nodes, 2] >= z_lo) & (coords[boundary_nodes, 2] < z_hi)
        slice_nodes = boundary_nodes[slice_mask]
        
        if len(slice_nodes) < 20:
            continue
        
        slice_coords = coords[slice_nodes, :2]
        center = np.median(slice_coords, axis=0)
        radii = np.linalg.norm(slice_coords - center, axis=1)
        
        r_30 = np.percentile(radii, 30)
        r_70 = np.percentile(radii, 70)
        
        for j, node in enumerate(slice_nodes):
            if radii[j] < r_30:
                endo_nodes.add(node)
            elif radii[j] > r_70:
                epi_nodes.add(node)
    
    # Classify faces
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


def build_fem_laplacian(coords, elements):
    """Build FEM Laplacian stiffness matrix"""
    n_nodes = len(coords)
    K = lil_matrix((n_nodes, n_nodes))
    
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
        
        for i in range(4):
            for j in range(4):
                K[elem[i], elem[j]] += Ke[i, j]
    
    return K.tocsr()


def solve_laplace_dirichlet(coords, elements, surfaces):
    """
    COMPONENT 1: Solve Laplace equation for transmural coordinate.
    
    Mathematical Formulation:
    ∇²φ = 0           in Ω (myocardium)
    φ = 0             on Γ_endo
    φ = 1             on Γ_epi
    
    Returns: φ(x) ∈ [0, 1] - smooth transmural scalar field
    """
    n_nodes = len(coords)
    
    print("      Building FEM Laplacian...")
    K = build_fem_laplacian(coords, elements)
    
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


def compute_phi_gradient(coords, elements, phi):
    """Compute gradient of transmural coordinate at each element"""
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


# COMPONENT 2: HELICAL FIBER RECONSTRUCTION VIA TRANSMURAL COORDINATE
def compute_local_coordinate_system(coords, elements, phi, grad_phi):
    """
    Compute local orthonormal coordinate system at each element.
    
    e_t: Transmural direction = ∇φ / |∇φ|
    e_l: Longitudinal direction (apex → base), orthogonalized
    e_c: Circumferential direction = e_l × e_t
    """
    n_elems = len(elements)
    
    # Transmural direction
    grad_norm = np.linalg.norm(grad_phi, axis=1, keepdims=True)
    grad_norm = np.maximum(grad_norm, 1e-10)
    e_t = grad_phi / grad_norm
    
    # Long axis direction (roughly +z, pointing base-ward)
    centroids = np.mean(coords[elements], axis=1)
    
    # Find apex and base
    z_vals = centroids[:, 2]
    apex_z = z_vals.min()
    base_z = z_vals.max()
    
    # Initial longitudinal direction (towards base)
    e_l_init = np.zeros((n_elems, 3))
    e_l_init[:, 2] = 1.0  # +z direction
    
    # Orthogonalize e_l to e_t
    e_l = np.zeros((n_elems, 3))
    for i in range(n_elems):
        # Gram-Schmidt: e_l = e_l_init - (e_l_init · e_t) e_t
        proj = np.dot(e_l_init[i], e_t[i])
        e_l[i] = e_l_init[i] - proj * e_t[i]
        norm = np.linalg.norm(e_l[i])
        if norm > 1e-10:
            e_l[i] /= norm
        else:
            # Fallback: use perpendicular direction
            e_l[i] = np.cross(e_t[i], [1, 0, 0])
            norm = np.linalg.norm(e_l[i])
            if norm > 1e-10:
                e_l[i] /= norm
    
    # Circumferential direction: e_c = e_l × e_t
    e_c = np.cross(e_l, e_t)
    e_c /= np.linalg.norm(e_c, axis=1, keepdims=True) + 1e-10
    
    return e_t, e_l, e_c


def reconstruct_helical_fibers(coords, elements, phi, grad_phi, tags=None,
                               alpha_endo=FIBER_ANGLE_ENDO_DEG,
                               alpha_epi=FIBER_ANGLE_EPI_DEG):
    """
    COMPONENT 2: Reconstruct myofibers via helical angle rotation.
    
    Mathematical Formulation:
    α(φ) = α_endo + (α_epi - α_endo) × φ
         = 60° - 120° × φ
    
    Fiber direction:
    f = cos(α) · e_c + sin(α) · e_l
    
    Sheet direction:
    s = f × e_t (perpendicular to fiber, in wall plane)
    
    In infarct regions:
    - Core: α = 0° (circumferential, no helical rotation)
    - Border: α reduced to ±30° (partial preservation)
    
    Reference: Streeter et al. "Fiber orientation in the canine left ventricle 
               during diastole and systole." Circ Res 1969.
    """
    n_elems = len(elements)
    
    print("      Computing local coordinate system...")
    e_t, e_l, e_c = compute_local_coordinate_system(coords, elements, phi, grad_phi)
    
    # Transmural position per element
    phi_elem = np.array([np.mean(phi[elem]) for elem in elements])
    
    # Fiber angle: linear interpolation through wall
    alpha_endo_rad = np.radians(alpha_endo)
    alpha_epi_rad = np.radians(alpha_epi)
    
    alpha = alpha_endo_rad + (alpha_epi_rad - alpha_endo_rad) * phi_elem
    
    # Modify in infarct regions
    if tags is not None:
        for i in range(n_elems):
            if tags[i] == TAG_INFARCT:
                # Core scar: no helical rotation (fibrotic, circumferential only)
                alpha[i] = 0.0
            elif tags[i] == TAG_BORDER:
                # Border zone: reduced rotation (50% of normal)
                alpha[i] = alpha[i] * 0.5
    
    # Compute fiber and sheet directions
    print("      Computing fiber orientations via helical rotation...")
    fibers = np.zeros((n_elems, 3))
    sheets = np.zeros((n_elems, 3))
    
    for i in range(n_elems):
        # Fiber: f = cos(α) e_c + sin(α) e_l
        fibers[i] = np.cos(alpha[i]) * e_c[i] + np.sin(alpha[i]) * e_l[i]
        fibers[i] /= np.linalg.norm(fibers[i]) + 1e-10
        
        # Sheet: s = f × e_t
        sheets[i] = np.cross(fibers[i], e_t[i])
        sheets[i] /= np.linalg.norm(sheets[i]) + 1e-10
    
    fiber_angles_deg = np.degrees(alpha)
    
    print(f"      Fiber angle range: {fiber_angles_deg.min():.1f}° to {fiber_angles_deg.max():.1f}°")
    
    return fibers, sheets, fiber_angles_deg, (e_t, e_l, e_c)


# COMPONENT 3: WALL STRESS FROM MODIFIED LAPLACE LAW WITH CURVATURE TENSOR
def compute_surface_curvature_tensor(coords, faces, vertex_normals):
    """
    Compute principal curvatures from the shape operator (Weingarten map).
    
    The shape operator S relates the change in normal to surface position:
    S = -dN/dX
    
    Principal curvatures κ₁, κ₂ are the eigenvalues of S.
    Mean curvature: H = (κ₁ + κ₂) / 2
    Gaussian curvature: K = κ₁ × κ₂
    """
    n_nodes = len(coords)
    
    # Build vertex-to-face connectivity
    node_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for node in face:
            node_faces[node].append(fi)
    
    kappa_1 = np.zeros(n_nodes)  # Max principal curvature
    kappa_2 = np.zeros(n_nodes)  # Min principal curvature
    mean_curvature = np.zeros(n_nodes)
    gaussian_curvature = np.zeros(n_nodes)
    
    for node in node_faces.keys():
        if len(node_faces[node]) < 3:
            continue
        
        p = coords[node]
        n = vertex_normals[node]
        
        if np.linalg.norm(n) < 0.1:
            continue
        
        # Find neighbors
        neighbors = set()
        for fi in node_faces[node]:
            neighbors.update(faces[fi])
        neighbors.discard(node)
        
        if len(neighbors) < 3:
            continue
        
        neighbor_list = list(neighbors)
        neighbor_coords = coords[neighbor_list]
        neighbor_normals = vertex_normals[neighbor_list]
        
        # Project to tangent plane
        diffs = neighbor_coords - p
        
        # Build tangent basis
        t1 = diffs[0] - np.dot(diffs[0], n) * n
        if np.linalg.norm(t1) < 1e-10:
            continue
        t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        
        # Fit shape operator using least squares
        # dn = S @ dx in tangent coordinates
        A = []
        b = []
        for i, neighbor in enumerate(neighbor_list):
            dx = diffs[i]
            dn = neighbor_normals[i] - n
            
            # Project to tangent plane
            dx_t = np.array([np.dot(dx, t1), np.dot(dx, t2)])
            dn_t = np.array([np.dot(dn, t1), np.dot(dn, t2)])
            
            if np.linalg.norm(dx_t) > 1e-10:
                # Shape operator equation: dn = -S @ dx
                A.append([dx_t[0], dx_t[1], 0, 0])
                A.append([0, 0, dx_t[0], dx_t[1]])
                b.append(-dn_t[0])
                b.append(-dn_t[1])
        
        if len(A) < 4:
            continue
        
        A = np.array(A)
        b = np.array(b)
        
        # Solve for shape operator components [S11, S12, S21, S22]
        try:
            S_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            S = np.array([[S_flat[0], S_flat[1]], 
                         [S_flat[2], S_flat[3]]])
            
            # Symmetrize (shape operator should be symmetric)
            S = (S + S.T) / 2
            
            # Eigenvalues = principal curvatures
            eigenvalues = np.linalg.eigvalsh(S)
            kappa_1[node] = np.max(eigenvalues)
            kappa_2[node] = np.min(eigenvalues)
            mean_curvature[node] = (kappa_1[node] + kappa_2[node]) / 2
            gaussian_curvature[node] = kappa_1[node] * kappa_2[node]
        except:
            pass
    
    return {
        'kappa_1': kappa_1,
        'kappa_2': kappa_2,
        'mean_curvature': mean_curvature,
        'gaussian_curvature': gaussian_curvature
    }


def compute_vertex_normals(coords, faces):
    """Compute area-weighted vertex normals from surface faces"""
    n_nodes = len(coords)
    vertex_normals = np.zeros((n_nodes, 3))
    
    for face in faces:
        v0, v1, v2 = coords[face]
        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        area = np.linalg.norm(n) / 2
        if area > 1e-10:
            n = n / (2 * area)  # Unit normal
            for node in face:
                vertex_normals[node] += n * area
    
    # Normalize
    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vertex_normals /= norms
    
    return vertex_normals


def compute_wall_stress_with_curvature(coords, elements, phi, grad_phi, surfaces,
                                        pressure=LV_PRESSURE_KPA, tags=None):
    """
    COMPONENT 3: Wall stress from modified Law of Laplace with curvature tensor.
    
    Mathematical Formulation:
    
    Classical Laplace Law (thin-walled sphere):
    σ = P × r / (2 × h)
    
    Modified with local curvature tensor:
    σ(x) = P × r_local(x) / (2 × h(x)) × f(κ₁, κ₂)
    
    Where:
    - h(x) = 1/|∇φ| = local wall thickness
    - r_local = 1/H = local radius of curvature (H = mean curvature)
    - κ₁, κ₂ = principal curvatures from shape operator
    - f(κ₁, κ₂) = curvature anisotropy factor = 1 + |κ₁/κ₂ - 1| × 0.5
    
    Reference: Zhong et al. "Finite element analysis of the stress distribution 
               in left ventricle aneurysm." Int J Cardiol 2008.
    """
    n_elems = len(elements)
    centroids = np.mean(coords[elements], axis=1)
    
    # Wall thickness from Laplace gradient: h = 1/|∇φ|
    grad_norm = np.linalg.norm(grad_phi, axis=1)
    grad_norm = np.maximum(grad_norm, 1e-8)
    wall_thickness = 1.0 / grad_norm
    
    # Clip to physiological bounds
    wall_thickness = np.clip(wall_thickness, 0.5, 25.0)
    
    print("      Computing surface curvature tensor...")
    
    # Compute curvature on endocardial surface
    if len(surfaces['endo_faces']) > 0:
        vertex_normals = compute_vertex_normals(coords, surfaces['endo_faces'])
        curvature = compute_surface_curvature_tensor(coords, surfaces['endo_faces'], vertex_normals)
    else:
        # Fallback: estimate from radial position
        curvature = {'kappa_1': np.zeros(len(coords)), 
                     'kappa_2': np.zeros(len(coords)),
                     'mean_curvature': np.zeros(len(coords))}
    
    # Interpolate curvature to elements
    kappa_1_elem = np.zeros(n_elems)
    kappa_2_elem = np.zeros(n_elems)
    mean_curv_elem = np.zeros(n_elems)
    
    for i, elem in enumerate(elements):
        kappa_1_elem[i] = np.mean(curvature['kappa_1'][elem])
        kappa_2_elem[i] = np.mean(curvature['kappa_2'][elem])
        mean_curv_elem[i] = np.mean(curvature['mean_curvature'][elem])
    
    # Local radius of curvature: r = 1/|H|
    mean_curv_elem = np.maximum(np.abs(mean_curv_elem), 1e-6)
    local_radius = 1.0 / mean_curv_elem
    local_radius = np.clip(local_radius, 10, 200)  # mm bounds
    
    # Curvature anisotropy factor: f(κ₁, κ₂) = 1 + |κ₁/κ₂ - 1| × 0.5
    kappa_2_safe = np.where(np.abs(kappa_2_elem) > 1e-8, kappa_2_elem, 1e-8)
    anisotropy = np.abs(kappa_1_elem / kappa_2_safe - 1.0)
    curvature_factor = 1.0 + anisotropy * 0.5
    curvature_factor = np.clip(curvature_factor, 1.0, 2.5)
    
    # For elements far from surface, use geometric estimate
    z_vals = centroids[:, 2]
    for i in range(n_elems):
        if local_radius[i] > 150 or local_radius[i] < 15:
            # Estimate from radial position at this z-level
            z = z_vals[i]
            z_mask = np.abs(z_vals - z) < 5.0
            if np.sum(z_mask) > 10:
                level_centroids = centroids[z_mask, :2]
                center = np.mean(level_centroids, axis=0)
                radii = np.linalg.norm(level_centroids - center, axis=1)
                local_radius[i] = np.mean(radii)
    
    # Modified Law of Laplace: σ = P × r / (2h) × f(κ)
    wall_stress = (pressure * local_radius) / (2 * wall_thickness) * curvature_factor
    
    # Modify for tissue type
    if tags is not None:
        for i in range(n_elems):
            if tags[i] == TAG_INFARCT:
                # Scar is stiffer, but load is transferred to border
                wall_stress[i] *= 0.7
            elif tags[i] == TAG_BORDER:
                # Stress concentration at border!
                wall_stress[i] *= 1.5
    
    print(f"      Wall thickness: {wall_thickness.min():.2f} - {wall_thickness.max():.2f} mm")
    print(f"      Local radius: {local_radius.min():.2f} - {local_radius.max():.2f} mm")
    print(f"      Wall stress: {wall_stress.min():.2f} - {wall_stress.max():.2f} kPa")
    
    return {
        'wall_stress': wall_stress,
        'wall_thickness': wall_thickness,
        'local_radius': local_radius,
        'curvature_factor': curvature_factor,
        'kappa_1': kappa_1_elem,
        'kappa_2': kappa_2_elem,
        'mean_curvature': mean_curv_elem
    }


# COMPONENT 4: LAPLACIAN-GEODESIC INJECTION SITE OPTIMIZATION
def build_mass_matrix(coords, elements):
    """Build lumped mass matrix for heat equation"""
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
    """
    Compute geodesic distance via the Heat Method (Crane et al. 2013).
    
    Algorithm:
    1. Solve heat equation: (M - tL)u = δ_source
    2. Compute normalized gradient: X = -∇u / |∇u|
    3. Solve Poisson equation: Lφ = ∇·X
    
    The solution φ gives approximate geodesic distances.
    
    Reference: Crane, Weischedel, Wardetzky. "Geodesics in Heat: A New Approach 
               to Computing Distance Based on Heat Flow." ACM TOG 2013.
    """
    n_nodes = len(coords)
    n_elems = len(elements)
    
    # Build mass matrix
    M = build_mass_matrix(coords, elements)
    
    # Estimate time step from mean edge length
    edge_lengths = []
    for elem in elements[:min(1000, n_elems)]:
        for i in range(4):
            for j in range(i+1, 4):
                edge_lengths.append(np.linalg.norm(coords[elem[i]] - coords[elem[j]]))
    h = np.mean(edge_lengths)
    t = t_factor * h * h
    
    # Step 1: Solve heat equation (M - tL)u = b
    A = M - t * L
    
    b = np.zeros(n_nodes)
    for node in source_nodes:
        b[node] = 1.0
    
    u = spsolve(A.tocsr(), b)
    
    # Step 2: Compute normalized gradient field
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
            X[i] = -grad_u / norm  # Normalize and flip
    
    # Step 3: Compute divergence and solve Poisson
    div_X = np.zeros(n_nodes)
    
    for i, elem in enumerate(elements):
        verts = coords[elem]
        J = np.array([verts[1] - verts[0], verts[2] - verts[0], verts[3] - verts[0]]).T
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
        
        for j in range(4):
            div_X[elem[j]] += vol * np.dot(dN[j], X[i])
    
    # Fix one node for Poisson equation
    L_mod = L.tolil()
    ref_node = source_nodes[0]
    L_mod[ref_node, :] = 0
    L_mod[ref_node, ref_node] = 1
    div_X[ref_node] = 0
    
    phi = spsolve(L_mod.tocsr(), div_X)
    phi = phi - phi[source_nodes].min()
    
    return phi


def optimize_injection_sites(coords, elements, tags, wall_stress, phi_trans,
                             geodesic_to_border, centroids, n_sites=5):
    """
    COMPONENT 4: Laplacian-geodesic optimization for injection sites.
    
    Optimization Objective:
    Maximize: score(x) = w₁ × proximity_to_border + w₂ × stress_reduction + 
                         w₃ × mid_wall_preference + w₄ × border_bonus
    
    Subject to:
    - Avoid core scar (no perfusion → injection won't diffuse)
    - Maintain spatial separation between sites (min_separation)
    - Prefer border zone (therapeutic target)
    - Prefer mid-wall (φ ≈ 0.5) for better distribution
    
    The geodesic distance ensures optimal path lengths on the manifold,
    while the stress component targets high-risk regions for remodeling.
    """
    n_elems = len(elements)
    
    # Transmural position per element
    phi_elem = np.array([np.mean(phi_trans[elem]) for elem in elements])
    
    # Geodesic distance to elements (average of node distances)
    geodesic_elem = np.array([np.mean(geodesic_to_border[elem]) for elem in elements])
    
    # Normalize stress
    stress_norm = wall_stress['wall_stress'] / np.median(wall_stress['wall_stress'])
    
    # Initialize scores
    scores = np.zeros(n_elems)
    
    for i in range(n_elems):
        # EXCLUDE: Core scar (no perfusion)
        if tags[i] == TAG_INFARCT:
            scores[i] = -np.inf
            continue
        
        # Component 1: Proximity to border zone (minimize geodesic distance)
        # Higher score when CLOSER to border
        dist_score = 1.0 / (geodesic_elem[i] + 1.0)
        
        # Component 2: Wall stress reduction potential (target high stress)
        stress_score = stress_norm[i]
        
        # Component 3: Mid-wall preference (φ ≈ 0.5)
        mid_wall_score = 1.0 - 2.0 * abs(phi_elem[i] - 0.5)
        
        # Component 4: Border zone bonus
        border_bonus = 2.0 if tags[i] == TAG_BORDER else 1.0
        
        # Combined score with weights
        scores[i] = (0.3 * dist_score + 
                     0.3 * stress_score + 
                     0.2 * mid_wall_score + 
                     0.2 * border_bonus)
    
    # Select top sites with spatial diversity
    selected_sites = []
    sorted_indices = np.argsort(scores)[::-1]
    
    for idx in sorted_indices:
        if len(selected_sites) >= n_sites:
            break
        
        if scores[idx] == -np.inf:
            continue
        
        # Check spatial separation
        centroid = centroids[idx]
        too_close = False
        
        for site_idx in selected_sites:
            dist = np.linalg.norm(centroid - centroids[site_idx])
            if dist < MIN_SITE_SEPARATION_MM:
                too_close = True
                break
        
        if not too_close:
            selected_sites.append(idx)
    
    # Compile results
    injection_sites = []
    for idx in selected_sites:
        injection_sites.append({
            'element_id': int(idx),
            'coordinates': centroids[idx].tolist(),
            'score': float(scores[idx]),
            'transmural_position': float(phi_elem[idx]),
            'wall_stress_kPa': float(wall_stress['wall_stress'][idx]),
            'geodesic_distance': float(geodesic_elem[idx]),
            'tissue_type': 'border' if tags[idx] == TAG_BORDER else 'healthy'
        })
    
    return injection_sites


# OUTPUT FUNCTIONS
def write_vtk_complete(filepath, coords, elements, scalars, vectors):
    """Write VTK with all computed fields"""
    n_nodes, n_elems = len(coords), len(elements)
    
    with open(filepath, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Complete Laplace-Dirichlet Analysis\n")
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
        f.write("2\n")  # 2 directions
        for fiber, sheet in zip(fibers, sheets):
            f.write(f"{fiber[0]:.6f} {fiber[1]:.6f} {fiber[2]:.6f} "
                   f"{sheet[0]:.6f} {sheet[1]:.6f} {sheet[2]:.6f}\n")


def write_injection_coords(filepath, sites):
    """Write injection site coordinates"""
    with open(filepath, 'w') as f:
        f.write("# Optimal injection sites from Laplacian-geodesic optimization\n")
        f.write("# X Y Z score stress_kPa geodesic_dist tissue_type\n")
        for site in sites:
            c = site['coordinates']
            f.write(f"{c[0]:.4f} {c[1]:.4f} {c[2]:.4f} "
                   f"{site['score']:.4f} {site['wall_stress_kPa']:.2f} "
                   f"{site['geodesic_distance']:.2f} {site['tissue_type']}\n")


# MAIN PIPELINE
def process_patient_complete(patient_id):
    """
    Complete Laplace-Dirichlet analysis implementing ALL components from the abstract.
    """
    print(f"PROCESSING: {patient_id}")
    
    results = {'patient_id': patient_id}
    
    # Load mesh
    print("\n  Loading mesh...")
    coords, elements = load_mesh(patient_id, BASE_DIR)
    n_nodes, n_elems = len(coords), len(elements)
    print(f"      {n_nodes:,} nodes, {n_elems:,} elements")
    
    centroids = np.mean(coords[elements], axis=1)
    
    # Load tissue classification
    tags = load_classification(patient_id, BASE_DIR)
    if tags is None:
        print("      No classification found - using all healthy")
        tags = np.ones(n_elems, dtype=np.int32)
    else:
        print(f"      Classification loaded: {np.sum(tags==TAG_INFARCT)} infarct, "
              f"{np.sum(tags==TAG_BORDER)} border")
    
    # Extract surfaces
    print("\n  [1] LAPLACE-DIRICHLET TRANSMURAL COORDINATE")
    print("      Extracting surfaces...")
    surfaces = extract_surfaces(coords, elements)
    print(f"      Endo: {len(surfaces['endo_nodes'])} nodes, "
          f"Epi: {len(surfaces['epi_nodes'])} nodes")
    
    # Solve Laplace-Dirichlet
    phi = solve_laplace_dirichlet(coords, elements, surfaces)
    print(f"      φ range: [{phi.min():.4f}, {phi.max():.4f}]")
    
    # Compute gradient
    grad_phi = compute_phi_gradient(coords, elements, phi)
    
    # COMPONENT 2: Helical Fiber Reconstruction
    print("\n  [2] HELICAL FIBER RECONSTRUCTION")
    fibers, sheets, fiber_angles, coord_system = reconstruct_helical_fibers(
        coords, elements, phi, grad_phi, tags,
        FIBER_ANGLE_ENDO_DEG, FIBER_ANGLE_EPI_DEG
    )
    
    # COMPONENT 3: Wall Stress with Curvature Tensor
    print("\n  [3] WALL STRESS (MODIFIED LAPLACE + CURVATURE TENSOR)")
    stress_data = compute_wall_stress_with_curvature(
        coords, elements, phi, grad_phi, surfaces, LV_PRESSURE_KPA, tags
    )
    
    # COMPONENT 4: Laplacian-Geodesic Injection Optimization
    print("\n  [4] LAPLACIAN-GEODESIC INJECTION OPTIMIZATION")
    
    # Compute geodesic distance to border zone
    border_elements = np.where(tags == TAG_BORDER)[0]
    if len(border_elements) > 0:
        border_nodes = list(set(elements[border_elements[:100]].flatten()))
        
        print("      Computing geodesic distances (Heat Method)...")
        L = build_fem_laplacian(coords, elements)
        geodesic_to_border = compute_geodesic_heat_method(coords, elements, border_nodes, L)
        print(f"      Geodesic range: [{geodesic_to_border.min():.2f}, "
              f"{geodesic_to_border.max():.2f}]")
    else:
        print("      No border zone - using endo distance")
        geodesic_to_border = np.zeros(n_nodes)
        endo_tree = cKDTree(coords[surfaces['endo_nodes']])
        for i, c in enumerate(coords):
            d, _ = endo_tree.query(c)
            geodesic_to_border[i] = d
    
    # Optimize injection sites
    print("      Optimizing injection sites...")
    injection_sites = optimize_injection_sites(
        coords, elements, tags, stress_data, phi,
        geodesic_to_border, centroids, N_INJECTION_SITES
    )
    
    print(f"      Selected {len(injection_sites)} optimal sites:")
    for site in injection_sites:
        print(f"        Element {site['element_id']}: "
              f"score={site['score']:.3f}, stress={site['wall_stress_kPa']:.1f}kPa, "
              f"type={site['tissue_type']}")
    
    # Save outputs
    print("\n  [5] SAVING OUTPUTS")
    
    out_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    
    # Transmural position per element
    phi_elem = np.array([np.mean(phi[elem]) for elem in elements])
    
    # VTK with all fields
    write_vtk_complete(
        os.path.join(out_dir, f"{patient_id}_complete_analysis.vtk"),
        coords, elements,
        scalars={
            'TransmuralPhi': phi,  # Node
            'TransmuralPhi_elem': phi_elem,  # Cell
            'TissueType': tags.astype(float),
            'FiberAngle_deg': fiber_angles,
            'WallThickness_mm': stress_data['wall_thickness'],
            'WallStress_kPa': stress_data['wall_stress'],
            'LocalRadius_mm': stress_data['local_radius'],
            'CurvatureFactor': stress_data['curvature_factor'],
            'Kappa1': stress_data['kappa_1'],
            'Kappa2': stress_data['kappa_2'],
        },
        vectors={
            'FiberDirection': fibers,
            'SheetDirection': sheets,
            'TransmuralDirection': coord_system[0],
        }
    )
    
    # Fiber file (CARP format)
    write_lon_file(os.path.join(out_dir, f"{patient_id}_reconstructed.lon"), fibers, sheets)
    
    # Injection sites
    with open(os.path.join(out_dir, f"{patient_id}_injection_sites.json"), 'w') as f:
        json.dump(injection_sites, f, indent=2)
    
    write_injection_coords(
        os.path.join(out_dir, f"{patient_id}_injection_coords.txt"),
        injection_sites
    )
    
    # Summary
    summary = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'methodology': {
            'component_1': 'Laplace-Dirichlet transmural coordinate (∇²φ=0)',
            'component_2': 'Helical fiber reconstruction (α(φ) = 60° - 120°×φ)',
            'component_3': 'Wall stress with curvature tensor (σ = Pr/(2h)×f(κ₁,κ₂))',
            'component_4': 'Laplacian-geodesic injection optimization (Heat Method)'
        },
        'mesh': {
            'n_nodes': int(n_nodes),
            'n_elements': int(n_elems),
        },
        'transmural': {
            'phi_range': [float(phi.min()), float(phi.max())],
        },
        'fibers': {
            'angle_range_deg': [float(fiber_angles.min()), float(fiber_angles.max())],
            'endo_angle': FIBER_ANGLE_ENDO_DEG,
            'epi_angle': FIBER_ANGLE_EPI_DEG,
        },
        'wall_stress': {
            'range_kPa': [float(stress_data['wall_stress'].min()), 
                         float(stress_data['wall_stress'].max())],
            'mean_kPa': float(np.mean(stress_data['wall_stress'])),
            'pressure_kPa': LV_PRESSURE_KPA,
        },
        'wall_thickness': {
            'range_mm': [float(stress_data['wall_thickness'].min()),
                        float(stress_data['wall_thickness'].max())],
            'mean_mm': float(np.mean(stress_data['wall_thickness'])),
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
        result = process_patient_complete(patient_id)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'patient_id': patient_id, 'status': 'FAILED', 'error': str(e)}


def main():
    """Main entry point with parallel processing"""
    print("COMPLETE LAPLACE-DIRICHLET CARDIAC ANALYSIS FRAMEWORK")
    
    print(f"\n  Using {N_CPUS} CPUs for parallel processing")
    
    print("\nIMPLEMENTED COMPONENTS (from abstract):")
    print("  [1] Laplace-Dirichlet transmural coordinate: ∇²φ = 0")
    print("  [2] Helical fiber reconstruction: α(φ) = α_endo + (α_epi - α_endo)×φ")
    print("  [3] Wall stress with curvature tensor: σ = Pr/(2h) × f(κ₁,κ₂)")
    print("  [4] Laplacian-geodesic injection optimization: Heat Method + stress")
    
    print("\n  Loading tissue tags from: infarct_coronary_territory/")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set environment variables for multi-threaded linear algebra
    import os as os_env
    threads_per_patient = max(1, N_CPUS // len(PATIENT_IDS))
    os_env.environ['OMP_NUM_THREADS'] = str(threads_per_patient)
    os_env.environ['MKL_NUM_THREADS'] = str(threads_per_patient)
    os_env.environ['OPENBLAS_NUM_THREADS'] = str(threads_per_patient)
    os_env.environ['NUMEXPR_NUM_THREADS'] = str(threads_per_patient)
    
    print(f"  Processing {len(PATIENT_IDS)} patients in parallel...")
    print(f"  Threads per patient: {threads_per_patient}")
    
    # Process patients in parallel
    n_workers = min(N_CPUS, len(PATIENT_IDS))
    
    with Pool(processes=n_workers) as pool:
        all_results = pool.map(process_patient_wrapper, PATIENT_IDS)
    
    # Summary table
    print("SUMMARY")
    
    print(f"\n{'Patient':<15} {'φ range':<15} {'Fiber°':<15} {'Stress kPa':<15} {'Sites':<10}")
    
    for r in all_results:
        if 'status' not in r:
            phi_r = r.get('transmural', {}).get('phi_range', [0, 1])
            fib_r = r.get('fibers', {}).get('angle_range_deg', [-60, 60])
            stress_r = r.get('wall_stress', {}).get('range_kPa', [0, 0])
            sites = len(r.get('injection_sites', []))
            print(f"{r['patient_id']:<15} "
                  f"[{phi_r[0]:.2f}, {phi_r[1]:.2f}]     "
                  f"[{fib_r[0]:.0f}°, {fib_r[1]:.0f}°]    "
                  f"[{stress_r[0]:.1f}, {stress_r[1]:.1f}]    "
                  f"{sites}")
        else:
            print(f"{r['patient_id']:<15} FAILED: {r.get('error', 'Unknown')[:40]}")
    
    # Save combined
    with open(os.path.join(OUTPUT_DIR, "complete_analysis_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


if __name__ == "__main__":
    results = main()


def main_notebook():
    """
    Alternative main function optimized for Jupyter notebooks.
    Uses joblib instead of multiprocessing Pool (more compatible with notebooks).
    """
    print("COMPLETE LAPLACE-DIRICHLET CARDIAC ANALYSIS FRAMEWORK")
    
    print(f"\n  Using {N_CPUS} CPUs for parallel processing (notebook mode)")
    
    print("\nIMPLEMENTED COMPONENTS (from abstract):")
    print("  [1] Laplace-Dirichlet transmural coordinate: ∇²φ = 0")
    print("  [2] Helical fiber reconstruction: α(φ) = α_endo + (α_epi - α_endo)×φ")
    print("  [3] Wall stress with curvature tensor: σ = Pr/(2h) × f(κ₁,κ₂)")
    print("  [4] Laplacian-geodesic injection optimization: Heat Method + stress")
    
    print("\n  Loading tissue tags from: infarct_coronary_territory/")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set environment variables for multi-threaded linear algebra
    os.environ['OMP_NUM_THREADS'] = str(N_CPUS)
    os.environ['MKL_NUM_THREADS'] = str(N_CPUS)
    os.environ['OPENBLAS_NUM_THREADS'] = str(N_CPUS)
    os.environ['NUMEXPR_NUM_THREADS'] = str(N_CPUS)
    
    print(f"  Processing {len(PATIENT_IDS)} patients...")
    
    if HAS_JOBLIB:
        # Use joblib for parallel processing (notebook-friendly)
        all_results = Parallel(n_jobs=min(N_CPUS, len(PATIENT_IDS)), verbose=10)(
            delayed(process_patient_wrapper)(pid) for pid in PATIENT_IDS
        )
    else:
        # Sequential fallback
        all_results = []
        for pid in PATIENT_IDS:
            result = process_patient_wrapper(pid)
            all_results.append(result)
    
    # Summary table
    print("SUMMARY")
    
    print(f"\n{'Patient':<15} {'φ range':<15} {'Fiber°':<15} {'Stress kPa':<15} {'Sites':<10}")
    
    for r in all_results:
        if 'status' not in r:
            phi_r = r.get('transmural', {}).get('phi_range', [0, 1])
            fib_r = r.get('fibers', {}).get('angle_range_deg', [-60, 60])
            stress_r = r.get('wall_stress', {}).get('range_kPa', [0, 0])
            sites = len(r.get('injection_sites', []))
            print(f"{r['patient_id']:<15} "
                  f"[{phi_r[0]:.2f}, {phi_r[1]:.2f}]     "
                  f"[{fib_r[0]:.0f}°, {fib_r[1]:.0f}°]    "
                  f"[{stress_r[0]:.1f}, {stress_r[1]:.1f}]    "
                  f"{sites}")
        else:
            print(f"{r['patient_id']:<15} FAILED: {r.get('error', 'Unknown')[:40]}")
    
    # Save combined
    with open(os.path.join(OUTPUT_DIR, "complete_analysis_summary.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return all_results


# For notebook usage:
# from laplace_dirichlet_complete import main_notebook
# results = main_notebook()