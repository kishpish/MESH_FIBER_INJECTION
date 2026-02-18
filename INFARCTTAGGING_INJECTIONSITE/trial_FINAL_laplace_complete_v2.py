#!/usr/bin/env python3
"""
LAPLACE-DIRICHLET CARDIAC ANALYSIS FRAMEWORK


Complete computational pipeline for cardiac mechanics analysis:

1. TRANSMURAL COORDINATE
   - Solve Laplace equation ∇²φ = 0 with Dirichlet boundary conditions
   - φ ∈ [0,1] defines position through myocardial wall

2. MYOFIBER RECONSTRUCTION  
   - Helical angle rotation: α(φ) = α_endo + (α_epi - α_endo) × φ
   - Fiber direction: f = cos(α)·e_c + sin(α)·e_l

3. WALL STRESS COMPUTATION
   - Modified Law of Laplace with local curvature tensor
   - Shape operator computation for principal curvatures κ₁, κ₂
   - σ = Pr/(2h) × f(κ₁, κ₂)

4. INJECTION SITE OPTIMIZATION
   - Geodesic distance computation via Heat Method
   - Multi-objective optimization: minimize geodesic, maximize stress reduction

References:
- Bishop et al. Am J Physiol Heart Circ Physiol 2010 (Laplace-Dirichlet)
- Streeter et al. Circ Res 1969 (Helical fiber architecture)
- Crane et al. ACM TOG 2013 (Heat Method for geodesics)
- Zhong et al. Int J Cardiol 2008 (Wall stress analysis)

"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
from collections import defaultdict
import os
import json
from datetime import datetime
from multiprocessing import Pool
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
N_CPUS = 44

PATIENT_IDS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]

BASE_DIR = "/home/shadeform/SCD_MODELS"
OUTPUT_DIR = "/home/shadeform/SCD_MODELS/laplace_complete_v2"

# Fiber angle parameters (Streeter et al. 1969)
FIBER_ANGLE_ENDO_DEG = 60.0
FIBER_ANGLE_EPI_DEG = -60.0

# Biomechanical parameters
LV_PRESSURE_KPA = 16.0
INJECTION_STIFFENING_FACTOR = 2.0

# Optimization parameters
N_INJECTION_SITES = 5
MIN_SITE_SEPARATION_MM = 15.0

# Tissue classification tags
TAG_HEALTHY = 1
TAG_BORDER = 2
TAG_INFARCT = 3


# MESH I/O
def load_mesh(patient_id, base_dir):
    """Load tetrahedral mesh from CARP format files."""
    pts = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.pts"
    elem = f"{base_dir}/simulation_ready/{patient_id}/{patient_id}_tet.elem"
    
    with open(pts) as f:
        n = int(f.readline())
        coords = np.array([[float(x) for x in f.readline().split()[:3]] for _ in range(n)])
    
    with open(elem) as f:
        n = int(f.readline())
        elements = np.array([[int(x) for x in f.readline().split()[1:5]] for _ in range(n)], dtype=np.int32)
    
    return coords, elements


def load_tags(patient_id, base_dir):
    """Load tissue classification tags."""
    path = f"{base_dir}/infarct_results_comprehensive/{patient_id}/{patient_id}_tagged.elem"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        n = int(f.readline())
        return np.array([int(f.readline().split()[5]) for _ in range(n)], dtype=np.int32)


# SURFACE EXTRACTION
def extract_surfaces(coords, elements):
    """Extract endocardial and epicardial surface meshes."""
    face_count = defaultdict(list)
    for ei, elem in enumerate(elements):
        for fi in [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]:
            face_count[tuple(sorted(elem[fi]))].append(ei)
    
    boundary_faces = [(f, e[0]) for f, e in face_count.items() if len(e) == 1]
    boundary_nodes = np.array(list(set(n for f, _ in boundary_faces for n in f)))
    
    z = coords[:, 2]
    z_min, z_max = z.min(), z.max()
    endo, epi = set(), set()
    endo_f, epi_f = [], []
    
    for i in range(25):
        lo, hi = z_min + i*(z_max-z_min)/25, z_min + (i+1)*(z_max-z_min)/25
        mask = (coords[boundary_nodes, 2] >= lo) & (coords[boundary_nodes, 2] < hi)
        nodes = boundary_nodes[mask]
        if len(nodes) < 20:
            continue
        
        xy = coords[nodes, :2]
        center = np.median(xy, axis=0)
        r = np.linalg.norm(xy - center, axis=1)
        r30, r70 = np.percentile(r, [30, 70])
        
        endo.update(nodes[r < r30])
        epi.update(nodes[r > r70])
    
    for face, _ in boundary_faces:
        if all(n in endo for n in face):
            endo_f.append(list(face))
        elif all(n in epi for n in face):
            epi_f.append(list(face))
    
    return {
        'endo_nodes': np.array(list(endo)),
        'epi_nodes': np.array(list(epi)),
        'endo_faces': np.array(endo_f) if endo_f else np.zeros((0, 3), dtype=int),
        'epi_faces': np.array(epi_f) if epi_f else np.zeros((0, 3), dtype=int),
    }


# LAPLACE-DIRICHLET TRANSMURAL COORDINATE
def build_laplacian(coords, elements):
    """
    Build FEM Laplacian stiffness matrix.
    
    For linear tetrahedral elements:
    K_ij = ∫_Ω ∇N_i · ∇N_j dV
    """
    n = len(coords)
    K = lil_matrix((n, n))
    
    for elem in elements:
        X = coords[elem]
        J = np.array([X[1] - X[0], X[2] - X[0], X[3] - X[0]]).T
        det = np.linalg.det(J)
        if abs(det) < 1e-15:
            continue
        
        vol = abs(det) / 6.0
        Jinv = np.linalg.inv(J)
        
        # Shape function gradients
        dN = np.zeros((4, 3))
        dN[0] = -Jinv.sum(axis=1)
        dN[1:4] = Jinv.T
        
        # Element stiffness matrix
        Ke = vol * (dN @ dN.T)
        for i in range(4):
            for j in range(4):
                K[elem[i], elem[j]] += Ke[i, j]
    
    return K.tocsr()


def solve_laplace_dirichlet(coords, elements, surfaces):
    """
    Solve Laplace equation for transmural coordinate.
    
    ∇²φ = 0 in Ω
    φ = 0 on Γ_endo (endocardium)
    φ = 1 on Γ_epi (epicardium)
    
    Returns transmural coordinate φ(x) ∈ [0,1].
    """
    n = len(coords)
    
    print("      Building FEM Laplacian...")
    K = build_laplacian(coords, elements)
    
    endo, epi = set(surfaces['endo_nodes']), set(surfaces['epi_nodes'])
    print(f"      Dirichlet BCs: {len(endo)} endo, {len(epi)} epi")
    
    # Apply Dirichlet BCs via penalty method
    K_mod = K.tolil()
    rhs = np.zeros(n)
    penalty = 1e12
    
    for node in endo:
        K_mod[node, :] = 0
        K_mod[node, node] = penalty
        rhs[node] = 0.0 * penalty
    
    for node in epi:
        K_mod[node, :] = 0
        K_mod[node, node] = penalty
        rhs[node] = 1.0 * penalty
    
    print("      Solving Laplace equation...")
    phi = spsolve(K_mod.tocsr(), rhs)
    return np.clip(phi, 0, 1)


def compute_gradient(coords, elements, phi):
    """Compute gradient ∇φ at element centroids."""
    grad = np.zeros((len(elements), 3))
    for i, elem in enumerate(elements):
        X = coords[elem]
        J = np.array([X[1]-X[0], X[2]-X[0], X[3]-X[0]]).T
        det = np.linalg.det(J)
        if abs(det) < 1e-15:
            continue
        Jinv = np.linalg.inv(J)
        dp = np.array([phi[elem[j]] - phi[elem[0]] for j in [1,2,3]])
        grad[i] = Jinv @ dp
    return grad


# HELICAL FIBER RECONSTRUCTION
def reconstruct_helical_fibers(coords, elements, phi, grad_phi, tags=None):
    """
    Reconstruct myofiber orientations using helical angle rotation.
    
    The fiber angle varies linearly through the wall:
    α(φ) = α_endo + (α_epi - α_endo) × φ
    
    Fiber direction in local coordinates:
    f = cos(α)·e_c + sin(α)·e_l
    
    Where:
    - e_t: transmural direction (∇φ/|∇φ|)
    - e_l: longitudinal direction (apex to base)
    - e_c: circumferential direction (e_l × e_t)
    
    In infarct regions, fibers are modified:
    - Core scar: α = 0° (circumferential, no rotation)
    - Border zone: α reduced by 50%
    """
    n = len(elements)
    
    # Transmural direction e_t = ∇φ/|∇φ|
    norm = np.linalg.norm(grad_phi, axis=1, keepdims=True)
    norm = np.maximum(norm, 1e-10)
    e_t = grad_phi / norm
    
    # Longitudinal direction (orthogonalized to e_t)
    e_l = np.zeros((n, 3))
    e_l[:, 2] = 1.0
    proj = np.sum(e_l * e_t, axis=1, keepdims=True)
    e_l = e_l - proj * e_t
    e_l = e_l / (np.linalg.norm(e_l, axis=1, keepdims=True) + 1e-10)
    
    # Circumferential direction e_c = e_l × e_t
    e_c = np.cross(e_l, e_t)
    e_c = e_c / (np.linalg.norm(e_c, axis=1, keepdims=True) + 1e-10)
    
    # Fiber angle: α(φ) = 60° - 120°×φ
    phi_elem = np.mean(phi[elements], axis=1)
    alpha = np.radians(FIBER_ANGLE_ENDO_DEG + (FIBER_ANGLE_EPI_DEG - FIBER_ANGLE_ENDO_DEG) * phi_elem)
    
    # Modify in infarct regions
    if tags is not None:
        alpha[tags == TAG_INFARCT] = 0.0
        alpha[tags == TAG_BORDER] *= 0.5
    
    # Fiber direction: f = cos(α)·e_c + sin(α)·e_l
    cos_a, sin_a = np.cos(alpha)[:, np.newaxis], np.sin(alpha)[:, np.newaxis]
    fibers = cos_a * e_c + sin_a * e_l
    fibers = fibers / (np.linalg.norm(fibers, axis=1, keepdims=True) + 1e-10)
    
    # Sheet direction: s = f × e_t
    sheets = np.cross(fibers, e_t)
    sheets = sheets / (np.linalg.norm(sheets, axis=1, keepdims=True) + 1e-10)
    
    return fibers, sheets, np.degrees(alpha), (e_t, e_l, e_c)


# CURVATURE TENSOR (SHAPE OPERATOR)
def compute_vertex_normals(coords, faces):
    """Compute area-weighted vertex normals from surface mesh."""
    n = len(coords)
    normals = np.zeros((n, 3))
    
    for face in faces:
        v0, v1, v2 = coords[face]
        fn = np.cross(v1 - v0, v2 - v0)
        area = np.linalg.norm(fn) / 2
        if area > 1e-10:
            fn = fn / (2 * area)
            for node in face:
                normals[node] += fn * area
    
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.maximum(norms, 1e-10)


def compute_shape_operator(coords, faces, vertex_normals):
    """
    Compute the shape operator (Weingarten map) at surface vertices.
    
    The shape operator S relates normal variation to surface position:
    S = -∂n/∂x
    
    Principal curvatures κ₁, κ₂ are the eigenvalues of S.
    Mean curvature: H = (κ₁ + κ₂)/2
    Gaussian curvature: K = κ₁ × κ₂
    """
    n = len(coords)
    
    node_faces = defaultdict(list)
    for fi, face in enumerate(faces):
        for node in face:
            node_faces[node].append(fi)
    
    kappa_1 = np.zeros(n)
    kappa_2 = np.zeros(n)
    mean_curv = np.zeros(n)
    gauss_curv = np.zeros(n)
    
    for node in node_faces.keys():
        if len(node_faces[node]) < 3:
            continue
        
        p = coords[node]
        normal = vertex_normals[node]
        
        if np.linalg.norm(normal) < 0.1:
            continue
        
        # Find one-ring neighbors
        neighbors = set()
        for fi in node_faces[node]:
            neighbors.update(faces[fi])
        neighbors.discard(node)
        
        if len(neighbors) < 3:
            continue
        
        neighbor_list = list(neighbors)
        diffs = coords[neighbor_list] - p
        neighbor_normals = vertex_normals[neighbor_list]
        
        # Build local tangent basis
        t1 = diffs[0] - np.dot(diffs[0], normal) * normal
        if np.linalg.norm(t1) < 1e-10:
            continue
        t1 = t1 / np.linalg.norm(t1)
        t2 = np.cross(normal, t1)
        
        # Fit shape operator via least squares: dn = -S @ dx
        A, b = [], []
        for i in range(len(neighbor_list)):
            dx = diffs[i]
            dn = neighbor_normals[i] - normal
            
            dx_t = np.array([np.dot(dx, t1), np.dot(dx, t2)])
            dn_t = np.array([np.dot(dn, t1), np.dot(dn, t2)])
            
            if np.linalg.norm(dx_t) > 1e-10:
                A.append([dx_t[0], dx_t[1], 0, 0])
                A.append([0, 0, dx_t[0], dx_t[1]])
                b.append(-dn_t[0])
                b.append(-dn_t[1])
        
        if len(A) < 4:
            continue
        
        try:
            S_flat, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
            S = np.array([[S_flat[0], S_flat[1]], [S_flat[2], S_flat[3]]])
            S = (S + S.T) / 2  # Symmetrize
            
            # Principal curvatures = eigenvalues
            eig = np.linalg.eigvalsh(S)
            kappa_1[node] = np.max(eig)
            kappa_2[node] = np.min(eig)
            mean_curv[node] = (kappa_1[node] + kappa_2[node]) / 2
            gauss_curv[node] = kappa_1[node] * kappa_2[node]
        except:
            pass
    
    return {'kappa_1': kappa_1, 'kappa_2': kappa_2, 
            'mean_curvature': mean_curv, 'gaussian_curvature': gauss_curv}


# WALL STRESS COMPUTATION
def compute_wall_stress(coords, elements, surfaces, tags=None):
    """
    Compute wall stress using modified Law of Laplace with curvature tensor.
    
    Classical Laplace law for thin-walled pressure vessel:
    σ = P × r / (2 × h)
    
    Modified with local curvature:
    σ = P × r_local / (2 × h) × f(κ₁, κ₂)
    
    Where:
    - r_local = 1/|H| (local radius from mean curvature)
    - h = wall thickness (geometric)
    - f(κ₁, κ₂) = 1 + |κ₁/κ₂ - 1| × 0.5 (curvature anisotropy factor)
    """
    n_elems = len(elements)
    centroids = np.mean(coords[elements], axis=1)
    
    # Geometric wall thickness
    print("      Computing geometric wall thickness...")
    endo_tree = cKDTree(coords[surfaces['endo_nodes']])
    epi_tree = cKDTree(coords[surfaces['epi_nodes']])
    
    endo_dist = np.array([endo_tree.query(c)[0] for c in centroids])
    epi_dist = np.array([epi_tree.query(c)[0] for c in centroids])
    wall_thickness = np.clip(endo_dist + epi_dist, 2.0, 25.0)
    
    # Shape operator for curvature tensor
    print("      Computing shape operator (curvature tensor)...")
    if len(surfaces['endo_faces']) > 0:
        vertex_normals = compute_vertex_normals(coords, surfaces['endo_faces'])
        curvature = compute_shape_operator(coords, surfaces['endo_faces'], vertex_normals)
    else:
        curvature = {'kappa_1': np.zeros(len(coords)), 'kappa_2': np.zeros(len(coords)),
                     'mean_curvature': np.zeros(len(coords))}
    
    # Interpolate curvature to elements
    kappa_1 = np.array([np.mean(curvature['kappa_1'][e]) for e in elements])
    kappa_2 = np.array([np.mean(curvature['kappa_2'][e]) for e in elements])
    mean_curv = np.array([np.mean(curvature['mean_curvature'][e]) for e in elements])
    
    # Local radius = 1/|H|
    local_radius = 1.0 / np.maximum(np.abs(mean_curv), 1e-6)
    local_radius = np.clip(local_radius, 10, 200)
    
    # Fallback for elements with poor curvature estimates
    z = centroids[:, 2]
    for i in range(n_elems):
        if local_radius[i] > 150:
            mask = np.abs(z - z[i]) < 5.0
            if np.sum(mask) > 10:
                xy = centroids[mask, :2]
                center = np.mean(xy, axis=0)
                local_radius[i] = np.mean(np.linalg.norm(xy - center, axis=1))
    
    # Curvature anisotropy factor: f(κ₁, κ₂) = 1 + |κ₁/κ₂ - 1| × 0.5
    k2_safe = np.where(np.abs(kappa_2) > 1e-8, kappa_2, 1e-8)
    anisotropy = np.abs(kappa_1 / k2_safe - 1.0)
    curv_factor = np.clip(1.0 + anisotropy * 0.5, 1.0, 2.5)
    
    # Modified Laplace: σ = P × r / (2h) × f(κ)
    stress = (LV_PRESSURE_KPA * local_radius) / (2 * wall_thickness) * curv_factor
    
    # Tissue-specific modifications
    if tags is not None:
        stress[tags == TAG_INFARCT] *= 0.7  # Scar is stiffer
        stress[tags == TAG_BORDER] *= 1.5   # Stress concentration at border
    
    print(f"      κ₁: [{kappa_1.min():.4f}, {kappa_1.max():.4f}]")
    print(f"      κ₂: [{kappa_2.min():.4f}, {kappa_2.max():.4f}]")
    print(f"      Wall stress: {stress.min():.2f} - {stress.max():.2f} kPa")
    
    return {
        'wall_stress': stress, 'wall_thickness': wall_thickness,
        'local_radius': local_radius, 'curvature_factor': curv_factor,
        'kappa_1': kappa_1, 'kappa_2': kappa_2, 'mean_curvature': mean_curv,
        'endo_dist': endo_dist, 'epi_dist': epi_dist,
    }


# GEODESIC DISTANCE (HEAT METHOD)
def compute_geodesic_distance(coords, elements, source_nodes, L):
    """
    Compute geodesic distances using the Heat Method (Crane et al. 2013).
    
    Algorithm:
    1. Solve heat equation: (M - tL)u = δ_source
    2. Normalize gradient: X = -∇u/|∇u|
    3. Solve Poisson equation: Lφ = ∇·X
    
    The solution φ gives approximate geodesic distances.
    """
    n_nodes, n_elems = len(coords), len(elements)
    
    # Lumped mass matrix
    M = np.zeros(n_nodes)
    for elem in elements:
        X = coords[elem]
        J = np.array([X[1]-X[0], X[2]-X[0], X[3]-X[0]]).T
        vol = abs(np.linalg.det(J)) / 6.0
        for node in elem:
            M[node] += vol / 4.0
    M = diags(M)
    
    # Time step from mean edge length
    h = np.mean([np.linalg.norm(coords[e[i]] - coords[e[j]]) 
                 for e in elements[:1000] for i in range(4) for j in range(i+1, 4)])
    t = h * h
    
    # Step 1: Solve heat equation
    b = np.zeros(n_nodes)
    for node in source_nodes:
        b[node] = 1.0
    u = spsolve((M - t * L).tocsr(), b)
    
    # Step 2: Compute normalized gradient field
    X = np.zeros((n_elems, 3))
    for i, elem in enumerate(elements):
        verts = coords[elem]
        J = np.array([verts[1]-verts[0], verts[2]-verts[0], verts[3]-verts[0]]).T
        det = np.linalg.det(J)
        if abs(det) < 1e-15:
            continue
        Jinv = np.linalg.inv(J)
        du = np.array([u[elem[j]] - u[elem[0]] for j in [1,2,3]])
        grad = Jinv @ du
        norm = np.linalg.norm(grad)
        if norm > 1e-10:
            X[i] = -grad / norm
    
    # Step 3: Compute divergence and solve Poisson
    div = np.zeros(n_nodes)
    for i, elem in enumerate(elements):
        verts = coords[elem]
        J = np.array([verts[1]-verts[0], verts[2]-verts[0], verts[3]-verts[0]]).T
        det = np.linalg.det(J)
        if abs(det) < 1e-15:
            continue
        vol = abs(det) / 6.0
        Jinv = np.linalg.inv(J)
        dN = np.zeros((4, 3))
        dN[0] = -Jinv.sum(axis=1)
        dN[1:4] = Jinv.T
        for j in range(4):
            div[elem[j]] += vol * np.dot(dN[j], X[i])
    
    L_mod = L.tolil()
    L_mod[source_nodes[0], :] = 0
    L_mod[source_nodes[0], source_nodes[0]] = 1
    div[source_nodes[0]] = 0
    
    phi = spsolve(L_mod.tocsr(), div)
    return phi - phi[source_nodes].min()


# STRESS REDUCTION PREDICTION
def compute_stress_reduction(stress, injection_elements, stiffening=2.0):
    """
    Predict wall stress reduction from therapeutic injection.
    
    Model: Injection stiffens local tissue, reducing local strain
    and redistributing stress.
    
    Δσ = σ_before × (1 - 1/stiffening_factor)
    """
    stress_after = stress.copy()
    for idx in injection_elements:
        stress_after[idx] *= (1.0 / stiffening)
    
    reduction = stress - stress_after
    total = np.sum(reduction[reduction > 0])
    return reduction, total


# INJECTION SITE OPTIMIZATION
def optimize_injection_sites(coords, elements, tags, stress_data, phi, geodesic, centroids, n_sites=5):
    """
    Optimize injection sites using multi-objective optimization.
    
    Objectives:
    1. MINIMIZE geodesic distance to border zone
    2. MAXIMIZE predicted wall stress reduction
    
    Constraints:
    - Avoid core scar (no perfusion)
    - Minimum spatial separation between sites
    - Prefer mid-wall transmural position
    """
    n = len(elements)
    
    phi_elem = np.mean(phi[elements], axis=1)
    geo_elem = np.mean(geodesic[elements], axis=1) if len(geodesic) == len(coords) else geodesic
    
    # Compute stress reduction potential
    stress_red = np.zeros(n)
    for i in range(n):
        if tags[i] != TAG_INFARCT:
            _, red = compute_stress_reduction(stress_data['wall_stress'], [i], INJECTION_STIFFENING_FACTOR)
            stress_red[i] = red
    
    # Normalize metrics
    geo_norm = geo_elem / (np.max(geo_elem) + 1e-10)
    stress_norm = stress_red / (np.max(stress_red[stress_red > 0]) + 1e-10)
    
    # Multi-objective: minimize geodesic, maximize stress reduction
    objective = 0.5 * geo_norm - 0.5 * stress_norm
    
    # Additional terms
    midwall_penalty = 2.0 * np.abs(phi_elem - 0.5)
    border_bonus = np.where(tags == TAG_BORDER, -0.5, 0)
    
    objective = objective + 0.2 * midwall_penalty + 0.2 * border_bonus
    objective[tags == TAG_INFARCT] = np.inf
    
    # Greedy selection with spatial constraints
    selected = []
    for idx in np.argsort(objective):
        if len(selected) >= n_sites or not np.isfinite(objective[idx]):
            break
        if not any(np.linalg.norm(centroids[idx] - centroids[s]) < MIN_SITE_SEPARATION_MM for s in selected):
            selected.append(idx)
    
    _, total_red = compute_stress_reduction(stress_data['wall_stress'], selected, INJECTION_STIFFENING_FACTOR)
    
    sites = [{
        'element_id': int(idx),
        'coordinates': centroids[idx].tolist(),
        'geodesic_distance': float(geo_elem[idx]),
        'stress_reduction_kPa': float(stress_red[idx]),
        'transmural_position': float(phi_elem[idx]),
        'wall_stress_kPa': float(stress_data['wall_stress'][idx]),
        'tissue_type': 'border' if tags[idx] == TAG_BORDER else 'healthy'
    } for idx in selected]
    
    return sites, total_red


# OUTPUT FUNCTIONS
def write_vtk(path, coords, elements, scalars, vectors):
    """Write VTK unstructured grid file."""
    n_nodes, n_elems = len(coords), len(elements)
    with open(path, 'w') as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Laplace-Dirichlet Cardiac Analysis\n")
        f.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        
        f.write(f"POINTS {n_nodes} float\n")
        for c in coords:
            f.write(f"{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")
        
        f.write(f"\nCELLS {n_elems} {n_elems*5}\n")
        for e in elements:
            f.write(f"4 {e[0]} {e[1]} {e[2]} {e[3]}\n")
        
        f.write(f"\nCELL_TYPES {n_elems}\n" + "10\n" * n_elems)
        
        pt = {k: v for k, v in scalars.items() if len(v) == n_nodes}
        if pt:
            f.write(f"\nPOINT_DATA {n_nodes}\n")
            for name, data in pt.items():
                f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                for v in data:
                    f.write(f"{float(v):.6f}\n")
        
        cl = {k: v for k, v in scalars.items() if len(v) == n_elems}
        if cl or vectors:
            f.write(f"\nCELL_DATA {n_elems}\n")
            for name, data in cl.items():
                f.write(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
                for v in data:
                    f.write(f"{float(v):.6f}\n")
            for name, data in vectors.items():
                if len(data) == n_elems:
                    f.write(f"\nVECTORS {name} float\n")
                    for v in data:
                        f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")


def write_lon(path, fibers, sheets):
    """Write CARP format fiber orientation file."""
    with open(path, 'w') as f:
        f.write("2\n")
        for fib, sht in zip(fibers, sheets):
            f.write(f"{fib[0]:.6f} {fib[1]:.6f} {fib[2]:.6f} {sht[0]:.6f} {sht[1]:.6f} {sht[2]:.6f}\n")


# PATIENT PROCESSING
def process_patient(patient_id):
    """Run complete analysis pipeline for one patient."""
    print(f"\nPROCESSING: {patient_id}")
    
    # Load mesh
    coords, elements = load_mesh(patient_id, BASE_DIR)
    n_nodes, n_elems = len(coords), len(elements)
    centroids = np.mean(coords[elements], axis=1)
    print(f"  Mesh: {n_nodes:,} nodes, {n_elems:,} elements")
    
    # Load tissue tags
    tags = load_tags(patient_id, BASE_DIR)
    if tags is None:
        tags = np.ones(n_elems, dtype=np.int32)
    else:
        print(f"  Tags: {np.sum(tags==TAG_INFARCT)} infarct, {np.sum(tags==TAG_BORDER)} border")
    
    # Extract surfaces
    print("\n  [1] TRANSMURAL COORDINATE")
    surfaces = extract_surfaces(coords, elements)
    print(f"      Endo: {len(surfaces['endo_nodes'])} nodes, Epi: {len(surfaces['epi_nodes'])} nodes")
    
    # Solve Laplace-Dirichlet
    phi = solve_laplace_dirichlet(coords, elements, surfaces)
    print(f"      φ: [{phi.min():.4f}, {phi.max():.4f}]")
    
    grad_phi = compute_gradient(coords, elements, phi)
    
    # Fiber reconstruction
    print("\n  [2] FIBER RECONSTRUCTION")
    fibers, sheets, angles, coord_sys = reconstruct_helical_fibers(coords, elements, phi, grad_phi, tags)
    print(f"      Fiber angle: [{angles.min():.1f}°, {angles.max():.1f}°]")
    
    # Wall stress with curvature tensor
    print("\n  [3] WALL STRESS COMPUTATION")
    stress_data = compute_wall_stress(coords, elements, surfaces, tags)
    
    # Geodesic optimization
    print("\n  [4] INJECTION SITE OPTIMIZATION")
    border_elem = np.where(tags == TAG_BORDER)[0]
    if len(border_elem) > 0:
        border_nodes = list(set(elements[border_elem[:100]].flatten()))
        print("      Computing geodesic distances...")
        L = build_laplacian(coords, elements)
        geodesic = compute_geodesic_distance(coords, elements, border_nodes, L)
    else:
        geodesic = stress_data['endo_dist']
    
    sites, total_red = optimize_injection_sites(coords, elements, tags, stress_data, phi, geodesic, centroids, N_INJECTION_SITES)
    
    print(f"      Selected {len(sites)} sites, total Δσ = {total_red:.2f} kPa")
    for s in sites:
        print(f"        Element {s['element_id']}: geodesic={s['geodesic_distance']:.2f}, Δσ={s['stress_reduction_kPa']:.2f}")
    
    # Save outputs
    print("\n  [5] SAVING OUTPUTS")
    out_dir = os.path.join(OUTPUT_DIR, patient_id)
    os.makedirs(out_dir, exist_ok=True)
    
    phi_elem = np.mean(phi[elements], axis=1)
    geo_elem = np.mean(geodesic[elements], axis=1) if len(geodesic) == n_nodes else np.zeros(n_elems)
    
    write_vtk(os.path.join(out_dir, f"{patient_id}_analysis.vtk"), coords, elements, {
        'TransmuralPhi': phi, 'TransmuralPhi_elem': phi_elem, 'TissueType': tags.astype(float),
        'FiberAngle_deg': angles, 'WallThickness_mm': stress_data['wall_thickness'],
        'WallStress_kPa': stress_data['wall_stress'], 'Kappa1': stress_data['kappa_1'],
        'Kappa2': stress_data['kappa_2'], 'MeanCurvature': stress_data['mean_curvature'],
        'CurvatureFactor': stress_data['curvature_factor'], 'GeodesicDistance': geo_elem,
    }, {'FiberDirection': fibers, 'SheetDirection': sheets, 'TransmuralDirection': coord_sys[0]})
    
    write_lon(os.path.join(out_dir, f"{patient_id}.lon"), fibers, sheets)
    
    summary = {
        'patient_id': patient_id,
        'timestamp': datetime.now().isoformat(),
        'mesh': {'n_nodes': n_nodes, 'n_elements': n_elems},
        'transmural': {'phi_range': [float(phi.min()), float(phi.max())]},
        'fibers': {'angle_range_deg': [float(angles.min()), float(angles.max())]},
        'curvature': {
            'kappa_1_range': [float(stress_data['kappa_1'].min()), float(stress_data['kappa_1'].max())],
            'kappa_2_range': [float(stress_data['kappa_2'].min()), float(stress_data['kappa_2'].max())],
        },
        'wall_stress': {
            'range_kPa': [float(stress_data['wall_stress'].min()), float(stress_data['wall_stress'].max())],
            'mean_kPa': float(stress_data['wall_stress'].mean()),
        },
        'optimization': {
            'total_stress_reduction_kPa': float(total_red),
            'n_sites': len(sites),
        },
        'injection_sites': sites,
    }
    
    with open(os.path.join(out_dir, f"{patient_id}_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Saved to: {out_dir}")
    
    return summary


def process_patient_wrapper(patient_id):
    """Wrapper for parallel processing with error handling."""
    try:
        return process_patient(patient_id)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'patient_id': patient_id, 'status': 'FAILED', 'error': str(e)}


# MAIN
def main():
    """Main entry point."""
    print("LAPLACE-DIRICHLET CARDIAC ANALYSIS FRAMEWORK")
    
    print("\n  Components:")
    print("  [1] Transmural coordinate (Laplace-Dirichlet)")
    print("  [2] Fiber reconstruction (helical angle rotation)")
    print("  [3] Wall stress (modified Laplace with curvature tensor)")
    print("  [4] Injection optimization (geodesic + stress reduction)")
    
    print(f"\n  Using {N_CPUS} CPUs")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process patients in parallel
    n_workers = min(N_CPUS, len(PATIENT_IDS))
    with Pool(n_workers) as pool:
        results = pool.map(process_patient_wrapper, PATIENT_IDS)
    
    # Summary
    print("SUMMARY")
    
    print(f"\n{'Patient':<15} {'κ₁ range':<25} {'Stress Red.':<15} {'Sites'}")
    
    for r in results:
        if 'error' not in r:
            k1 = r['curvature']['kappa_1_range']
            red = r['optimization']['total_stress_reduction_kPa']
            sites = r['optimization']['n_sites']
            print(f"{r['patient_id']:<15} [{k1[0]:.4f}, {k1[1]:.4f}]       {red:.2f} kPa         {sites}")
        else:
            print(f"{r['patient_id']:<15} FAILED: {r['error'][:40]}")
    
    # Save combined results
    with open(os.path.join(OUTPUT_DIR, "summary.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")
    
    return results


if __name__ == "__main__":
    main()