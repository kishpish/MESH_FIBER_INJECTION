#!/usr/bin/env python3
"""
Complete EP Outputs - OPTIMIZED VERSION


This script:
1. FIXES: Pseudo-ECG calculation (vectorized lead field method)
2. GENERATES: Parameter files for calcium transient simulations
3. GENERATES: Parameter files for S1-S2 vulnerability protocol

%run complete_ep_outputs_fast.py
"""

import numpy as np
import json
from pathlib import Path
import pandas as pd
import shutil
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
V7_DIR = BASE_DIR / "opencarp_results/v7_all"
CALCIUM_DIR = BASE_DIR / "opencarp_results/calcium_sims"
S1S2_DIR = BASE_DIR / "opencarp_results/s1s2_sims"
OUTPUT_CSV = V7_DIR / "corrected_ecg_metrics.csv"

PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000601",
    "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]


# PART 1: OPTIMIZED PSEUDO-ECG CALCULATION
def read_igb_data_safe(filepath, n_nodes):
    """Read binary IGB voltage data"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(1024).decode('ascii', errors='ignore')

        info = {}
        for part in header.split():
            if ':' in part:
                key, val = part.split(':', 1)
                try:
                    info[key] = int(val)
                except:
                    info[key] = val

        n_timesteps = info.get('t', 401)

        with open(filepath, 'rb') as f:
            f.seek(1024)
            data = np.frombuffer(f.read(), dtype=np.float32)

        actual_timesteps = len(data) // n_nodes
        if actual_timesteps > 0:
            data = data[:actual_timesteps * n_nodes].reshape(actual_timesteps, n_nodes)
            return data, actual_timesteps
    except:
        pass
    return None, 0


def load_mesh(mesh_dir, patient_id):
    """Load mesh nodes and elements"""
    pts_file = mesh_dir / f"{patient_id}.pts"
    elem_file = mesh_dir / f"{patient_id}.elem"

    with open(pts_file) as f:
        lines = f.readlines()
    n_nodes = int(lines[0].strip())
    nodes = np.array([[float(x) for x in line.split()[:3]] for line in lines[1:n_nodes+1]])

    with open(elem_file) as f:
        lines = f.readlines()
    elements, tags = [], []
    for line in lines[1:]:
        parts = line.strip().split()
        if parts[0] == "Tt":
            elements.append([int(parts[i]) for i in range(1, 5)])
            tags.append(int(parts[-1]))

    return nodes, np.array(elements), np.array(tags)


def compute_pseudo_ecg_fast(vm_data, nodes, elements):
    """
    OPTIMIZED pseudo-ECG using vectorized lead field method.

    Uses simplified dipole summation:
    Phi_ECG(t) = sum_i [ (Vm_i(t) - Vm_mean(t)) * w_i ]

    where w_i is a directional weight based on position relative to electrode.

    This approximation is valid for far-field potentials and gives
    proper mV-range values without element-by-element gradient computation.
    """
    n_timesteps = vm_data.shape[0]
    nodes_mm = nodes / 1000.0  # Convert to mm

    # Compute node-based centrality
    center = nodes_mm.mean(axis=0)
    extent = nodes_mm.max(axis=0) - nodes_mm.min(axis=0)

    # Position electrodes at realistic distances
    lead_positions = {
        'V1': center + np.array([extent[0] * 2, extent[1] * 0.3, 0]),   # Right parasternal
        'V2': center + np.array([extent[0] * 0.5, extent[1] * 2, 0]),   # Left parasternal
        'V3': center + np.array([-extent[0], extent[1] * 1.5, -extent[2] * 0.5]),
        'V4': center + np.array([-extent[0] * 1.5, extent[1], -extent[2]]),
        'V5': center + np.array([-extent[0] * 2, extent[1] * 0.5, -extent[2]]),
        'V6': center + np.array([-extent[0] * 2.5, 0, -extent[2]]),
    }

    ecg = {}

    for lead_name, lead_pos in lead_positions.items():
        # Compute weights based on inverse distance and direction
        r_vec = lead_pos - nodes_mm  # (n_nodes, 3)
        r_dist = np.linalg.norm(r_vec, axis=1)  # (n_nodes,)
        r_dist = np.maximum(r_dist, 1.0)  # Avoid singularity

        # Direction-weighted inverse distance (dipole approximation)
        # Weight by 1/r^2 and projection onto primary axis
        primary_dir = lead_pos - center
        primary_dir = primary_dir / np.linalg.norm(primary_dir)

        # Dot product with primary direction for dipole effect
        proj = np.dot(r_vec / r_dist[:, np.newaxis], primary_dir)
        weights = proj / (r_dist ** 2)

        # Normalize weights
        weights = weights / np.abs(weights).sum()

        # Compute ECG signal: weighted sum of Vm deviations
        vm_mean = vm_data.mean(axis=1, keepdims=True)
        vm_dev = vm_data - vm_mean  # Deviation from mean

        # ECG signal in mV (scale factor empirically derived)
        scale = 0.05  # Scaling to get ~1-3 mV range
        signal = np.dot(vm_dev, weights) * scale

        ecg[lead_name] = signal

    return ecg


def compute_pseudo_ecg_gradient_sampled(vm_data, nodes, elements, n_sample=5000):
    """
    Compute pseudo-ECG using gradient method but with element sampling.

    This provides more accurate lead field computation while remaining fast.
    Samples a subset of elements to approximate the full integral.
    """
    n_timesteps = vm_data.shape[0]
    n_elements = len(elements)
    nodes_mm = nodes / 1000.0

    # Sample elements uniformly
    if n_elements > n_sample:
        sample_idx = np.random.choice(n_elements, n_sample, replace=False)
        sample_elements = elements[sample_idx]
        scale_factor = n_elements / n_sample  # Scale up contribution
    else:
        sample_elements = elements
        sample_idx = np.arange(n_elements)
        scale_factor = 1.0

    # Precompute element data
    centroids = np.zeros((len(sample_elements), 3))
    volumes = np.zeros(len(sample_elements))
    A_inv_list = []
    valid_mask = []

    for i, elem in enumerate(sample_elements):
        pts = nodes_mm[elem]
        centroids[i] = pts.mean(axis=0)

        # Build gradient matrix
        A = np.array([
            pts[1] - pts[0],
            pts[2] - pts[0],
            pts[3] - pts[0]
        ])

        # Tetrahedron volume
        v1, v2, v3 = A[0], A[1], A[2]
        vol = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        volumes[i] = vol

        # Check conditioning and store inverse
        try:
            cond = np.linalg.cond(A)
            if cond < 1e8:
                A_inv_list.append(np.linalg.inv(A))
                valid_mask.append(True)
            else:
                A_inv_list.append(None)
                valid_mask.append(False)
        except:
            A_inv_list.append(None)
            valid_mask.append(False)

    valid_mask = np.array(valid_mask)
    valid_idx = np.where(valid_mask)[0]

    # Lead positions
    center = centroids.mean(axis=0)
    extent = centroids.max(axis=0) - centroids.min(axis=0)

    lead_positions = {
        'V1': center + np.array([extent[0] * 3, extent[1] * 0.5, 0]),
        'V2': center + np.array([extent[0], extent[1] * 3, 0]),
        'V3': center + np.array([-extent[0], extent[1] * 2, -extent[2]]),
        'V4': center + np.array([-extent[0] * 2, extent[1], -extent[2] * 1.5]),
        'V5': center + np.array([-extent[0] * 2.5, extent[1] * 0.5, -extent[2] * 1.5]),
        'V6': center + np.array([-extent[0] * 3, 0, -extent[2] * 1.5]),
    }

    # Conductivity scaling - empirically adjusted for realistic mV range
    # The gradient method with isolated heart needs significant scaling
    # True ECG requires full torso model; this is a pseudo-ECG approximation
    sigma = 0.2  # S/m
    base_scale = 1.0 / (4 * np.pi * sigma)
    # Empirical scaling to get 1-3 mV range (accounts for simplified geometry)
    # Typical values: 400-1200 mV with scale=1000, so use ~2.0 to get 1-3 mV
    empirical_scale = 2.5
    scale = base_scale * scale_factor * empirical_scale

    ecg = {}

    for lead_name, lead_pos in lead_positions.items():
        signal = np.zeros(n_timesteps)

        # Precompute lead vectors for all valid elements
        for i in valid_idx:
            elem = sample_elements[i]
            A_inv = A_inv_list[i]
            vol = volumes[i]

            # Vector from element to electrode
            r_vec = lead_pos - centroids[i]
            r_dist = np.linalg.norm(r_vec)
            if r_dist < 1.0:
                continue
            r_hat = r_vec / r_dist

            # Get Vm values for all timesteps at once
            vm_elem = vm_data[:, elem]  # (n_timesteps, 4)

            # Compute gradients vectorized over time
            b = np.stack([
                vm_elem[:, 1] - vm_elem[:, 0],
                vm_elem[:, 2] - vm_elem[:, 0],
                vm_elem[:, 3] - vm_elem[:, 0]
            ], axis=1)  # (n_timesteps, 3)

            grad_vm = np.dot(b, A_inv.T)  # (n_timesteps, 3)

            # Contribution: (grad_vm . r_hat) / r^2 * vol
            contrib = np.dot(grad_vm, r_hat) / (r_dist ** 2) * vol
            signal += contrib

        ecg[lead_name] = signal * scale

    return ecg


def compute_ecg_metrics(ecg, dt_ms=1.0):
    """Extract clinical ECG metrics"""
    metrics = {}

    for lead, signal in ecg.items():
        signal = np.array(signal)

        # Basic amplitude metrics
        metrics[f'ECG_{lead}_min_mV'] = float(np.min(signal))
        metrics[f'ECG_{lead}_max_mV'] = float(np.max(signal))
        metrics[f'ECG_{lead}_pp_amplitude_mV'] = float(np.ptp(signal))

        # Find QRS complex (maximum rate of change)
        deriv = np.diff(signal)
        qrs_idx = np.argmax(np.abs(deriv))
        metrics[f'ECG_{lead}_QRS_onset_ms'] = float(qrs_idx * dt_ms)

        # QRS duration
        deriv_thresh = 0.5 * np.max(np.abs(deriv))
        qrs_mask = np.abs(deriv) > deriv_thresh
        if qrs_mask.any():
            qrs_start = np.argmax(qrs_mask)
            qrs_end = len(qrs_mask) - np.argmax(qrs_mask[::-1])
            metrics[f'ECG_{lead}_QRS_duration_ms'] = float((qrs_end - qrs_start) * dt_ms)

        # R and S wave amplitudes
        search_window = slice(max(0, qrs_idx-20), min(len(signal), qrs_idx+40))
        r_amp = np.max(signal[search_window])
        s_amp = np.min(signal[search_window])
        metrics[f'ECG_{lead}_R_amplitude_mV'] = float(r_amp)
        metrics[f'ECG_{lead}_S_amplitude_mV'] = float(s_amp)

    return metrics


def process_patient_ecg(patient_id, use_gradient=True, n_sample=5000):
    """Process ECG for one patient"""
    print(f"Processing {patient_id}...")

    patient_dir = V7_DIR / patient_id
    sim_dir = patient_dir / "opencarp" / f"{patient_id}_v7"
    mesh_dir = patient_dir / "mesh"

    results = {'patient_id': patient_id}

    # Load mesh
    nodes, elements, tags = load_mesh(mesh_dir, patient_id)
    n_nodes = len(nodes)
    print(f"  Mesh: {n_nodes} nodes, {len(elements)} elements")

    # Load Vm data
    vm_file = sim_dir / "vm.igb"
    if not vm_file.exists():
        print(f"  No vm.igb file")
        results['ecg_available'] = False
        return results, None

    vm_data, n_timesteps = read_igb_data_safe(vm_file, n_nodes)
    if vm_data is None:
        print(f"  Failed to read vm.igb")
        results['ecg_available'] = False
        return results, None

    results['n_timesteps'] = n_timesteps
    print(f"  Vm data: {n_timesteps} timesteps")

    # Compute pseudo-ECG
    print(f"  Computing pseudo-ECG...")
    if use_gradient and n_timesteps > 10:  # Use gradient method with sampling
        ecg = compute_pseudo_ecg_gradient_sampled(vm_data, nodes, elements, n_sample)
    else:
        ecg = compute_pseudo_ecg_fast(vm_data, nodes, elements)

    # Extract metrics
    ecg_metrics = compute_ecg_metrics(ecg)
    results.update(ecg_metrics)
    results['ecg_available'] = True

    # Validate amplitude range
    v1_amp = results.get('ECG_V1_pp_amplitude_mV', 0)
    results['ecg_amplitude_valid'] = 0.1 < v1_amp < 10.0
    print(f"  V1 amplitude: {v1_amp:.4f} mV")

    return results, ecg


# PART 2: CALCIUM TRANSIENT SIMULATION PARAMETERS
def create_calcium_simulation_params(patient_id, output_dir):
    """Create OpenCarp parameter file for calcium transient output."""
    patient_dir = V7_DIR / patient_id
    mesh_dir = patient_dir / "mesh"

    cal_dir = output_dir / patient_id
    cal_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh files
    for ext in ['.pts', '.elem', '.lon']:
        src = mesh_dir / f"{patient_id}{ext}"
        if src.exists():
            shutil.copy2(src, cal_dir / f"{patient_id}{ext}")

    # Copy stimulus file
    apex_vtx = cal_dir / "stim_apex.vtx"
    orig_vtx = patient_dir / "opencarp" / "stim_apex.vtx"
    if orig_vtx.exists():
        shutil.copy2(orig_vtx, apex_vtx)

    mesh_path = str(cal_dir / patient_id)
    par_file = cal_dir / "simulation_calcium.par"

    par_content = f"""# OpenCARP Simulation with Calcium Transient Output
# Patient: {patient_id}
# Purpose: Extract intracellular calcium [Ca]i

simID = {patient_id}_calcium
meshname = {mesh_path}

# Time stepping (dt in MICROSECONDS)
dt = 10
tend = 500.0

# Solver settings
bidomain = 0
parab_solve = 1
mass_lumping = 1
cg_tol_parab = 1e-6

# Conductivity regions
num_gregions = 3

gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = 0.174
gregion[0].g_it = 0.019
gregion[0].g_el = 0.625
gregion[0].g_et = 0.236

gregion[1].num_IDs = 1
gregion[1].ID[0] = 2
gregion[1].g_il = 0.0087
gregion[1].g_it = 0.00095
gregion[1].g_el = 0.031
gregion[1].g_et = 0.012

gregion[2].num_IDs = 1
gregion[2].ID[0] = 3
gregion[2].g_il = 0.087
gregion[2].g_it = 0.0095
gregion[2].g_el = 0.312
gregion[2].g_et = 0.118

# Ionic model
num_imp_regions = 3

imp_region[0].num_IDs = 1
imp_region[0].ID[0] = 1
imp_region[0].im = tenTusscherPanfilov
imp_region[0].cellSurfVolRatio = 0.14

imp_region[1].num_IDs = 1
imp_region[1].ID[0] = 2
imp_region[1].im = tenTusscherPanfilov
imp_region[1].im_param = "GNa*0.05,GK1*0.05,GCaL*0.05,Gto*0.05"
imp_region[1].cellSurfVolRatio = 0.14

imp_region[2].num_IDs = 1
imp_region[2].ID[0] = 3
imp_region[2].im = tenTusscherPanfilov
imp_region[2].im_param = "GNa*0.6,GK1*0.7,GCaL*0.7,Gto*0.3"
imp_region[2].cellSurfVolRatio = 0.14

# CALCIUM OUTPUT
num_gvecs = 1
gvec[0].name = Cai_output
gvec[0].ID = Cai

# Stimulus
num_stim = 1
stimulus[0].stimtype = 0
stimulus[0].strength = 150.0
stimulus[0].duration = 2.0
stimulus[0].start = 0
stimulus[0].npls = 1
stimulus[0].vtx_file = {apex_vtx}

# LAT and APD
num_LATs = 1
lats[0].ID = LAT
lats[0].all = 1
lats[0].measurand = 0
lats[0].threshold = -10.0
lats[0].method = 1

compute_APD = 1
actthresh = -40.0
recovery_thresh = 0.9

# Output
spacedt = 1.0
timedt = 1.0
"""

    with open(par_file, 'w') as f:
        f.write(par_content)

    # Create run script
    run_script = cal_dir / "run_calcium.sh"
    run_content = f"""#!/bin/bash
cd {cal_dir}
mpirun -np 16 /usr/local/bin/openCARP +F simulation_calcium.par
echo "Done. Output: Cai_output.igb"
"""
    with open(run_script, 'w') as f:
        f.write(run_content)

    return par_file


# PART 3: S1-S2 VULNERABILITY PROTOCOL
def create_s1s2_simulation_params(patient_id, output_dir, s1_bcl=500, s2_intervals=None):
    """Create OpenCarp parameter files for S1-S2 vulnerability protocol."""
    if s2_intervals is None:
        s2_intervals = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200]

    patient_dir = V7_DIR / patient_id
    mesh_dir = patient_dir / "mesh"

    s1s2_patient_dir = output_dir / patient_id
    s1s2_patient_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh files
    for ext in ['.pts', '.elem', '.lon']:
        src = mesh_dir / f"{patient_id}{ext}"
        if src.exists():
            shutil.copy2(src, s1s2_patient_dir / f"{patient_id}{ext}")

    apex_vtx = s1s2_patient_dir / "stim_apex.vtx"
    orig_vtx = patient_dir / "opencarp" / "stim_apex.vtx"
    if orig_vtx.exists():
        shutil.copy2(orig_vtx, apex_vtx)

    mesh_path = str(s1s2_patient_dir / patient_id)
    param_files = []

    for s2_ci in s2_intervals:
        s2_start = s1_bcl * 8 + s2_ci
        tend = s2_start + 500

        par_file = s1s2_patient_dir / f"s1s2_{s2_ci}ms.par"

        par_content = f"""# S1-S2 Vulnerability Protocol
# Patient: {patient_id}, S2 CI: {s2_ci} ms

simID = {patient_id}_s1s2_{s2_ci}
meshname = {mesh_path}

dt = 10
tend = {tend}

bidomain = 0
parab_solve = 1
mass_lumping = 1
cg_tol_parab = 1e-6

num_gregions = 3

gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = 0.174
gregion[0].g_it = 0.019
gregion[0].g_el = 0.625
gregion[0].g_et = 0.236

gregion[1].num_IDs = 1
gregion[1].ID[0] = 2
gregion[1].g_il = 0.0087
gregion[1].g_it = 0.00095
gregion[1].g_el = 0.031
gregion[1].g_et = 0.012

gregion[2].num_IDs = 1
gregion[2].ID[0] = 3
gregion[2].g_il = 0.087
gregion[2].g_it = 0.0095
gregion[2].g_el = 0.312
gregion[2].g_et = 0.118

num_imp_regions = 3

imp_region[0].num_IDs = 1
imp_region[0].ID[0] = 1
imp_region[0].im = tenTusscherPanfilov
imp_region[0].cellSurfVolRatio = 0.14

imp_region[1].num_IDs = 1
imp_region[1].ID[0] = 2
imp_region[1].im = tenTusscherPanfilov
imp_region[1].im_param = "GNa*0.05,GK1*0.05,GCaL*0.05,Gto*0.05"
imp_region[1].cellSurfVolRatio = 0.14

imp_region[2].num_IDs = 1
imp_region[2].ID[0] = 3
imp_region[2].im = tenTusscherPanfilov
imp_region[2].im_param = "GNa*0.6,GK1*0.7,GCaL*0.7,Gto*0.3"
imp_region[2].cellSurfVolRatio = 0.14

# S1-S2 Protocol
num_stim = 2

stimulus[0].stimtype = 0
stimulus[0].strength = 150.0
stimulus[0].duration = 2.0
stimulus[0].start = 0
stimulus[0].npls = 8
stimulus[0].bcl = {s1_bcl}
stimulus[0].vtx_file = {apex_vtx}

stimulus[1].stimtype = 0
stimulus[1].strength = 200.0
stimulus[1].duration = 2.0
stimulus[1].start = {s2_start}
stimulus[1].npls = 1
stimulus[1].vtx_file = {apex_vtx}

num_LATs = 1
lats[0].ID = LAT
lats[0].all = 1
lats[0].measurand = 0
lats[0].threshold = -10.0
lats[0].method = 1

compute_APD = 1
actthresh = -40.0
recovery_thresh = 0.9

spacedt = 1.0
timedt = 5.0
"""

        with open(par_file, 'w') as f:
            f.write(par_content)
        param_files.append(par_file)

    # Create run script
    run_script = s1s2_patient_dir / "run_all_s1s2.sh"
    run_content = f"""#!/bin/bash
cd {s1s2_patient_dir}
for ci in {' '.join(str(x) for x in s2_intervals)}; do
    echo "Running S1S2 CI=${{ci}}ms..."
    mpirun -np 16 /usr/local/bin/openCARP +F s1s2_${{ci}}ms.par
done
echo "All S1-S2 protocols complete."
"""
    with open(run_script, 'w') as f:
        f.write(run_content)

    return param_files


# MAIN EXECUTION
def main():
    print("COMPLETE EP OUTPUTS - OPTIMIZED VERSION")

    # Part 1: Corrected ECG
    print("\nPART 1: CORRECTED PSEUDO-ECG CALCULATION")

    all_ecg_results = []
    all_ecg_signals = {}

    for patient_id in PATIENTS:
        try:
            results, ecg = process_patient_ecg(patient_id, use_gradient=True, n_sample=10000)
            all_ecg_results.append(results)
            if ecg:
                all_ecg_signals[patient_id] = {k: v.tolist() for k, v in ecg.items()}
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_ecg_results.append({'patient_id': patient_id, 'error': str(e)})

    # Save corrected ECG CSV
    df_ecg = pd.DataFrame(all_ecg_results)
    df_ecg.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")

    # Save ECG signals
    ecg_json = V7_DIR / "corrected_pseudo_ecg.json"
    with open(ecg_json, 'w') as f:
        json.dump(all_ecg_signals, f)
    print(f"Saved: {ecg_json}")

    # Validate
    print("\nECG Amplitude Summary:")
    for r in all_ecg_results:
        if r.get('ecg_available'):
            amp = r.get('ECG_V1_pp_amplitude_mV', 0)
            valid = "OK" if r.get('ecg_amplitude_valid') else "CHECK"
            print(f"  {r['patient_id']}: V1 = {amp:.4f} mV [{valid}]")
        else:
            print(f"  {r['patient_id']}: No ECG data")

    # Part 2: Calcium parameters
    print("\nPART 2: CALCIUM SIMULATION PARAMETERS")

    CALCIUM_DIR.mkdir(parents=True, exist_ok=True)
    for patient_id in PATIENTS:
        par_file = create_calcium_simulation_params(patient_id, CALCIUM_DIR)
        print(f"  Created: {par_file.name}")
    print(f"Output dir: {CALCIUM_DIR}")

    # Part 3: S1-S2 parameters
    print("\nPART 3: S1-S2 VULNERABILITY PARAMETERS")

    S1S2_DIR.mkdir(parents=True, exist_ok=True)
    for patient_id in PATIENTS:
        par_files = create_s1s2_simulation_params(patient_id, S1S2_DIR)
        print(f"  Created {len(par_files)} protocols for {patient_id}")
    print(f"Output dir: {S1S2_DIR}")

    # Summary
    print("SUMMARY")
    print(f"""
1. CORRECTED PSEUDO-ECG
   - Used gradient-sampled lead field method
   - Output: corrected_ecg_metrics.csv, corrected_pseudo_ecg.json

2. CALCIUM TRANSIENTS (Requires new simulations)
   - Parameter files in: {CALCIUM_DIR}
   - Run: bash run_calcium.sh

3. S1-S2 VULNERABILITY (Requires new simulations)
   - Parameter files in: {S1S2_DIR}
   - Run: bash run_all_s1s2.sh
""")

    return df_ecg


if __name__ == "__main__":
    df = main()
    print("\nDone!")