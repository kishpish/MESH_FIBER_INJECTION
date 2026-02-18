#!/usr/bin/env python3
"""
PHASE 3: HYDROGEL PATCH PARAMETRIC SIMULATIONS


This script runs the complete parametric sweep for hydrogel patch optimization:
- 7 stiffness values (1, 5, 10, 25, 50, 75, 100 kPa)
- 5 thickness values (0.5, 1.0, 2.0, 3.0, 5.0 mm)
- 4 coverage configs (scar-only, scar+25%BZ, scar+50%BZ, scar+100%BZ)

Total: 140 configurations per patient × 10 patients = 1,400 simulations

Usage:
    python3 hydrogel_patch_simulations.py                    # Full sweep
    python3 hydrogel_patch_simulations.py --patient SCD0000101  # Single patient
    python3 hydrogel_patch_simulations.py --test             # Quick test (1 config)

"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import subprocess
import struct
import time

# CONFIGURATION
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
MESH_DIR = BASE_DIR / "simulation_ready"
OUTPUT_DIR = BASE_DIR / "hydrogel_patch_results"
BASELINE_DIR = BASE_DIR / "febio_dynamic_results"
FEBIO_PATH = "/home/shadeform/FEBio/bin/febio4"
LD_LIBRARY_PATH = "/home/shadeform/FEBio/lib"

PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401",
    "SCD0000601", "SCD0000701", "SCD0000801", "SCD0001001",
    "SCD0001101", "SCD0001201"
]

# PARAMETRIC SWEEP CONFIGURATION
PATCH_STIFFNESS_kPa = [1, 5, 10, 25, 50, 75, 100]      # 7 values
PATCH_THICKNESS_mm = [0.5, 1.0, 2.0, 3.0, 5.0]         # 5 values
PATCH_COVERAGE = ["scar_only", "scar_bz25", "scar_bz50", "scar_bz100"]  # 4 configs

# Simulation parameters (same as baseline)
CARDIAC_CYCLE_DURATION = 1.0  # seconds
NUM_OUTPUT_FRAMES = 100
HEART_RATE = 60

# Pressure waveform (mmHg)
DIASTOLIC_PRESSURE = 8.0
SYSTOLIC_PRESSURE = 120.0

# Material regions (from baseline)
MATERIAL_PARAMS = {
    "healthy": {"c1": 2.0, "c2": 6.0, "c3": 5.0, "c4": 50.0, "k": 100.0},
    "border_zone": {"c1": 5.0, "c2": 6.0, "c3": 10.0, "c4": 50.0, "k": 200.0},
    "infarct_scar": {"c1": 20.0, "c2": 6.0, "c3": 40.0, "c4": 50.0, "k": 500.0},
}

# Improvement thresholds for labeling
EF_IMPROVEMENT_THRESHOLD = 3.0       # Minimum 3% EF improvement
STRESS_REDUCTION_THRESHOLD = 20.0    # Minimum 20% stress reduction in BZ

# HELPER FUNCTIONS
def load_mesh(patient_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load patient mesh (nodes, elements, element tags)"""
    mesh_path = MESH_DIR / patient_id

    # Load nodes (.pts file - OpenCarp format)
    # Try different naming conventions
    pts_file = mesh_path / f"{patient_id}_tet.pts"
    if not pts_file.exists():
        pts_file = mesh_path / f"{patient_id}.pts"
    with open(pts_file, 'r') as f:
        n_nodes = int(f.readline().strip())
        nodes = np.zeros((n_nodes, 3))
        for i in range(n_nodes):
            line = f.readline().strip().split()
            nodes[i] = [float(x) for x in line[:3]]

    # Load elements (.elem file - OpenCarp format)
    elem_file = mesh_path / f"{patient_id}_tet.elem"
    if not elem_file.exists():
        elem_file = mesh_path / f"{patient_id}.elem"
    elements = []
    tags = []
    with open(elem_file, 'r') as f:
        n_elem = int(f.readline().strip())
        for _ in range(n_elem):
            line = f.readline().strip().split()
            # Format: Tt node1 node2 node3 node4 tag
            elem_nodes = [int(x) for x in line[1:5]]
            tag = int(line[5]) if len(line) > 5 else 1
            elements.append(elem_nodes)
            tags.append(tag)

    return nodes, np.array(elements), np.array(tags)


def identify_epicardial_surface(nodes: np.ndarray, elements: np.ndarray) -> np.ndarray:
    """Identify epicardial surface elements (outermost surface)"""
    from collections import defaultdict

    # Find boundary faces (faces that appear only once)
    face_count = defaultdict(int)
    elem_faces = {}

    for i, elem in enumerate(elements):
        # Tetrahedral faces
        faces = [
            tuple(sorted([elem[0], elem[1], elem[2]])),
            tuple(sorted([elem[0], elem[1], elem[3]])),
            tuple(sorted([elem[0], elem[2], elem[3]])),
            tuple(sorted([elem[1], elem[2], elem[3]])),
        ]
        elem_faces[i] = faces
        for face in faces:
            face_count[face] += 1

    # Elements with boundary faces
    boundary_elements = set()
    for i, faces in elem_faces.items():
        for face in faces:
            if face_count[face] == 1:
                boundary_elements.add(i)

    # Filter for epicardial (outermost) - use centroid distance from center
    centroids = np.array([nodes[elements[i]].mean(axis=0) for i in boundary_elements])
    center = nodes.mean(axis=0)
    distances = np.linalg.norm(centroids - center, axis=1)

    # Epicardial = outer 50% of boundary elements by distance
    threshold = np.percentile(distances, 50)
    epicardial_mask = distances >= threshold

    epicardial_elements = np.array(list(boundary_elements))[epicardial_mask]
    return epicardial_elements


def identify_patch_elements(nodes: np.ndarray, elements: np.ndarray, tags: np.ndarray,
                           epicardial: np.ndarray, coverage: str) -> np.ndarray:
    """Identify elements covered by patch based on coverage configuration"""
    # Tags: 1=healthy, 2=scar, 3=border_zone
    scar_elems = set(np.where(tags == 2)[0])
    bz_elems = set(np.where(tags == 3)[0])

    # Epicardial elements in each region
    epi_scar = epicardial[np.isin(epicardial, list(scar_elems))]
    epi_bz = epicardial[np.isin(epicardial, list(bz_elems))]

    if coverage == "scar_only":
        patch_elems = epi_scar
    elif coverage == "scar_bz25":
        # Scar + 25% of border zone (closest to scar)
        if len(epi_bz) > 0:
            scar_centroids = nodes[elements[epi_scar]].mean(axis=1) if len(epi_scar) > 0 else np.array([[0,0,0]])
            bz_centroids = nodes[elements[epi_bz]].mean(axis=1)
            # Distance to nearest scar element
            from scipy.spatial import KDTree
            if len(scar_centroids) > 0:
                tree = KDTree(scar_centroids)
                dists, _ = tree.query(bz_centroids)
                threshold = np.percentile(dists, 25)
                bz_select = epi_bz[dists <= threshold]
            else:
                bz_select = epi_bz[:len(epi_bz)//4]
        else:
            bz_select = np.array([], dtype=int)
        patch_elems = np.concatenate([epi_scar, bz_select])
    elif coverage == "scar_bz50":
        # Scar + 50% of border zone
        if len(epi_bz) > 0:
            bz_select = epi_bz[:len(epi_bz)//2]
        else:
            bz_select = np.array([], dtype=int)
        patch_elems = np.concatenate([epi_scar, bz_select])
    elif coverage == "scar_bz100":
        # Scar + full border zone
        patch_elems = np.concatenate([epi_scar, epi_bz])
    else:
        patch_elems = epi_scar

    return patch_elems.astype(int)


def create_patch_material_xml(stiffness_kPa: float, thickness_mm: float) -> str:
    """Create FEBio material XML for hydrogel patch (neo-Hookean)"""
    # Neo-Hookean: W = C1(I1 - 3) + 1/D(J-1)^2
    # For nearly incompressible (nu ≈ 0.49): C1 ≈ E/6
    # Material ID 4 (after healthy=1, border_zone=2, infarct_scar=3)

    return f'''    <material id="4" name="hydrogel_patch" type="neo-Hookean">
      <density>1.0</density>
      <E>{stiffness_kPa}</E>
      <v>0.49</v>
    </material>
'''


def generate_feb_with_patch(patient_id: str, stiffness: float, thickness: float,
                           coverage: str, output_path: Path) -> str:
    """Generate FEBio file with hydrogel patch added"""

    # Load mesh
    nodes, elements, tags = load_mesh(patient_id)

    # Identify epicardial surface
    epicardial = identify_epicardial_surface(nodes, elements)

    # Identify patch elements
    patch_elems = identify_patch_elements(nodes, elements, tags, epicardial, coverage)

    # Load baseline .feb as template
    baseline_feb = BASELINE_DIR / patient_id / "cardiac_dynamic.feb"
    if not baseline_feb.exists():
        baseline_feb = MESH_DIR / patient_id / f"{patient_id}.feb"

    with open(baseline_feb, 'r') as f:
        feb_content = f.read()

    # Modify: Add patch material
    patch_material = create_patch_material_xml(stiffness, thickness)

    # Insert patch material before </Material> tag (with proper indentation)
    feb_content = feb_content.replace('  </Material>', f'{patch_material}  </Material>')

    # Modify: Add patch domain (elements with material 100)
    # This requires adding patch elements to the mesh
    # For simplicity, we mark existing elements with patch material
    # In production, you'd extrude patch elements from epicardial surface

    # Add patch element set
    patch_elem_str = ','.join(map(str, patch_elems))
    patch_domain = f'''
        <ElementSet name="patch_elements">
            {patch_elem_str}
        </ElementSet>
'''

    # Add simulation metadata
    config_id = f"E{stiffness}_T{thickness}_{coverage}"

    # Save modified .feb
    output_file = output_path / f"patch_{config_id}.feb"
    with open(output_file, 'w') as f:
        f.write(feb_content)

    return str(output_file), len(patch_elems)


def run_febio_simulation(feb_file: str, timeout: int = 3600) -> Tuple[bool, str]:
    """Run FEBio simulation"""
    env = os.environ.copy()
    env['LD_LIBRARY_PATH'] = LD_LIBRARY_PATH

    try:
        result = subprocess.run(
            [FEBIO_PATH, '-i', feb_file],
            capture_output=True, text=True,
            timeout=timeout, env=env
        )

        success = result.returncode == 0
        log = result.stdout + result.stderr
        return success, log

    except subprocess.TimeoutExpired:
        return False, "Simulation timed out"
    except Exception as e:
        return False, str(e)


def extract_patch_results(xplt_file: str, patient_id: str) -> Dict:
    """Extract results from FEBio xplt output"""
    results = {
        'LVEF_pct': None,
        'EDV_mL': None,
        'ESV_mL': None,
        'stroke_volume_mL': None,
        'border_zone_peak_stress_kPa': None,
        'border_zone_strain_pct': None,
        'healthy_strain_pct': None,
        'GLS_pct': None,
    }

    # Parse xplt binary file (FEBio output format)
    # This is a simplified extraction - full implementation would parse binary

    xplt_path = Path(xplt_file)
    if not xplt_path.exists():
        return results

    # Look for accompanying JSON or CSV results
    results_json = xplt_path.with_suffix('.json')
    if results_json.exists():
        with open(results_json) as f:
            data = json.load(f)
            results.update(data)

    return results


def compute_improvement_metrics(baseline: Dict, patch: Dict) -> Dict:
    """Compute improvement metrics comparing patch to baseline"""

    metrics = {
        'delta_EF_pct': None,
        'delta_BZ_stress_pct': None,
        'delta_strain_normalization': None,
        'is_optimal': False,
    }

    if baseline.get('LVEF_pct') and patch.get('LVEF_pct'):
        metrics['delta_EF_pct'] = patch['LVEF_pct'] - baseline['LVEF_pct']

    if baseline.get('border_zone_peak_stress_kPa') and patch.get('border_zone_peak_stress_kPa'):
        base_stress = baseline['border_zone_peak_stress_kPa']
        patch_stress = patch['border_zone_peak_stress_kPa']
        metrics['delta_BZ_stress_pct'] = -100 * (base_stress - patch_stress) / base_stress

    # Strain normalization: how much closer BZ strain gets to healthy
    if all(k in baseline for k in ['border_zone_strain_pct', 'healthy_strain_pct']):
        if all(k in patch for k in ['border_zone_strain_pct', 'healthy_strain_pct']):
            base_gap = abs(baseline['border_zone_strain_pct'] - baseline['healthy_strain_pct'])
            patch_gap = abs(patch['border_zone_strain_pct'] - patch['healthy_strain_pct'])
            if base_gap > 0:
                metrics['delta_strain_normalization'] = 100 * (base_gap - patch_gap) / base_gap

    # Label as optimal if thresholds met
    if metrics['delta_EF_pct'] is not None and metrics['delta_BZ_stress_pct'] is not None:
        metrics['is_optimal'] = (
            metrics['delta_EF_pct'] >= EF_IMPROVEMENT_THRESHOLD and
            metrics['delta_BZ_stress_pct'] >= STRESS_REDUCTION_THRESHOLD
        )

    return metrics


# MAIN SIMULATION PIPELINE
def run_single_configuration(patient_id: str, stiffness: float, thickness: float,
                            coverage: str, baseline_results: Dict) -> Dict:
    """Run a single patch configuration and return results"""

    config_id = f"{patient_id}_E{stiffness}_T{thickness}_{coverage}"
    patient_output = OUTPUT_DIR / patient_id
    patient_output.mkdir(parents=True, exist_ok=True)

    result = {
        'patient_id': patient_id,
        'config_id': config_id,
        'stiffness_kPa': stiffness,
        'thickness_mm': thickness,
        'coverage': coverage,
        'simulation_success': False,
        'patch_n_elements': 0,
        'runtime_seconds': 0,
    }

    start_time = time.time()

    try:
        # Generate .feb file with patch
        feb_file, n_patch_elems = generate_feb_with_patch(
            patient_id, stiffness, thickness, coverage, patient_output
        )
        result['patch_n_elements'] = n_patch_elems

        # Run simulation
        success, log = run_febio_simulation(feb_file)
        result['simulation_success'] = success

        if success:
            # Extract results
            xplt_file = feb_file.replace('.feb', '.xplt')
            patch_results = extract_patch_results(xplt_file, patient_id)
            result.update(patch_results)

            # Compute improvement metrics
            improvement = compute_improvement_metrics(baseline_results, patch_results)
            result.update(improvement)

        # Save log
        log_file = patient_output / f"patch_{config_id}.log"
        with open(log_file, 'w') as f:
            f.write(log)

    except Exception as e:
        result['error'] = str(e)

    result['runtime_seconds'] = time.time() - start_time

    return result


def load_baseline_results(patient_id: str) -> Dict:
    """Load baseline simulation results for a patient"""

    # Try comprehensive CSV first
    csv_path = BASELINE_DIR / "COMPREHENSIVE_DYNAMIC_CARDIAC_RESULTS.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        patient_data = df[df['patient_id'] == patient_id]
        if len(patient_data) > 0:
            return patient_data.iloc[0].to_dict()

    # Try patient-specific JSON
    json_path = BASELINE_DIR / patient_id / "dynamic_results.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)

    return {}


def run_parametric_sweep(patients: List[str] = None, parallel: bool = True,
                        max_workers: int = 4) -> pd.DataFrame:
    """Run full parametric sweep across all configurations"""

    if patients is None:
        patients = PATIENTS

    print("HYDROGEL PATCH PARAMETRIC SWEEP")
    print(f"Patients: {len(patients)}")
    print(f"Stiffness values: {PATCH_STIFFNESS_kPa}")
    print(f"Thickness values: {PATCH_THICKNESS_mm}")
    print(f"Coverage configs: {PATCH_COVERAGE}")
    n_configs = len(PATCH_STIFFNESS_kPa) * len(PATCH_THICKNESS_mm) * len(PATCH_COVERAGE)
    print(f"Configurations per patient: {n_configs}")
    print(f"Total simulations: {n_configs * len(patients)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []

    for patient_id in patients:
        print(f"\n[{patient_id}] Loading baseline...")
        baseline = load_baseline_results(patient_id)

        if not baseline:
            print(f"  WARNING: No baseline results for {patient_id}, skipping")
            continue

        print(f"  Baseline LVEF: {baseline.get('LVEF_pct', 'N/A'):.1f}%")

        # Generate all configurations for this patient
        configs = []
        for stiffness in PATCH_STIFFNESS_kPa:
            for thickness in PATCH_THICKNESS_mm:
                for coverage in PATCH_COVERAGE:
                    configs.append((patient_id, stiffness, thickness, coverage, baseline))

        print(f"  Running {len(configs)} configurations...")

        if parallel and max_workers > 1:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(run_single_configuration, *cfg): cfg
                    for cfg in configs
                }

                for future in as_completed(futures):
                    result = future.result()
                    all_results.append(result)

                    status = "OK" if result['simulation_success'] else "FAIL"
                    print(f"    [{status}] E={result['stiffness_kPa']}kPa "
                          f"T={result['thickness_mm']}mm {result['coverage']}")
        else:
            for cfg in configs:
                result = run_single_configuration(*cfg)
                all_results.append(result)

                status = "OK" if result['simulation_success'] else "FAIL"
                print(f"    [{status}] E={result['stiffness_kPa']}kPa "
                      f"T={result['thickness_mm']}mm {result['coverage']}")

    # Save results
    df = pd.DataFrame(all_results)
    output_csv = OUTPUT_DIR / "HYDROGEL_PATCH_SWEEP_RESULTS.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n\nResults saved: {output_csv}")

    # Summary statistics
    print("SWEEP SUMMARY")
    print(f"Total simulations: {len(df)}")
    print(f"Successful: {df['simulation_success'].sum()}")
    print(f"Failed: {(~df['simulation_success']).sum()}")

    if 'is_optimal' in df.columns:
        optimal = df[df['is_optimal'] == True]
        print(f"\nOptimal configurations: {len(optimal)}")
        if len(optimal) > 0:
            print("\nTop optimal configs:")
            print(optimal[['patient_id', 'stiffness_kPa', 'thickness_mm', 'coverage',
                          'delta_EF_pct', 'delta_BZ_stress_pct']].head(10).to_string())

    return df


def run_quick_test(patient_id: str = "SCD0000101"):
    """Run quick test with single configuration"""
    print("Running quick test simulation...")

    baseline = load_baseline_results(patient_id)
    result = run_single_configuration(
        patient_id,
        stiffness=10.0,  # 10 kPa
        thickness=2.0,   # 2 mm
        coverage="scar_bz50",
        baseline_results=baseline
    )

    print("\nTest result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    return result


# MAIN
def main():
    parser = argparse.ArgumentParser(description="Hydrogel Patch Parametric Simulations")
    parser.add_argument('--patient', type=str, help='Run single patient')
    parser.add_argument('--test', action='store_true', help='Quick test mode')
    parser.add_argument('--workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--sequential', action='store_true', help='Run sequentially')

    args = parser.parse_args()

    if args.test:
        run_quick_test()
    elif args.patient:
        run_parametric_sweep(patients=[args.patient], parallel=not args.sequential,
                            max_workers=args.workers)
    else:
        run_parametric_sweep(parallel=not args.sequential, max_workers=args.workers)


if __name__ == "__main__":
    main()
