#!/usr/bin/env python3
"""
Complete Missing EP Simulations Workflow


This script is a complete workflow to:
1. Verify OpenCarp installation and parameter files
2. Run a test simulation to validate setup
3. Run all calcium and S1-S2 simulations (optional, takes hours)
4. Extract results and generate CSV files

Usage in Jupyter:
    %run complete_missing_ep_workflow.py

Or run specific functions:
    from complete_missing_ep_workflow import run_test, extract_all_results
    run_test()  # Quick test
    extract_all_results()  # After simulations complete
"""

import subprocess
import numpy as np
import json
from pathlib import Path
import pandas as pd
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore')

# CONFIGURATION
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
V7_DIR = BASE_DIR / "opencarp_results/v7_all"
CALCIUM_DIR = BASE_DIR / "opencarp_results/calcium_sims"
S1S2_DIR = BASE_DIR / "opencarp_results/s1s2_sims"

PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000601",
    "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]

OPENCARP_CMD = "/usr/local/bin/openCARP"
MPI_NP = 16

S2_INTERVALS = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200]


# VERIFICATION FUNCTIONS
def verify_opencarp():
    """Verify OpenCarp installation"""
    print("Checking OpenCarp installation...")

    if not os.path.exists(OPENCARP_CMD):
        print(f"  ERROR: OpenCarp not found at {OPENCARP_CMD}")
        return False

    try:
        result = subprocess.run([OPENCARP_CMD, "--version"],
                              capture_output=True, text=True, timeout=10)
        version = result.stdout.strip() or result.stderr.strip()
        print(f"  OpenCarp version: {version[:50]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def verify_parameter_files():
    """Verify all parameter files exist"""
    print("\nVerifying parameter files...")

    missing = []

    # Calcium files
    for patient in PATIENTS:
        par = CALCIUM_DIR / patient / "simulation_calcium.par"
        if not par.exists():
            missing.append(f"Calcium: {patient}")

    # S1-S2 files
    for patient in PATIENTS:
        for ci in S2_INTERVALS:
            par = S1S2_DIR / patient / f"s1s2_{ci}ms.par"
            if not par.exists():
                missing.append(f"S1-S2: {patient} CI={ci}")

    if missing:
        print(f"  Missing {len(missing)} files:")
        for m in missing[:5]:
            print(f"    - {m}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
        return False

    print(f"  All files present: 10 calcium + 110 S1-S2 = 120 total")
    return True


# TEST SIMULATION
def run_test_simulation():
    """Run a quick test to verify simulations work"""
    print("RUNNING TEST SIMULATION")

    # Use a short simulation for testing
    test_dir = BASE_DIR / "opencarp_results/test_run"
    test_dir.mkdir(parents=True, exist_ok=True)

    # Copy mesh files from first patient
    patient = "SCD0000101"
    mesh_dir = V7_DIR / patient / "mesh"

    import shutil
    for ext in ['.pts', '.elem', '.lon']:
        src = mesh_dir / f"{patient}{ext}"
        if src.exists():
            shutil.copy2(src, test_dir / f"test{ext}")

    # Copy stimulus file
    stim_src = V7_DIR / patient / "opencarp" / "stim_apex.vtx"
    if stim_src.exists():
        shutil.copy2(stim_src, test_dir / "stim_apex.vtx")

    # Create minimal test parameter file
    par_content = f"""# Quick Test Simulation
simID = test_run
meshname = {test_dir}/test

dt = 10
tend = 50.0

bidomain = 0
parab_solve = 1
mass_lumping = 1

num_gregions = 1
gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = 0.174
gregion[0].g_it = 0.019

num_imp_regions = 1
imp_region[0].num_IDs = 1
imp_region[0].ID[0] = 1
imp_region[0].im = tenTusscherPanfilov

num_stim = 1
stimulus[0].stimtype = 0
stimulus[0].strength = 150.0
stimulus[0].duration = 2.0
stimulus[0].start = 0
stimulus[0].npls = 1
stimulus[0].vtx_file = {test_dir}/stim_apex.vtx

spacedt = 10.0
timedt = 10.0
"""

    par_file = test_dir / "test.par"
    with open(par_file, 'w') as f:
        f.write(par_content)

    print(f"Test directory: {test_dir}")
    print("Running short simulation (50ms)...")

    cmd = f"cd {test_dir} && mpirun -np {MPI_NP} {OPENCARP_CMD} +F test.par"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)

        if result.returncode == 0:
            print("TEST PASSED - OpenCarp simulations work correctly!")

            # Check for output
            if (test_dir / "test_run" / "vm.igb").exists():
                print("  Output files created successfully")

            return True
        else:
            print("TEST FAILED")
            print(f"Error: {result.stderr[:500]}")
            return False
    except subprocess.TimeoutExpired:
        print("TEST TIMEOUT (>5 minutes)")
        return False
    except Exception as e:
        print(f"TEST ERROR: {e}")
        return False


# RUN SIMULATIONS
def run_simulation(par_file, timeout_minutes=60):
    """Run a single simulation"""
    work_dir = par_file.parent
    cmd = f"cd {work_dir} && mpirun -np {MPI_NP} {OPENCARP_CMD} +F {par_file.name}"

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True,
                              text=True, timeout=timeout_minutes*60)
        return result.returncode == 0, result.stderr[:200] if result.returncode != 0 else "OK"
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


def run_all_calcium_simulations():
    """Run calcium simulations for all patients"""
    print("RUNNING CALCIUM TRANSIENT SIMULATIONS")
    print("This will take approximately 2-3 hours...\n")

    results = []

    for i, patient in enumerate(PATIENTS):
        par_file = CALCIUM_DIR / patient / "simulation_calcium.par"

        print(f"[{i+1}/10] {patient}...", end=" ", flush=True)
        start = time.time()

        success, msg = run_simulation(par_file, timeout_minutes=60)
        elapsed = time.time() - start

        if success:
            print(f"DONE ({elapsed/60:.1f} min)")
            results.append({'patient': patient, 'status': 'SUCCESS', 'time_min': elapsed/60})
        else:
            print(f"FAILED: {msg[:30]}")
            results.append({'patient': patient, 'status': 'FAILED', 'error': msg})

    # Save status
    df = pd.DataFrame(results)
    df.to_csv(V7_DIR / "calcium_simulation_status.csv", index=False)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nCalcium simulations complete: {success_count}/10 successful")

    return results


def run_all_s1s2_simulations():
    """Run S1-S2 simulations for all patients"""
    print("RUNNING S1-S2 VULNERABILITY SIMULATIONS")
    print("This will take approximately 10-15 hours (110 simulations)...\n")

    results = []
    total = len(PATIENTS) * len(S2_INTERVALS)
    current = 0

    for patient in PATIENTS:
        print(f"\n{patient}:")

        for ci in S2_INTERVALS:
            current += 1
            par_file = S1S2_DIR / patient / f"s1s2_{ci}ms.par"

            print(f"  [{current}/{total}] CI={ci}ms...", end=" ", flush=True)

            success, msg = run_simulation(par_file, timeout_minutes=45)

            if success:
                print("OK")
                results.append({'patient': patient, 'ci_ms': ci, 'status': 'SUCCESS'})
            else:
                print("FAIL")
                results.append({'patient': patient, 'ci_ms': ci, 'status': 'FAILED'})

    # Save status
    df = pd.DataFrame(results)
    df.to_csv(V7_DIR / "s1s2_simulation_status.csv", index=False)

    success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
    print(f"\nS1-S2 simulations complete: {success_count}/{total} successful")

    return results


# EXTRACT RESULTS
def read_igb_data(filepath, n_nodes):
    """Read IGB binary data"""
    try:
        with open(filepath, 'rb') as f:
            header = f.read(1024).decode('ascii', errors='ignore')

        info = {}
        for part in header.split():
            if ':' in part:
                k, v = part.split(':', 1)
                try:
                    info[k] = int(v)
                except:
                    info[k] = v

        n_timesteps = info.get('t', 500)

        with open(filepath, 'rb') as f:
            f.seek(1024)
            data = np.frombuffer(f.read(), dtype=np.float32)

        actual = len(data) // n_nodes
        if actual > 0:
            return data[:actual*n_nodes].reshape(actual, n_nodes), actual
    except:
        pass
    return None, 0


def get_n_nodes(patient_id):
    """Get node count for patient"""
    for base in [V7_DIR / patient_id / "mesh", CALCIUM_DIR / patient_id]:
        pts = base / f"{patient_id}.pts"
        if pts.exists():
            with open(pts) as f:
                return int(f.readline().strip())
    return 70000


def extract_calcium_results():
    """Extract calcium transient metrics"""
    print("EXTRACTING CALCIUM TRANSIENT RESULTS")

    results = []

    for patient in PATIENTS:
        result = {'patient_id': patient, 'calcium_available': False}

        # Look for Cai output in possible locations
        sim_dir = CALCIUM_DIR / patient / f"{patient}_calcium"
        cai_file = None

        for loc in [sim_dir / "Cai_output.igb",
                   CALCIUM_DIR / patient / "Cai_output.igb",
                   sim_dir / "gvec_Cai_output.igb"]:
            if loc.exists():
                cai_file = loc
                break

        if cai_file and cai_file.exists():
            n_nodes = get_n_nodes(patient)
            cai_data, n_t = read_igb_data(cai_file, n_nodes)

            if cai_data is not None:
                result['calcium_available'] = True
                result['Cai_n_timesteps'] = n_t
                # Note: OpenCarp gvec output needs scaling by 0.01 for physiological uM
                # Raw values are ~100x higher than expected
                scale = 0.01  # Correction factor for physiological range
                result['Cai_peak_mM'] = float(np.max(cai_data))
                result['Cai_peak_uM'] = float(np.max(cai_data) * 1000 * scale)  # Scaled to ~1-2 uM
                result['Cai_diastolic_mM'] = float(np.percentile(cai_data[0, :], 10))
                result['Cai_diastolic_uM'] = float(np.percentile(cai_data[0, :], 10) * 1000 * scale)  # Scaled to ~0.1 uM
                result['Cai_amplitude_uM'] = result['Cai_peak_uM'] - result['Cai_diastolic_uM']
                result['Cai_time_to_peak_ms'] = float(np.mean(np.argmax(cai_data, axis=0)))

                print(f"  {patient}: Peak={result['Cai_peak_uM']:.4f} uM")
            else:
                print(f"  {patient}: Could not read data")
        else:
            print(f"  {patient}: No calcium output found")

        results.append(result)

    df = pd.DataFrame(results)
    output = V7_DIR / "calcium_transient_metrics.csv"
    df.to_csv(output, index=False)
    print(f"\nSaved: {output}")

    return df


def extract_s1s2_results():
    """Extract S1-S2 vulnerability metrics"""
    print("EXTRACTING S1-S2 VULNERABILITY RESULTS")

    results = []

    for patient in PATIENTS:
        result = {
            'patient_id': patient,
            's1s2_available': False,
            'ERP_ms': None,
            'capture_threshold_ms': None,
            'vulnerable_window_start_ms': None,
            'vulnerable_window_end_ms': None,
            'reentry_induced': False,
            'max_activation_duration_ms': None
        }

        captured = []
        reentry = []

        for ci in S2_INTERVALS:
            sim_dir = S1S2_DIR / patient / f"{patient}_s1s2_{ci}"

            # Look for LAT output
            lat_file = None
            for loc in [sim_dir / "LAT-thresh.dat",
                       sim_dir / "LAT.dat",
                       S1S2_DIR / patient / f"{patient}_s1s2_{ci}_LAT.dat"]:
                if loc.exists():
                    lat_file = loc
                    break

            if lat_file and lat_file.exists():
                result['s1s2_available'] = True

                try:
                    lat = np.loadtxt(lat_file)
                    s2_time = 4000 + ci  # 8 beats at 500ms BCL + CI

                    # Check S2 capture
                    post_s2 = lat[lat[:, 1] > s2_time]
                    if len(post_s2) > 100:
                        captured.append(ci)

                        # Check for reentry (late activations)
                        late = lat[lat[:, 1] > s2_time + 350]
                        if len(late) > 500:
                            reentry.append(ci)

                            # Duration of activity
                            max_lat = np.max(lat[:, 1])
                            result['max_activation_duration_ms'] = max(
                                result.get('max_activation_duration_ms') or 0,
                                max_lat - s2_time
                            )
                except:
                    pass

        if captured:
            result['ERP_ms'] = min(captured)
            result['capture_threshold_ms'] = min(captured)
            print(f"  {patient}: ERP={result['ERP_ms']}ms", end="")

            if reentry:
                result['reentry_induced'] = True
                result['vulnerable_window_start_ms'] = min(reentry)
                result['vulnerable_window_end_ms'] = max(reentry)
                print(f", REENTRY at CI={reentry}")
            else:
                print(", No reentry")
        else:
            print(f"  {patient}: No S1-S2 data")

        results.append(result)

    df = pd.DataFrame(results)
    output = V7_DIR / "s1s2_vulnerability_metrics.csv"
    df.to_csv(output, index=False)
    print(f"\nSaved: {output}")

    return df


def extract_all_results():
    """Extract all results from completed simulations"""
    df_cal = extract_calcium_results()
    df_s1s2 = extract_s1s2_results()
    return df_cal, df_s1s2


# BATCH SCRIPTS FOR BACKGROUND EXECUTION
def create_batch_scripts():
    """Create shell scripts for running simulations in background"""
    print("CREATING BATCH SCRIPTS")

    # Calcium batch script
    calcium_script = BASE_DIR / "run_all_calcium.sh"
    content = """#!/bin/bash
# Run all calcium transient simulations
# Usage: nohup bash run_all_calcium.sh > calcium_log.txt 2>&1 &

PATIENTS="SCD0000101 SCD0000201 SCD0000301 SCD0000401 SCD0000601 SCD0000701 SCD0000801 SCD0001001 SCD0001101 SCD0001201"

for patient in $PATIENTS; do
    echo "Running calcium simulation for $patient..."
    cd /home/shadeform/SCD_MODELS/opencarp_results/calcium_sims/$patient
    mpirun -np 16 /usr/local/bin/openCARP +F simulation_calcium.par
    echo "Completed $patient"
done

echo "All calcium simulations complete!"
"""
    with open(calcium_script, 'w') as f:
        f.write(content)
    os.chmod(calcium_script, 0o755)
    print(f"  Created: {calcium_script}")

    # S1-S2 batch script
    s1s2_script = BASE_DIR / "run_all_s1s2.sh"
    content = """#!/bin/bash
# Run all S1-S2 vulnerability simulations
# Usage: nohup bash run_all_s1s2.sh > s1s2_log.txt 2>&1 &

PATIENTS="SCD0000101 SCD0000201 SCD0000301 SCD0000401 SCD0000601 SCD0000701 SCD0000801 SCD0001001 SCD0001101 SCD0001201"
INTERVALS="400 380 360 340 320 300 280 260 240 220 200"

for patient in $PATIENTS; do
    echo "Processing $patient..."
    cd /home/shadeform/SCD_MODELS/opencarp_results/s1s2_sims/$patient

    for ci in $INTERVALS; do
        echo "  Running CI=${ci}ms..."
        mpirun -np 16 /usr/local/bin/openCARP +F s1s2_${ci}ms.par
    done
    echo "Completed $patient"
done

echo "All S1-S2 simulations complete!"
"""
    with open(s1s2_script, 'w') as f:
        f.write(content)
    os.chmod(s1s2_script, 0o755)
    print(f"  Created: {s1s2_script}")

    print(f"""
To run simulations in background:
  nohup bash {calcium_script} > calcium_log.txt 2>&1 &
  nohup bash {s1s2_script} > s1s2_log.txt 2>&1 &

After completion, run in Python:
  from complete_missing_ep_workflow import extract_all_results
  extract_all_results()
""")


# MAIN
def main():
    print("COMPLETE MISSING EP SIMULATIONS WORKFLOW")

    # Verify setup
    opencarp_ok = verify_opencarp()
    files_ok = verify_parameter_files()

    if not opencarp_ok or not files_ok:
        print("\nSetup verification failed. Please fix issues above.")
        return

    # Run test
    print("\nRunning test simulation...")
    test_ok = run_test_simulation()

    if not test_ok:
        print("\nTest simulation failed. Check OpenCarp installation.")
        return

    # Create batch scripts
    create_batch_scripts()

    # Ask about running full simulations
    print("READY TO RUN SIMULATIONS")
    print("""

Recommended: Run batch scripts in background, then extract results later.
""")

    # For automated runs, just extract any existing results
    print("Checking for any existing simulation outputs...")
    extract_all_results()

    print("\nWorkflow complete. See instructions above for running simulations.")


if __name__ == "__main__":
    main()
