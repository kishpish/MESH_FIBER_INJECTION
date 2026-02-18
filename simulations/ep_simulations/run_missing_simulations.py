#!/usr/bin/env python3
"""
Run Missing OpenCarp Simulations and Extract Results

This script:
1. Runs calcium transient simulations for all 10 patients
2. Runs S1-S2 vulnerability protocol simulations for all 10 patients
3. Extracts results and generates CSV files

: This requires OpenCarp v18.1 to be installed and accessible.

 %run run_missing_simulations.py
"""

import subprocess
import numpy as np
import json
from pathlib import Path
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
V7_DIR = BASE_DIR / "opencarp_results/v7_all"
CALCIUM_DIR = BASE_DIR / "opencarp_results/calcium_sims"
S1S2_DIR = BASE_DIR / "opencarp_results/s1s2_sims"

PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000601",
    "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]

# OpenCarp executable path
OPENCARP_CMD = "/usr/local/bin/openCARP"
MPI_NP = 16  # Number of MPI processes


def check_opencarp_available():
    """Check if OpenCarp is available"""
    try:
        result = subprocess.run([OPENCARP_CMD, "--version"],
                              capture_output=True, text=True, timeout=10)
        print(f"OpenCarp found: {result.stdout.strip()}")
        return True
    except Exception as e:
        print(f"OpenCarp not found or not accessible: {e}")
        print("Please ensure OpenCarp is installed and in PATH")
        return False


def run_simulation(par_file, timeout_minutes=30):
    """Run a single OpenCarp simulation"""
    work_dir = par_file.parent

    cmd = f"cd {work_dir} && mpirun -np {MPI_NP} {OPENCARP_CMD} +F {par_file.name}"

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=timeout_minutes * 60
        )
        if result.returncode == 0:
            return True, "Success"
        else:
            return False, result.stderr[:500]
    except subprocess.TimeoutExpired:
        return False, "Timeout"
    except Exception as e:
        return False, str(e)


# CALCIUM TRANSIENT SIMULATIONS
def run_all_calcium_simulations():
    """Run calcium transient simulations for all patients"""
    print("RUNNING CALCIUM TRANSIENT SIMULATIONS")

    results = []

    for patient_id in PATIENTS:
        par_file = CALCIUM_DIR / patient_id / "simulation_calcium.par"

        if not par_file.exists():
            print(f"  {patient_id}: Parameter file not found")
            results.append({'patient_id': patient_id, 'status': 'MISSING_PAR'})
            continue

        print(f"  Running {patient_id}...", end=" ", flush=True)
        start_time = time.time()

        success, msg = run_simulation(par_file, timeout_minutes=60)

        elapsed = time.time() - start_time

        if success:
            print(f"DONE ({elapsed:.1f}s)")
            results.append({'patient_id': patient_id, 'status': 'SUCCESS', 'time_s': elapsed})
        else:
            print(f"FAILED: {msg[:50]}")
            results.append({'patient_id': patient_id, 'status': 'FAILED', 'error': msg})

    return results


def extract_calcium_results():
    """Extract calcium transient metrics from simulation outputs"""
    print("EXTRACTING CALCIUM TRANSIENT RESULTS")

    all_results = []

    for patient_id in PATIENTS:
        sim_dir = CALCIUM_DIR / patient_id / f"{patient_id}_calcium"
        cai_file = sim_dir / "Cai_output.igb"

        result = {'patient_id': patient_id}

        # Check for output files
        if not cai_file.exists():
            # Try alternative location
            cai_file = CALCIUM_DIR / patient_id / "Cai_output.igb"

        if cai_file.exists():
            result['calcium_available'] = True

            # Read calcium data
            try:
                cai_data, n_timesteps = read_igb_data(cai_file, get_n_nodes(patient_id))

                if cai_data is not None:
                    # Extract metrics
                    result['Cai_n_timesteps'] = n_timesteps
                    result['Cai_peak_uM'] = float(np.max(cai_data))
                    result['Cai_diastolic_uM'] = float(np.min(cai_data[0, :]))
                    result['Cai_amplitude_uM'] = result['Cai_peak_uM'] - result['Cai_diastolic_uM']

                    # Time to peak (per node, then average)
                    peak_times = np.argmax(cai_data, axis=0)
                    result['Cai_time_to_peak_ms'] = float(np.mean(peak_times))

                    # Calcium transient duration (time above 50% amplitude)
                    threshold = result['Cai_diastolic_uM'] + 0.5 * result['Cai_amplitude_uM']
                    durations = []
                    for node in range(min(1000, cai_data.shape[1])):  # Sample nodes
                        trace = cai_data[:, node]
                        above = trace > threshold
                        if above.any():
                            start = np.argmax(above)
                            end = len(above) - np.argmax(above[::-1])
                            durations.append(end - start)
                    if durations:
                        result['Cai_duration_50_ms'] = float(np.mean(durations))

                    print(f"  {patient_id}: Peak={result['Cai_peak_uM']:.3f} uM, Amplitude={result['Cai_amplitude_uM']:.3f} uM")
                else:
                    result['calcium_available'] = False
                    result['error'] = "Failed to read IGB"
                    print(f"  {patient_id}: Failed to read calcium data")
            except Exception as e:
                result['calcium_available'] = False
                result['error'] = str(e)
                print(f"  {patient_id}: Error - {e}")
        else:
            result['calcium_available'] = False
            result['error'] = "No output file"
            print(f"  {patient_id}: No calcium output found")

        all_results.append(result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_csv = V7_DIR / "calcium_transient_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return df


# S1-S2 VULNERABILITY SIMULATIONS
def run_all_s1s2_simulations():
    """Run S1-S2 vulnerability simulations for all patients"""
    print("RUNNING S1-S2 VULNERABILITY SIMULATIONS")

    s2_intervals = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200]
    results = []

    for patient_id in PATIENTS:
        patient_results = {'patient_id': patient_id, 'protocols_run': 0, 'protocols_success': 0}

        print(f"\n  {patient_id}:")

        for ci in s2_intervals:
            par_file = S1S2_DIR / patient_id / f"s1s2_{ci}ms.par"

            if not par_file.exists():
                continue

            print(f"    CI={ci}ms...", end=" ", flush=True)
            patient_results['protocols_run'] += 1

            success, msg = run_simulation(par_file, timeout_minutes=45)

            if success:
                print("OK")
                patient_results['protocols_success'] += 1
            else:
                print(f"FAIL")

        results.append(patient_results)

    return results


def extract_s1s2_results():
    """Extract S1-S2 vulnerability metrics from simulation outputs"""
    print("EXTRACTING S1-S2 VULNERABILITY RESULTS")

    s2_intervals = [400, 380, 360, 340, 320, 300, 280, 260, 240, 220, 200]
    all_results = []

    for patient_id in PATIENTS:
        result = {
            'patient_id': patient_id,
            's1s2_available': False,
            'ERP_ms': None,
            'vulnerable_window_start_ms': None,
            'vulnerable_window_end_ms': None,
            'reentry_induced': False
        }

        captured_intervals = []
        reentry_intervals = []

        for ci in s2_intervals:
            sim_dir = S1S2_DIR / patient_id / f"{patient_id}_s1s2_{ci}"
            lat_file = sim_dir / "LAT-thresh.dat"

            if not lat_file.exists():
                # Try alternative location
                lat_file = S1S2_DIR / patient_id / f"LAT_s1s2_{ci}.dat"

            if lat_file.exists():
                result['s1s2_available'] = True

                # Analyze LAT data
                try:
                    lat_data = np.loadtxt(lat_file)

                    # S2 timing: S1*8 + CI = 4000 + CI
                    s2_time = 4000 + ci

                    # Check if S2 captured (activations after S2 time)
                    late_activations = lat_data[lat_data[:, 1] > s2_time]

                    if len(late_activations) > 100:  # S2 captured
                        captured_intervals.append(ci)

                        # Check for reentry (activations much later than expected)
                        # Normal APD ~250ms, so activations after S2+300ms suggest reentry
                        very_late = lat_data[lat_data[:, 1] > s2_time + 350]
                        if len(very_late) > 500:  # Reentry likely
                            reentry_intervals.append(ci)
                except Exception as e:
                    pass

        if captured_intervals:
            result['ERP_ms'] = min(captured_intervals)
            print(f"  {patient_id}: ERP = {result['ERP_ms']} ms")

            if reentry_intervals:
                result['reentry_induced'] = True
                result['vulnerable_window_start_ms'] = min(reentry_intervals)
                result['vulnerable_window_end_ms'] = max(reentry_intervals)
                print(f"    Reentry induced at CI = {reentry_intervals}")
            else:
                print(f"    No reentry induced")
        else:
            print(f"  {patient_id}: No S1-S2 data available")

        all_results.append(result)

    # Save to CSV
    df = pd.DataFrame(all_results)
    output_csv = V7_DIR / "s1s2_vulnerability_metrics.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved: {output_csv}")

    return df


# HELPER FUNCTIONS
def get_n_nodes(patient_id):
    """Get number of nodes for a patient mesh"""
    pts_file = V7_DIR / patient_id / "mesh" / f"{patient_id}.pts"
    if pts_file.exists():
        with open(pts_file) as f:
            return int(f.readline().strip())
    # Fallback to calcium sim mesh
    pts_file = CALCIUM_DIR / patient_id / f"{patient_id}.pts"
    if pts_file.exists():
        with open(pts_file) as f:
            return int(f.readline().strip())
    return 70000  # Default estimate


def read_igb_data(filepath, n_nodes):
    """Read IGB binary data file"""
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

        n_timesteps = info.get('t', 500)

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


# MAIN EXECUTION
def main():
    print("COMPLETE MISSING EP SIMULATIONS")
    print(f"""
This script will:
1. Run calcium transient simulations (10 patients)
2. Run S1-S2 vulnerability simulations (10 patients x 11 intervals = 110 runs)
3. Extract results and generate CSV files

IMPORTANT: This requires OpenCarp v18.1 and significant compute time.
Estimated time: 2-4 hours depending on hardware.
""")

    # Check if OpenCarp is available
    print("\nChecking OpenCarp availability...")
    if not check_opencarp_available():
        print("OPENCARP NOT AVAILABLE - GENERATING EXTRACTION SCRIPTS ONLY")
        print("""
OpenCarp is not available in the current environment.
The parameter files have been created. To run the simulations:

1. Ensure OpenCarp v18.1 is installed
2. Run calcium simulations:
   for patient in SCD0000101 SCD0000201 ...; do
       cd /home/shadeform/SCD_MODELS/opencarp_results/calcium_sims/$patient
       mpirun -np 16 openCARP +F simulation_calcium.par
   done

3. Run S1-S2 simulations:
   for patient in SCD0000101 SCD0000201 ...; do
       cd /home/shadeform/SCD_MODELS/opencarp_results/s1s2_sims/$patient
       bash run_all_s1s2.sh
   done

4. After simulations complete, run this script again to extract results.
""")
        # Just try to extract any existing results
        print("\nChecking for existing simulation outputs...")
        extract_calcium_results()
        extract_s1s2_results()
        return

    # Run simulations
    print("PHASE 1: CALCIUM TRANSIENT SIMULATIONS")
    calcium_status = run_all_calcium_simulations()

    print("PHASE 2: S1-S2 VULNERABILITY SIMULATIONS")
    s1s2_status = run_all_s1s2_simulations()

    # Extract results
    print("PHASE 3: EXTRACTING RESULTS")
    df_calcium = extract_calcium_results()
    df_s1s2 = extract_s1s2_results()

    # Summary
    print("SUMMARY")

    calcium_success = sum(1 for r in calcium_status if r.get('status') == 'SUCCESS')
    s1s2_success = sum(r.get('protocols_success', 0) for r in s1s2_status)

    print(f"""
CALCIUM TRANSIENTS:
  - Simulations run: {len(calcium_status)}
  - Successful: {calcium_success}/10
  - Output: calcium_transient_metrics.csv

S1-S2 VULNERABILITY:
  - Total protocols: {sum(r.get('protocols_run', 0) for r in s1s2_status)}
  - Successful: {s1s2_success}
  - Output: s1s2_vulnerability_metrics.csv

Output files in: {V7_DIR}
""")

    return df_calcium, df_s1s2


if __name__ == "__main__":
    main()
    print("\nDone!")
