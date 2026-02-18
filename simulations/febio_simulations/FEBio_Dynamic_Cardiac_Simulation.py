#!/usr/bin/env python3
"""
FEBio DYNAMIC CARDIAC SIMULATION - JUPYTER NOTEBOOK RUNNER
Run   in a Jupyter notebook cell to execute the dynamic FEBio simulations.

Usage:
    %run FEBio_Dynamic_Cardiac_Simulation.py

Or import and run specific functions:
    from FEBio_Dynamic_Cardiac_Simulation import run_single_patient, run_all_patients

"""

# %%
# CELL 1: IMPORTS AND CONFIGURATION
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from IPython.display import display, 'HTML'

# Add parent directory to path
sys.path.insert(0, '/home/shadeform/SCD_MODELS')

# Import the main simulation module
from run_dynamic_febio_simulations import (
    MeshLoader, SurfaceExtractor, DynamicFEBGenerator,
    SimulationRunner, ResultsExtractor, BatchProcessor,
    PATIENTS, OUTPUT_DIR, MATERIAL_PARAMS, generate_pressure_waveform
)

print("FEBio Dynamic Cardiac Simulation Pipeline")
print(f"\nPatients: {len(PATIENTS)}")
print(f"Output directory: {OUTPUT_DIR}")

# %%
# CELL 2: RUN SINGLE PATIENT (for testing)
def run_single_patient(patient_id: str, verbose: bool = True):
    """
    Run dynamic simulation for a single patient.

    Args:
        patient_id: Patient identifier (e.g., "SCD0000101")
        verbose: Print detailed output

    Returns:
        dict: Simulation results
    """
    print(f"Processing {patient_id}")

    start_time = datetime.now()

    # Load mesh
    print("\n[1/5] Loading mesh data")
    mesh = MeshLoader(patient_id)
    if not mesh.load_all():
        return {"success": False, "error": "Failed to load mesh"}

    # Extract surfaces
    print("\n[2/5] Extracting surfaces")
    surfaces = SurfaceExtractor(mesh.nodes, mesh.elements)
    surfaces.extract_all()

    # Generate FEBio file
    print("\n[3/5] Generating FEBio input file")
    generator = DynamicFEBGenerator(mesh, surfaces, patient_id)
    feb_file = generator.generate()

    # Run simulation
    print("\n[4/5] Running FEBio simulation")
    runner = SimulationRunner(patient_id)
    success = runner.run(feb_file, timeout=1800)  # 30 min timeout

    # Extract results
    print("\n[5/5] Extracting results")
    extractor = ResultsExtractor(patient_id, mesh, surfaces)
    results = extractor.extract_all()

    elapsed = (datetime.now() - start_time).total_seconds()

    print(f"Completed in {elapsed:.1f} seconds")

    # Display key results
    if verbose and "LVEF_pct" in results:
        print(f"\nKey Results:")
        print(f"  EDV: {results['EDV_mL']:.1f} mL")
        print(f"  ESV: {results['ESV_mL']:.1f} mL")
        print(f"  Stroke Volume: {results['stroke_volume_mL']:.1f} mL")
        print(f"  LVEF: {results['LVEF_pct']:.1f}%")
        print(f"  Cardiac Output: {results['cardiac_output_L_min']:.2f} L/min")
        print(f"  GLS: {results.get('GLS_pct', 'N/A')}")

    return {
        "success": True,
        "patient_id": patient_id,
        "elapsed_s": elapsed,
        **results
    }


# Test with first patient
# result = run_single_patient("SCD0000101")

# %%
# CELL 3: RUN ALL PATIENTS
def run_all_patients():
    """
    Run dynamic simulations for all patients.

    Returns:
        dict: Results for all patients
    """
    processor = BatchProcessor()
    processor.process_all()
    return processor.results

# Uncomment to run all patients:
# all_results = run_all_patients()

# %%
# CELL 4: VISUALIZE PRESSURE-VOLUME LOOPS
def plot_pv_loops(patient_ids: list = None, save_path: str = None):
    """
    Plot P-V loops for specified patients.

    Args:
        patient_ids: List of patient IDs to plot (default: all)
        save_path: Path to save figure (optional)
    """
    if patient_ids is None:
        patient_ids = PATIENTS

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for idx, patient_id in enumerate(patient_ids):
        ax = axes[idx]

        # Load P-V data
        pv_file = OUTPUT_DIR / patient_id / "pv_loop_data.csv"
        if pv_file.exists():
            df = pd.read_csv(pv_file)
            ax.plot(df['volume_mL'], df['pressure_mmHg'], 'b-', linewidth=2)
            ax.fill(df['volume_mL'], df['pressure_mmHg'], alpha=0.3)

            # Mark ED and ES points
            ed_idx = df['volume_mL'].idxmax()
            es_idx = df['volume_mL'].idxmin()
            ax.plot(df.loc[ed_idx, 'volume_mL'], df.loc[ed_idx, 'pressure_mmHg'],
                   'go', markersize=10, label='ED')
            ax.plot(df.loc[es_idx, 'volume_mL'], df.loc[es_idx, 'pressure_mmHg'],
                   'ro', markersize=10, label='ES')
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                   transform=ax.transAxes)

        ax.set_xlabel('Volume (mL)')
        ax.set_ylabel('Pressure (mmHg)')
        ax.set_title(patient_id)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    plt.show()

# Uncomment to plot P-V loops:
# plot_pv_loops()

# %%
# CELL 5: VISUALIZE REGIONAL STRAIN
def plot_regional_strain(save_path: str = None):
    """
    Plot regional strain comparison across tissue types.
    """
    # Load regional data
    regional_file = OUTPUT_DIR / "regional_mechanics_summary.csv"

    if not regional_file.exists():
        print("Regional data not found. Run simulations first.")
        return

    df = pd.read_csv(regional_file)

    # Create grouped bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ['circumferential_strain_pct', 'longitudinal_strain_pct', 'radial_strain_pct']
    titles = ['Circumferential Strain', 'Longitudinal Strain', 'Radial Strain']

    regions = ['healthy', 'border_zone', 'infarct_scar']
    colors = ['green', 'orange', 'red']

    for ax, metric, title in zip(axes, metrics, titles):
        for i, (region, color) in enumerate(zip(regions, colors)):
            region_data = df[df['region'] == region][metric]
            patients = df[df['region'] == region]['patient_id']

            x = np.arange(len(patients)) + i * 0.25
            ax.bar(x, region_data, width=0.25, label=region, color=color, alpha=0.7)

        ax.set_xlabel('Patient')
        ax.set_ylabel('Strain (%)')
        ax.set_title(title)
        ax.legend()
        ax.set_xticks(np.arange(len(PATIENTS)) + 0.25)
        ax.set_xticklabels([p[-3:] for p in PATIENTS], rotation=45)
        ax.grid(True, axis='y', alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    plt.show()

# Uncomment to plot:
# plot_regional_strain()

# %%
# CELL 6: SUMMARY STATISTICS
def display_summary():
    """
    Display summary statistics from simulations.
    """
    summary_file = OUTPUT_DIR / "dynamic_simulation_summary.csv"

    if not summary_file.exists():
        print("Summary file not found. Run simulations first.")
        return

    df = pd.read_csv(summary_file)

    print("DYNAMIC SIMULATION SUMMARY")

    # Display as formatted table
    display(df.style.format({
        'EDV_mL': '{:.1f}',
        'ESV_mL': '{:.1f}',
        'stroke_volume_mL': '{:.1f}',
        'LVEF_pct': '{:.1f}',
        'cardiac_output_L_min': '{:.2f}',
        'stroke_work_J': '{:.3f}',
        'GLS_pct': '{:.1f}',
        'GCS_pct': '{:.1f}',
        'elapsed_s': '{:.1f}',
    }))

    # Statistics
    print("\nStatistical Summary:")
    for col in ['LVEF_pct', 'EDV_mL', 'ESV_mL', 'GLS_pct', 'cardiac_output_L_min']:
        if col in df.columns:
            print(f"{col}:")
            print(f"  Mean: {df[col].mean():.2f}")
            print(f"  Std:  {df[col].std():.2f}")
            print(f"  Range: {df[col].min():.2f} - {df[col].max():.2f}")

# Uncomment to display:
# display_summary()

# %%
# CELL 7: EXPORT ALL RESULTS TO COMPREHENSIVE CSV
def export_comprehensive_csv():
    """
    Export all simulation results to a single comprehensive CSV.
    """
    all_data = []

    for patient_id in PATIENTS:
        result_file = OUTPUT_DIR / patient_id / "dynamic_results.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                data = json.load(f)

            row = {
                "patient_id": patient_id,
                "simulation_type": "dynamic_cardiac_cycle",
                "EDV_mL": data.get("EDV_mL"),
                "ESV_mL": data.get("ESV_mL"),
                "stroke_volume_mL": data.get("stroke_volume_mL"),
                "LVEF_pct": data.get("LVEF_pct"),
                "cardiac_output_L_min": data.get("cardiac_output_L_min"),
                "stroke_work_J": data.get("stroke_work_J"),
                "GLS_pct": data.get("GLS_pct"),
                "GCS_pct": data.get("GCS_pct"),
                "dPdt_max_kPa_s": data.get("dPdt_max_kPa_s"),
                "dPdt_min_kPa_s": data.get("dPdt_min_kPa_s"),
                "ES_pressure_kPa": data.get("ES_pressure_kPa"),
                "ES_volume_mL": data.get("ES_volume_mL"),
                "ED_pressure_kPa": data.get("ED_pressure_kPa"),
                "ED_volume_mL": data.get("ED_volume_mL"),
            }

            # Add regional data
            if "regional" in data:
                for region in ["healthy", "border_zone", "infarct_scar"]:
                    if region in data["regional"]:
                        for metric, value in data["regional"][region].items():
                            row[f"{region}_{metric}"] = value

            all_data.append(row)

    if all_data:
        df = pd.DataFrame(all_data)
        output_path = OUTPUT_DIR / "COMPREHENSIVE_DYNAMIC_FEBIO_RESULTS.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
        print(f"Columns: {len(df.columns)}")
        return df
    else:
        print("No results found. Run simulations first.")
        return None

# Uncomment to export:
# df_comprehensive = export_comprehensive_csv()

# %%
# CELL 8: QUICK START - RUN EVERYTHING
def run_complete_pipeline():
    """
    Run the complete pipeline: simulations, visualization, and export.
    """
    print("STARTING COMPLETE DYNAMIC SIMULATION PIPELINE")

    # Run all simulations
    print("\n[STEP 1] Running simulations")
    all_results = run_all_patients()

    # Generate visualizations
    print("\n[STEP 2] Generating visualizations")
    plot_pv_loops(save_path=str(OUTPUT_DIR / "all_pv_loops.png"))
    plot_regional_strain(save_path=str(OUTPUT_DIR / "regional_strain_comparison.png"))

    # Export comprehensive CSV
    print("\n[STEP 3] Exporting comprehensive results")
    df = export_comprehensive_csv()

    # Display summary
    print("\n[STEP 4] Summary")
    display_summary()

    print("PIPELINE COMPLETE")
    print(f"All results saved to: {OUTPUT_DIR}")

    return all_results, df

# Uncomment to run everything:
# results, df = run_complete_pipeline()

# %%
# CELL 9: INTERACTIVE - RUN WITH CUSTOM PARAMETERS
# To run with custom parameters, modify these and execute:

CUSTOM_PARAMS = {
    "patients_to_run": ["SCD0000101"],  # List of patients or None for all
    "timeout_seconds": 1800,             # Simulation timeout
    "verbose": True,                     # Detailed output
}

def run_custom():
    """Run with custom parameters."""
    patients = CUSTOM_PARAMS["patients_to_run"] or PATIENTS

    results = {}
    for pid in patients:
        results[pid] = run_single_patient(pid, verbose=CUSTOM_PARAMS["verbose"])

    return results

# Uncomment to run custom:
# custom_results = run_custom()
