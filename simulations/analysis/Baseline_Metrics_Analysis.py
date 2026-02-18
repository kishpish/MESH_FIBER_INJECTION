#!/usr/bin/env python3
"""
PHASE TWO: Baseline Metrics Analysis - Jupyter Notebook Runner

This script is designed to run in Jupyter notebooks for interactive analysis
and visualization of baseline cardiac mechanics metrics.

Usage in Jupyter:
    %run Baseline_Metrics_Analysis.py

Or import specific functions:
    from Baseline_Metrics_Analysis import (
        run_extraction, plot_global_function, plot_regional_strains,
        plot_border_zone_analysis, plot_pv_loops_comparison
    )
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the extraction module
from extract_baseline_metrics import (
    BaselineMetricsExtractor, PatientBaselineMetrics,
    flatten_metrics, PATIENTS, DYNAMIC_RESULTS_DIR, OUTPUT_DIR
)

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# EXTRACTION RUNNER
def run_extraction() -> Tuple[pd.DataFrame, List[PatientBaselineMetrics]]:
    """
    Run baseline metrics extraction for all patients.

    Returns:
        Tuple of (DataFrame with all metrics, List of PatientBaselineMetrics objects)
    """
    print("Extracting baseline metrics for all patients
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for patient_id in PATIENTS:
        extractor = BaselineMetricsExtractor(patient_id, DYNAMIC_RESULTS_DIR)
        metrics = extractor.extract_all_metrics()
        all_metrics.append(metrics)
        print(f"  {patient_id}: LVEF={metrics.global_function.ejection_fraction_pct:.1f}%")

    # Create DataFrame
    rows = [flatten_metrics(m) for m in all_metrics]
    df = pd.DataFrame(rows)

    # Save to CSV
    csv_path = OUTPUT_DIR / "BASELINE_METRICS_ALL_PATIENTS.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved to: {csv_path}")

    return df, all_metrics


def load_existing_metrics() -> Optional[pd.DataFrame]:
    """Load existing baseline metrics CSV if available."""
    csv_path = OUTPUT_DIR / "BASELINE_METRICS_ALL_PATIENTS.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# GLOBAL FUNCTION VISUALIZATIONS
def plot_global_function(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Create comprehensive global function metrics visualization.

    Args:
        df: DataFrame with baseline metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Ejection Fraction
    ax = axes[0, 0]
    colors = ['#2ecc71' if ef > 40 else '#e74c3c' for ef in df['ejection_fraction_pct']]
    bars = ax.bar(range(len(df)), df['ejection_fraction_pct'], color=colors, edgecolor='black')
    ax.axhline(y=55, color='green', linestyle='--', label='Normal (>55%)')
    ax.axhline(y=40, color='orange', linestyle='--', label='HFrEF threshold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Ejection Fraction (%)')
    ax.set_title('Left Ventricular Ejection Fraction')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 70)

    # 2. Stroke Volume and Cardiac Output
    ax = axes[0, 1]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['stroke_volume_mL'], width, label='Stroke Volume (mL)', color='#3498db')
    ax2 = ax.twinx()
    ax2.bar(x + width/2, df['cardiac_output_L_min'], width, label='Cardiac Output (L/min)', color='#e67e22')
    ax.set_xticks(x)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stroke Volume (mL)', color='#3498db')
    ax2.set_ylabel('Cardiac Output (L/min)', color='#e67e22')
    ax.set_title('Stroke Volume & Cardiac Output')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 3. dP/dt max (Contractility)
    ax = axes[0, 2]
    ax.bar(range(len(df)), df['dPdt_max_mmHg_s'], color='#9b59b6', edgecolor='black')
    ax.axhline(y=1200, color='green', linestyle='--', label='Normal >1200 mmHg/s')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('dP/dt max (mmHg/s)')
    ax.set_title('Contractility (dP/dt max)')
    ax.legend()

    # 4. Tau (Relaxation)
    ax = axes[1, 0]
    colors = ['#2ecc71' if t < 50 else '#e74c3c' for t in df['tau_ms']]
    ax.bar(range(len(df)), df['tau_ms'], color=colors, edgecolor='black')
    ax.axhline(y=48, color='green', linestyle='--', label='Normal <48ms')
    ax.axhline(y=55, color='red', linestyle='--', label='Abnormal >55ms')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Tau (ms)')
    ax.set_title('Relaxation Time Constant')
    ax.legend()

    # 5. Stroke Work
    ax = axes[1, 1]
    ax.bar(range(len(df)), df['stroke_work_J'], color='#1abc9c', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stroke Work (J)')
    ax.set_title('Stroke Work (P-V Loop Area)')

    # 6. EDV vs ESV
    ax = axes[1, 2]
    x = np.arange(len(df))
    width = 0.35
    ax.bar(x - width/2, df['EDV_mL'], width, label='EDV', color='#3498db')
    ax.bar(x + width/2, df['ESV_mL'], width, label='ESV', color='#e74c3c')
    ax.set_xticks(x)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Volume (mL)')
    ax.set_title('End-Diastolic vs End-Systolic Volume')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_regional_strains(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Visualize regional strain metrics by tissue type.

    Args:
        df: DataFrame with baseline metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    tissue_types = ['healthy', 'border_zone', 'infarct_scar']
    colors = {'healthy': '#27ae60', 'border_zone': '#f39c12', 'infarct_scar': '#c0392b'}
    labels = {'healthy': 'Healthy', 'border_zone': 'Border Zone', 'infarct_scar': 'Infarct Scar'}

    x = np.arange(len(df))
    width = 0.25

    # 1. Circumferential Strain
    ax = axes[0, 0]
    for i, tissue in enumerate(tissue_types):
        col = f'{tissue}_circumferential_strain_pct'
        ax.bar(x + i*width, df[col], width, label=labels[tissue], color=colors[tissue])
    ax.axhline(y=-18, color='green', linestyle='--', alpha=0.7, label='Normal healthy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Strain (%)')
    ax.set_title('Circumferential Strain by Region')
    ax.legend(loc='lower right')
    ax.set_ylim(-25, 5)

    # 2. Longitudinal Strain
    ax = axes[0, 1]
    for i, tissue in enumerate(tissue_types):
        col = f'{tissue}_longitudinal_strain_pct'
        ax.bar(x + i*width, df[col], width, label=labels[tissue], color=colors[tissue])
    ax.axhline(y=-20, color='green', linestyle='--', alpha=0.7, label='Normal healthy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Strain (%)')
    ax.set_title('Longitudinal Strain by Region')
    ax.legend(loc='lower right')
    ax.set_ylim(-25, 5)

    # 3. Radial Strain (Wall Thickening)
    ax = axes[1, 0]
    for i, tissue in enumerate(tissue_types):
        col = f'{tissue}_radial_strain_pct'
        ax.bar(x + i*width, df[col], width, label=labels[tissue], color=colors[tissue])
    ax.axhline(y=45, color='green', linestyle='--', alpha=0.7, label='Normal healthy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Strain (%)')
    ax.set_title('Radial Strain (Wall Thickening) by Region')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 55)

    # 4. Peak Stress
    ax = axes[1, 1]
    for i, tissue in enumerate(tissue_types):
        col = f'{tissue}_peak_stress_kPa'
        ax.bar(x + i*width, df[col], width, label=labels[tissue], color=colors[tissue])
    ax.set_xticks(x + width)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Peak Stress (kPa)')
    ax.set_title('Peak Systolic Stress by Region')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_border_zone_analysis(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Detailed border zone analysis visualization.

    Args:
        df: DataFrame with baseline metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Stress Concentration Factor
    ax = axes[0, 0]
    colors = ['#e74c3c' if scf > 2.0 else '#f39c12' if scf > 1.5 else '#27ae60'
              for scf in df['bz_stress_concentration_factor']]
    ax.bar(range(len(df)), df['bz_stress_concentration_factor'], color=colors, edgecolor='black')
    ax.axhline(y=1.5, color='orange', linestyle='--', label='Target (<1.5)')
    ax.axhline(y=2.0, color='red', linestyle='--', label='High risk (>2.0)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stress Concentration Factor')
    ax.set_title('Border Zone Stress Concentration\n(BZ peak / Healthy mean)')
    ax.legend()

    # 2. Strain Mismatch
    ax = axes[0, 1]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['bz_circumferential_strain_mismatch_pct'], width,
           label='Circumferential', color='#3498db')
    ax.bar(x, df['bz_longitudinal_strain_mismatch_pct'], width,
           label='Longitudinal', color='#9b59b6')
    ax.bar(x + width, df['bz_radial_strain_mismatch_pct'], width,
           label='Radial', color='#e67e22')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Strain Mismatch (%)')
    ax.set_title('Border Zone - Healthy Strain Mismatch')
    ax.legend()

    # 3. Mechanical Disadvantage Index
    ax = axes[0, 2]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    sorted_idx = df['bz_mechanical_disadvantage_index'].argsort()
    ax.barh(range(len(df)), df['bz_mechanical_disadvantage_index'].iloc[sorted_idx],
            color=[colors[i] for i in range(len(df))])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['patient_id'].iloc[sorted_idx])
    ax.set_xlabel('Mechanical Disadvantage Index')
    ax.set_title('Border Zone Mechanical Burden\n(Higher = More vulnerable)')

    # 4. Stress Gradient at Interface
    ax = axes[1, 0]
    ax.bar(range(len(df)), df['bz_stress_gradient_scar_bz_kPa_mm'], color='#8e44ad', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stress Gradient (kPa/mm)')
    ax.set_title('Stress Gradient at Scar-BZ Interface')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # 5. Work Density Ratio
    ax = axes[1, 1]
    colors = ['#e74c3c' if wr > 1.5 else '#27ae60' for wr in df['bz_work_density_ratio']]
    ax.bar(range(len(df)), df['bz_work_density_ratio'], color=colors, edgecolor='black')
    ax.axhline(y=1.0, color='green', linestyle='--', label='Equal work')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Work Density Ratio')
    ax.set_title('BZ Work / Healthy Work Ratio')
    ax.legend()

    # 6. Wall Stress Index
    ax = axes[1, 2]
    ax.bar(range(len(df)), df['bz_wall_stress_index_kPa'], color='#16a085', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Wall Stress Index (kPa)')
    ax.set_title('Laplace Wall Stress Estimate')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_pv_loops_comparison(patients: List[str] = None, save_path: Optional[str] = None):
    """
    Plot P-V loops for multiple patients on the same axes.

    Args:
        patients: List of patient IDs (default: all patients)
        save_path: Optional path to save figure
    """
    if patients is None:
        patients = PATIENTS

    fig, ax = plt.subplots(figsize=(12, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(patients)))

    for i, patient_id in enumerate(patients):
        json_path = DYNAMIC_RESULTS_DIR / patient_id / "dynamic_results.json"
        if not json_path.exists():
            continue

        with open(json_path) as f:
            data = json.load(f)

        volumes = data.get('volume_mL', [])
        pressures = data.get('pressure_kPa', [])

        if len(volumes) > 0 and len(pressures) > 0:
            # Convert pressure to mmHg for clinical relevance
            pressures_mmHg = np.array(pressures) * 7.50062
            ax.plot(volumes, pressures_mmHg, '-', linewidth=2,
                   color=colors[i], label=patient_id[-3:], alpha=0.8)

            # Mark ED and ES points
            ax.scatter([volumes[0]], [pressures_mmHg[0]], s=50, color=colors[i],
                      marker='o', zorder=5)  # ED
            min_vol_idx = np.argmin(volumes)
            ax.scatter([volumes[min_vol_idx]], [pressures_mmHg[min_vol_idx]], s=50,
                      color=colors[i], marker='s', zorder=5)  # ES

    ax.set_xlabel('Volume (mL)', fontsize=12)
    ax.set_ylabel('Pressure (mmHg)', fontsize=12)
    ax.set_title('Pressure-Volume Loops - All Patients', fontsize=14)
    ax.legend(title='Patient', loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add direction arrow
    ax.annotate('', xy=(200, 100), xytext=(300, 100),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.text(250, 105, 'Ejection', ha='center', fontsize=10, color='gray')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def plot_global_strain_summary(df: pd.DataFrame, save_path: Optional[str] = None):
    """
    Plot global strain indices (GLS, GCS, GRS).

    Args:
        df: DataFrame with baseline metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GLS
    ax = axes[0]
    colors = ['#27ae60' if gls < -18 else '#e74c3c' for gls in df['GLS_pct']]
    ax.bar(range(len(df)), df['GLS_pct'], color=colors, edgecolor='black')
    ax.axhline(y=-18, color='green', linestyle='--', label='Normal (<-18%)')
    ax.axhline(y=-16, color='orange', linestyle='--', label='Abnormal (>-16%)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('GLS (%)')
    ax.set_title('Global Longitudinal Strain')
    ax.legend()
    ax.set_ylim(-25, 0)

    # GCS
    ax = axes[1]
    ax.bar(range(len(df)), df['GCS_pct'], color='#3498db', edgecolor='black')
    ax.axhline(y=-20, color='green', linestyle='--', label='Normal')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('GCS (%)')
    ax.set_title('Global Circumferential Strain')
    ax.legend()
    ax.set_ylim(-25, 0)

    # GRS
    ax = axes[2]
    ax.bar(range(len(df)), df['GRS_pct'], color='#9b59b6', edgecolor='black')
    ax.axhline(y=40, color='green', linestyle='--', label='Normal')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('GRS (%)')
    ax.set_title('Global Radial Strain')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def generate_clinical_report(df: pd.DataFrame) -> str:
    """
    Generate a clinical summary report of baseline metrics.

    Args:
        df: DataFrame with baseline metrics

    Returns:
        Formatted clinical report string
    """
    report = []
    report.append("CLINICAL SUMMARY: Baseline Cardiac Mechanics Metrics")

    # Summary statistics
    report.append("\n1. GLOBAL FUNCTION SUMMARY")
    report.append(f"  LVEF: {df['ejection_fraction_pct'].mean():.1f} +/- {df['ejection_fraction_pct'].std():.1f}%")
    report.append(f"  Range: {df['ejection_fraction_pct'].min():.1f} - {df['ejection_fraction_pct'].max():.1f}%")

    hfref_count = (df['ejection_fraction_pct'] < 40).sum()
    report.append(f"  Patients with HFrEF (EF<40%): {hfref_count}/{len(df)}")

    report.append(f"\n  Cardiac Output: {df['cardiac_output_L_min'].mean():.2f} +/- {df['cardiac_output_L_min'].std():.2f} L/min")
    report.append(f"  dP/dt max: {df['dPdt_max_mmHg_s'].mean():.0f} +/- {df['dPdt_max_mmHg_s'].std():.0f} mmHg/s")
    report.append(f"  Tau: {df['tau_ms'].mean():.1f} +/- {df['tau_ms'].std():.1f} ms")

    report.append("\n2. GLOBAL STRAIN INDICES")
    report.append(f"  GLS: {df['GLS_pct'].mean():.1f} +/- {df['GLS_pct'].std():.1f}%")
    gls_abnormal = (df['GLS_pct'] > -16).sum()
    report.append(f"  Patients with abnormal GLS (>-16%): {gls_abnormal}/{len(df)}")
    report.append(f"  GCS: {df['GCS_pct'].mean():.1f} +/- {df['GCS_pct'].std():.1f}%")
    report.append(f"  GRS: {df['GRS_pct'].mean():.1f} +/- {df['GRS_pct'].std():.1f}%")

    report.append("\n3. BORDER ZONE ANALYSIS")
    report.append(f"  Stress Concentration Factor: {df['bz_stress_concentration_factor'].mean():.2f} +/- {df['bz_stress_concentration_factor'].std():.2f}")
    high_scf = (df['bz_stress_concentration_factor'] > 2.0).sum()
    report.append(f"  High stress concentration (>2.0): {high_scf}/{len(df)} patients")

    report.append(f"\n  Circumferential Strain Mismatch: {df['bz_circumferential_strain_mismatch_pct'].mean():.1f}%")
    report.append(f"  Mechanical Disadvantage Index: {df['bz_mechanical_disadvantage_index'].mean():.2f}")

    report.append("\n4. HYDROGEL DESIGN IMPLICATIONS")
    avg_scf = df['bz_stress_concentration_factor'].mean()
    if avg_scf > 2.0:
        report.append("  - HIGH priority: Significant stress concentration in border zone")
        report.append("  - Recommend: Stiffer hydrogel patches for mechanical support")
    elif avg_scf > 1.5:
        report.append("  - MODERATE priority: Elevated stress concentration")
        report.append("  - Recommend: Medium stiffness hydrogel with good compliance matching")
    else:
        report.append("  - LOWER priority: Acceptable stress concentration")
        report.append("  - Recommend: Focus on biological integration over mechanical support")

    # Individual patient flags
    report.append("\n5. INDIVIDUAL PATIENT FLAGS")
    for _, row in df.iterrows():
        flags = []
        if row['ejection_fraction_pct'] < 35:
            flags.append("Severe HFrEF")
        if row['GLS_pct'] > -14:
            flags.append("Severely reduced GLS")
        if row['bz_stress_concentration_factor'] > 2.5:
            flags.append("Critical BZ stress")
        if row['bz_mechanical_disadvantage_index'] > 3.0:
            flags.append("High mechanical burden")

        if flags:
            report.append(f"  {row['patient_id']}: {', '.join(flags)}")


    return "\n".join(report)


# MAIN EXECUTION
def run_complete_analysis():
    """Run complete baseline metrics analysis with all visualizations."""
    print("PHASE TWO: Complete Baseline Metrics Analysis")

    # Extract or load metrics
    existing_df = load_existing_metrics()
    if existing_df is not None:
        print("Loading existing metrics
        df = existing_df
        all_metrics = None
    else:
        print("Extracting metrics from simulation results
        df, all_metrics = run_extraction()

    # Create output directory for figures
    fig_dir = OUTPUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Generate all visualizations
    print("\nGenerating visualizations

    plot_global_function(df, str(fig_dir / "global_function_metrics.png"))
    plot_regional_strains(df, str(fig_dir / "regional_strain_metrics.png"))
    plot_border_zone_analysis(df, str(fig_dir / "border_zone_analysis.png"))
    plot_pv_loops_comparison(save_path=str(fig_dir / "pv_loops_comparison.png"))
    plot_global_strain_summary(df, str(fig_dir / "global_strain_summary.png"))

    # Generate clinical report
    print("\nGenerating clinical report
    report = generate_clinical_report(df)
    print(report)

    # Save report
    report_path = OUTPUT_DIR / "CLINICAL_SUMMARY_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    print("Analysis Complete")
    print(f"Output directory: {OUTPUT_DIR}")

    return df


# Run if executed directly
if __name__ == "__main__":
    df = run_complete_analysis()
