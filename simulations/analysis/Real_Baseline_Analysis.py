#!/usr/bin/env python3
"""
REAL Baseline Metrics Analysis - Jupyter Notebook Runner

This script runs in Jupyter notebooks to analyze REAL baseline
metrics extracted from actual FEBio simulation output.

Usage in Jupyter:
    %run Real_Baseline_Analysis.py

Or import functions:
    from Real_Baseline_Analysis import (
        run_extraction, plot_real_strains,
        compare_tissue_strains, generate_summary
    )
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
RESULTS_DIR = BASE_DIR / "real_baseline_metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


def run_extraction():
    """Run the real metrics extraction script."""
    print("Running real FEBio metrics extraction...")
    exec(open(BASE_DIR / "extract_real_baseline_metrics.py").read())
    return load_results()


def load_results() -> pd.DataFrame:
    """Load existing results."""
    csv_path = RESULTS_DIR / "REAL_BASELINE_METRICS_ALL_PATIENTS.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print("No results found. Running extraction...")
        return run_extraction()


def plot_real_strains(df: pd.DataFrame, save: bool = True):
    """Plot REAL strain values from FEBio simulation."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Global Circumferential Strain
    ax = axes[0, 0]
    ax.bar(range(len(df)), df['GCS_pct'], color='#3498db', edgecolor='black')
    ax.axhline(y=-20, color='green', linestyle='--', alpha=0.7, label='Normal reference')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('GCS (%)')
    ax.set_title('Global Circumferential Strain (REAL from FEBio)')
    ax.legend()
    ax.set_ylim(-25, 0)

    # 2. Strain by Tissue Type - per patient
    ax = axes[0, 1]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['healthy_circumferential_strain_pct'], width,
           label='Healthy', color='#27ae60')
    ax.bar(x, df['border_zone_circumferential_strain_pct'], width,
           label='Border Zone', color='#f39c12')
    ax.bar(x + width, df['infarct_scar_circumferential_strain_pct'], width,
           label='Infarct Scar', color='#c0392b')
    ax.set_xticks(x)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Circumferential Strain (%)')
    ax.set_title('Regional Strain by Tissue Type (REAL)')
    ax.legend()
    ax.set_ylim(-25, 0)

    # 3. Strain Distribution (box plot style)
    ax = axes[1, 0]
    tissue_data = {
        'Healthy': df['healthy_circumferential_strain_pct'],
        'Border Zone': df['border_zone_circumferential_strain_pct'],
        'Infarct Scar': df['infarct_scar_circumferential_strain_pct']
    }
    colors = ['#27ae60', '#f39c12', '#c0392b']
    bp = ax.boxplot([tissue_data['Healthy'], tissue_data['Border Zone'],
                     tissue_data['Infarct Scar']], labels=tissue_data.keys(),
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel('Circumferential Strain (%)')
    ax.set_title('Strain Distribution by Tissue Type (REAL)')
    ax.axhline(y=-18, color='gray', linestyle='--', alpha=0.5, label='Normal healthy')

    # 4. Strain Variability (error bars)
    ax = axes[1, 1]
    tissues = ['healthy', 'border_zone', 'infarct_scar']
    means = [df[f'{t}_Ecc_mean'].mean() * 100 for t in tissues]
    stds = [df[f'{t}_Ecc_std'].mean() * 100 for t in tissues]
    x_pos = np.arange(len(tissues))

    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Healthy', 'Border Zone', 'Infarct Scar'])
    ax.set_ylabel('Circumferential Strain (%)')
    ax.set_title('Mean Strain with Variability (REAL)')

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "real_strain_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'real_strain_analysis.png'}")

    plt.show()
    return fig


def plot_functional_metrics(df: pd.DataFrame, save: bool = True):
    """Plot functional cardiac metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Ejection Fraction
    ax = axes[0, 0]
    colors = ['#27ae60' if ef > 40 else '#e74c3c' for ef in df['ejection_fraction_pct']]
    ax.bar(range(len(df)), df['ejection_fraction_pct'], color=colors, edgecolor='black')
    ax.axhline(y=55, color='green', linestyle='--', label='Normal (>55%)')
    ax.axhline(y=40, color='orange', linestyle='--', label='HFrEF threshold')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('LVEF (%)')
    ax.set_title('Left Ventricular Ejection Fraction')
    ax.legend()
    ax.set_ylim(0, 70)

    # 2. Cavity Volume (EDV)
    ax = axes[0, 1]
    ax.bar(range(len(df)), df['EDV_mL'], color='#3498db', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Volume (mL)')
    ax.set_title('End-Diastolic Volume (EDV)')

    # 3. Stroke Volume
    ax = axes[1, 0]
    ax.bar(range(len(df)), df['stroke_volume_mL'], color='#9b59b6', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stroke Volume (mL)')
    ax.set_title('Stroke Volume')

    # 4. Tissue Fractions
    ax = axes[1, 1]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['healthy_fraction_pct'], width, label='Healthy', color='#27ae60')
    ax.bar(x, df['border_zone_fraction_pct'], width, label='Border Zone', color='#f39c12')
    ax.bar(x + width, df['infarct_scar_fraction_pct'], width, label='Infarct Scar', color='#c0392b')
    ax.set_xticks(x)
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Fraction (%)')
    ax.set_title('Tissue Composition')
    ax.legend()

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "functional_metrics.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'functional_metrics.png'}")

    plt.show()
    return fig


def plot_border_zone_analysis(df: pd.DataFrame, save: bool = True):
    """Plot border zone specific metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. BZ Stress Concentration Factor
    ax = axes[0, 0]
    colors = ['#e74c3c' if scf > 1.2 else '#27ae60' for scf in df['bz_stress_concentration_factor']]
    ax.bar(range(len(df)), df['bz_stress_concentration_factor'], color=colors, edgecolor='black')
    ax.axhline(y=1.0, color='green', linestyle='--', label='Ideal (1.0)')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Stress Concentration Factor')
    ax.set_title('BZ Stress Concentration (Healthy/BZ strain ratio)')
    ax.legend()

    # 2. Strain Mismatch
    ax = axes[0, 1]
    ax.bar(range(len(df)), df['bz_circ_strain_mismatch_pct'], color='#8e44ad', edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Strain Mismatch (%)')
    ax.set_title('BZ - Healthy Circumferential Strain Mismatch')

    # 3. Mechanical Disadvantage Index
    ax = axes[1, 0]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df)))
    sorted_idx = df['bz_mechanical_disadvantage_index'].argsort()
    ax.barh(range(len(df)), df['bz_mechanical_disadvantage_index'].iloc[sorted_idx],
            color=[colors[i] for i in range(len(df))])
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['patient_id'].iloc[sorted_idx])
    ax.set_xlabel('Mechanical Disadvantage Index')
    ax.set_title('BZ Mechanical Burden (Higher = More vulnerable)')

    # 4. Wall Stress
    ax = axes[1, 1]
    ax.bar(range(len(df)), df['bz_wall_stress_kPa'], color='#16a085', edgecolor='black')
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels([p[-3:] for p in df['patient_id']], rotation=45)
    ax.set_ylabel('Wall Stress (kPa)')
    ax.set_title('Estimated Wall Stress (Laplace)')

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "border_zone_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'border_zone_analysis.png'}")

    plt.show()
    return fig


def generate_summary(df: pd.DataFrame) -> str:
    """Generate clinical summary of REAL baseline metrics."""
    summary = []
    summary.append("REAL BASELINE METRICS SUMMARY")
    summary.append("(From Actual FEBio Simulation Output)")

    summary.append("\n1. DATA SOURCE VERIFICATION")
    completed = df['simulation_completed'].sum()
    summary.append(f"  Simulations completed: {completed}/{len(df)}")
    summary.append(f"  Data source: VTK files with real Ecc strain values")

    summary.append("\n2. GLOBAL STRAIN (REAL)")
    summary.append(f"  GCS: {df['GCS_pct'].mean():.2f} +/- {df['GCS_pct'].std():.2f}%")
    summary.append(f"  Range: {df['GCS_pct'].min():.2f}% to {df['GCS_pct'].max():.2f}%")
    summary.append(f"  GLS (estimated): {df['GLS_pct'].mean():.2f}%")

    summary.append("\n3. REGIONAL STRAIN BY TISSUE (REAL)")
    for tissue in ['healthy', 'border_zone', 'infarct_scar']:
        col = f'{tissue}_circumferential_strain_pct'
        summary.append(f"  {tissue.replace('_', ' ').title()}:")
        summary.append(f"    Mean: {df[col].mean():.2f}%")
        summary.append(f"    Range: {df[col].min():.2f}% to {df[col].max():.2f}%")

    summary.append("\n4. FUNCTIONAL METRICS")
    summary.append(f"  LVEF: {df['ejection_fraction_pct'].mean():.1f} +/- {df['ejection_fraction_pct'].std():.1f}%")
    summary.append(f"  EDV: {df['EDV_mL'].mean():.1f} +/- {df['EDV_mL'].std():.1f} mL")
    summary.append(f"  SV: {df['stroke_volume_mL'].mean():.1f} +/- {df['stroke_volume_mL'].std():.1f} mL")

    summary.append("\n5. BORDER ZONE ANALYSIS")
    summary.append(f"  Stress Concentration: {df['bz_stress_concentration_factor'].mean():.3f}")
    summary.append(f"  Strain Mismatch: {df['bz_circ_strain_mismatch_pct'].mean():.2f}%")
    summary.append(f"  Mechanical Disadvantage: {df['bz_mechanical_disadvantage_index'].mean():.2f}")


    return "\n".join(summary)


def run_complete_analysis():
    """Run complete analysis with all visualizations."""
    print("Loading REAL baseline metrics...")
    df = load_results()

    print("\nGenerating visualizations...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_real_strains(df)
    plot_functional_metrics(df)
    plot_border_zone_analysis(df)

    print("\n" + generate_summary(df))

    return df


# MAIN
if __name__ == "__main__":
    df = run_complete_analysis()
