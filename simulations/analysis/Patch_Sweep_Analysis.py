#!/usr/bin/env python3
"""
Parametric Patch Sweep Analysis - Jupyter Notebook Runner

This script provides functions for analyzing patch sweep results
in Jupyter notebooks with comprehensive visualizations.

Usage in Jupyter:
    %run /home/shadeform/SCD_MODELS/Patch_Sweep_Analysis.py

Or import specific functions:
    from Patch_Sweep_Analysis import (
        load_sweep_results, plot_stiffness_optimization,
        plot_coverage_analysis, find_optimal_configurations
    )
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Configuration
BASE_DIR = Path("/home/shadeform/SCD_MODELS")
RESULTS_DIR = BASE_DIR / "patch_sweep_results"
FIGURES_DIR = RESULTS_DIR / "figures"
BASELINE_DIR = BASE_DIR / "real_baseline_metrics"

# Set matplotlib style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


# DATA LOADING
def load_sweep_results(synthetic: bool = True) -> pd.DataFrame:
    """Load patch sweep results."""
    if synthetic:
        csv_path = RESULTS_DIR / "PATCH_SWEEP_SYNTHETIC_RESULTS.csv"
    else:
        csv_path = RESULTS_DIR / "PATCH_SWEEP_RESULTS.csv"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} configurations from {csv_path.name}")
        return df
    else:
        print(f"Results file not found: {csv_path}")
        print("Run parametric_patch_sweep.py first to generate results")
        return None


def load_baseline_metrics() -> pd.DataFrame:
    """Load baseline metrics for comparison."""
    csv_path = BASELINE_DIR / "REAL_BASELINE_METRICS_ALL_PATIENTS.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


# STATISTICAL ANALYSIS
def analyze_parameter_effects(df: pd.DataFrame) -> Dict:
    """Analyze the effect of each parameter on improvement."""
    results = {}

    # Stiffness effect
    stiffness_groups = df.groupby('stiffness_kPa').agg({
        'GCS_improvement_pct': ['mean', 'std', 'min', 'max'],
        'LVEF_improvement_pct': ['mean', 'std'],
        'bz_stress_reduction_pct': 'mean'
    }).round(4)
    results['stiffness'] = stiffness_groups

    # Thickness effect
    thickness_groups = df.groupby('thickness_mm').agg({
        'GCS_improvement_pct': ['mean', 'std', 'min', 'max'],
        'LVEF_improvement_pct': ['mean', 'std'],
    }).round(4)
    results['thickness'] = thickness_groups

    # Coverage effect
    coverage_groups = df.groupby('coverage_fraction').agg({
        'GCS_improvement_pct': ['mean', 'std', 'min', 'max'],
        'LVEF_improvement_pct': ['mean', 'std'],
    }).round(4)
    results['coverage'] = coverage_groups

    return results


def find_optimal_configurations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Find top N optimal configurations."""
    optimal = df.nlargest(n, 'GCS_improvement_pct')[
        ['patient_id', 'config_name', 'stiffness_kPa', 'thickness_mm',
         'coverage_fraction', 'GCS_improvement_pct', 'LVEF_improvement_pct',
         'bz_stress_reduction_pct']
    ].copy()
    return optimal


def find_patient_optimal(df: pd.DataFrame) -> pd.DataFrame:
    """Find optimal configuration for each patient."""
    optimal = df.loc[df.groupby('patient_id')['GCS_improvement_pct'].idxmax()]
    return optimal[['patient_id', 'stiffness_kPa', 'thickness_mm',
                    'coverage_fraction', 'GCS_improvement_pct', 'LVEF_improvement_pct']]


# VISUALIZATION FUNCTIONS
def plot_stiffness_optimization(df: pd.DataFrame, save: bool = True):
    """Plot stiffness optimization analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    stiffness_vals = sorted(df['stiffness_kPa'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(stiffness_vals)))

    # 1. GCS Improvement by Stiffness
    ax = axes[0, 0]
    gcs_means = df.groupby('stiffness_kPa')['GCS_improvement_pct'].mean()
    gcs_stds = df.groupby('stiffness_kPa')['GCS_improvement_pct'].std()

    bars = ax.bar(range(len(stiffness_vals)), gcs_means.values, yerr=gcs_stds.values,
                  capsize=5, color=colors, edgecolor='black')

    # Mark optimal
    optimal_idx = np.argmax(gcs_means.values)
    bars[optimal_idx].set_edgecolor('red')
    bars[optimal_idx].set_linewidth(3)

    ax.set_xticks(range(len(stiffness_vals)))
    ax.set_xticklabels([f'{s:.0f}' for s in stiffness_vals])
    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('GCS Improvement (%)')
    ax.set_title('GCS Improvement by Patch Stiffness')
    ax.axhline(y=gcs_means.values[optimal_idx], color='red', linestyle='--', alpha=0.5)

    # 2. Stiffness Effect Score
    ax = axes[0, 1]
    effect_scores = df.groupby('stiffness_kPa')['stiffness_effect'].mean()
    ax.plot(stiffness_vals, effect_scores.values, 'o-', markersize=10, linewidth=2, color='#e74c3c')
    ax.fill_between(stiffness_vals, effect_scores.values, alpha=0.3, color='#e74c3c')
    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('Stiffness Effect Score')
    ax.set_title('Stiffness Effect (Gaussian Model)')
    ax.axvline(x=15, color='green', linestyle='--', label='Optimal (15 kPa)')
    ax.legend()

    # 3. Box plot by stiffness
    ax = axes[1, 0]
    data_by_stiffness = [df[df['stiffness_kPa'] == s]['GCS_improvement_pct'].values
                         for s in stiffness_vals]
    bp = ax.boxplot(data_by_stiffness, labels=[f'{s:.0f}' for s in stiffness_vals],
                    patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('GCS Improvement (%)')
    ax.set_title('GCS Improvement Distribution by Stiffness')

    # 4. LVEF Improvement by Stiffness
    ax = axes[1, 1]
    lvef_means = df.groupby('stiffness_kPa')['LVEF_improvement_pct'].mean()
    ax.bar(range(len(stiffness_vals)), lvef_means.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(stiffness_vals)))
    ax.set_xticklabels([f'{s:.0f}' for s in stiffness_vals])
    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('LVEF Improvement (%)')
    ax.set_title('LVEF Improvement by Patch Stiffness')

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "stiffness_optimization.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'stiffness_optimization.png'}")

    plt.show()
    return fig


def plot_thickness_optimization(df: pd.DataFrame, save: bool = True):
    """Plot thickness optimization analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    thickness_vals = sorted(df['thickness_mm'].unique())
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(thickness_vals)))

    # 1. GCS Improvement by Thickness
    ax = axes[0]
    gcs_means = df.groupby('thickness_mm')['GCS_improvement_pct'].mean()
    ax.bar(range(len(thickness_vals)), gcs_means.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(thickness_vals)))
    ax.set_xticklabels([f'{t:.1f}' for t in thickness_vals])
    ax.set_xlabel('Thickness (mm)')
    ax.set_ylabel('GCS Improvement (%)')
    ax.set_title('GCS Improvement by Patch Thickness')

    # 2. Thickness Effect (logarithmic)
    ax = axes[1]
    effect_scores = df.groupby('thickness_mm')['thickness_effect'].mean()
    ax.plot(thickness_vals, effect_scores.values, 'o-', markersize=10, linewidth=2, color='#9b59b6')
    ax.fill_between(thickness_vals, effect_scores.values, alpha=0.3, color='#9b59b6')
    ax.set_xlabel('Thickness (mm)')
    ax.set_ylabel('Thickness Effect Score')
    ax.set_title('Thickness Effect (Logarithmic Model)')

    # 3. Combined with stiffness (heatmap-like)
    ax = axes[2]
    pivot = df.pivot_table(values='GCS_improvement_pct',
                          index='thickness_mm',
                          columns='stiffness_kPa',
                          aggfunc='mean')
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn',
                   extent=[0, len(pivot.columns), 0, len(pivot.index)])
    ax.set_xticks(np.arange(len(pivot.columns)) + 0.5)
    ax.set_xticklabels([f'{s:.0f}' for s in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)) + 0.5)
    ax.set_yticklabels([f'{t:.1f}' for t in pivot.index])
    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('Thickness (mm)')
    ax.set_title('GCS Improvement: Thickness vs Stiffness')
    plt.colorbar(im, ax=ax, label='GCS Improvement (%)')

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "thickness_optimization.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'thickness_optimization.png'}")

    plt.show()
    return fig


def plot_coverage_analysis(df: pd.DataFrame, save: bool = True):
    """Plot coverage analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    coverage_vals = sorted(df['coverage_fraction'].unique())
    colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

    # 1. GCS Improvement by Coverage
    ax = axes[0]
    gcs_means = df.groupby('coverage_fraction')['GCS_improvement_pct'].mean()
    ax.bar(range(len(coverage_vals)), gcs_means.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(coverage_vals)))
    ax.set_xticklabels([f'{c*100:.0f}%' for c in coverage_vals])
    ax.set_xlabel('Coverage')
    ax.set_ylabel('GCS Improvement (%)')
    ax.set_title('GCS Improvement by BZ Coverage')

    # 2. Coverage Effect
    ax = axes[1]
    effect_scores = df.groupby('coverage_fraction')['coverage_effect'].mean()
    ax.plot([c*100 for c in coverage_vals], effect_scores.values, 'o-',
            markersize=10, linewidth=2, color='#e67e22')
    ax.fill_between([c*100 for c in coverage_vals], effect_scores.values,
                    alpha=0.3, color='#e67e22')
    ax.set_xlabel('Coverage (%)')
    ax.set_ylabel('Coverage Effect Score')
    ax.set_title('Coverage Effect (tanh Model)')

    # 3. Stress reduction by coverage
    ax = axes[2]
    stress_means = df.groupby('coverage_fraction')['bz_stress_reduction_pct'].mean()
    ax.bar(range(len(coverage_vals)), stress_means.values, color=colors, edgecolor='black')
    ax.set_xticks(range(len(coverage_vals)))
    ax.set_xticklabels([f'{c*100:.0f}%' for c in coverage_vals])
    ax.set_xlabel('Coverage')
    ax.set_ylabel('BZ Stress Reduction (%)')
    ax.set_title('Border Zone Stress Reduction by Coverage')

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "coverage_analysis.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'coverage_analysis.png'}")

    plt.show()
    return fig


def plot_patient_comparison(df: pd.DataFrame, save: bool = True):
    """Compare patch effects across patients."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    patients = df['patient_id'].unique()
    n_patients = len(patients)
    colors = plt.cm.tab10(np.linspace(0, 1, n_patients))

    # 1. Max GCS improvement per patient
    ax = axes[0, 0]
    max_gcs = df.groupby('patient_id')['GCS_improvement_pct'].max()
    ax.bar(range(n_patients), max_gcs.values, color=colors, edgecolor='black')
    ax.set_xticks(range(n_patients))
    ax.set_xticklabels([p[-3:] for p in patients], rotation=45)
    ax.set_ylabel('Max GCS Improvement (%)')
    ax.set_title('Maximum GCS Improvement by Patient')

    # 2. Baseline BZ fraction vs improvement
    ax = axes[0, 1]
    patient_data = df.groupby('patient_id').agg({
        'baseline_bz_fraction_pct': 'first',
        'GCS_improvement_pct': 'max'
    })
    ax.scatter(patient_data['baseline_bz_fraction_pct'],
               patient_data['GCS_improvement_pct'], s=100, c=colors, edgecolors='black')
    for i, patient in enumerate(patients):
        ax.annotate(patient[-3:],
                   (patient_data.loc[patient, 'baseline_bz_fraction_pct'],
                    patient_data.loc[patient, 'GCS_improvement_pct']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    # Fit line
    z = np.polyfit(patient_data['baseline_bz_fraction_pct'],
                   patient_data['GCS_improvement_pct'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(patient_data['baseline_bz_fraction_pct'].min(),
                        patient_data['baseline_bz_fraction_pct'].max(), 100)
    ax.plot(x_line, p(x_line), '--', color='red', alpha=0.7, label='Linear fit')

    ax.set_xlabel('Border Zone Fraction (%)')
    ax.set_ylabel('Max GCS Improvement (%)')
    ax.set_title('BZ Size vs Improvement Potential')
    ax.legend()

    # 3. Optimal stiffness per patient
    ax = axes[1, 0]
    optimal_per_patient = find_patient_optimal(df)
    ax.bar(range(n_patients), optimal_per_patient['stiffness_kPa'].values,
           color=colors, edgecolor='black')
    ax.set_xticks(range(n_patients))
    ax.set_xticklabels([p[-3:] for p in optimal_per_patient['patient_id']], rotation=45)
    ax.set_ylabel('Optimal Stiffness (kPa)')
    ax.set_title('Optimal Patch Stiffness by Patient')
    ax.axhline(y=10, color='red', linestyle='--', label='10 kPa (most common)')
    ax.legend()

    # 4. Improvement factor distribution
    ax = axes[1, 1]
    for i, patient in enumerate(patients):
        patient_df = df[df['patient_id'] == patient]
        ax.hist(patient_df['improvement_factor'], bins=20, alpha=0.5,
                color=colors[i], label=patient[-3:])
    ax.set_xlabel('Improvement Factor')
    ax.set_ylabel('Frequency')
    ax.set_title('Improvement Factor Distribution by Patient')
    ax.legend(ncol=2, fontsize=8)

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "patient_comparison.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'patient_comparison.png'}")

    plt.show()
    return fig


def plot_3d_parameter_space(df: pd.DataFrame, save: bool = True):
    """Create 3D visualization of parameter space."""
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(14, 6))

    # Aggregate across patients
    agg_df = df.groupby(['stiffness_kPa', 'thickness_mm', 'coverage_fraction']).agg({
        'GCS_improvement_pct': 'mean'
    }).reset_index()

    # 3D scatter plot
    ax = fig.add_subplot(121, projection='3d')

    scatter = ax.scatter(agg_df['stiffness_kPa'],
                        agg_df['thickness_mm'],
                        agg_df['coverage_fraction'],
                        c=agg_df['GCS_improvement_pct'],
                        cmap='RdYlGn', s=100, alpha=0.8)

    ax.set_xlabel('Stiffness (kPa)')
    ax.set_ylabel('Thickness (mm)')
    ax.set_zlabel('Coverage')
    ax.set_title('Parameter Space Exploration')
    plt.colorbar(scatter, ax=ax, label='GCS Improvement (%)', shrink=0.6)

    # Contour plot (stiffness vs thickness, averaged over coverage)
    ax2 = fig.add_subplot(122)

    pivot = df.pivot_table(values='GCS_improvement_pct',
                          index='thickness_mm',
                          columns='stiffness_kPa',
                          aggfunc='mean')

    contour = ax2.contourf(pivot.columns, pivot.index, pivot.values,
                          levels=20, cmap='RdYlGn')
    ax2.set_xlabel('Stiffness (kPa)')
    ax2.set_ylabel('Thickness (mm)')
    ax2.set_title('GCS Improvement Contour Map')
    plt.colorbar(contour, ax=ax2, label='GCS Improvement (%)')

    # Mark optimal region
    optimal_s = 10
    optimal_t = 5
    ax2.scatter([optimal_s], [optimal_t], marker='*', s=300, c='red',
               edgecolors='black', zorder=10, label='Optimal')
    ax2.legend()

    plt.tight_layout()

    if save:
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURES_DIR / "parameter_space_3d.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGURES_DIR / 'parameter_space_3d.png'}")

    plt.show()
    return fig


def generate_sweep_report(df: pd.DataFrame) -> str:
    """Generate comprehensive sweep report."""
    report = []

    report.append("PARAMETRIC PATCH SWEEP REPORT")

    # Overview
    report.append("\n1. SWEEP OVERVIEW")
    report.append(f"  Total configurations: {len(df)}")
    report.append(f"  Patients: {df['patient_id'].nunique()}")
    report.append(f"  Configurations per patient: {len(df) // df['patient_id'].nunique()}")

    # Parameter ranges
    report.append("\n2. PARAMETER RANGES")
    report.append(f"  Stiffness: {df['stiffness_kPa'].min():.1f} - {df['stiffness_kPa'].max():.1f} kPa")
    report.append(f"  Thickness: {df['thickness_mm'].min():.1f} - {df['thickness_mm'].max():.1f} mm")
    report.append(f"  Coverage: {df['coverage_fraction'].min()*100:.0f}% - {df['coverage_fraction'].max()*100:.0f}%")

    # GCS Results
    report.append("\n3. GCS IMPROVEMENT RESULTS")
    report.append(f"  Mean: {df['GCS_improvement_pct'].mean():.3f}%")
    report.append(f"  Std: {df['GCS_improvement_pct'].std():.3f}%")
    report.append(f"  Range: {df['GCS_improvement_pct'].min():.3f}% to {df['GCS_improvement_pct'].max():.3f}%")

    # LVEF Results
    report.append("\n4. LVEF IMPROVEMENT RESULTS")
    report.append(f"  Mean: {df['LVEF_improvement_pct'].mean():.2f}%")
    report.append(f"  Range: {df['LVEF_improvement_pct'].min():.2f}% to {df['LVEF_improvement_pct'].max():.2f}%")

    # Optimal stiffness
    report.append("\n5. OPTIMAL STIFFNESS ANALYSIS")
    stiffness_means = df.groupby('stiffness_kPa')['GCS_improvement_pct'].mean()
    optimal_stiffness = stiffness_means.idxmax()
    report.append(f"  Optimal stiffness: {optimal_stiffness} kPa")
    report.append(f"  At optimal: {stiffness_means[optimal_stiffness]:.3f}% GCS improvement")
    report.append("\n  All stiffness values:")
    for s, gcs in stiffness_means.items():
        report.append(f"    {s:6.1f} kPa: {gcs:.4f}%")

    # Top configurations
    report.append("\n6. TOP 5 CONFIGURATIONS")
    top5 = find_optimal_configurations(df, 5)
    for i, row in top5.iterrows():
        report.append(f"  {row['patient_id']}: S={row['stiffness_kPa']}kPa, "
                     f"T={row['thickness_mm']}mm, C={row['coverage_fraction']*100:.0f}% "
                     f"-> GCS +{row['GCS_improvement_pct']:.2f}%")


    return "\n".join(report)


# MAIN EXECUTION
def run_complete_analysis():
    """Run complete patch sweep analysis."""
    print("Loading patch sweep results...")
    df = load_sweep_results()

    if df is None:
        return None

    print("\nGenerating visualizations...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_stiffness_optimization(df)
    plot_thickness_optimization(df)
    plot_coverage_analysis(df)
    plot_patient_comparison(df)
    plot_3d_parameter_space(df)

    print("\n" + generate_sweep_report(df))

    # Save report
    report = generate_sweep_report(df)
    report_path = RESULTS_DIR / "SWEEP_REPORT.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {report_path}")

    return df


if __name__ == "__main__":
    df = run_complete_analysis()
