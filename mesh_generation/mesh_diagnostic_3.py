import numpy as np
import csv
from pathlib import Path
from datetime import datetime


def generate_quality_report_for_existing_meshes(patient_ids, output_dir=OUTPUT_DIR):
    """Generate quality reports for already-completed meshes."""
    
    print("QUALITY REPORTS FOR EXISTING MESHES")
    
    reports = []
    
    for i, patient_id in enumerate(patient_ids):
        print(f"\n[{i+1}/{len(patient_ids)}] Analyzing {patient_id}...")
        
        try:
            # Read the existing mesh
            vertices, elements, tags = read_mesh_carp(patient_id, output_dir)
            
            if vertices is None:
                print(f"  ✗ Could not load mesh for {patient_id}")
                continue
            
            print(f"  Loaded: {len(vertices)} vertices, {len(elements)} elements")
            
            # Evaluate quality
            report = evaluate_mesh_quality(vertices, elements, tags, patient_id)
            
            # Try to load additional metadata if available
            patient_dir = Path(output_dir) / patient_id
            
            # Check for transmural depth file
            transmural_file = patient_dir / f"{patient_id}_transmural.dat"
            if transmural_file.exists():
                print(f"  ✓ Found transmural depth data")
            
            # Check for surface node lists
            endo_file = patient_dir / f"{patient_id}_endo.vtx"
            epi_file = patient_dir / f"{patient_id}_epi.vtx"
            base_file = patient_dir / f"{patient_id}_base.vtx"
            
            if endo_file.exists():
                with open(endo_file, 'r') as f:
                    lines = f.readlines()
                    report.n_endo_nodes = int(lines[0].strip())
            
            if epi_file.exists():
                with open(epi_file, 'r') as f:
                    lines = f.readlines()
                    report.n_epi_nodes = int(lines[0].strip())
            
            if base_file.exists():
                with open(base_file, 'r') as f:
                    lines = f.readlines()
                    report.n_base_nodes = int(lines[0].strip())
            
            # Set method (likely TetGen based on previous outputs)
            report.method = "TetGen"
            
            # Print summary
            print(f"  Quality Metrics:")
            print(f"    Elements: {report.n_elements}")
            print(f"    Min Jacobian: {report.min_jacobian:.6f}")
            print(f"    Dihedral: [{report.min_dihedral:.2f}, {report.max_dihedral:.2f}]°")
            print(f"    Max AR: {report.max_aspect_ratio:.2f}")
            print(f"    OpenCarp Ready: {'YES' if report.opencarp_ready else 'NO'}")
            print(f"    FEBio Ready: {'YES' if report.febio_ready else 'NO'}")
            
            reports.append(report)
            
        except Exception as e:
            print(f"  Error analyzing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save combined report
    if reports:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_path = f"{output_dir}/quality_report_first5_{timestamp}.csv"
        
        with open(csv_path, 'w', newline='') as f:
            fieldnames = list(reports[0].to_dict().keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in reports:
                writer.writerow(r.to_dict())
        
        print(f"\n Quality report saved to: {csv_path}")
        
        # Print summary table
        print("QUALITY SUMMARY - FIRST 5 PATIENTS")
        print(f"{'Patient':<14} {'Elements':<12} {'Inv':<6} {'MinJac':<10} {'MinDih':<10} "
              f"{'MaxDih':<10} {'MaxAR':<10} {'OC':<6} {'FB':<6}")
        
        for r in reports:
            oc = "YES" if r.opencarp_ready else "NO"
            fb = "YES" if r.febio_ready else "NO"
            
            print(f"{r.patient_id:<14} {r.n_elements:<12} {r.n_inverted:<6} "
                  f"{r.min_jacobian:<10.6f} {r.min_dihedral:<10.2f} "
                  f"{r.max_dihedral:<10.2f} {r.max_aspect_ratio:<10.2f} "
                  f"{oc:<6} {fb:<6}")
        
        n_oc = sum(1 for r in reports if r.opencarp_ready)
        n_fb = sum(1 for r in reports if r.febio_ready)
        print(f"OpenCarp Ready: {n_oc}/{len(reports)}")
        print(f"FEBio Ready: {n_fb}/{len(reports)}")
    
    return reports