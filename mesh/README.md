# Mesh Generation Pipeline

Algorithms extracted from `3_SCD_MESH_GENERATION.ipynb` for generating simulation-ready tetrahedral cardiac meshes for 10 SCD patients.

## Pipeline Flow

```
processed_triangular_meshes/ (STLs)
    |
    v
mesh_trial_1.py --> high_resolution_meshes/
    |
    v
mesh_trial_2.py --> improved_meshes/ (abandoned)
mesh_trial_3.py --> repaired_meshes/
mesh_trial_4.py --> repaired_meshes/ (updated version of trial 3)
    |
    v
final_mesh_gen.py --> simulation_ready/   <-- FINAL OUTPUT
```

## File Descriptions

### Trial Files (Iterative Development)
#### `mesh_trial_1.py` (54 KB, 1,614 lines)
**Comprehensive High-Resolution Mesh Generation & Tissue Tagging Pipeline**
- First stage of the pipeline: generates high-resolution tetrahedral meshes from triangular surface meshes (STLs)
- Uses fTetWild as the primary meshing engine
- Produces 100K+ element meshes with 3-region tissue tagging (healthy, border zone, infarct/scar)
- Generates wall definitions (endocardium, epicardium, base)
- Computes quality metrics (scaled Jacobian, dihedral angles, aspect ratio, radius-edge ratio)
- Reads from: `tetrahedral_meshes/`, `processed_triangular_meshes/`
- Writes to: `high_resolution_meshes/`

#### `mesh_trial_2.py` (31 KB, 874 lines)
**Tetrahedral Mesh Quality Improvement**
- Attempts to improve mesh quality WITHOUT regenerating elements
- Methods: MMG3D optimization, PyMesh optimization, Gmsh optimization, custom Laplacian smoothing
- Fixes high aspect ratios, poor Jacobians, bad dihedral angles
- Reads from: `high_resolution_meshes/`
- Writes to: `improved_meshes/`
- Status: Abandoned in favor of the full repair approach (trial 3)

#### `mesh_trial_3.py` (38 KB, 1,030 lines)
**Optimal Tetrahedral Mesh Quality Repair**
- Multi-stage repair pipeline:
  - Stage 1: Surface extraction & repair (PyMeshLab/PyMesh)
  - Stage 2: MMG3D optimization (WITH vertex insertion allowed)
  - Stage 3: If still bad, fTetWild complete remesh
  - Stage 4: Final cleanup and validation
- Key fix: Previous approach used `-noinsert` which blocked MMG3D from adding vertices
- Reads from: `high_resolution_meshes/`
- Writes to: `repaired_meshes/`

#### `mesh_trial_4.py` (40 KB, 1,058 lines)
**Updated Optimal Tetrahedral Mesh Quality Repair**
- Updated version of trial 3 with refinements to the multi-stage repair pipeline
- Same 4-stage approach with improved parameters and handling
- Reads from: `high_resolution_meshes/`
- Writes to: `repaired_meshes/`

### Diagnostic Files (Quality Analysis)

#### `mesh_diagnostic_1.py` (20 KB, 499 lines)
**Mesh Quality Analysis**
- Analyzes ALL quality metrics to identify what prevents FEBio/OpenCarp readiness
- Metrics: aspect ratio, scaled Jacobian, dihedral angles, radius-edge ratio, edge length ratio, volume distribution, inverted elements
- Reads from: `repaired_meshes/` (or `high_resolution_meshes/`)
- Writes to: `quality_analysis/` (reports only, no mesh output)

#### `mesh_diagnostic_2.py` (15 KB, 419 lines)
**Mesh Quality Validation & Export**
- Validates mesh quality against adjustable thresholds
- Exports meshes in FEBio and OpenCarp formats
- Key finding: radius-edge ratio threshold of 10 was too strict; revised to 100 per literature
- Confirmed: aspect ratio excellent (all < 50), Jacobian excellent (all > 0.06), dihedral angles good
- Reads from: `repaired_meshes/`
- Writes to: `final_validation/` (reports only, no mesh output)

#### `mesh_diagnostic_3.py` (4.5 KB, 110 lines)
**Quality Report Generator**
- Quick utility to generate quality reports for already-completed meshes
- Helper function called from the main generation pipeline
- Depends on `read_mesh_carp()` and `evaluate_mesh_quality()` from the parent pipeline context

### Final Algorithm

#### `final_mesh_gen.py` (35 KB, 954 lines)
**Evidence-Based Cardiac Mesh Optimization for FEBio & OpenCarp**
- THE FINAL ALGORITHM that produces the `simulation_ready/` meshes
- Based on FEBio developer guidance (Steve Maas) and published cardiac simulation literature
- Applies 7-stage optimization to repaired meshes:
  - Laplacian smoothing, Jacobian-targeted optimization, aspect ratio reduction, dihedral angle correction, volume equalization, boundary preservation, final validation
- All 10 patients pass both FEBio AND OpenCarp readiness thresholds
- Results: zero inversions, min Jacobian 0.060-0.084, mean Jacobian 0.69-0.76, max aspect ratio 26.3-43.1
- Reads from: `repaired_meshes/`
- Writes to: `simulation_ready/`