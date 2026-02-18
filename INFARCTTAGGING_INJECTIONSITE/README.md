# Infarct Tagging & Laplace-Dirichlet Analysis Pipeline

Algorithms for infarct tissue classification, Laplace-Dirichlet transmural analysis, fiber reconstruction, wall stress computation, and injection site optimization across 10 SCD patients.

## Pipeline Flow

```
simulation_ready/ (meshes)
    |
    v
1_infarct_coronary_territory.py --> infarct_coronary_territory/
    |
    v
2_laplace_dirichlet_complete.py --> laplace_complete/   (first Laplace attempt)
    |
    v
3_infarct_detection_comprehensive.py --> infarct_results_comprehensive/
    |                                     (tagged .elem files used by FEBio/OpenCarp)
    v
4_laplace_dirichlet_complete_v2.py --> laplace_complete/  (re-run, nearly identical to #2)
    |
    v
5_laplace_fixed.py --> laplace_fixed/   (bug fix for wall thickness artifacts)
    |
    v
6_laplace_complete_v2_FINAL.py --> laplace_complete_v2/   <-- FINAL OUTPUT
```

## File Descriptions

### `1_infarct_coronary_territory.py` (34 KB, 1,001 lines)
**Anatomically-Correct Infarct Assignment**
- First approach: assigns infarct regions using the AHA 17-segment model and coronary artery territory mapping
- Coronary territories: LAD (anterior wall, septum, apex, ~40-50% LV), RCA (inferior wall, septum), LCx (lateral wall, posterior)
- Geometric classification based on angular position relative to septum and longitudinal position
- Reads from: `simulation_ready/`
- Writes to: `infarct_coronary_territory/`
- Status: Initial approach, superseded by multi-metric detection in file #3

### `2_laplace_dirichlet_complete.py` (44 KB, 1,292 lines)
**Complete Laplace-Dirichlet Cardiac Analysis Framework (v1)**
- First implementation of the full Laplace-Dirichlet pipeline with 4 components:
  - Component 1: Transmural coordinate via Laplace equation (nabla^2 phi = 0, phi in [0,1])
  - Component 2: Helical fiber reconstruction (alpha(phi) = alpha_endo + (alpha_epi - alpha_endo) * phi)
  - Component 3: Wall stress from modified Laplace law with curvature tensor (kappa_1, kappa_2)
  - Component 4: Geodesic injection site optimization via Heat Method (Crane et al. 2013)
- Produces .lon fiber files, analysis VTK, and summary JSON per patient
- Reads from: `simulation_ready/`, `infarct_results_comprehensive/`
- Writes to: `laplace_complete/`
- Status: First working version, later found wall thickness artifacts

### `3_infarct_detection_comprehensive.py` (51 KB, 1,506 lines)
**Multi-Metric Infarct Detection Framework**
- Integrates FIVE complementary methodologies for tissue classification without LGE-MRI:
  1. Laplace-Dirichlet transmural coordinate (Bishop et al. 2010)
  2. Robust wall thickness from gradient (infarcted ~2.86mm vs healthy ~8.73mm)
  3. Coronary territory probability mapping
  4. Transmural depth analysis
  5. Spectral clustering for spatial coherence
- Outputs tagged .elem files (Tag 1=Healthy, Tag 2=Border Zone, Tag 3=Infarct)
- These tagged elements are consumed downstream by FEBio and OpenCarp simulations
- Reads from: `simulation_ready/`
- Writes to: `infarct_results_comprehensive/`
- Status: This is NOT a trial of the Laplace pipeline -- it is a separate, complementary algorithm that produces the tissue-tagged meshes used by the simulation notebooks

### `4_laplace_dirichlet_complete_v2.py` (44 KB, 1,291 lines)
**Laplace-Dirichlet Framework (re-run)**
- Nearly identical to file #2 (1 character difference)
- Likely a re-run or minor parameter tweak of the original Laplace framework
- No cell output in the notebook, meaning it was not executed or did not complete
- Reads from: `simulation_ready/`, `infarct_results_comprehensive/`
- Writes to: `laplace_complete/`
- Status: Abandoned re-run

### `5_laplace_fixed.py` (33 KB, 1,028 lines)
**Laplace Framework with Bug Fixes**
- Key fix: Wall thickness computed directly (endo-to-epi distance along transmural direction) instead of from Laplace gradient
- This eliminates the vertical stripe artifacts found in files #2 and #4
- Streamlined version of the framework with the critical correction applied
- No cell output in the notebook, meaning it was an intermediate iteration
- Reads from: `simulation_ready/`, `infarct_results_comprehensive/`
- Writes to: `laplace_fixed/`
- Status: Intermediate fix, refined further into the final version

### `6_laplace_complete_v2_FINAL.py` (30 KB, 884 lines)
**THE FINAL ALGORITHM -- Laplace-Dirichlet Cardiac Analysis Framework**
- Clean, refined version incorporating all fixes and optimizations from the prior iterations
- Complete pipeline:
  1. Transmural coordinate (Laplace solve)
  2. Myofiber reconstruction (helical angle rotation per Streeter et al. 1969)
  3. Wall stress with curvature correction (modified Law of Laplace with shape operator)
  4. Geodesic distance computation (Heat Method)
  5. Injection site optimization (stress-weighted geodesic scoring)
- Successfully ran on all 10 patients with confirmed output
- Produces per-patient: `.lon` fiber file, `_analysis.vtk`, `_summary.json`
- Reads from: `simulation_ready/`, `infarct_results_comprehensive/`
- Writes to: `laplace_complete_v2/`
- Status: **FINAL** -- this is the version whose output is consumed by all downstream algorithms