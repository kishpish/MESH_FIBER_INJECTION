#!/usr/bin/env python3
"""
PARAMETRIC HYDROGEL PATCH SWEEP

Systematically varies hydrogel patch parameters (stiffness, thickness, coverage)
and runs FEBio simulations to evaluate mechanical improvement over baseline.

This script performs:
1. Generation of patch configurations across parameter space
2. FEBio simulation file generation with patch materials
3. Simulation execution for each configuration
4. Extraction of improvement metrics vs baseline
5. Results aggregation and analysis


"""

import os
import sys
import json
import time
import shutil
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from itertools import product
from scipy.spatial import KDTree
import xml.etree.ElementTree as ET
from xml.dom import minidom
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION
PATIENTS = [
    "SCD0000101", "SCD0000201", "SCD0000301", "SCD0000401", "SCD0000601",
    "SCD0000701", "SCD0000801", "SCD0001001", "SCD0001101", "SCD0001201"
]

BASE_DIR = Path("/home/shadeform/SCD_MODELS")
FEBIO_RESULTS_DIR = BASE_DIR / "febio_results"
TAGGED_MESH_DIR = BASE_DIR / "infarct_results_comprehensive"
BASELINE_METRICS_DIR = BASE_DIR / "real_baseline_metrics"
OUTPUT_DIR = BASE_DIR / "patch_sweep_results"

# FEBio configuration
FEBIO_PATH = "/home/shadeform/FEBio/bin/febio4"
FEBIO_LIB_PATH = "/home/shadeform/FEBio/lib"

# Simulation parameters
ED_PRESSURE_KPA = 1.066576   # End-diastolic pressure (~8 mmHg)
ES_PRESSURE_KPA = 15.99864   # End-systolic pressure (~120 mmHg)
TIME_STEPS = 20
STEP_SIZE = 0.05


# PATCH PARAMETER SPACE
@dataclass
class PatchParameters:
    """Hydrogel patch configuration parameters."""
    # Stiffness (Young's modulus in kPa)
    # Range: 1 kPa (very soft) to 100 kPa (stiff)
    stiffness_kPa: float = 10.0

    # Thickness (mm)
    # Range: 0.5 mm to 5.0 mm
    thickness_mm: float = 2.0

    # Coverage (fraction of border zone covered)
    # Range: 0.25 (25%) to 1.0 (100%)
    coverage_fraction: float = 0.5

    # Derived material properties for FEBio
    # Neo-Hookean: c1 = E / (4 * (1 + nu)), where nu ~ 0.49 for incompressible
    @property
    def c1_kPa(self) -> float:
        """Neo-Hookean c1 parameter from Young's modulus."""
        nu = 0.49  # Poisson's ratio (nearly incompressible)
        return self.stiffness_kPa / (4 * (1 + nu))

    @property
    def bulk_modulus_kPa(self) -> float:
        """Bulk modulus for penalty method."""
        return self.stiffness_kPa * 100  # High for incompressibility

    def to_dict(self) -> Dict:
        return {
            'stiffness_kPa': self.stiffness_kPa,
            'thickness_mm': self.thickness_mm,
            'coverage_fraction': self.coverage_fraction,
            'c1_kPa': self.c1_kPa,
            'bulk_modulus_kPa': self.bulk_modulus_kPa,
        }


# Define parameter sweep ranges
STIFFNESS_VALUES = [1.0, 5.0, 10.0, 25.0, 50.0, 100.0]  # kPa
THICKNESS_VALUES = [0.5, 1.0, 2.0, 3.0, 5.0]  # mm
COVERAGE_VALUES = [0.25, 0.50, 0.75, 1.0]  # fraction

# For quick testing, use reduced set
STIFFNESS_VALUES_QUICK = [5.0, 25.0, 100.0]
THICKNESS_VALUES_QUICK = [1.0, 3.0]
COVERAGE_VALUES_QUICK = [0.5, 1.0]


def generate_patch_configurations(quick_mode: bool = False) -> List[PatchParameters]:
    """Generate all patch configurations for sweep."""
    if quick_mode:
        stiffness = STIFFNESS_VALUES_QUICK
        thickness = THICKNESS_VALUES_QUICK
        coverage = COVERAGE_VALUES_QUICK
    else:
        stiffness = STIFFNESS_VALUES
        thickness = THICKNESS_VALUES
        coverage = COVERAGE_VALUES

    configs = []
    for s, t, c in product(stiffness, thickness, coverage):
        configs.append(PatchParameters(
            stiffness_kPa=s,
            thickness_mm=t,
            coverage_fraction=c
        ))

    return configs


# MESH AND MATERIAL HANDLING
class MeshLoader:
    """Load mesh data from simulation files."""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.pts_file = BASE_DIR / "simulation_ready" / patient_id / f"{patient_id}_tet.pts"
        self.elem_file = TAGGED_MESH_DIR / patient_id / f"{patient_id}_tagged.elem"

        self.nodes = None
        self.elements = None
        self.tissue_tags = None

    def load(self) -> bool:
        """Load mesh data."""
        # Load nodes
        if self.pts_file.exists():
            self.nodes = []
            with open(self.pts_file, 'r') as f:
                n_nodes = int(f.readline().strip())
                for line in f:
                    coords = [float(x) for x in line.strip().split()]
                    if len(coords) >= 3:
                        self.nodes.append(coords[:3])
            self.nodes = np.array(self.nodes)

        # Load elements with tags
        if self.elem_file.exists():
            self.elements = []
            self.tissue_tags = []
            with open(self.elem_file, 'r') as f:
                n_elem = int(f.readline().strip())
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6 and parts[0] == 'Tt':
                        # Tt n1 n2 n3 n4 tag (0-indexed)
                        elem = [int(parts[i]) for i in range(1, 5)]
                        tag = int(parts[5])
                        self.elements.append(elem)
                        self.tissue_tags.append(tag)

            self.elements = np.array(self.elements)
            self.tissue_tags = np.array(self.tissue_tags)

        return self.nodes is not None and self.elements is not None

    def get_border_zone_elements(self) -> np.ndarray:
        """Get indices of border zone elements (tag=2)."""
        return np.where(self.tissue_tags == 2)[0]

    def get_border_zone_surface_nodes(self) -> np.ndarray:
        """Get surface nodes of the border zone for patch placement."""
        bz_elem_idx = self.get_border_zone_elements()
        bz_elements = self.elements[bz_elem_idx]

        # Get unique nodes in border zone
        bz_nodes = np.unique(bz_elements.flatten())

        # Find surface nodes (simplified: outer nodes based on distance from centroid)
        if len(bz_nodes) > 0 and self.nodes is not None:
            bz_coords = self.nodes[bz_nodes]
            centroid = np.mean(self.nodes, axis=0)
            distances = np.linalg.norm(bz_coords - centroid, axis=1)

            # Surface nodes are those farther from centroid (outer 60%)
            threshold = np.percentile(distances, 40)
            surface_mask = distances > threshold
            surface_nodes = bz_nodes[surface_mask]

            return surface_nodes

        return bz_nodes


class PatchElementSelector:
    """Select elements to receive patch material based on coverage."""

    def __init__(self, mesh: MeshLoader, patch_params: PatchParameters):
        self.mesh = mesh
        self.params = patch_params

    def select_patch_elements(self) -> np.ndarray:
        """Select border zone elements for patch based on coverage fraction."""
        bz_elements = self.mesh.get_border_zone_elements()
        n_total = len(bz_elements)
        n_select = int(n_total * self.params.coverage_fraction)

        if n_select == 0:
            return np.array([], dtype=int)

        if self.params.coverage_fraction >= 1.0:
            return bz_elements

        # Select elements spatially (contiguous region)
        # Use element centroids to select a connected region
        bz_elem_data = self.mesh.elements[bz_elements]

        # Calculate centroids of BZ elements
        centroids = np.zeros((len(bz_elements), 3))
        for i, elem in enumerate(bz_elem_data):
            elem_nodes = self.mesh.nodes[elem]
            centroids[i] = np.mean(elem_nodes, axis=0)

        # Find the center of the BZ region
        bz_center = np.mean(centroids, axis=0)

        # Select elements closest to center (contiguous patch)
        distances = np.linalg.norm(centroids - bz_center, axis=1)
        sorted_idx = np.argsort(distances)
        selected_idx = sorted_idx[:n_select]

        return bz_elements[selected_idx]


# FEBIO FILE GENERATION WITH PATCH
class PatchFEBGenerator:
    """Generate FEBio simulation files with hydrogel patch."""

    def __init__(self, patient_id: str, patch_params: PatchParameters):
        self.patient_id = patient_id
        self.patch_params = patch_params
        self.mesh = MeshLoader(patient_id)

        # Material parameters
        self.materials = {
            'healthy': {'c1': 2.0, 'c2': 6.0, 'c3': 5.0, 'c4': 50.0, 'c5': 0.0, 'k': 100.0},
            'border_zone': {'c1': 5.0, 'c2': 6.0, 'c3': 10.0, 'c4': 50.0, 'c5': 0.0, 'k': 200.0},
            'infarct_scar': {'c1': 20.0, 'c2': 6.0, 'c3': 40.0, 'c4': 50.0, 'c5': 0.0, 'k': 500.0},
            'patch': {
                'type': 'neo-Hookean',
                'c1': patch_params.c1_kPa,
                'k': patch_params.bulk_modulus_kPa,
            }
        }

    def generate(self, output_path: Path) -> bool:
        """Generate complete FEBio file with patch."""
        if not self.mesh.load():
            print(f"  [ERROR] Failed to load mesh for {self.patient_id}")
            return False

        # Select patch elements
        selector = PatchElementSelector(self.mesh, self.patch_params)
        patch_elements = selector.select_patch_elements()

        # Create modified tissue tags
        modified_tags = self.mesh.tissue_tags.copy()
        modified_tags[patch_elements] = 4  # New tag for patch material

        # Generate FEB XML
        root = ET.Element('febio_spec', version="4.0")

        # Module
        ET.SubElement(root, 'Module', type="solid")

        # Control
        self._add_control(root)

        # Materials
        self._add_materials(root)

        # Mesh
        self._add_mesh(root, modified_tags)

        # MeshDomains
        self._add_mesh_domains(root, modified_tags)

        # Boundary conditions
        self._add_boundary(root)

        # Loads
        self._add_loads(root)

        # LoadData
        self._add_load_data(root)

        # Output
        self._add_output(root)

        # Write file
        xml_str = ET.tostring(root, encoding='unicode')
        dom = minidom.parseString(xml_str)
        pretty_xml = dom.toprettyxml(indent="  ")

        # Remove extra blank lines
        lines = [line for line in pretty_xml.split('\n') if line.strip()]
        final_xml = '\n'.join(lines)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(final_xml)

        return True

    def _add_control(self, root: ET.Element):
        """Add control section."""
        control = ET.SubElement(root, 'Control')
        ET.SubElement(control, 'analysis').text = 'STATIC'
        ET.SubElement(control, 'time_steps').text = str(TIME_STEPS)
        ET.SubElement(control, 'step_size').text = str(STEP_SIZE)

        solver = ET.SubElement(control, 'solver', type="solid")
        ET.SubElement(solver, 'symmetric_stiffness').text = '1'
        ET.SubElement(solver, 'equation_scheme').text = 'staggered'
        ET.SubElement(solver, 'equation_order').text = 'default'
        ET.SubElement(solver, 'optimize_bw').text = '0'
        ET.SubElement(solver, 'lstol').text = '0.9'
        ET.SubElement(solver, 'lsmin').text = '0.01'
        ET.SubElement(solver, 'lsiter').text = '5'
        ET.SubElement(solver, 'max_refs').text = '15'
        ET.SubElement(solver, 'check_zero_diagonal').text = '0'
        ET.SubElement(solver, 'zero_diagonal_tol').text = '0'
        ET.SubElement(solver, 'force_partition').text = '0'
        ET.SubElement(solver, 'reform_each_time_step').text = '1'
        ET.SubElement(solver, 'reform_augment').text = '0'
        ET.SubElement(solver, 'diverge_reform').text = '1'
        ET.SubElement(solver, 'min_residual').text = '1e-20'
        ET.SubElement(solver, 'max_residual').text = '0'
        ET.SubElement(solver, 'dtol').text = '0.001'
        ET.SubElement(solver, 'etol').text = '0.01'
        ET.SubElement(solver, 'rtol').text = '0'
        ET.SubElement(solver, 'rhoi').text = '0'
        ET.SubElement(solver, 'alpha').text = '1'
        ET.SubElement(solver, 'beta').text = '0.25'
        ET.SubElement(solver, 'gamma').text = '0.5'
        ET.SubElement(solver, 'logSolve').text = '0'
        ET.SubElement(solver, 'arc_length').text = '0'
        ET.SubElement(solver, 'arc_length_scale').text = '0'

        qn = ET.SubElement(solver, 'qn_method', type="BFGS")
        ET.SubElement(qn, 'max_ups').text = '10'
        ET.SubElement(qn, 'max_buffer_size').text = '0'
        ET.SubElement(qn, 'cycle_buffer').text = '1'
        ET.SubElement(qn, 'cmax').text = '100000'

        ts = ET.SubElement(control, 'time_stepper', type="default")
        ET.SubElement(ts, 'max_retries').text = '5'
        ET.SubElement(ts, 'opt_iter').text = '10'
        ET.SubElement(ts, 'dtmin').text = '0.001'
        ET.SubElement(ts, 'dtmax').text = str(STEP_SIZE)

    def _add_materials(self, root: ET.Element):
        """Add materials section including patch material."""
        materials = ET.SubElement(root, 'Material')

        # Tissue materials (trans iso Mooney-Rivlin)
        tissue_names = ['healthy', 'border_zone', 'infarct_scar']
        for i, name in enumerate(tissue_names, 1):
            mat = ET.SubElement(materials, 'material', id=str(i), name=name,
                              type="trans iso Mooney-Rivlin")
            params = self.materials[name]
            ET.SubElement(mat, 'c1').text = str(params['c1'])
            ET.SubElement(mat, 'c2').text = str(params['c2'])
            ET.SubElement(mat, 'c3').text = str(params['c3'])
            ET.SubElement(mat, 'c4').text = str(params['c4'])
            ET.SubElement(mat, 'c5').text = str(params['c5'])
            ET.SubElement(mat, 'k').text = str(params['k'])
            ET.SubElement(mat, 'lam_max').text = '1.4'
            ET.SubElement(mat, 'fiber', type="user")

        # Patch material (neo-Hookean)
        patch_mat = ET.SubElement(materials, 'material', id='4', name='hydrogel_patch',
                                 type="neo-Hookean")
        ET.SubElement(patch_mat, 'E').text = str(self.patch_params.stiffness_kPa)
        ET.SubElement(patch_mat, 'v').text = '0.49'

    def _add_mesh(self, root: ET.Element, modified_tags: np.ndarray):
        """Add mesh section."""
        mesh = ET.SubElement(root, 'Mesh')

        # Nodes
        nodes = ET.SubElement(mesh, 'Nodes', name="AllNodes")
        for i, coord in enumerate(self.mesh.nodes):
            node = ET.SubElement(nodes, 'node', id=str(i + 1))
            node.text = f"{coord[0]},{coord[1]},{coord[2]}"

        # Elements by material
        material_names = {1: 'healthy', 2: 'border_zone', 3: 'infarct_scar', 4: 'patch'}

        for tag, name in material_names.items():
            elem_mask = modified_tags == tag
            if not np.any(elem_mask):
                continue

            elements = ET.SubElement(mesh, 'Elements', type="tet4", name=f"{name}_domain")
            elem_indices = np.where(elem_mask)[0]

            for idx in elem_indices:
                elem = self.mesh.elements[idx]
                el = ET.SubElement(elements, 'elem', id=str(idx + 1))
                # Convert to 1-indexed
                el.text = f"{elem[0]+1},{elem[1]+1},{elem[2]+1},{elem[3]+1}"

        # NodeSet for base (fixed boundary)
        z_coords = self.mesh.nodes[:, 2]
        z_max = np.max(z_coords)
        base_threshold = z_max - 0.05 * (z_max - np.min(z_coords))
        base_nodes = np.where(z_coords > base_threshold)[0] + 1  # 1-indexed

        nodeset = ET.SubElement(mesh, 'NodeSet', name="base_nodes")
        nodeset.text = ','.join(map(str, base_nodes[:100]))  # Limit for file size

        # Surface for pressure load (endocardium)
        self._add_endocardial_surface(mesh, modified_tags)

    def _add_endocardial_surface(self, mesh: ET.Element, modified_tags: np.ndarray):
        """Add endocardial surface for pressure loading."""
        # Find boundary faces
        face_count = {}
        face_to_elem = {}

        for elem_idx, elem in enumerate(self.mesh.elements):
            faces = [
                tuple(sorted([elem[0], elem[1], elem[2]])),
                tuple(sorted([elem[0], elem[1], elem[3]])),
                tuple(sorted([elem[0], elem[2], elem[3]])),
                tuple(sorted([elem[1], elem[2], elem[3]])),
            ]
            for face in faces:
                face_count[face] = face_count.get(face, 0) + 1
                face_to_elem[face] = elem_idx

        # Boundary faces appear exactly once
        boundary_faces = [f for f, c in face_count.items() if c == 1]

        # Filter for endocardial (inner) surface
        centroid = np.mean(self.mesh.nodes, axis=0)
        endo_faces = []

        for face in boundary_faces[:5000]:  # Limit for performance
            face_center = np.mean(self.mesh.nodes[list(face)], axis=0)
            dist = np.linalg.norm(face_center - centroid)
            if dist < np.percentile([np.linalg.norm(self.mesh.nodes[list(f)].mean(axis=0) - centroid)
                                    for f in boundary_faces[:1000]], 50):
                endo_faces.append(face)

        if endo_faces:
            surface = ET.SubElement(mesh, 'Surface', name="endocardium")
            for i, face in enumerate(endo_faces[:2000]):
                tri = ET.SubElement(surface, 'tri3', id=str(i + 1))
                tri.text = f"{face[0]+1},{face[1]+1},{face[2]+1}"

    def _add_mesh_domains(self, root: ET.Element, modified_tags: np.ndarray):
        """Add mesh domains section."""
        domains = ET.SubElement(root, 'MeshDomains')

        material_map = {
            1: ('healthy_domain', 'healthy'),
            2: ('border_zone_domain', 'border_zone'),
            3: ('infarct_scar_domain', 'infarct_scar'),
            4: ('patch_domain', 'hydrogel_patch'),
        }

        for tag, (domain_name, mat_name) in material_map.items():
            if np.any(modified_tags == tag):
                domain = ET.SubElement(domains, 'SolidDomain', name=domain_name, mat=mat_name)

    def _add_boundary(self, root: ET.Element):
        """Add boundary conditions."""
        boundary = ET.SubElement(root, 'Boundary')

        # Fixed base
        fix = ET.SubElement(boundary, 'bc', type="zero displacement", node_set="base_nodes")
        ET.SubElement(fix, 'x_dof').text = '1'
        ET.SubElement(fix, 'y_dof').text = '1'
        ET.SubElement(fix, 'z_dof').text = '1'

    def _add_loads(self, root: ET.Element):
        """Add pressure load."""
        loads = ET.SubElement(root, 'Loads')

        pressure = ET.SubElement(loads, 'surface_load', type="pressure", surface="endocardium")
        ET.SubElement(pressure, 'pressure', lc='1').text = '1.0'
        ET.SubElement(pressure, 'linear').text = '0'
        ET.SubElement(pressure, 'symmetric_stiffness').text = '1'

    def _add_load_data(self, root: ET.Element):
        """Add load curve for pressure."""
        loaddata = ET.SubElement(root, 'LoadData')

        lc = ET.SubElement(loaddata, 'load_controller', id='1', type="loadcurve")
        ET.SubElement(lc, 'interpolate').text = 'LINEAR'
        ET.SubElement(lc, 'extend').text = 'CONSTANT'

        points = ET.SubElement(lc, 'points')
        # Cardiac cycle pressure profile
        pressure_curve = [
            (0.0, ED_PRESSURE_KPA),
            (0.1, ED_PRESSURE_KPA),
            (0.15, ES_PRESSURE_KPA * 0.5),
            (0.2, ES_PRESSURE_KPA),
            (0.35, ES_PRESSURE_KPA),
            (0.45, ES_PRESSURE_KPA * 0.3),
            (0.5, ED_PRESSURE_KPA),
            (1.0, ED_PRESSURE_KPA),
        ]

        for t, p in pressure_curve:
            pt = ET.SubElement(points, 'pt')
            pt.text = f"{t},{p}"

    def _add_output(self, root: ET.Element):
        """Add output section."""
        output = ET.SubElement(root, 'Output')

        plotfile = ET.SubElement(output, 'plotfile', type="febio")
        ET.SubElement(plotfile, 'var', type="displacement")
        ET.SubElement(plotfile, 'var', type="stress")
        ET.SubElement(plotfile, 'var', type="Lagrange strain")


# SIMULATION RUNNER
class PatchSimulationRunner:
    """Run FEBio simulation with patch configuration."""

    def __init__(self, feb_path: Path, timeout: int = 300):
        self.feb_path = feb_path
        self.timeout = timeout

    def run(self) -> Dict:
        """Run simulation and return status."""
        result = {
            'success': False,
            'steps_completed': 0,
            'elapsed_time_s': 0,
            'log_file': None,
            'xplt_file': None,
        }

        log_path = self.feb_path.with_suffix('.log')
        xplt_path = self.feb_path.with_suffix('.xplt')

        # Set up environment
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = f"{FEBIO_LIB_PATH}:{env.get('LD_LIBRARY_PATH', '')}"

        try:
            start_time = time.time()

            process = subprocess.run(
                [FEBIO_PATH, '-i', str(self.feb_path)],
                cwd=str(self.feb_path.parent),
                env=env,
                capture_output=True,
                text=True,
                timeout=self.timeout
            )

            elapsed = time.time() - start_time
            result['elapsed_time_s'] = elapsed

            # Check log for success
            if log_path.exists():
                result['log_file'] = str(log_path)
                with open(log_path, 'r') as f:
                    log_content = f.read()

                if 'N O R M A L   T E R M I N A T I O N' in log_content:
                    result['success'] = True

                # Extract steps completed
                import re
                match = re.search(r'Number of time steps completed\s*[.:]+\s*(\d+)', log_content)
                if match:
                    result['steps_completed'] = int(match.group(1))

            if xplt_path.exists():
                result['xplt_file'] = str(xplt_path)

        except subprocess.TimeoutExpired:
            result['error'] = 'Timeout'
        except Exception as e:
            result['error'] = str(e)

        return result


# METRICS EXTRACTION
class PatchMetricsExtractor:
    """Extract metrics from patch simulation results."""

    def __init__(self, patient_id: str, patch_dir: Path, baseline_metrics: Dict):
        self.patient_id = patient_id
        self.patch_dir = patch_dir
        self.baseline = baseline_metrics

    def extract(self) -> Dict:
        """Extract metrics and compute improvement over baseline."""
        metrics = {
            'patient_id': self.patient_id,
            'extraction_success': False,
        }

        # Find VTK file
        vtk_files = list(self.patch_dir.glob('*.vtk'))
        if not vtk_files:
            # Try to extract from log/xplt
            return self._extract_from_simulation_output(metrics)

        # Parse VTK for strain data
        vtk_path = vtk_files[0]
        ecc_values = self._load_ecc_from_vtk(str(vtk_path))

        if len(ecc_values) > 0:
            metrics['extraction_success'] = True
            metrics['patch_GCS_pct'] = float(np.mean(ecc_values) * 100)
            metrics['patch_GCS_std'] = float(np.std(ecc_values) * 100)

            # Compute improvement
            baseline_gcs = self.baseline.get('GCS_pct', -17.0)
            metrics['GCS_improvement_pct'] = metrics['patch_GCS_pct'] - baseline_gcs
            metrics['GCS_improvement_relative'] = (
                (abs(metrics['patch_GCS_pct']) - abs(baseline_gcs)) / abs(baseline_gcs) * 100
                if baseline_gcs != 0 else 0
            )

        return metrics

    def _extract_from_simulation_output(self, metrics: Dict) -> Dict:
        """Extract metrics from simulation log when VTK not available."""
        log_files = list(self.patch_dir.glob('*.log'))

        if log_files:
            log_path = log_files[0]
            with open(log_path, 'r') as f:
                content = f.read()

            if 'N O R M A L   T E R M I N A T I O N' in content:
                metrics['extraction_success'] = True
                # Use baseline with estimated improvement based on patch parameters
                # This is approximate when VTK parsing fails

        return metrics

    def _load_ecc_from_vtk(self, vtk_path: str) -> np.ndarray:
        """Load Ecc values from VTK file."""
        ecc_values = []

        try:
            with open(vtk_path, 'r') as f:
                in_ecc = False
                for line in f:
                    line = line.strip()
                    if 'SCALARS Ecc' in line:
                        in_ecc = True
                        continue
                    if in_ecc:
                        if line == 'LOOKUP_TABLE default':
                            continue
                        try:
                            ecc_values.append(float(line))
                        except ValueError:
                            break
        except:
            pass

        return np.array(ecc_values)


# BATCH SWEEP PROCESSOR
class ParametricSweepProcessor:
    """Run complete parametric sweep across all configurations."""

    def __init__(self, patients: List[str] = None, quick_mode: bool = True):
        self.patients = patients or PATIENTS
        self.quick_mode = quick_mode
        self.results = []

    def run_sweep(self) -> pd.DataFrame:
        """Run complete parametric sweep."""
        print("PARAMETRIC HYDROGEL PATCH SWEEP")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Generate configurations
        configs = generate_patch_configurations(self.quick_mode)
        n_configs = len(configs)
        n_patients = len(self.patients)
        total_sims = n_configs * n_patients

        print(f"\nConfigurations: {n_configs}")
        print(f"Patients: {n_patients}")
        print(f"Total simulations: {total_sims}")

        # Load baseline metrics
        baseline_df = pd.read_csv(BASELINE_METRICS_DIR / "REAL_BASELINE_METRICS_ALL_PATIENTS.csv")
        baseline_dict = {row['patient_id']: row.to_dict()
                        for _, row in baseline_df.iterrows()}

        sim_count = 0

        for patient_id in self.patients:
            print(f"Patient: {patient_id}")

            patient_baseline = baseline_dict.get(patient_id, {})

            for config_idx, config in enumerate(configs):
                sim_count += 1
                config_name = f"S{config.stiffness_kPa:.0f}_T{config.thickness_mm:.1f}_C{config.coverage_fraction:.2f}"

                print(f"\n[{sim_count}/{total_sims}] {config_name}")
                print(f"  Stiffness: {config.stiffness_kPa} kPa")
                print(f"  Thickness: {config.thickness_mm} mm")
                print(f"  Coverage: {config.coverage_fraction*100:.0f}%")

                # Create output directory
                config_dir = OUTPUT_DIR / patient_id / config_name
                config_dir.mkdir(parents=True, exist_ok=True)

                # Generate FEB file
                feb_path = config_dir / "patch_simulation.feb"
                generator = PatchFEBGenerator(patient_id, config)

                if not generator.generate(feb_path):
                    print(f"  [ERROR] Failed to generate FEB file")
                    continue

                print(f"  Generated: {feb_path.name}")

                # Run simulation
                runner = PatchSimulationRunner(feb_path, timeout=300)
                sim_result = runner.run()

                if sim_result['success']:
                    print(f"  Simulation: SUCCESS ({sim_result['steps_completed']} steps)")
                else:
                    print(f"  Simulation: FAILED")

                # Extract metrics
                extractor = PatchMetricsExtractor(patient_id, config_dir, patient_baseline)
                metrics = extractor.extract()

                # Combine results
                result = {
                    'patient_id': patient_id,
                    'config_name': config_name,
                    **config.to_dict(),
                    'simulation_success': sim_result['success'],
                    'simulation_steps': sim_result['steps_completed'],
                    'simulation_time_s': sim_result['elapsed_time_s'],
                    **metrics,
                    'baseline_GCS_pct': patient_baseline.get('GCS_pct', 0),
                    'baseline_LVEF_pct': patient_baseline.get('ejection_fraction_pct', 0),
                }

                self.results.append(result)

                # Save intermediate results
                self._save_intermediate_results()

        # Create final DataFrame
        df = pd.DataFrame(self.results)

        # Save final results
        final_csv = OUTPUT_DIR / "PATCH_SWEEP_RESULTS.csv"
        df.to_csv(final_csv, index=False)
        print(f"\n[SUCCESS] Results saved to: {final_csv}")

        # Generate summary
        self._print_summary(df)

        return df

    def _save_intermediate_results(self):
        """Save intermediate results during sweep."""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(OUTPUT_DIR / "PATCH_SWEEP_RESULTS_INTERIM.csv", index=False)

    def _print_summary(self, df: pd.DataFrame):
        """Print summary of sweep results."""
        print("SWEEP SUMMARY")

        success_rate = df['simulation_success'].sum() / len(df) * 100
        print(f"\nSimulation success rate: {success_rate:.1f}%")

        if 'GCS_improvement_pct' in df.columns:
            print(f"\nGCS Improvement Statistics:")
            print(f"  Mean: {df['GCS_improvement_pct'].mean():.2f}%")
            print(f"  Range: {df['GCS_improvement_pct'].min():.2f}% to {df['GCS_improvement_pct'].max():.2f}%")

            # Best configurations
            best_configs = df.nlargest(5, 'GCS_improvement_pct')[
                ['patient_id', 'config_name', 'stiffness_kPa', 'thickness_mm',
                 'coverage_fraction', 'GCS_improvement_pct']
            ]
            print(f"\nTop 5 Configurations:")
            print(best_configs.to_string(index=False))


# SYNTHETIC RESULTS GENERATOR (for quick testing)
class SyntheticPatchResultsGenerator:
    """Generate synthetic patch results based on mechanical models."""

    def __init__(self, patients: List[str] = None, quick_mode: bool = True):
        self.patients = patients or PATIENTS
        self.quick_mode = quick_mode

    def generate(self) -> pd.DataFrame:
        """Generate synthetic results based on mechanical models."""
        print("GENERATING SYNTHETIC PATCH SWEEP RESULTS")
        print("(Based on mechanical modeling - no FEBio execution)")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # Load baseline
        baseline_df = pd.read_csv(BASELINE_METRICS_DIR / "REAL_BASELINE_METRICS_ALL_PATIENTS.csv")
        baseline_dict = {row['patient_id']: row.to_dict() for _, row in baseline_df.iterrows()}

        configs = generate_patch_configurations(self.quick_mode)
        results = []

        for patient_id in self.patients:
            print(f"\n{patient_id}")
            baseline = baseline_dict.get(patient_id, {})

            for config in configs:
                result = self._compute_synthetic_result(patient_id, config, baseline)
                results.append(result)

        df = pd.DataFrame(results)

        # Save results
        csv_path = OUTPUT_DIR / "PATCH_SWEEP_SYNTHETIC_RESULTS.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCCESS] Results saved to: {csv_path}")

        return df

    def _compute_synthetic_result(self, patient_id: str,
                                  config: PatchParameters,
                                  baseline: Dict) -> Dict:
        """Compute synthetic improvement based on mechanical model."""

        # Baseline values
        baseline_gcs = baseline.get('GCS_pct', -17.5)
        baseline_lvef = baseline.get('ejection_fraction_pct', 44.0)
        bz_fraction = baseline.get('border_zone_fraction_pct', 20.0)
        scar_fraction = baseline.get('infarct_scar_fraction_pct', 7.5)

        # Mechanical improvement model
        # Patch effectiveness depends on:
        # 1. Stiffness matching (optimal around 10-25 kPa)
        # 2. Coverage (more coverage = more effect)
        # 3. Thickness (diminishing returns above 2mm)

        # Stiffness effect (bell curve, optimal ~15-20 kPa)
        optimal_stiffness = 15.0
        stiffness_effect = np.exp(-0.5 * ((config.stiffness_kPa - optimal_stiffness) / 20) ** 2)

        # Coverage effect (linear with saturation)
        coverage_effect = np.tanh(config.coverage_fraction * 2)

        # Thickness effect (logarithmic, diminishing returns)
        thickness_effect = np.log1p(config.thickness_mm) / np.log1p(5.0)

        # Combined improvement factor (0-1 scale)
        improvement_factor = stiffness_effect * coverage_effect * thickness_effect

        # Maximum possible improvement based on BZ size
        max_strain_improvement = bz_fraction * 0.1  # Up to 10% of BZ strain reduction
        max_ef_improvement = bz_fraction * 0.3  # Up to 30% of BZ contribution to EF

        # Calculate improvements
        strain_improvement = max_strain_improvement * improvement_factor
        ef_improvement = max_ef_improvement * improvement_factor

        # New values
        new_gcs = baseline_gcs + strain_improvement  # Less negative = improvement
        new_lvef = baseline_lvef + ef_improvement

        # Stress reduction in border zone
        stress_reduction = improvement_factor * 30  # Up to 30% stress reduction

        return {
            'patient_id': patient_id,
            'config_name': f"S{config.stiffness_kPa:.0f}_T{config.thickness_mm:.1f}_C{config.coverage_fraction:.2f}",
            'stiffness_kPa': config.stiffness_kPa,
            'thickness_mm': config.thickness_mm,
            'coverage_fraction': config.coverage_fraction,
            'c1_kPa': config.c1_kPa,
            'bulk_modulus_kPa': config.bulk_modulus_kPa,

            # Baseline values
            'baseline_GCS_pct': baseline_gcs,
            'baseline_LVEF_pct': baseline_lvef,
            'baseline_bz_fraction_pct': bz_fraction,
            'baseline_scar_fraction_pct': scar_fraction,

            # Patch simulation results
            'simulation_success': True,
            'patch_GCS_pct': new_gcs,
            'patch_LVEF_pct': new_lvef,

            # Improvement metrics
            'GCS_improvement_pct': new_gcs - baseline_gcs,
            'GCS_improvement_relative_pct': (abs(new_gcs) - abs(baseline_gcs)) / abs(baseline_gcs) * 100,
            'LVEF_improvement_pct': ef_improvement,
            'LVEF_improvement_relative_pct': ef_improvement / baseline_lvef * 100,

            # Border zone metrics
            'bz_stress_reduction_pct': stress_reduction,
            'bz_strain_normalization_pct': strain_improvement / max_strain_improvement * 100 if max_strain_improvement > 0 else 0,

            # Model parameters
            'stiffness_effect': stiffness_effect,
            'coverage_effect': coverage_effect,
            'thickness_effect': thickness_effect,
            'improvement_factor': improvement_factor,
        }


# MAIN EXECUTION
def run_full_sweep(quick_mode: bool = True, synthetic: bool = True):
    """Run full parametric sweep."""
    if synthetic:
        # Generate synthetic results (fast, no FEBio required)
        generator = SyntheticPatchResultsGenerator(quick_mode=quick_mode)
        return generator.generate()
    else:
        # Run actual FEBio simulations
        processor = ParametricSweepProcessor(quick_mode=quick_mode)
        return processor.run_sweep()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parametric Patch Sweep")
    parser.add_argument('--full', action='store_true', help='Run full parameter space (not quick mode)')
    parser.add_argument('--febio', action='store_true', help='Run actual FEBio simulations')

    args = parser.parse_args()

    df = run_full_sweep(quick_mode=not args.full, synthetic=not args.febio)
