"""Transient IPM simulation helper using FEMM.

This version drives balanced three-phase sine currents, rotates the rotor with a
sliding band, and solves a set of quasi-static steps that span 60 electrical
degrees (one-sixth of an electrical cycle).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import math
import sys
import csv
from datetime import datetime
import numpy as np

try:
    import femm
except ImportError:
    sys.exit(1)

try:
    from live_plotter_dash import DashLivePlotter
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    DashLivePlotter = None


# ----- User-set simulation inputs -----
FEM_FILE = r"d:\\Knowledge Upgradation\\03 Github\\IPM Simulation\\TeslaModel3.fem"
SLIDING_BAND_NAME = "AGap"  # boundary name of the sliding band in your model
MOTOR_DEPTH_MM = 134.0  # axial length of motor in mm (from TeslaModel3.fem [Depth] = 134)
RPM = 3000.0  # mechanical speed
CURRENT_RMS = 400.0  # A, per phase RMS
CURRENT_ANGLE_DEG = 51.0  # electrical angle offset of Phase A
STEPS = 18  # number of solves across the simulation span
POLE_PAIRS = 3  # Tesla Model 3 has 6 poles = 3 pole pairs
ROTOR_GROUP = 2  # group id for rotor + magnets (not used in sliding band approach)
FULL_CYCLE = False  # True = simulate full electrical cycle (360°), False = 1/6 cycle (60°)
INITIAL_ROTOR_POSITION_DEG = 10.0  # Starting mechanical rotor position in degrees
ENABLE_LIVE_PLOT = True  # True = show live updating plot during simulation
SAVE_CSV = True  # True = save results to CSV file
ENABLE_INDUCTANCE_CALC = True  # True = calculate Ld, Lq, L0 using frozen permeability method (slower)
MAGNET_MATERIAL_NAME = "BMN-52UH"  # Name of magnet material in FEMM model
INDUCTANCE_TYPE = "incremental"  # "apparent" = L=ψ/I, "incremental" = L=dψ/dI
DELTA_I_PERCENT = 1.0  # Percentage of test current for incremental inductance (1% recommended)
# Output directory for generated artifacts
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
# --------------------------------------


@dataclass
class TimeStepResult:
    step_index: int
    time_s: float
    electrical_angle_deg: float
    mechanical_angle_deg: float
    ia: float
    ib: float
    ic: float
    torque_nm: Optional[float]
    flux_linkage_a: Optional[float] = None
    flux_linkage_b: Optional[float] = None
    flux_linkage_c: Optional[float] = None
    backmf_a: Optional[float] = None
    backmf_b: Optional[float] = None
    backmf_c: Optional[float] = None
    # Inductance parameters
    Ld: Optional[float] = None
    Lq: Optional[float] = None
    L0: Optional[float] = None
    saliency_ratio: Optional[float] = None
    Laa: Optional[float] = None
    Lab: Optional[float] = None
    Lac: Optional[float] = None
    Lba: Optional[float] = None
    Lbb: Optional[float] = None
    Lbc: Optional[float] = None
    Lca: Optional[float] = None
    Lcb: Optional[float] = None
    Lcc: Optional[float] = None


class IPMTransientSimulator:
    """Runs a transient-like stepped solution in FEMM."""

    def __init__(self, fem_file: str, pole_pairs: int = 4, rotor_group: int = 2, enable_live_plot: bool = True, use_dash: bool = True,
                 enable_inductance_calc: bool = False, magnet_material_name: str = "BMN-52UH",
                 inductance_type: str = "apparent", delta_i_percent: float = 1.0):
        self.fem_file = fem_file
        self.model_name = Path(fem_file).stem
        self.pole_pairs = pole_pairs
        self.rotor_group = rotor_group
        self.results: List[TimeStepResult] = []
        self.working_file = f"{self.model_name}_transient_working.fem"
        self._initialized = False
        self.enable_inductance_calc = enable_inductance_calc
        self.magnet_material_name = magnet_material_name
        self.inductance_type = inductance_type.lower()
        self.delta_i_percent = delta_i_percent
        
        # Use Dash live plotter when available; otherwise disable live plotting
        if enable_live_plot and use_dash and DASH_AVAILABLE:
            self.live_plotter = DashLivePlotter(enable_live_plot=True)
            self.using_dash = True
        else:
            self.live_plotter = None
            self.using_dash = False
            if enable_live_plot:
                print("Dash not available; live plotting disabled.")

    def initialize(self) -> bool:
        try:
            femm.openfemm(0)
            femm.opendocument(self.fem_file)
            femm.mi_saveas(self.working_file)
            self._initialized = True
            return True
        except Exception:
            return False

    def close(self) -> None:
        try:
            femm.closefemm()
        except Exception:
            pass

    def _set_phase_currents(self, ia: float, ib: float, ic: float) -> None:
        """Apply instantaneous currents to the three phase circuits."""
        for name, amps in ("fase1", ia), ("fase2", ib), ("fase3", ic):
            try:
                femm.mi_modifycircprop(name, 1, amps)
            except Exception:
                # If the circuit name mismatches, FEMM will raise. Caller can inspect logs.
                pass

    def _get_flux_linkages(self, step_index: int) -> tuple:
        """Extract flux linkage from each phase circuit."""
        try:
            flux_linkages = {}
            for phase in ["fase1", "fase2", "fase3"]:
                props = femm.mo_getcircuitproperties(phase)
                flux_linkages[phase] = props[2]
            return (flux_linkages.get("fase1"), flux_linkages.get("fase2"), flux_linkages.get("fase3"))
        except Exception:
            return (None, None, None)

    def _get_torque(self, step_index: int) -> Optional[float]:
        """Extract torque via sliding band gap integral."""
        try:
            torque_val = femm.mo_gapintegral(SLIDING_BAND_NAME, 0)
            return torque_val
        except Exception:
            return None

    def calculate_inductances_frozen_permeability(self, mech_angle_deg: float, ia: float, ib: float, ic: float,
                                                   psi_a_total: float, psi_b_total: float, psi_c_total: float,
                                                   magnet_material_name: str = "BMN-52UH",
                                                   step_index: int = 0,
                                                   inductance_type: str = "apparent",
                                                   delta_i_percent: float = 1.0) -> dict:
        """
        Calculate self and mutual inductances using the frozen permeability method.
        
        This method assumes the current solution is already loaded with all three phase currents.
        It:
        1. Uses the current loaded solution state as the frozen permeability base
        2. Disables magnet contribution in individual phase files
        3. Re-solves with each phase individually to extract inductance matrix
        
        Args:
            mech_angle_deg: Mechanical rotor position in degrees
            ia, ib, ic: Phase currents in Amperes
            psi_a_total, psi_b_total, psi_c_total: Total flux linkages (already computed)
            magnet_material_name: Name of the magnet material in FEMM model
            step_index: Index for temporary file naming (reused, not unique per step)
            
        Returns:
            Dictionary containing:
                - 'flux_linkages': [psi_a, psi_b, psi_c] total flux linkages in Wb
                - 'inductance_matrix': 3x3 numpy array [[Laa, Lab, Lac],
                                                          [Lba, Lbb, Lbc],
                                                          [Lca, Lcb, Lcc]] in H
                - 'Ld': d-axis inductance in H
                - 'Lq': q-axis inductance in H
                - 'saliency_ratio': Lq/Ld
        """
        try:
            # This solution is already computed with all three phase currents            
            # Disable magnetization (freeze permeability, remove magnet contribution)
            try:
                femm.mi_modifymaterial(magnet_material_name, 3, 0)  # Set Hc to 0
            except Exception as e:
                print(f"Warning: Could not modify material '{magnet_material_name}': {e}")
            
            no_mag_fem = "temp_inductance_frozen_no_mag.fem"
            working_ans = str(Path(self.working_file).with_suffix(".ans"))

            femm.mi_saveas(no_mag_fem)  # Save with Hc = 0

            try:
                femm.mi_close()
            except Exception:
                pass

            try:
                femm.opendocument(no_mag_fem)
            except Exception as e:
                print(f"Warning: Could not open no-mag model '{no_mag_fem}': {e}")
            
            # Initialize inductance matrix
            L_matrix = np.zeros((3, 3))
            i_test = max(abs(ia), abs(ib), abs(ic), 1e-6)
            
            if inductance_type == "incremental":
                # Incremental inductance: L = dψ/dI
                delta_i = i_test * (delta_i_percent / 100.0)
                
                # Phase A excitation: solve at i_test and i_test+delta_i
                femm.mi_setprevious(working_ans, 2)
                self._set_phase_currents(i_test, 0, 0)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:
                    try:
                        psi_a1 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b1 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c1 = femm.mo_getcircuitproperties('fase3')[2]
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test for Phase A: {e}")
                        psi_a1, psi_b1, psi_c1 = 0, 0, 0
                    
                    # Solve at i_test + delta_i
                    femm.mi_setprevious(working_ans, 2)
                    self._set_phase_currents(i_test + delta_i, 0, 0)
                    femm.mi_analyze(1)
                    femm.mi_loadsolution()
                    
                    try:
                        psi_a2 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b2 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c2 = femm.mo_getcircuitproperties('fase3')[2]
                        
                        L_matrix[0, 0] = (psi_a2 - psi_a1) / delta_i  # dLaa/dI
                        L_matrix[1, 0] = (psi_b2 - psi_b1) / delta_i  # dLba/dI
                        L_matrix[2, 0] = (psi_c2 - psi_c1) / delta_i  # dLca/dI
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test+delta for Phase A: {e}")
                
                # Phase B excitation: solve at i_test and i_test+delta_i
                femm.mi_setprevious(working_ans, 2)
                self._set_phase_currents(0, i_test, 0)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:
                    try:
                        psi_a1 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b1 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c1 = femm.mo_getcircuitproperties('fase3')[2]
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test for Phase B: {e}")
                        psi_a1, psi_b1, psi_c1 = 0, 0, 0
                    
                    femm.mi_setprevious(working_ans, 2)
                    self._set_phase_currents(0, i_test + delta_i, 0)
                    femm.mi_analyze(1)
                    femm.mi_loadsolution()
                    
                    try:
                        psi_a2 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b2 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c2 = femm.mo_getcircuitproperties('fase3')[2]
                        
                        L_matrix[0, 1] = (psi_a2 - psi_a1) / delta_i  # dLab/dI
                        L_matrix[1, 1] = (psi_b2 - psi_b1) / delta_i  # dLbb/dI
                        L_matrix[2, 1] = (psi_c2 - psi_c1) / delta_i  # dLcb/dI
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test+delta for Phase B: {e}")
                
                # Phase C excitation: solve at i_test and i_test+delta_i
                femm.mi_setprevious(working_ans, 2)
                self._set_phase_currents(0, 0, i_test)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:
                    try:
                        psi_a1 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b1 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c1 = femm.mo_getcircuitproperties('fase3')[2]
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test for Phase C: {e}")
                        psi_a1, psi_b1, psi_c1 = 0, 0, 0
                    
                    femm.mi_setprevious(working_ans, 2)
                    self._set_phase_currents(0, 0, i_test + delta_i)
                    femm.mi_analyze(1)
                    femm.mi_loadsolution()
                    
                    try:
                        psi_a2 = femm.mo_getcircuitproperties('fase1')[2]
                        psi_b2 = femm.mo_getcircuitproperties('fase2')[2]
                        psi_c2 = femm.mo_getcircuitproperties('fase3')[2]
                        
                        L_matrix[0, 2] = (psi_a2 - psi_a1) / delta_i  # dLac/dI
                        L_matrix[1, 2] = (psi_b2 - psi_b1) / delta_i  # dLbc/dI
                        L_matrix[2, 2] = (psi_c2 - psi_c1) / delta_i  # dLcc/dI
                    except Exception as e:
                        print(f"Warning: Could not extract flux at i_test+delta for Phase C: {e}")
            
            else:  # apparent inductance (default)
                # Apparent inductance: L = ψ/I
                # Phase A excitation only (with frozen permeability from current solution)
                femm.mi_setprevious(working_ans, 2)  # Use frozen permeability
                
                self._set_phase_currents(i_test, 0, 0)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:  # Avoid division by zero
                    try:
                        prop_a = femm.mo_getcircuitproperties('fase1')
                        prop_b = femm.mo_getcircuitproperties('fase2')
                        prop_c = femm.mo_getcircuitproperties('fase3')
                        
                        L_matrix[0, 0] = prop_a[2] / i_test  # Laa
                        L_matrix[1, 0] = prop_b[2] / i_test  # Lba
                        L_matrix[2, 0] = prop_c[2] / i_test  # Lca
                    except Exception as e:
                        print(f"Warning: Could not extract flux linkages for Phase A: {e}")
                
                # Phase B excitation only (with frozen permeability)
                femm.mi_setprevious(working_ans, 2)
                
                self._set_phase_currents(0, i_test, 0)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:
                    try:
                        prop_a = femm.mo_getcircuitproperties('fase1')
                        prop_b = femm.mo_getcircuitproperties('fase2')
                        prop_c = femm.mo_getcircuitproperties('fase3')
                        
                        L_matrix[0, 1] = prop_a[2] / i_test  # Lab
                        L_matrix[1, 1] = prop_b[2] / i_test  # Lbb
                        L_matrix[2, 1] = prop_c[2] / i_test  # Lcb
                    except Exception as e:
                        print(f"Warning: Could not extract flux linkages for Phase B: {e}")
                
                # Phase C excitation only (with frozen permeability)
                femm.mi_setprevious(working_ans, 2)
                
                self._set_phase_currents(0, 0, i_test)
                femm.mi_analyze(1)
                femm.mi_loadsolution()
                
                if i_test > 1e-6:
                    try:
                        prop_a = femm.mo_getcircuitproperties('fase1')
                        prop_b = femm.mo_getcircuitproperties('fase2')
                        prop_c = femm.mo_getcircuitproperties('fase3')
                        
                        L_matrix[0, 2] = prop_a[2] / i_test  # Lac
                        L_matrix[1, 2] = prop_b[2] / i_test  # Lbc
                        L_matrix[2, 2] = prop_c[2] / i_test  # Lcc
                    except Exception as e:
                        print(f"Warning: Could not extract flux linkages for Phase C: {e}")
            
            # Calculate dq inductances using Park transformation
            # Electrical angle in radians
            elec_angle_rad = math.radians(mech_angle_deg * self.pole_pairs)
            
            # Park transformation matrix (abc to dq0)
            k_park = (2/3) * np.array([
                [math.cos(elec_angle_rad), math.cos(elec_angle_rad - 2*math.pi/3), math.cos(elec_angle_rad + 2*math.pi/3)],
                [math.sin(elec_angle_rad), math.sin(elec_angle_rad - 2*math.pi/3), math.sin(elec_angle_rad + 2*math.pi/3)],
                [0.5, 0.5, 0.5]
            ])
            
            # Inverse Park transformation
            k_park_inv = np.array([
                [math.cos(elec_angle_rad), math.sin(elec_angle_rad), 1],
                [math.cos(elec_angle_rad - 2*math.pi/3), math.sin(elec_angle_rad - 2*math.pi/3), 1],
                [math.cos(elec_angle_rad + 2*math.pi/3), math.sin(elec_angle_rad + 2*math.pi/3), 1]
            ])
            
            # Transform inductance matrix to dq0 frame
            L_dq0 = k_park @ L_matrix @ k_park_inv
            
            Ld = L_dq0[0, 0]  # d-axis inductance
            Lq = L_dq0[1, 1]  # q-axis inductance
            L0 = L_dq0[2, 2]  # zero-sequence inductance
            
            saliency_ratio = Lq / Ld if abs(Ld) > 1e-9 else 0
            
            try:
                femm.mi_close()
            except Exception:
                pass

            try:
                femm.opendocument(self.working_file)
            except Exception as e:
                print(f"Warning: Could not reopen working model '{self.working_file}': {e}")

            return {
                'flux_linkages': [psi_a_total, psi_b_total, psi_c_total],
                'inductance_matrix': L_matrix,
                'Ld': Ld,
                'Lq': Lq,
                'L0': L0,
                'L_dq0': L_dq0,
                'saliency_ratio': saliency_ratio
            }
            
        except Exception as e:
            print(f"Error calculating inductances: {e}")
            return {
                'flux_linkages': [psi_a_total, psi_b_total, psi_c_total],
                'inductance_matrix': None,
                'Ld': None,
                'Lq': None,
                'L0': None,
                'L_dq0': None,
                'saliency_ratio': None
            }

    def _set_rotor_position(self, mech_angle_deg: float) -> None:
        """Set rotor position via sliding band boundary innerangle (per FEMM sliding band approach)."""
        try:
            # Modify the sliding band boundary innerangle to match rotor position
            # This is the standard FEMM approach: mi_modifyboundprop(boundary_name, property_id, value)
            # property_id 10 = innerangle for sliding band boundaries
            femm.mi_modifyboundprop(SLIDING_BAND_NAME, 10, mech_angle_deg)
        except Exception:
            pass

    def _solve_and_record(self, step_index: int, time_s: float, elec_angle_deg: float, mech_angle_deg: float, ia: float, ib: float, ic: float) -> None:
        """Set phase currents, solve, and extract torque via sliding band gap integral."""
        try:
            # Set the phase currents for this step
            self._set_phase_currents(ia, ib, ic)
            
            # Set rotor position via sliding band
            self._set_rotor_position(mech_angle_deg)
            
            # Mesh, solve and load solution
            femm.mi_createmesh()
            femm.mi_analyze(0)  # verbose=1
            femm.mi_loadsolution()

            # Extract flux linkage for each phase
            flux_a, flux_b, flux_c = self._get_flux_linkages(step_index)
            
            # Extract torque using sliding band gap integral
            torque_val = self._get_torque(step_index)
            
            # Calculate back-EMF from flux derivative (e = dΦ/dt)
            backmf_a = None
            backmf_b = None
            backmf_c = None
            if len(self.results) > 0:
                prev_result = self.results[-1]
                dtime = time_s - prev_result.time_s
                if dtime > 0:
                    flux_a_val = flux_a if flux_a is not None else 0
                    flux_b_val = flux_b if flux_b is not None else 0
                    flux_c_val = flux_c if flux_c is not None else 0
                    prev_flux_a = prev_result.flux_linkage_a if prev_result.flux_linkage_a is not None else 0
                    prev_flux_b = prev_result.flux_linkage_b if prev_result.flux_linkage_b is not None else 0
                    prev_flux_c = prev_result.flux_linkage_c if prev_result.flux_linkage_c is not None else 0
                    
                    backmf_a = (flux_a_val - prev_flux_a) / dtime
                    backmf_b = (flux_b_val - prev_flux_b) / dtime
                    backmf_c = (flux_c_val - prev_flux_c) / dtime
            
            # Calculate inductances if enabled
            Ld, Lq, L0, saliency_ratio = None, None, None, None
            Laa, Lab, Lac, Lba, Lbb, Lbc, Lca, Lcb, Lcc = None, None, None, None, None, None, None, None, None
            
            if self.enable_inductance_calc:
                try:
                    ind_result = self.calculate_inductances_frozen_permeability(
                        mech_angle_deg, ia, ib, ic, flux_a, flux_b, flux_c,
                        self.magnet_material_name, step_index,
                        inductance_type=self.inductance_type,
                        delta_i_percent=self.delta_i_percent
                    )
                    
                    Ld = ind_result.get('Ld')
                    Lq = ind_result.get('Lq')
                    L0 = ind_result.get('L0')
                    saliency_ratio = ind_result.get('saliency_ratio')
                    
                    # Extract inductance matrix elements if available
                    L_matrix = ind_result.get('inductance_matrix')
                    if L_matrix is not None:
                        Laa, Lab, Lac = L_matrix[0, 0], L_matrix[0, 1], L_matrix[0, 2]
                        Lba, Lbb, Lbc = L_matrix[1, 0], L_matrix[1, 1], L_matrix[1, 2]
                        Lca, Lcb, Lcc = L_matrix[2, 0], L_matrix[2, 1], L_matrix[2, 2]
                except Exception as e:
                    print(f"Warning: Inductance calculation failed at step {step_index}: {e}")
            
            # Close solution after all extractions complete
            femm.mo_close()
            
            result = TimeStepResult(
                step_index=step_index,
                time_s=time_s,
                electrical_angle_deg=elec_angle_deg,
                mechanical_angle_deg=mech_angle_deg,
                ia=ia,
                ib=ib,
                ic=ic,
                torque_nm=torque_val,
                flux_linkage_a=flux_a,
                flux_linkage_b=flux_b,
                flux_linkage_c=flux_c,
                backmf_a=backmf_a,
                backmf_b=backmf_b,
                backmf_c=backmf_c,
                Ld=Ld,
                Lq=Lq,
                L0=L0,
                saliency_ratio=saliency_ratio,
                Laa=Laa,
                Lab=Lab,
                Lac=Lac,
                Lba=Lba,
                Lbb=Lbb,
                Lbc=Lbc,
                Lca=Lca,
                Lcb=Lcb,
                Lcc=Lcc,
            )
            
            self.results.append(result)
            
            # Update live plot if enabled
            if self.live_plotter is not None:
                self.live_plotter.update(result)
            
        except Exception as e:
            print(f"Error at step {step_index}: {e}")

    def run(self, rpm: float, current_rms: float, current_angle_deg: float,
            steps: int, full_cycle: bool = False, initial_rotor_pos_deg: float = 0.0) -> List[TimeStepResult]:
        if not self._initialized and not self.initialize():
            raise RuntimeError("FEMM failed to open the model.")

        mech_hz = rpm / 60.0
        elec_hz = mech_hz * self.pole_pairs
        if elec_hz <= 0:
            raise ValueError("Electrical frequency must be positive.")

        # Determine electrical span based on full_cycle flag
        electrical_span_deg = 360.0 if full_cycle else 60.0
        electrical_span_cycles = electrical_span_deg / 360.0
        sim_time = electrical_span_cycles / elec_hz
        
        # Calculate step sizes
        mechanical_span_deg = electrical_span_deg / self.pole_pairs
        mechanical_step_deg = mechanical_span_deg / steps if steps > 0 else 0

        i_peak = current_rms * math.sqrt(2.0)
        phase_offset_rad = math.radians(current_angle_deg)

        print(f"  Sample mechanical angle progression:")
        print(f"  [Electrical angle should span 0° to {electrical_span_deg:.0f}°; Mechanical {mechanical_span_deg:.0f}°]")
        print(f"  [Initial rotor position: {initial_rotor_pos_deg:.2f}°]\n")

        electrical_step_deg = electrical_span_deg / steps if steps > 0 else 0
        
        # Start Dash server if available
        if self.using_dash and self.live_plotter is not None and hasattr(self.live_plotter, 'start_server'):
            self.live_plotter.start_server()
        
        for idx in range(steps + 1):
            # Deterministic angles by index; start at 0° electrical and INITIAL_ROTOR_POSITION_DEG mechanical
            t = idx * (sim_time / steps)
            mech_deg_total = initial_rotor_pos_deg + idx * mechanical_step_deg
            elec_deg = idx * electrical_step_deg  # No modulo to prevent wrap-around in plots

            # Currents based on electrical angle
            elec_rad = math.radians(elec_deg)
            ia = i_peak * math.sin(elec_rad + phase_offset_rad)
            ib = i_peak * math.sin(elec_rad + phase_offset_rad - 2.0 * math.pi / 3.0)
            ic = i_peak * math.sin(elec_rad + phase_offset_rad + 2.0 * math.pi / 3.0)

            self._solve_and_record(idx, t, elec_deg, mech_deg_total, ia, ib, ic)

        return self.results


def main() -> None:
    fem_path = FEM_FILE
    rpm = RPM
    current_rms = CURRENT_RMS
    current_angle = CURRENT_ANGLE_DEG
    steps = STEPS
    pole_pairs = POLE_PAIRS
    rotor_group = ROTOR_GROUP
    full_cycle = FULL_CYCLE
    initial_rotor_pos = INITIAL_ROTOR_POSITION_DEG
    enable_inductance = ENABLE_INDUCTANCE_CALC
    magnet_material = MAGNET_MATERIAL_NAME
    inductance_type = INDUCTANCE_TYPE
    delta_i_percent = DELTA_I_PERCENT

    sim = IPMTransientSimulator(fem_path, pole_pairs=pole_pairs, rotor_group=rotor_group, 
                                 enable_live_plot=ENABLE_LIVE_PLOT, use_dash=True,
                                 enable_inductance_calc=enable_inductance, 
                                 magnet_material_name=magnet_material,
                                 inductance_type=inductance_type,
                                 delta_i_percent=delta_i_percent)
    
    if enable_inductance:
        print(f"\n*** Inductance calculation ENABLED (using frozen permeability method) ***")
        print(f"*** Inductance type: {inductance_type.upper()} ***")
        print(f"*** Magnet material: {magnet_material} ***")
        if inductance_type == "incremental":
            print(f"*** Delta I: {delta_i_percent}% ***")
            print(f"*** This will significantly increase simulation time (8x solves per step) ***\n")
        else:
            print(f"*** This will significantly increase simulation time (4x solves per step) ***\n")
    
    try:
        results = sim.run(rpm, current_rms, current_angle, steps, full_cycle, initial_rotor_pos)

        # Summary statistics
        torque_vals = [r.torque_nm for r in results if r.torque_nm is not None]
        flux_a_vals = [r.flux_linkage_a for r in results if r.flux_linkage_a is not None]
        flux_b_vals = [r.flux_linkage_b for r in results if r.flux_linkage_b is not None]
        flux_c_vals = [r.flux_linkage_c for r in results if r.flux_linkage_c is not None]

        if torque_vals:
            torque_avg = sum(torque_vals) / len(torque_vals)
            torque_min = min(torque_vals)
            torque_max = max(torque_vals)
            torque_p2p = torque_max - torque_min
            print("")
            print(f"Torque avg = {torque_avg:.4f} Nm, p2p = {torque_p2p:.4f} Nm (min {torque_min:.4f}, max {torque_max:.4f})")

        def _print_flux_stats(label: str, values: list) -> None:
            if not values:
                return
            f_min = min(values)
            f_max = max(values)
            print(f"Flux {label}: min = {f_min:.6f} Wb, max = {f_max:.6f} Wb")

        _print_flux_stats("A", flux_a_vals)
        _print_flux_stats("B", flux_b_vals)
        _print_flux_stats("C", flux_c_vals)
        
        # Inductance statistics if enabled
        if enable_inductance:
            ld_vals = [r.Ld for r in results if r.Ld is not None]
            lq_vals = [r.Lq for r in results if r.Lq is not None]
            saliency_vals = [r.saliency_ratio for r in results if r.saliency_ratio is not None]
            
            if ld_vals and lq_vals:
                ld_avg = sum(ld_vals) / len(ld_vals)
                lq_avg = sum(lq_vals) / len(lq_vals)
                saliency_avg = sum(saliency_vals) / len(saliency_vals) if saliency_vals else 0
                
                print(f"\nInductance Statistics:")
                print(f"  Ld avg = {ld_avg*1e6:.2f} µH (min {min(ld_vals)*1e6:.2f}, max {max(ld_vals)*1e6:.2f})")
                print(f"  Lq avg = {lq_avg*1e6:.2f} µH (min {min(lq_vals)*1e6:.2f}, max {max(lq_vals)*1e6:.2f})")
                print(f"  Saliency ratio (Lq/Ld) avg = {saliency_avg:.3f}")
        
        # Save to CSV
        if SAVE_CSV:
            csv_filename = OUTPUT_DIR / f"simulation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            save_results_to_csv(results, csv_filename)
        
        # Finalize live plot (save and display)
        if ENABLE_LIVE_PLOT and sim.live_plotter is not None:
            sim.live_plotter.finalize()
        elif ENABLE_LIVE_PLOT:
            print("Live plot disabled (Dash not available).")
    finally:
        sim.close()


def save_results_to_csv(results: List[TimeStepResult], filename: str = "simulation_results.csv") -> None:
    """Save simulation results to CSV file."""
    try:
        output_path = Path(filename)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'Step', 'Time_ms', 'Electrical_Angle_deg', 'Mechanical_Angle_deg',
                'Current_A_Phase', 'Current_B_Phase', 'Current_C_Phase',
                'Flux_Linkage_A_Wb', 'Flux_Linkage_B_Wb', 'Flux_Linkage_C_Wb',
                'BackEMF_A_V', 'BackEMF_B_V', 'BackEMF_C_V',
                'Torque_Nm',
                'Ld_H', 'Lq_H', 'L0_H', 'Saliency_Ratio',
                'Laa_H', 'Lab_H', 'Lac_H',
                'Lba_H', 'Lbb_H', 'Lbc_H',
                'Lca_H', 'Lcb_H', 'Lcc_H'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                writer.writerow({
                    'Step': r.step_index,
                    'Time_ms': f"{r.time_s * 1e3:.6f}",
                    'Electrical_Angle_deg': f"{r.electrical_angle_deg:.4f}",
                    'Mechanical_Angle_deg': f"{r.mechanical_angle_deg:.4f}",
                    'Current_A_Phase': f"{r.ia:.6f}",
                    'Current_B_Phase': f"{r.ib:.6f}",
                    'Current_C_Phase': f"{r.ic:.6f}",
                    'Flux_Linkage_A_Wb': f"{r.flux_linkage_a:.8f}" if r.flux_linkage_a is not None else "N/A",
                    'Flux_Linkage_B_Wb': f"{r.flux_linkage_b:.8f}" if r.flux_linkage_b is not None else "N/A",
                    'Flux_Linkage_C_Wb': f"{r.flux_linkage_c:.8f}" if r.flux_linkage_c is not None else "N/A",
                    'BackEMF_A_V': f"{r.backmf_a:.6f}" if r.backmf_a is not None else "N/A",
                    'BackEMF_B_V': f"{r.backmf_b:.6f}" if r.backmf_b is not None else "N/A",
                    'BackEMF_C_V': f"{r.backmf_c:.6f}" if r.backmf_c is not None else "N/A",
                    'Torque_Nm': f"{r.torque_nm:.6f}" if r.torque_nm is not None else "N/A",
                    'Ld_H': f"{r.Ld:.9f}" if r.Ld is not None else "N/A",
                    'Lq_H': f"{r.Lq:.9f}" if r.Lq is not None else "N/A",
                    'L0_H': f"{r.L0:.9f}" if r.L0 is not None else "N/A",
                    'Saliency_Ratio': f"{r.saliency_ratio:.4f}" if r.saliency_ratio is not None else "N/A",
                    'Laa_H': f"{r.Laa:.9f}" if r.Laa is not None else "N/A",
                    'Lab_H': f"{r.Lab:.9f}" if r.Lab is not None else "N/A",
                    'Lac_H': f"{r.Lac:.9f}" if r.Lac is not None else "N/A",
                    'Lba_H': f"{r.Lba:.9f}" if r.Lba is not None else "N/A",
                    'Lbb_H': f"{r.Lbb:.9f}" if r.Lbb is not None else "N/A",
                    'Lbc_H': f"{r.Lbc:.9f}" if r.Lbc is not None else "N/A",
                    'Lca_H': f"{r.Lca:.9f}" if r.Lca is not None else "N/A",
                    'Lcb_H': f"{r.Lcb:.9f}" if r.Lcb is not None else "N/A",
                    'Lcc_H': f"{r.Lcc:.9f}" if r.Lcc is not None else "N/A",
                })
        
        print(f"Results saved to: {filename}\n")
    except Exception as e:
        print(f"Error saving CSV file: {e}\n")
if __name__ == "__main__":
    main()
