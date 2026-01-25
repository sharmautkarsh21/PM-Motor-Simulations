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
STEPS = 96  # number of solves across the simulation span
POLE_PAIRS = 3  # Tesla Model 3 has 6 poles = 3 pole pairs
ROTOR_GROUP = 2  # group id for rotor + magnets (not used in sliding band approach)
FULL_CYCLE = True  # True = simulate full electrical cycle (360°), False = 1/6 cycle (60°)
INITIAL_ROTOR_POSITION_DEG = 10.0  # Starting mechanical rotor position in degrees
ENABLE_LIVE_PLOT = True  # True = show live updating plot during simulation
SAVE_CSV = True  # True = save results to CSV file
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




class IPMTransientSimulator:
    """Runs a transient-like stepped solution in FEMM."""

    def __init__(self, fem_file: str, pole_pairs: int = 4, rotor_group: int = 2, enable_live_plot: bool = True, use_dash: bool = True):
        self.fem_file = fem_file
        self.model_name = Path(fem_file).stem
        self.pole_pairs = pole_pairs
        self.rotor_group = rotor_group
        self.results: List[TimeStepResult] = []
        self.working_file = f"{self.model_name}_transient_working.fem"
        self._initialized = False
        
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
            
            femm.mo_close()  # Close solution after extracting results
            
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

    sim = IPMTransientSimulator(fem_path, pole_pairs=pole_pairs, rotor_group=rotor_group, 
                                 enable_live_plot=ENABLE_LIVE_PLOT, use_dash=True)
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
                'Torque_Nm'
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
                })
        
        print(f"Results saved to: {filename}\n")
    except Exception as e:
        print(f"Error saving CSV file: {e}\n")
if __name__ == "__main__":
    main()
