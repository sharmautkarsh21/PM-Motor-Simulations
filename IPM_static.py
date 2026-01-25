# IPM Motor Analysis using FEMM - Flux Density Visualization and Field Extraction
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import femm  # FEMM Python interface for 2D magnetostatic analysis
except ImportError:
    sys.exit(1)


class IPMMotorAnalyzer:
    """Class to handle FEMM IPM motor model initialization and analysis"""
    
    def __init__(self, fem_file_path):
        """
        Initialize the IPM Motor Analyzer
        
        Parameters:
        -----------
        fem_file_path : str
            Path to the .fem FEMM file
        """
        self.fem_file = fem_file_path
        self.model_name = Path(fem_file_path).stem
        self.femm_initialized = False
        self.simulation_results = {}
        
        # Model geometry parameters extracted from TeslaModel3.fem
        self.depth = 134  # mm - axial length of motor
        self.precision = 1e-8  # mesh precision for FEMM solver
        self.frequency = 0  # Hz - static analysis (no time-varying fields)
        self.problem_type = "planar"  # 2D planar magnetostatic problem
        
        # Circuit parameters - 3-phase current distribution
        self.circuits = {
            'PhaseA': 400,      # Phase A current (Amps)
            'PhaseB': -200,      # Phase B current (Amps)
            'PhaseC': -200       # Phase C current (Amps)
        }
        
        # Material properties - relative magnetic permeability
        self.materials = {
            'M270-35A': 1.0,    # Silicon steel lamination
            'Copper': 1.0,      # Winding conductors (non-magnetic)
            'Aluminium': 1.0,   # Cage/housing (non-magnetic)
            'BMN-52UH': 1.05,   # NdFeB permanent magnet
            'Air': 1.0          # Air gap and surrounding regions
        }
        
    def initialize_femm(self):
        """Initialize FEMM environment and load the model"""
        try:
            femm.openfemm(0)  # Open FEMM in normal (non-minimized) mode
            femm.opendocument(self.fem_file)  # Load the .fem model file
            self.femm_initialized = True
            return True
        except Exception as e:
            return False
    
    def analyze_model(self, run_femm_solver=True):
        """
        Run analysis on the model
        
        Parameters:
        -----------
        run_femm_solver : bool
            If True, solves the FEM problem. If False, loads pre-computed results.
        """
        if not self.femm_initialized:
            print("✗ FEMM not initialized. Call initialize_femm() first.")
            return False
        
        try:
            femm.mi_saveas(f"{self.model_name}_working.fem")  # Save working copy of model
            
            if run_femm_solver:
                femm.mi_createmesh()  # Discretize geometry into finite elements
                femm.mi_analyze()  # Solve the magnetostatic equations (A-V formulation)
            
            femm.mi_loadsolution()  # Load computed results into postprocessor
            self._extract_results()  # Extract field values at sample points
            
            return True
        except Exception as e:
            return False
    
    def _extract_results(self):
        """Extract magnetic field results from the FEMM solution"""
        try:
            self.simulation_results['field_samples'] = {}
            
            # Define sample points in rotor, air-gap, and stator regions
            sample_points = [(75, 0), (112.5, 0), (80, 10),(75, 20), (50, 30)]
            
            # Retrieve magnetic field magnitude at each sample point
            for x, y in sample_points:
                try:
                    B_x = femm.mo_getb(x, y)[0]  # Field component in X direction (Tesla)
                    B_y = femm.mo_getb(x, y)[1]  # Field component in Y direction (Tesla)
                    B_mag = np.sqrt(B_x**2 + B_y**2)  # Magnitude |B|
                    self.simulation_results['field_samples'][(x, y)] = B_mag
                except:
                    pass
        except Exception as e:
            pass
    
    
    def plot_field_distribution(self, filename=None):
        """
        Create flux density distribution with geometry overlay for one pole (60° sector)
        
        Parameters:
        -----------
        filename : str, optional
            If provided, save the figure to this file
        """
        if not self.femm_initialized:
            print("✗ FEMM not initialized. Cannot plot.")
            return False
        
        try:
            print("\nGenerating flux density distribution for one pole...")
            
            # Define the simulation domain - 60° rotor sector from TeslaModel3.fem
            theta_start = 0  # degrees
            theta_end = 60   # degrees
            r_min = 0  # origin
            r_max = 112.5  # outer stator boundary
            
            # Create dense Cartesian grid for smooth contour rendering (reduces discretization artifacts)
            print("Sampling magnetic field on dense grid...")
            n_x = 400  # Resolution in X direction (pixels)
            n_y = 400  # Resolution in Y direction (pixels)
            
            # Set grid bounds in Cartesian coordinates
            x_max_cart = r_max * np.cos(np.deg2rad(theta_start))
            y_max_cart = r_max * np.sin(np.deg2rad(theta_end))
            
            # Create evenly-spaced coordinate arrays
            x_array = np.linspace(0, x_max_cart, n_x)
            y_array = np.linspace(0, y_max_cart, n_y)
            X, Y = np.meshgrid(x_array, y_array)  # 2D mesh of (x,y) coordinates
            
            # Initialize flux density magnitude array
            B_mag = np.zeros_like(X)
            
            # Sample FEMM field at each grid point
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    x_pt = X[i, j]  # Cartesian X coordinate
                    y_pt = Y[i, j]  # Cartesian Y coordinate
                    
                    # Convert to polar coordinates for sector boundary check
                    r_pt = np.hypot(x_pt, y_pt)  # Radial distance from origin
                    if r_pt < 1e-6:
                        theta_pt = 0
                    else:
                        theta_pt = np.rad2deg(np.arctan2(y_pt, x_pt))  # Angle from positive X-axis
                    
                    # Only query FEMM solution within the defined sector
                    if theta_start <= theta_pt <= theta_end and r_pt <= r_max:
                        try:
                            B_field = femm.mo_getb(x_pt, y_pt)  # Get (Bx, By) from FEMM postprocessor
                            B_mag[i, j] = np.sqrt(B_field[0]**2 + B_field[1]**2)  # Compute magnitude |B|
                        except:
                            B_mag[i, j] = np.nan  # Mark invalid points
                    else:
                        B_mag[i, j] = np.nan  # Mark points outside sector
            
            # Setup figure and axes
            fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))
            
            # Mask out-of-domain points (NaN values) for cleaner visualization
            B_mag_masked = np.ma.masked_invalid(B_mag)
            
            # Compute color scale limits from sampled data
            b_min = np.nanmin(B_mag)  # Minimum flux density in sector
            b_max = np.nanmax(B_mag)  # Maximum flux density in sector
            
            # Use pcolormesh with gouraud shading for smooth FEMM-like rendering (matches native FEMM visualization)
            im1 = ax1.pcolormesh(X, Y, B_mag_masked, cmap='jet', shading='gouraud', vmin=b_min, vmax=b_max)
            
            x_min, x_max = 0, x_max_cart
            y_min, y_max = 0, y_max_cart
            
            # Extract and overlay actual geometry boundaries from FEMM model
            print("Extracting actual geometry from model...")
            try:
                # Read geometry directly from .fem file (binary format with text sections)
                import os
                fem_file = f"{self.model_name}.fem"
                
                # Lists to store geometry data from .fem file
                geometry_points = []  # (x, y) coordinates of vertices
                geometry_segments = []  # Line and arc segment definitions
                
                if os.path.exists(fem_file):
                    with open(fem_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Parse .fem file sections: [NumPoints], [NumSegments], [NumArcSegments]
                    in_points = False
                    in_segments = False
                    in_arcs = False
                    
                    for i, line in enumerate(lines):
                        # Track which section we're in
                        if '[NumPoints]' in line:
                            in_points = True
                            continue
                        elif '[NumSegments]' in line:
                            in_points = False
                            in_segments = True
                            continue
                        elif '[NumArcSegments]' in line:
                            in_segments = False
                            in_arcs = True
                            continue
                        elif '[NumHoles]' in line or '[NumBlockLabels]' in line:
                            in_arcs = False
                            continue
                        
                        # Parse vertex coordinates from [NumPoints] section
                        if in_points and line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                try:
                                    x = float(parts[0])  # X coordinate (mm)
                                    y = float(parts[1])  # Y coordinate (mm)
                                    geometry_points.append((x, y))
                                except:
                                    pass
                        
                        # Parse straight line segments from [NumSegments] section
                        if in_segments and line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 2:
                                try:
                                    n1 = int(parts[0])  # First vertex index
                                    n2 = int(parts[1])  # Second vertex index
                                    geometry_segments.append(('line', n1, n2))
                                except:
                                    pass
                        
                        # Parse arc segments from [NumArcSegments] section
                        if in_arcs and line.strip():
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                try:
                                    n1 = int(parts[0])  # Start vertex index
                                    n2 = int(parts[1])  # End vertex index
                                    arc_angle = float(parts[2])  # Arc angle in degrees (+ = CCW, - = CW)
                                    geometry_segments.append(('arc', n1, n2, arc_angle))
                                except:
                                    pass
                
                # Draw geometry segments on the flux density plot
                for seg in geometry_segments:
                    try:
                        if seg[0] == 'line':
                            # Draw straight line segment (stator slots, rotor boundaries, etc.)
                            n1, n2 = seg[1], seg[2]
                            if n1 < len(geometry_points) and n2 < len(geometry_points):
                                p1 = geometry_points[n1]  # First endpoint
                                p2 = geometry_points[n2]  # Second endpoint
                                
                                # Cull segments outside viewing window for cleaner plot
                                if (0 <= p1[0] <= 120 and 0 <= p1[1] <= 110 and 
                                    0 <= p2[0] <= 120 and 0 <= p2[1] <= 110):
                                    ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],'black', linewidth=1, alpha=1)
                        
                        elif seg[0] == 'arc':
                            # Draw arc segment (rotor air-gap boundary, curved slot edges, etc.)
                            n1, n2, arc_angle = seg[1], seg[2], seg[3]  # Arc angle in degrees
                            if n1 < len(geometry_points) and n2 < len(geometry_points):
                                p1 = geometry_points[n1]  # Start point of arc
                                p2 = geometry_points[n2]  # End point of arc
                                
                                # Cull arcs outside viewing window
                                if (0 <= p1[0] <= 120 and 0 <= p1[1] <= 110 and 
                                    0 <= p2[0] <= 120 and 0 <= p2[1] <= 110):
                                    
                                    # For very small angles, approximate as straight line for efficiency
                                    if abs(arc_angle) < 0.1:
                                        ax1.plot([p1[0], p2[0]], [p1[1], p2[1]],'black', linewidth=1, alpha=1)
                                    else:
                                        # Compute arc from endpoints and angle using geometric formulas
                                        chord_length = np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)  # Distance between endpoints
                                        if chord_length > 0:
                                            arc_rad = np.deg2rad(arc_angle)  # Convert angle to radians
                                            
                                            # Arc radius from chord length and angle (R = c / (2*sin(θ/2)))
                                            radius = chord_length / (2 * np.sin(abs(arc_rad)/2))
                                            
                                            # Chord midpoint
                                            mid_x = (p1[0] + p2[0]) / 2
                                            mid_y = (p1[1] + p2[1]) / 2
                                            
                                            # Distance from chord midpoint to arc center
                                            h = radius * np.cos(abs(arc_rad)/2)
                                            
                                            # Unit vector perpendicular to chord
                                            dx = p2[0] - p1[0]
                                            dy = p2[1] - p1[1]
                                            perp_x = -dy / chord_length  # Perpendicular X component
                                            perp_y = dx / chord_length    # Perpendicular Y component
                                            
                                            # Arc center location (perpendicular direction determined by arc_angle sign)
                                            if arc_angle > 0:
                                                center_x = mid_x + h * perp_x  # Counterclockwise arc
                                                center_y = mid_y + h * perp_y
                                            else:
                                                center_x = mid_x - h * perp_x  # Clockwise arc
                                                center_y = mid_y - h * perp_y
                                            
                                            # Compute angles from center to start/end points
                                            start_angle = np.arctan2(p1[1] - center_y, p1[0] - center_x)
                                            end_angle = np.arctan2(p2[1] - center_y, p2[0] - center_x)
                                            
                                            # Generate smooth arc using 30 points along the curve
                                            if arc_angle > 0:
                                                if end_angle < start_angle:
                                                    end_angle += 2*np.pi  # Handle angle wraparound
                                                angles = np.linspace(start_angle, end_angle, 30)
                                            else:
                                                if start_angle < end_angle:
                                                    start_angle += 2*np.pi  # Handle angle wraparound
                                                angles = np.linspace(end_angle, start_angle, 30)
                                            
                                            # Convert polar arc to Cartesian coordinates and plot
                                            arc_x = center_x + radius * np.cos(angles)
                                            arc_y = center_y + radius * np.sin(angles)
                                            
                                            ax1.plot(arc_x, arc_y, 'black', linewidth=1, alpha=1)
                    except Exception as e:
                        pass
                
            except Exception as e:
                # Fallback to simple circular approximation if .fem file parsing fails
                theta_circle = np.linspace(0, np.deg2rad(60), 150)  # 60° sector
                
                # Draw rotor outer surface (r = 75 mm)
                r_rotor = 75
                x_rotor = r_rotor * np.cos(theta_circle)
                y_rotor = r_rotor * np.sin(theta_circle)
                ax1.plot(x_rotor, y_rotor, 'black', linewidth=1.2, alpha=0.7)
                
                # Draw air-gap outer boundary / stator inner surface (r = 85 mm)
                r_gap_outer = 85
                x_gap = r_gap_outer * np.cos(theta_circle)
                y_gap = r_gap_outer * np.sin(theta_circle)
                ax1.plot(x_gap, y_gap, 'black', linewidth=1.2, alpha=0.7)
                
                # Draw stator outer surface (r = 112.5 mm)
                r_stator = 112.5
                x_stator = r_stator * np.cos(theta_circle)
                y_stator = r_stator * np.sin(theta_circle)
                ax1.plot(x_stator, y_stator, 'black', linewidth=1.2, alpha=0.7)
                
                # Draw radial boundaries at θ=0° and θ=60° (sector edges)
                for angle_deg in [0, 60]:
                    angle_rad = np.deg2rad(angle_deg)
                    ax1.plot([0, 120*np.cos(angle_rad)], [0, 120*np.sin(angle_rad)], 
                           'black', linewidth=1.0, alpha=0.6)
            
            # Configure plot appearance and axes
            ax1.set_title('Flux Density Distribution with Geometry (One Pole)', fontsize=15, fontweight='bold')
            ax1.set_xlabel('X (mm)', fontsize=13)
            ax1.set_ylabel('Y (mm)', fontsize=13)
            ax1.set_aspect('equal')  # Equal axis scaling to preserve geometry proportions
            ax1.set_xlim([x_min - 5, x_max + 5])
            ax1.set_ylim([y_min - 5, y_max + 5])
            
            # Create colorbar with fine tick resolution for accurate field magnitude reading
            cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label('Flux Density |B| (T)', fontsize=12, fontweight='bold')
            
            # Set colorbar ticks to 20 levels for fine color resolution
            from matplotlib.ticker import MaxNLocator
            cbar1.locator = MaxNLocator(nbins=20)  # Generate 20 evenly-spaced tick positions
            cbar1.update_ticks()
            
            # Format colorbar tick labels to 2 decimal places for readability
            cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
            cbar1.ax.tick_params(labelsize=10)
            
            plt.tight_layout()
            
            # Save high-resolution PNG output
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            else:
                plot_file = f"{self.model_name}_flux_distribution.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            plt.close()  # Free memory by closing figure
            return True
            
        except Exception as e:
            return False
    
    def plot_spatial_flux_density(self, filename=None):
        """
        Create spatial flux density curve B vs theta at air gap for one pole
        
        Parameters:
        -----------
        filename : str, optional
            If provided, save the figure to this file
        """
        try:
            r_gap_inner = 75
            r_gap_outer = 85
            r_airgap = (r_gap_inner + r_gap_outer) / 2
            
            theta_start = 0
            theta_end = 60
            
            theta_resolution = 0.1
            n_points = int((theta_end - theta_start) / theta_resolution) + 1
            theta_deg = np.linspace(theta_start, theta_end, n_points)
            theta_rad = np.deg2rad(theta_deg)
            
            B_magnitude = np.zeros(n_points)
            
            for i, theta in enumerate(theta_rad):
                x = r_airgap * np.cos(theta)
                y = r_airgap * np.sin(theta)
                
                try:
                    B_field = femm.mo_getb(x, y)
                    Bx = B_field[0]
                    By = B_field[1]
                    B_magnitude[i] = np.sqrt(Bx**2 + By**2)
                except:
                    B_magnitude[i] = 0
            
            # Create single plot for B magnitude vs theta
            fig, ax = plt.subplots(1, 1, figsize=(14, 7))
            
            # Plot magnitude
            ax.plot(theta_deg, B_magnitude, 'b-', linewidth=2, label='Air Gap Flux Density |B|')
            ax.fill_between(theta_deg, 0, B_magnitude, alpha=0.25, color='blue')
            ax.grid(True, alpha=0.35, linestyle='--', linewidth=0.8)
            ax.set_xlabel('Mechanical Angle θ (degrees)', fontsize=13)
            ax.set_ylabel('Flux Density Magnitude |B| (T)', fontsize=13)
            ax.set_title(f'Air Gap Flux Density vs Mechanical Angle (r = {r_airgap:.1f} mm)', 
                         fontsize=15, fontweight='bold')
            ax.legend(loc='best', fontsize=12, framealpha=0.9)
            ax.set_xlim([theta_start, theta_end])
            ax.tick_params(labelsize=11)
            
            # Add statistics
            max_B = np.max(B_magnitude)
            mean_B = np.mean(B_magnitude)
            min_B = np.min(B_magnitude)
            
            stats_text = f'Max: {max_B:.3f} T  |  Mean: {mean_B:.3f} T  |  Min: {min_B:.3f} T  |  Resolution: {theta_resolution}°'
            fig.text(0.5, 0.02, stats_text, ha='center', fontsize=11, 
                    style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout(rect=[0, 0.05, 1, 1])
            
            if filename:
                plt.savefig(filename, dpi=300, bbox_inches='tight')
            else:
                plot_file = f"{self.model_name}_spatial_flux_density.png"
                plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            
            plt.close()
            return True
            
        except Exception as e:
            return False
    
    def run_full_analysis(self, plot=True, detailed_plot=True):
        """Run complete FEMM analysis and generate plots"""
        if not self.initialize_femm():  # Load model into FEMM
            return False
        
        if not self.analyze_model(run_femm_solver=True):  # Solve and extract field data
            return False
        
        if plot:
            self.plot_field_distribution()  # Generate contour plot of flux density
            # self.plot_spatial_flux_density()  # Optional: B vs angle curve
        
        return True
    
    def close_femm(self):
        """Close FEMM application and cleanup resources"""
        try:
            femm.closefemm()  # Gracefully shut down FEMM
        except Exception as e:
            pass  # Silently ignore errors if FEMM already closed


def main():
    """Entry point: initialize analyzer, run simulation, and generate plots"""
    model_path = r"d:\Knowledge Upgradation\03 Github\IPM Simulation\TeslaModel3.fem"
    
    if not os.path.exists(model_path):  # Validate model file exists
        sys.exit(1)
    
    analyzer = IPMMotorAnalyzer(model_path)  # Create analyzer instance
    analyzer.run_full_analysis(plot=True)  # Execute full FEMM analysis with plots
    analyzer.close_femm()  # Clean shutdown

if __name__ == "__main__":
    main()
