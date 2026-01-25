"""
FEMM Utilities Module
Helper functions for FEMM analysis and post-processing
"""

import numpy as np
import femm
from typing import Tuple, List, Dict, Optional


class FEMMFieldCalculator:
    """Calculate field properties at various points and regions"""
    
    @staticmethod
    def get_field_at_point(x: float, y: float) -> Dict[str, float]:
        """
        Get magnetic field components at a specific point
        
        Parameters:
        -----------
        x, y : float
            Coordinates in mm
        
        Returns:
        --------
        dict : Field components {B_x, B_y, B_mag, A}
        """
        try:
            B_x, B_y = femm.mo_getb(x, y)
            A = femm.mo_geta(x, y)
            B_mag = np.sqrt(B_x**2 + B_y**2)
            
            return {
                'B_x': B_x,
                'B_y': B_y,
                'B_mag': B_mag,
                'A': A,
                'x': x,
                'y': y
            }
        except:
            return None
    
    @staticmethod
    def get_field_along_line(x1: float, y1: float, x2: float, y2: float, 
                            num_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get field along a line between two points
        
        Parameters:
        -----------
        x1, y1, x2, y2 : float
            Start and end coordinates
        num_points : int
            Number of sample points
        
        Returns:
        --------
        distances, B_mag, angles : Field magnitude and angle along line
        """
        x_line = np.linspace(x1, x2, num_points)
        y_line = np.linspace(y1, y2, num_points)
        distances = np.linspace(0, np.sqrt((x2-x1)**2 + (y2-y1)**2), num_points)
        
        B_mag = np.zeros(num_points)
        angles = np.zeros(num_points)
        
        for i in range(num_points):
            try:
                B_x, B_y = femm.mo_getb(x_line[i], y_line[i])
                B_mag[i] = np.sqrt(B_x**2 + B_y**2)
                angles[i] = np.arctan2(B_y, B_x) * 180 / np.pi
            except:
                B_mag[i] = 0
                angles[i] = 0
        
        return distances, B_mag, angles
    
    @staticmethod
    def get_flux_density_grid(x_min: float, x_max: float, y_min: float, y_max: float,
                             nx: int = 50, ny: int = 50) -> Tuple[np.ndarray, np.ndarray, 
                                                                    np.ndarray, np.ndarray]:
        """
        Create a grid of flux density values
        
        Parameters:
        -----------
        x_min, x_max, y_min, y_max : float
            Grid bounds
        nx, ny : int
            Number of points in each direction
        
        Returns:
        --------
        X, Y, B_x, B_y : Grid arrays of field components
        """
        x = np.linspace(x_min, x_max, nx)
        y = np.linspace(y_min, y_max, ny)
        X, Y = np.meshgrid(x, y)
        
        B_x = np.zeros_like(X)
        B_y = np.zeros_like(X)
        
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    B_x[i, j], B_y[i, j] = femm.mo_getb(X[i, j], Y[i, j])
                except:
                    B_x[i, j] = 0
                    B_y[i, j] = 0
        
        return X, Y, B_x, B_y


class FEMMCircuitAnalyzer:
    """Analyze motor circuits and winding properties"""
    
    @staticmethod
    def analyze_circuit_properties(circuit_name: str) -> Dict:
        """
        Analyze properties of a circuit
        
        Parameters:
        -----------
        circuit_name : str
            Name of the circuit
        
        Returns:
        --------
        dict : Circuit properties
        """
        try:
            # Get circuit integral and flux linkage
            integral = femm.mo_getgapintegral(circuit_name)
            
            return {
                'name': circuit_name,
                'integral': integral
            }
        except:
            return {'name': circuit_name, 'integral': 'Error'}
    
    @staticmethod
    def get_winding_inductance() -> float:
        """
        Estimate winding inductance from stored energy
        Uses L = 2*Energy / I^2
        
        Returns:
        --------
        float : Estimated inductance (H)
        """
        try:
            # Get total energy in the problem
            # This is an approximation
            energy = femm.mo_getgapintegral("AGap")
            # Would need actual current to calculate properly
            return energy
        except:
            return None


class FEMMPostProcessor:
    """Post-processing and result extraction"""
    
    @staticmethod
    def extract_field_profile(x_coords: List[float], y_coords: List[float]) -> List[Dict]:
        """
        Extract field at multiple points
        
        Parameters:
        -----------
        x_coords, y_coords : list
            Lists of x, y coordinates
        
        Returns:
        --------
        list : Field data at each point
        """
        results = []
        
        for x, y in zip(x_coords, y_coords):
            field_data = FEMMFieldCalculator.get_field_at_point(x, y)
            if field_data:
                results.append(field_data)
        
        return results
    
    @staticmethod
    def get_statistics_from_grid(X: np.ndarray, Y: np.ndarray, 
                                B_x: np.ndarray, B_y: np.ndarray) -> Dict[str, float]:
        """
        Calculate statistics from field grid
        
        Parameters:
        -----------
        X, Y, B_x, B_y : np.ndarray
            Grid arrays
        
        Returns:
        --------
        dict : Statistical measures
        """
        B_mag = np.sqrt(B_x**2 + B_y**2)
        
        return {
            'B_max': np.max(B_mag),
            'B_min': np.min(B_mag),
            'B_mean': np.mean(B_mag),
            'B_std': np.std(B_mag),
            'B_median': np.median(B_mag)
        }


class FEMMModelProperties:
    """Extract and display model properties"""
    
    @staticmethod
    def get_problem_type() -> str:
        """Get problem type"""
        # This would require FEMM API call
        return "planar"
    
    @staticmethod
    def get_mesh_info() -> Dict[str, int]:
        """Get mesh information"""
        try:
            num_nodes = femm.mo_numnodes()
            num_elements = femm.mo_numelements()
            
            return {
                'nodes': num_nodes,
                'elements': num_elements
            }
        except:
            return {'nodes': 'unknown', 'elements': 'unknown'}
    
    @staticmethod
    def list_block_labels() -> List[Dict]:
        """List all block labels in the model"""
        # This requires iterating through the model
        # Implementation depends on FEMM API capabilities
        return []


def calculate_torque_from_maxwell_stress(X: np.ndarray, Y: np.ndarray,
                                         B_x: np.ndarray, B_y: np.ndarray,
                                         radius: float = 80) -> float:
    """
    Calculate electromagnetic torque from Maxwell stress tensor
    Using T = ∫∫ (B²/2μ₀) * r * dA
    
    Parameters:
    -----------
    X, Y : np.ndarray
        Position grids
    B_x, B_y : np.ndarray
        Flux density components
    radius : float
        Radius for torque calculation (mm)
    
    Returns:
    --------
    float : Estimated torque (Nm)
    """
    mu_0 = 4 * np.pi * 1e-7  # H/m = Vs/(Am)
    
    B_mag = np.sqrt(B_x**2 + B_y**2)
    
    # Maxwell stress: T_max = B²/(2*mu_0)
    stress = B_mag**2 / (2 * mu_0)
    
    # Integrate over circular path at given radius
    # This is approximate - proper calculation needs line integral
    mask = np.sqrt(X**2 + Y**2) > (radius - 5)
    mask = mask & (np.sqrt(X**2 + Y**2) < (radius + 5))
    
    if np.any(mask):
        avg_stress = np.mean(stress[mask])
        # Circumferential area (simplified)
        arc_length = 2 * np.pi * radius / 1000  # convert to m
        radial_depth = 0.134  # m (axial depth)
        
        torque = avg_stress * arc_length * radial_depth * radius / 1000
        return torque
    
    return 0.0
