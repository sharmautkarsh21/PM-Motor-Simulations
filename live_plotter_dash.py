"""Interactive live plotting using Plotly Dash for FEMM simulation.

Opens a web-based dashboard in your browser with real-time updates,
full zoom/pan/resize capabilities during simulation.
"""

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
import threading
import time
import logging

print(f"Loading live_plotter_dash from: {__file__}")

try:
    from dash import Dash, dcc, html
    from dash.dependencies import Input, Output
    import plotly.graph_objs as go
    import plotly.io as pio
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False


@dataclass
class TimeStepResult:
    """Matches the main simulation result structure."""
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


class DashLivePlotter:
    """Interactive web-based live plotter using Plotly Dash."""
    
    def __init__(self, enable_live_plot: bool = True, update_interval_ms: int = 100):
        self.enable_live_plot = enable_live_plot and DASH_AVAILABLE
        self.update_interval_ms = update_interval_ms
        self.app = None
        self.server_thread = None
        self.data = {
            'elec_angles': [],
            'mech_angles': [],
            'time_vals': [],
            'flux_a': [],
            'flux_b': [],
            'flux_c': [],
            'backmf_a': [],
            'backmf_b': [],
            'backmf_c': [],
            'torques': [],
            'ia_vals': [],
            'ib_vals': [],
            'ic_vals': [],
        }
        self.running = False
        self.lock = threading.Lock()
        
        if self.enable_live_plot:
            self._initialize_dash()
    
    def _initialize_dash(self) -> None:
        """Initialize Dash app with layout."""
        self.app = Dash(__name__)
        
        self.app.layout = html.Div([
            html.H1("IPM Transient Simulation - Live Results", 
                   style={'textAlign': 'center', 'color': '#2c3e50'}),
            
            dcc.Graph(id='flux-plot', style={'height': '300px'}),
            dcc.Graph(id='backmf-plot', style={'height': '300px'}),
            dcc.Graph(id='torque-plot', style={'height': '300px'}),
            dcc.Graph(id='current-plot', style={'height': '300px'}),
            
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval_ms,
                n_intervals=0
            )
        ], style={'padding': '20px'})
        
        @self.app.callback(
            [Output('flux-plot', 'figure'),
             Output('backmf-plot', 'figure'),
             Output('torque-plot', 'figure'),
             Output('current-plot', 'figure')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_plots(n):
            with self.lock:
                elec_angles = self.data['elec_angles'].copy()
                time_vals = self.data['time_vals'].copy()
                flux_a = self.data['flux_a'].copy()
                flux_b = self.data['flux_b'].copy()
                flux_c = self.data['flux_c'].copy()
                backmf_a = self.data['backmf_a'].copy()
                backmf_b = self.data['backmf_b'].copy()
                backmf_c = self.data['backmf_c'].copy()
                torques = self.data['torques'].copy()
                ia_vals = self.data['ia_vals'].copy()
                ib_vals = self.data['ib_vals'].copy()
                ic_vals = self.data['ic_vals'].copy()
            
            # Skip first point for back-emf (derivative starts at second point)
            time_backmf_ms = [t * 1000 for t in time_vals[1:]] if len(time_vals) > 1 else []
            
            # Flux linkage plot
            flux_fig = go.Figure()
            if len(elec_angles) > 0:
                flux_fig.add_trace(go.Scatter(
                    x=elec_angles, y=flux_a,
                    mode='lines', name='Phase A',
                    line=dict(color='red', width=2),
                    connectgaps=False
                ))
                flux_fig.add_trace(go.Scatter(
                    x=elec_angles, y=flux_b,
                    mode='lines', name='Phase B',
                    line=dict(color='green', width=2),
                    connectgaps=False
                ))
                flux_fig.add_trace(go.Scatter(
                    x=elec_angles, y=flux_c,
                    mode='lines', name='Phase C',
                    line=dict(color='blue', width=2),
                    connectgaps=False
                ))
            
            # Set dynamic x-axis range
            x_range_flux = None
            if len(elec_angles) > 0:
                x_min = min(elec_angles)
                x_max = max(elec_angles)
                x_range_flux = [x_min, x_max + (x_max - x_min) * 0.02]  # Add 2% margin
            
            flux_fig.update_layout(
                title='Flux Linkage vs Electrical Rotor Position',
                xaxis_title='Electrical Rotor Position (°)',
                yaxis_title='Flux Linkage (Wb)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(x=1.02, y=1),
                margin=dict(l=50, r=150, t=50, b=50),
                xaxis_range=x_range_flux
            )
            
            # Back-emf plot
            backmf_fig = go.Figure()
            if len(time_backmf_ms) > 0:
                backmf_fig.add_trace(go.Scatter(
                    x=time_backmf_ms, y=backmf_a,
                    mode='lines', name='Phase A',
                    line=dict(color='red', width=2),
                    connectgaps=False
                ))
                backmf_fig.add_trace(go.Scatter(
                    x=time_backmf_ms, y=backmf_b,
                    mode='lines', name='Phase B',
                    line=dict(color='green', width=2),
                    connectgaps=False
                ))
                backmf_fig.add_trace(go.Scatter(
                    x=time_backmf_ms, y=backmf_c,
                    mode='lines', name='Phase C',
                    line=dict(color='blue', width=2),
                    connectgaps=False
                ))
            
            # Set dynamic x-axis range for back-emf (time-based)
            x_range_backmf = None
            if len(time_backmf_ms) > 0:
                x_min = min(time_backmf_ms)
                x_max = max(time_backmf_ms)
                x_range_backmf = [x_min, x_max + (x_max - x_min) * 0.02]  # Add 2% margin
            
            backmf_fig.update_layout(
                title='Back-emf vs Time',
                xaxis_title='Time (ms)',
                yaxis_title='Back-emf (V)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(x=1.02, y=1),
                margin=dict(l=50, r=150, t=50, b=50),
                xaxis_range=x_range_backmf
            )
            
            # Torque plot
            torque_fig = go.Figure()
            if len(elec_angles) > 0:
                torque_fig.add_trace(go.Scatter(
                    x=elec_angles, y=torques,
                    mode='lines', name='Torque',
                    line=dict(color='magenta', width=2),
                    connectgaps=False
                ))
            
            # Set dynamic x-axis range for torque
            x_range_torque = None
            if len(elec_angles) > 0:
                x_min = min(elec_angles)
                x_max = max(elec_angles)
                x_range_torque = [x_min, x_max + (x_max - x_min) * 0.02]  # Add 2% margin
            
            torque_fig.update_layout(
                title='Electromagnetic Torque vs Electrical Rotor Position',
                xaxis_title='Electrical Rotor Position (°)',
                yaxis_title='Torque (Nm)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(x=1.02, y=1),
                margin=dict(l=50, r=150, t=50, b=50),
                xaxis_range=x_range_torque
            )
            
            # Current plot
            current_fig = go.Figure()
            if len(elec_angles) > 0:
                current_fig.add_trace(go.Scatter(
                    x=elec_angles, y=ia_vals,
                    mode='lines', name='Ia',
                    line=dict(color='red', width=2),
                    connectgaps=False
                ))
                current_fig.add_trace(go.Scatter(
                    x=elec_angles, y=ib_vals,
                    mode='lines', name='Ib',
                    line=dict(color='green', width=2),
                    connectgaps=False
                ))
                current_fig.add_trace(go.Scatter(
                    x=elec_angles, y=ic_vals,
                    mode='lines', name='Ic',
                    line=dict(color='blue', width=2),
                    connectgaps=False
                ))
            
            # Set dynamic x-axis range for currents
            x_range_current = None
            if len(elec_angles) > 0:
                x_min = min(elec_angles)
                x_max = max(elec_angles)
                x_range_current = [x_min, x_max + (x_max - x_min) * 0.02]  # Add 2% margin
            
            current_fig.update_layout(
                title='Three-Phase Currents vs Electrical Rotor Position',
                xaxis_title='Electrical Rotor Position (°)',
                yaxis_title='Current (A)',
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(x=1.02, y=1),
                margin=dict(l=50, r=150, t=50, b=50),
                xaxis_range=x_range_current
            )
            
            return flux_fig, backmf_fig, torque_fig, current_fig
    
    def start_server(self, port: int = 8050) -> None:
        """Start Dash server in background thread."""
        if not self.enable_live_plot or self.app is None:
            return
        
        # Suppress Dash/Flask logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        
        def run_server():
            self.app.run(debug=False, port=port, use_reloader=False)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        print(f"\n{'='*70}")
        print(f"Live plot server starting...")
        print(f"Open your browser to: http://127.0.0.1:{port}")
        print(f"The plot will update automatically as the simulation runs.")
        print(f"You can zoom, pan, and resize the window freely!")
        print(f"{'='*70}\n")
        
        time.sleep(2)  # Give server time to start
    
    def update(self, result: TimeStepResult) -> None:
        """Update plots with new result data during simulation."""
        if not self.enable_live_plot:
            return
        
        with self.lock:
            elec_angle = result.electrical_angle_deg
            mech_angle = result.mechanical_angle_deg
            time_s = result.time_s
            
            self.data['elec_angles'].append(elec_angle)
            self.data['mech_angles'].append(mech_angle)
            self.data['time_vals'].append(time_s)
            
            # Store flux linkage
            flux_a = result.flux_linkage_a if result.flux_linkage_a is not None else 0
            flux_b = result.flux_linkage_b if result.flux_linkage_b is not None else 0
            flux_c = result.flux_linkage_c if result.flux_linkage_c is not None else 0
            self.data['flux_a'].append(flux_a)
            self.data['flux_b'].append(flux_b)
            self.data['flux_c'].append(flux_c)
            
            # Add pre-calculated back-EMF (skip first point which is None)
            if result.backmf_a is not None:
                self.data['backmf_a'].append(result.backmf_a)
                self.data['backmf_b'].append(result.backmf_b)
                self.data['backmf_c'].append(result.backmf_c)
            
            # Add torque and current data
            torque = result.torque_nm if result.torque_nm is not None else 0
            self.data['torques'].append(torque)
            self.data['ia_vals'].append(result.ia)
            self.data['ib_vals'].append(result.ib)
            self.data['ic_vals'].append(result.ic)
    
    def _create_plots(self):
        """Create final plot figures for saving."""
        with self.lock:
            elec_angles = self.data['elec_angles'].copy()
            time_vals = self.data['time_vals'].copy()
            flux_a = self.data['flux_a'].copy()
            flux_b = self.data['flux_b'].copy()
            flux_c = self.data['flux_c'].copy()
            backmf_a = self.data['backmf_a'].copy()
            backmf_b = self.data['backmf_b'].copy()
            backmf_c = self.data['backmf_c'].copy()
            torques = self.data['torques'].copy()
            ia_vals = self.data['ia_vals'].copy()
            ib_vals = self.data['ib_vals'].copy()
            ic_vals = self.data['ic_vals'].copy()
        
        time_backmf_ms = [t * 1000 for t in time_vals[1:]] if len(time_vals) > 1 else []
        
        # Flux linkage plot
        flux_fig = go.Figure()
        if len(elec_angles) > 0:
            flux_fig.add_trace(go.Scatter(
                x=elec_angles, y=flux_a,
                mode='lines', name='Phase A',
                line=dict(color='red', width=2),
                connectgaps=False
            ))
            flux_fig.add_trace(go.Scatter(
                x=elec_angles, y=flux_b,
                mode='lines', name='Phase B',
                line=dict(color='green', width=2),
                connectgaps=False
            ))
            flux_fig.add_trace(go.Scatter(
                x=elec_angles, y=flux_c,
                mode='lines', name='Phase C',
                line=dict(color='blue', width=2),
                connectgaps=False
            ))
        
        flux_fig.update_layout(
            title='Flux Linkage vs Electrical Rotor Position',
            xaxis_title='Electrical Rotor Position (°)',
            yaxis_title='Flux Linkage (Wb)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(l=50, r=150, t=50, b=50),
            width=1400,
            height=400
        )
        flux_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        flux_fig.add_vline(x=360, line_dash="dash", line_color="gray", opacity=0.3)
        
        # Back-emf plot
        backmf_fig = go.Figure()
        if len(time_backmf_ms) > 0:
            backmf_fig.add_trace(go.Scatter(
                x=time_backmf_ms, y=backmf_a,
                mode='lines', name='Phase A',
                line=dict(color='red', width=2),
                connectgaps=False
            ))
            backmf_fig.add_trace(go.Scatter(
                x=time_backmf_ms, y=backmf_b,
                mode='lines', name='Phase B',
                line=dict(color='green', width=2),
                connectgaps=False
            ))
            backmf_fig.add_trace(go.Scatter(
                x=time_backmf_ms, y=backmf_c,
                mode='lines', name='Phase C',
                line=dict(color='blue', width=2),
                connectgaps=False
            ))
        
        backmf_fig.update_layout(
            title='Back-emf vs Time',
            xaxis_title='Time (ms)',
            yaxis_title='Back-emf (V)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(l=50, r=150, t=50, b=50),
            width=1400,
            height=400
        )
        
        # Torque plot
        torque_fig = go.Figure()
        if len(elec_angles) > 0:
            torque_fig.add_trace(go.Scatter(
                x=elec_angles, y=torques,
                mode='lines', name='Torque',
                line=dict(color='magenta', width=2),
                connectgaps=False
            ))
        
        torque_fig.update_layout(
            title='Electromagnetic Torque vs Electrical Rotor Position',
            xaxis_title='Electrical Rotor Position (°)',
            yaxis_title='Torque (Nm)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(l=50, r=150, t=50, b=50),
            width=1400,
            height=400
        )
        torque_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        torque_fig.add_vline(x=360, line_dash="dash", line_color="gray", opacity=0.3)
        
        # Current plot
        current_fig = go.Figure()
        if len(elec_angles) > 0:
            current_fig.add_trace(go.Scatter(
                x=elec_angles, y=ia_vals,
                mode='lines', name='Ia',
                line=dict(color='red', width=2),
                connectgaps=False
            ))
            current_fig.add_trace(go.Scatter(
                x=elec_angles, y=ib_vals,
                mode='lines', name='Ib',
                line=dict(color='green', width=2),
                connectgaps=False
            ))
            current_fig.add_trace(go.Scatter(
                x=elec_angles, y=ic_vals,
                mode='lines', name='Ic',
                line=dict(color='blue', width=2),
                connectgaps=False
            ))
        
        current_fig.update_layout(
            title='Three-Phase Currents vs Electrical Rotor Position',
            xaxis_title='Electrical Rotor Position (°)',
            yaxis_title='Current (A)',
            hovermode='x unified',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=1.02, y=1),
            margin=dict(l=50, r=150, t=50, b=50),
            width=1400,
            height=400
        )
        current_fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.3)
        current_fig.add_vline(x=360, line_dash="dash", line_color="gray", opacity=0.3)
        
        return flux_fig, backmf_fig, torque_fig, current_fig
    
    def finalize(self) -> None:
        """Save plots and keep server running."""
        if self.enable_live_plot and self.running:
            with self.lock:
                elec_min = min(self.data['elec_angles']) if self.data['elec_angles'] else 0
                elec_max = max(self.data['elec_angles']) if self.data['elec_angles'] else 0
                mech_min = min(self.data['mech_angles']) if self.data['mech_angles'] else 0
                mech_max = max(self.data['mech_angles']) if self.data['mech_angles'] else 0
            
            # Create and save plots
            try:
                print("\nSaving plots to PNG files...")
                flux_fig, backmf_fig, torque_fig, current_fig = self._create_plots()
                
                pio.write_image(flux_fig, "flux_linkage.png", width=1400, height=400)
                print("  ✓ flux_linkage.png")
                
                pio.write_image(backmf_fig, "back_emf.png", width=1400, height=400)
                print("  ✓ back_emf.png")
                
                pio.write_image(torque_fig, "torque.png", width=1400, height=400)
                print("  ✓ torque.png")
                
                pio.write_image(current_fig, "currents.png", width=1400, height=400)
                print("  ✓ currents.png")
                
                print("\nAll plots saved successfully!")
            except Exception as e:
                print(f"\nWarning: Could not save plots as PNG: {e}")
                print("You may need to install: pip install kaleido")
            
            print(f"\n{'='*70}")
            print(f"Simulation complete!")
            print(f"Electrical angle range: {elec_min:.2f}° to {elec_max:.2f}°")
            print(f"Mechanical angle range: {mech_min:.2f}° to {mech_max:.2f}°")
            print(f"\nThe live plot is still running in your browser.")
            print(f"Press Ctrl+C in the terminal to stop the server.")
            print(f"{'='*70}\n")
            
            # Keep main thread alive so server continues running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down live plot server...")
