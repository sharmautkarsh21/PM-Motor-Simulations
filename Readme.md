# IPM Motor Transient Simulation

Transient simulation of Interior Permanent Magnet (IPM) motors using FEMM with interactive live plotting.

## Features
- Balanced three-phase sine current excitation
- Rotor position control via sliding band
- Real-time flux linkage, back-EMF, torque, and current visualization
- Interactive web-based Dash dashboard
- CSV data export
- Static PNG plot export

## Installation

```bash
pip install -r requirements.txt
pip install -r requirements_dash.txt
```

## Usage

```bash
python IPM_transient.py
```

Open browser to: `http://127.0.0.1:8050`

## Configuration

Edit `IPM_transient.py`:
```python
RPM = 3000.0
CURRENT_RMS = 400.0
STEPS = 60
FULL_CYCLE = True
INITIAL_ROTOR_POSITION_DEG = 10.0
```

## Output
- Live interactive plots (browser-based)
- CSV simulation results with timestamp
- PNG exports (flux, back-EMF, torque, currents)

## Requirements
- FEMM (Finite Element Method Magnetics)
- Python 3.8+
- See `requirements.txt` and `requirements_dash.txt`