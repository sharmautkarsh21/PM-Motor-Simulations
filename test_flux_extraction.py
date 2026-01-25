"""Simple test to extract flux linkage from FEMM model."""

import femm

FEM_FILE = r"d:\\Knowledge Upgradation\\03 Github\\IPM Simulation\\TeslaModel3.fem"

try:
    print("Opening FEMM and loading model...")
    femm.openfemm(0)
    femm.opendocument(FEM_FILE)
    
    print("Meshing and solving...")
    femm.mi_createmesh()
    femm.mi_analyze(1)
    femm.mi_loadsolution()
    
    print("\nFlux linkage extraction:")
    print("=" * 60)
    
    # Extract flux linkage for each phase
    for phase in ["fase1", "fase2", "fase3"]:
        props = femm.mo_getcircuitproperties(phase)
        flux = props[2]
        print(f"{phase}: {flux:.6f} Wb")
    
    femm.mo_close()
    femm.closefemm()
    print("\nDone!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    femm.closefemm()
