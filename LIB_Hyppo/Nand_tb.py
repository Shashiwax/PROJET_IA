from pathlib import Path
from pyngs.core import NGSpiceInstance
import numpy as np
import os

class NANDTestbench:
    def __init__(self, netlist_path: Path):
        # CORRECTION 1 : Conversion en chemin absolu et en string
        # NGSpice a souvent du mal avec les chemins relatifs ou les objets Path
        self.netlist_path = str(Path(netlist_path).resolve())
        
        if not os.path.exists(self.netlist_path):
            raise FileNotFoundError(f"Le fichier netlist n'existe pas : {self.netlist_path}")

        self.inst = NGSpiceInstance()
        
        # Chargement du circuit
        try:
            self.inst.load(self.netlist_path)
            print(f"Netlist chargée avec succès : {self.netlist_path}")
        except Exception as e:
            print(f"Erreur lors du chargement de la netlist : {e}")
            raise

    def run_simulation(self, w_values : np.ndarray):
        """Lance la simulation NGSpice."""
        self.inst.set_parameter('wn_val', w_values[0])
        self.inst.set_parameter('wp_val', w_values[1])
        self.inst.run()

    def get_delay_measurements(self):
        """Récupère les mesures de délai de la NAND."""
        delays = {
            "trise1": self.inst.get_measure('trise1') * 1e9,  # Convert to ns
            "trise2": self.inst.get_measure('trise2') * 1e9,
            "tfall1": self.inst.get_measure('tfall1') * 1e9,
            "tfall2": self.inst.get_measure('tfall2') * 1e9,
        }
        assert all(value is not None for value in delays.values()), "One or more delay measurements not found in the simulation results."
        return delays

    def get_area_measurement(self):
        """Récupère la mesure de surface de la NAND."""
        area = self.inst.get_measure('cell_area')  # Assuming 'area' is a valid measure
        assert area is not None, "Area measurement not found in the simulation results."
        return area
    
    def get_power_measurement(self):
        """Récupère la mesure de puissance de la NAND."""
        power = self.inst.get_measure('power')  # Assuming 'power' is a valid measure
        assert power is not None, "Power measurement not found in the simulation results."
        return power
    
    def get_all_measurements(self):
        """Récupère toutes les mesures de la NAND."""
        measurements = {}
        measurements['delays'] = self.get_delay_measurements()
        measurements['area'] = self.get_area_measurement()
        return measurements

    def stop_simulation(self):
        """Arrête la simulation NGSpice."""
        self.inst.stop()

"""if __name__ == "__main__":
    nand_tb = NANDTestbench(Path("LIB_Hyppo/nand.cir"))

    # Example transistor widths in micrometers
    w_values = ["1.0e+06u", "2.0e+06u"]  # [W_nmos, W_pmos]

    nand_tb.run_simulation(w_values)
    measurements = nand_tb.get_all_measurements()
    print("NAND Measurements:", measurements)
    nand_tb.stop_simulation()"""