from pathlib import Path
from pyngs.core import NGSpiceInstance
import os
from loguru import logger

class NANDTestbench:
    def __init__(self, delay_netlist_path: str, power_netlist_path: str):
        self.delay_netlist = str(Path(delay_netlist_path).resolve())
        self.power_netlist = str(Path(power_netlist_path).resolve())
        
        # --- OPTIMISATION : DOUBLE BUFFERING ---
        # On charge les deux simulations EN MÉMOIRE dès le début.
        # On ne fera plus jamais de .load() ensuite.
        
        logger.info("Chargement des instances NGSpice (Delay & Power)...")
        
        # Instance 1 : Délai
        self.inst_delay = NGSpiceInstance()
        if not os.path.exists(self.delay_netlist):
            raise FileNotFoundError(f"Netlist introuvable: {self.delay_netlist}")
        self.inst_delay.load(self.delay_netlist)
        
        # Instance 2 : Puissance
        self.inst_power = NGSpiceInstance()
        if not os.path.exists(self.power_netlist):
            raise FileNotFoundError(f"Netlist introuvable: {self.power_netlist}")
        self.inst_power.load(self.power_netlist)
        
        logger.info("Instances chargées et prêtes.")

    def run_specific_simulation(self, mode, w_values: list):
        """
        Sélectionne l'instance déjà chargée, met à jour W, et relance.
        """
        # 1. Sélection de l'instance active
        if mode == "DELAY":
            inst = self.inst_delay
        elif mode == "POWER":
            inst = self.inst_power
        else:
            raise ValueError(f"Mode inconnu : {mode}")

        # 2. Mise à jour des paramètres (Rapide, en mémoire)
        # w_values = ["1.0e-6", "2.0e-6"]
        inst.set_parameter('wn_val', w_values[0])
        inst.set_parameter('wp_val', w_values[1])

        # 3. Exécution
        # Comme la netlist est déjà chargée, run() devrait juste relancer l'analyse
        inst.run()

    def get_measurements(self, mode):
        """Récupère les mesures sur l'instance concernée."""
        measurements = {}
        
        if mode == "DELAY":
            inst = self.inst_delay
            try:
                # Area
                area = inst.get_measure('cell_area')
                if area is not None: 
                    measurements['area'] = area

                # Delays
                delays = {}
                for key in ['trise1', 'trise2', 'tfall1', 'tfall2']:
                    val = inst.get_measure(key)
                    if val is not None:
                        delays[key] = val * 1e9 # ns
                
                if delays:
                    measurements['delays'] = delays
            except Exception:
                pass

        elif mode == "POWER":
            inst = self.inst_power
            try:
                p_static = inst.get_measure('static_power')
                p_dyn = inst.get_measure('dyn_power')
                
                p_data = {}
                if p_static is not None: p_data['static'] = float(p_static)
                if p_dyn is not None: p_data['dynamic'] = float(p_dyn)
                
                if p_data:
                    measurements['power'] = p_data
            except Exception:
                pass

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