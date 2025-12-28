from pathlib import Path
from pyngs.core import NGSpiceInstance
import os
import numpy as np
from loguru import logger

class NANDTestbench:
    def __init__(self, delay_netlist_path: str, power_netlist_path: str):
        # Conversion des chemins
        self.delay_netlist = str(Path(delay_netlist_path).resolve())
        self.power_netlist = str(Path(power_netlist_path).resolve())
        
        # --- CORRECTION ICI : Initialisation de la variable manquante ---
        self.active_instance = None 
        # ----------------------------------------------------------------

        logger.info("Chargement initial des instances NGSpice...")
        
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
        
        logger.info("Instances chargées.")

    def run_specific_simulation(self, mode, w_values: list):
        """
        Sélectionne l'instance, met à jour W, et relance.
        """
        # 1. Sélection de l'instance active et mise à jour de la référence
        if mode == "DELAY":
            self.active_instance = self.inst_delay
        elif mode == "POWER":
            self.active_instance = self.inst_power
        else:
            raise ValueError(f"Mode inconnu : {mode}")

        # 2. Mise à jour des paramètres (Rapide, en mémoire)
        # w_values = ["1.0e-6", "2.0e-6"]
        self.active_instance.set_parameter('wn_val', w_values[0])
        self.active_instance.set_parameter('wp_val', w_values[1])

        # 3. Exécution
        self.active_instance.run()

    def check_simulation_health(self):
        """
        Vérifie si la dernière simulation a convergé correctement.
        """
        # Si aucune simulation n'a encore tourné, c'est 'sain' par défaut ou on attend
        if self.active_instance is None:
            return True

        try:
            # On vérifie si le vecteur temps existe et est complet
            time_vec = self.active_instance.get_vector('time')
            
            if time_vec is None or len(time_vec) == 0:
                return False
            
            # Vérification de la durée finale
            last_time = time_vec[-1]
            if last_time < 1e-12: # Seuil minimal de sécurité
                return False
                
            return True
            
        except Exception:
            return False

    def get_measurements(self, mode):
        """Récupère les mesures sur l'instance concernée."""
        measurements = {}
        
        # On utilise l'instance correspondant au mode demandé
        if mode == "DELAY":
            inst = self.inst_delay
        elif mode == "POWER":
            inst = self.inst_power
        else:
            return {}

        try:
            # --- Mesures Delay & Area ---
            if mode == "DELAY":
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

            # --- Mesures Power ---
            elif mode == "POWER":
                p_static = inst.get_measure('static_power')
                p_dyn = inst.get_measure('dyn_power')
                
                p_data = {}
                if p_static is not None: p_data['static'] = float(p_static)
                if p_dyn is not None: p_data['dynamic'] = float(p_dyn)
                
                if p_data:
                    measurements['power'] = p_data

        except Exception as e:
            logger.warning(f"Erreur lecture mesures ({mode}): {e}")
            pass

        return measurements