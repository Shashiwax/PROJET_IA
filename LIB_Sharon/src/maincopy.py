from pathlib import Path
from pyngs.core import NGSpiceInstance
import time

# 1. Configuration des chemins PDK et Netlist [cite: 56, 70]
NETLIST_PATH = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")

def simulate_circuit(wn_value, wp_value):
    """
    Fonction atomique pour l'Agent RL : 
    Prend des variables, simule et renvoie les spécifications.
    """
    # Création d'une nouvelle instance pour éviter 'double free or corruption'
    inst = NGSpiceInstance()
    try:
        inst.load(NETLIST_PATH)
        
        # Application des variables de l'agent (W varie, L est fixe) 
        inst.set_parameter("wn", f"{wn_value}u")
        inst.set_parameter("wp", f"{wp_value}u")
        
        # Lancement de la simulation transitoire [cite: 191]
        inst.run()
        
        # Extraction des performances selon le document de caractérisation [cite: 82]
        results = {
            "area": inst.get_measure("cell_area"),   # Surface [cite: 83, 92]
            "delay": inst.get_measure("delay_fall"), # Délai [cite: 84, 95]
            "pstat": inst.get_measure("pstat_in1"),  # Conso statique [cite: 85, 107]
            "edyn": inst.get_measure("edyn_val")     # Conso dynamique [cite: 114]
        }
        return results
    finally:
        inst.stop() # Libération immédiate de la mémoire Ngspice

# --- TEST DU FLUX D'OPTIMISATION ---

print("--- Étape 1 : Caractérisation Initiale ---")
res_init = simulate_circuit(0.65, 1.0)
print(f"Délai initial : {res_init['delay']} s")

print("\n--- Étape 2 : Action de l'Agent (wn = 0.8u) ---")
res_new = simulate_circuit(0.8, 1.0)
print(f"Nouveau délai : {res_new['delay']} s")

# Vérification de l'amélioration
diff = res_init['delay'] - res_new['delay']
print(f"Amélioration du délai : {diff} s")
