from pathlib import Path
from pyngs.core import NGSpiceInstance
import time

# 1. Chemin vers la netlist
netlist_path = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
inst = NGSpiceInstance()

try:
    print(f"Chargement de la netlist (Modèles Sky130A)...", flush=True)
    inst.load(netlist_path)
    print(f"Netlist chargée avec succès.", flush=True)

    # 2. Simulation initiale (Caractérisation de base)
    print("\n--- Simulation initiale en cours ---", flush=True)
    inst.run()

    # Extraction des specs initiales
    specs_init = {
        "Surface (cell_area)": inst.get_measure("cell_area"),
        "Délai (delay_fall)": inst.get_measure("delay_fall"),
        "Conso Statique": inst.get_measure("pstat_in1"),
        "Énergie Dynamique": inst.get_measure("edyn_val")
    }

    for name, val in specs_init.items():
        print(f"{name} : {val}", flush=True)

    # 3. MODIFICATION DES PARAMÈTRES
    # On augmente la largeur du NMOS (wn)
    new_wn = 0000000.8
    print(f"\n[AGENT] Action : Modification de wn à {new_wn}", flush=True)
    inst.set_parameter("wn", new_wn)

    # 4. RÉ-EXÉCUTION DE LA SIMULATION (Étape cruciale)
    # On reset et on relance pour que NGSpice recalcule les points de fonctionnement
    try:
        inst.cmd("reset")
    except AttributeError:
        pass
    
    print("Relance de la simulation avec les nouveaux paramètres...", flush=True)
    inst.run()

    # 5. EXTRACTION ET AFFICHAGE DES NOUVELLES SPECS
    print("\n--- Caractéristiques mises à jour ---", flush=True)
    
    specs_updated = {
        "Surface (cell_area)": inst.get_measure("cell_area"),
        "Délai (delay_fall)": inst.get_measure("delay_fall"),
        "Conso Statique": inst.get_measure("pstat_in1"),
        "Énergie Dynamique": inst.get_measure("edyn_val")
    }

    for key, value in specs_updated.items():
        print(f"{key} : {value}", flush=True)
    
    print("--------------------------------------\n")

    # Petit comparatif rapide pour le délai
    diff = specs_init["Délai (delay_fall)"] - specs_updated["Délai (delay_fall)"]
    print(f"Gain de rapidité constaté : {diff:.4e} s")

except Exception as e:
    print(f"\nErreur détectée : {e}", flush=True)

finally:
    inst.stop()
    print("\nSimulateur arrêté.", flush=True)
