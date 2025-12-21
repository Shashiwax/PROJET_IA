from pathlib import Path
from pyngs.core import NGSpiceInstance

netlist_path = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
inst = NGSpiceInstance()

def extraire_toutes_specs(instance):
    """Récupère l'ensemble des mesures définies dans la netlist."""
    return {
        "Surface (cell_area)": instance.get_measure("cell_area"),
        "Délai (delay_fall)": instance.get_measure("delay_fall"),
        "Conso Statique": instance.get_measure("pstat_in1"),
        "Énergie Dynamique": instance.get_measure("edyn_val")
    }

def afficher_tableau(titre, specs, wn, wp):
    """Affiche les résultats proprement."""
    print(f"\n--- {titre} (wn={wn}, wp={wp}) ---")
    for nom, val in specs.items():
        # Formatage scientifique pour la lisibilité
        print(f"  {nom:<20} : {val:.4e}")
    print("-" * (len(titre) + 20))

try:
    inst.load(netlist_path)

    # 1. ÉTAT INITIAL
    inst.run()
    wn_init = inst.get_parameter("wn")
    wp_init = inst.get_parameter("wp")
    specs_initiales = extraire_toutes_specs(inst)
    
    afficher_tableau("SPÉCIFICATIONS INITIALES", specs_initiales, wn_init, wp_init)

    # 2. MODIFICATION (Exemple : Augmentation de la largeur du NMOS)
    target_wn = 0.8  # On passe à 800nm
    print(f"\n[AGENT] Action : Modification de wn de {wn_init} à {target_wn}...")
    inst.set_parameter("wn", target_wn)

    # 3. RELANCE DE LA SIMULATION
    try:
        inst.cmd("reset")
    except:
        pass
    inst.run()

    # 4. ÉTAT FINAL ET COMPARAISON
    specs_finales = extraire_toutes_specs(inst)
    afficher_tableau("SPÉCIFICATIONS FINALES", specs_finales, target_wn, wp_init)

    # 5. RÉSUMÉ DES VARIATIONS (%)
    print("\n--- Analyse de l'impact ---")
    for key in specs_initiales:
        v_init = specs_initiales[key]
        v_final = specs_finales[key]
        if v_init != 0:
            variation = ((v_final - v_init) / v_init) * 100
            print(f"  {key:<20} : {variation:+.2f} %")

except Exception as e:
    print(f"\nErreur : {e}")
finally:
    inst.stop()
