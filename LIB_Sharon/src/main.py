from pathlib import Path
from pyngs.core import NGSpiceInstance
import math

# --- CONFIGURATION TECHNOLOGIQUE (Sky130 HD) ---
H_CELL = 2.72      # Hauteur fixe (8 Grids)
X_GRID = 0.46      # Pas horizontal
WN_MAX = 0.65      # Limite physique NMOS
WP_MAX = 1.00      # Limite physique PMOS
W_FIXED = 0.16     # Estimation des marges/contacts


def get_all_metrics(instance):
    pstat_in0_w = instance.get_measure("pstat_in0")
    pstat_in1_w = instance.get_measure("pstat_in1")
    pstat_vdd_in0_w = instance.get_measure("pstat_vdd_in0")
    pstat_vdd_in1_w = instance.get_measure("pstat_vdd_in1")
    pstat_wc_w = max(instance.get_measure("pstat_vdd_in0"), instance.get_measure("pstat_vdd_in1"))
    return {
        # On multiplie par 1e6 si tes unités SPICE sont déjà en "micros" 
        # ou 1e12 si elles sont en mètres.
        "Surface Réelle (µm²)": instance.get_measure("cell_area"), #x1
        "Surface Transistors (µm²)": instance.get_measure("active_area"), #x1
        "Délai Fall (ps)": instance.get_measure("delay_fall") * 1e12,
        "Délai Rise (ps)": instance.get_measure("delay_rise") * 1e12,
        # Total static power (VDD+VIN as defined in your .meas)
        "Conso Statique IN1 (pW)": pstat_in1_w * 1e12,
        "Conso Statique IN0 (µW)": pstat_in0_w * 1e6,

        # VDD-only static power (cleaner for analysis/RL)
        "Conso Statique VDD IN1 (pW)": pstat_vdd_in1_w * 1e12,
        "Conso Statique VDD IN0 (µW)": pstat_vdd_in0_w * 1e6,
        "Énergie Dyn (fJ)": instance.get_measure("edyn_val") * 1e15,
        "Conso Statique WC (µW)": pstat_wc_w * 1e6

    }



# 1. Initialisation
netlist_path = Path("/home/sharo/PROJET_IA/LIB_Sharon/netlists/inv.cir")
inst = NGSpiceInstance()

try:
    inst.load(netlist_path)
    
    # --- ÉTAPE A : ÉTAT INITIAL ---
    inst.run()
    wn_init = inst.get_parameter("wn")
    wp_init = inst.get_parameter("wp")
    metrics_init = get_all_metrics(inst)
    
    print(f"--- ÉTAT INITIAL (wn={wn_init}, wp={wp_init}) ---")
    for k, v in metrics_init.items():
        print(f"  {k:<25} : {v:.4f}")

    # --- ÉTAPE B : MODIFICATION MANUELLE ---
    # Changez ces valeurs pour tester l'impact (ex: wn=0.5, wp=0.8)
    new_wn = 0.65 
    new_wp = 1.20 # Testons un dépassement de limite pour voir l'impact surface
    
    print(f"\n[MANUEL] Modification : wn -> {new_wn}, wp -> {new_wp}")
    
    # Vérification des limites du tableau
    if new_wn > WN_MAX or new_wp > WP_MAX:
        print(f"  /!\\ ALERTE : Valeurs supérieures au High Density Standard ({WN_MAX}/{WP_MAX})")

    inst.set_parameter("wn", new_wn)
    inst.set_parameter("wp", new_wp)

    # --- ÉTAPE C : RELANCE ET ANALYSE ---
    try:
        inst.cmd("reset")
    except AttributeError:
        pass
    
    inst.run()
    metrics_final = get_all_metrics(inst)

    print(f"\n--- ÉTAT MODIFIÉ (wn={new_wn}, wp={new_wp}) ---")
    for k, v in metrics_final.items():
        val_init = metrics_init[k]
        diff = ((v - val_init) / val_init * 100) if val_init != 0 else 0
        print(f"  {k:<25} : {v:.4f} ({diff:+.2f} %)")

    # Calcul du nombre de sites théorique pour vérification
    largeur_estim = max(new_wn, new_wp) + W_FIXED
    sites = math.ceil(largeur_estim / X_GRID)
    print(f"\nEstimation physique : {sites} sites de {X_GRID}µm occupés.")

except Exception as e:
    print(f"\nErreur : {e}")
finally:
    inst.stop()
