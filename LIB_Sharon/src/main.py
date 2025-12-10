from pathlib import Path
from pyngs.core import NGSpiceInstance

# Path relatif comme dans leur exemple
netlist_path = Path("../netlists/rc.cir")

print(">>> Python script started")

# Créer l’instance ngspice
inst = NGSpiceInstance()

# Charger la netlist
inst.load(netlist_path)

# Lancer une première simulation
inst.run()

# Lister paramètres et mesures
print("Parameters:", inst.list_parameters())
print("Measures:", inst.list_measures())

# Liste de paires (R, C)
rc_values = [
    (1e3, 1e-6),   # 1 kΩ, 1 µF
    (1e4, 1e-6),   # 10 kΩ, 1 µF
    (1e3, 1e-7),   # 1 kΩ, 0.1 µF
    (5e3, 2e-6),   # 5 kΩ, 2 µF
]

for R in rc_values:
    pass
