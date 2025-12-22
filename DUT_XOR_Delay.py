from pathlib import Path
from pyngs.core import NGSpiceInstance
import numpy as np

netlist_path = Path("xor.cir")
inst = NGSpiceInstance()
inst.load(netlist_path)

# On lance la simulation
inst.run()

print(f"Temps de montée 1: {inst.get_measure('trise1')*10e9:.3f} ns")
print(f"Temps de montée 1: {inst.get_measure('trise2')*10e9:.3f} ns")
print(f"Temps de descente 1: {inst.get_measure('tfall1')*10e9:.3f} ns")
print(f"Temps de descente 2: {inst.get_measure('tfall2')*10e9:.3f} ns")

inst.stop()
