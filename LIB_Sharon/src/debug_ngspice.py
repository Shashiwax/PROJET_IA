import ctypes

# Chemin exact que nous avons vérifié
path = "/usr/lib/x86_64-linux-gnu/libngspice.so"

print(f"Tentative de chargement de : {path}")

try:
    # On essaie de charger la librairie manuellement
    lib = ctypes.CDLL(path)
    print("✅ SUCCÈS : La librairie s'est chargée correctement !")
except OSError as e:
    print(f"❌ ÉCHEC : {e}")
