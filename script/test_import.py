import os
import sys
print(f"Répertoire courant : {os.getcwd()}")
print(f"Fichiers dans le répertoire : {os.listdir('.')}")

try:
    import lunar_cycle_analyzer
    print("Import lunar_cycle_analyzer réussi!")
except ImportError as e:
    print(f"Erreur d'import lunar_cycle_analyzer: {e}")

try:
    import euromillions_predictor_optimizer
    print("Import euromillions_predictor_optimizer réussi!")
except ImportError as e:
    print(f"Erreur d'import euromillions_predictor_optimizer: {e}")
