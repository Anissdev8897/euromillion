from euromillions_analyzer import EuromillionsAdvancedAnalyzer
from incremental_learning import EuromillionsIncrementalLearning

# Configuration de base
config = {
    "csv_file": "tirage_euromillions.csv",
    "output_dir": "resultats_euromillions"
}

# Créer l'analyseur
analyzer = EuromillionsAdvancedAnalyzer(config)
analyzer.load_data()

# Créer l'apprentissage incrémental
incremental_learner = EuromillionsIncrementalLearning(analyzer)

# Nouveau tirage à ajouter (exemple)
nouveau_tirage = {
    "N1": 12, "N2": 25, "N3": 37, "N4": 42, "N5": 50,
    "E1": 3, "E2": 9
}

# Mettre à jour les modèles avec le nouveau tirage
incremental_learner.update_models(nouveau_tirage)
