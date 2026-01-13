import pandas as pd

# --- Configuration ---
# Essayez de laisser input_filename tel quel si votre fichier est 'euromillions.csv'
# et que le script est dans le même dossier.
# Sinon, mettez le chemin complet vers votre fichier.
input_filename = 'euromillions.csv'
output_filename = 'euromillions_dates_converties.csv'

# Liste des mois en français pour la conversion
french_months = [
    "Janvier", "Février", "Mars", "Avril", "Mai", "Juin",
    "Juillet", "Août", "Septembre", "Octobre", "Novembre", "Décembre"
]

# --- Début du Script ---

def convert_csv_dates(input_file, output_file):
    """
    Charge un fichier CSV, détecte la colonne de date, convertit les dates
    au format français "JJ Mois AAAA", et sauvegarde le résultat.
    """
    print(f"--- Traitement du fichier : {input_file} ---")

    # 1. Tentative de lecture du fichier CSV (détection du séparateur)
    df = None
    try:
        print("1. Chargement du fichier CSV...")
        # Essai avec le point-virgule (courant en Europe)
        df = pd.read_csv(input_file, sep=';', dtype=str, keep_default_na=False)
        if df.shape[1] <= 1: # Si une seule colonne, ce n'est probablement pas le bon séparateur
            print("   Le séparateur ';' a produit une seule colonne. Essai avec ','...")
            df = pd.read_csv(input_file, sep=',', dtype=str, keep_default_na=False)
            if df.shape[1] <= 1:
                print("   Avertissement : Le fichier semble n'avoir qu'une seule colonne, même avec le séparateur ','.")
                print("   Veuillez vérifier la structure de votre fichier CSV.")
        print(f"   Fichier chargé. {df.shape[0]} lignes et {df.shape[1]} colonnes détectées.")
    except FileNotFoundError:
        print(f"   ERREUR : Le fichier '{input_file}' est introuvable.")
        print("   Veuillez vérifier que le nom du fichier est correct et qu'il se trouve au bon endroit.")
        return
    except Exception as e:
        print(f"   ERREUR inattendue lors de la lecture du fichier : {e}")
        return

    # 2. Détection de la colonne de dates
    print("\n2. Recherche de la colonne des dates...")
    date_col_name = None
    potential_date_cols = []

    # Priorité 1: Colonne nommée 'DATE' (ignorant la casse)
    for col in df.columns:
        if 'DATE' in col.upper():
            date_col_name = col
            print(f"   Colonne nommée 'DATE' trouvée : '{date_col_name}'.")
            break

    # Priorité 2: Détection automatique basée sur le format 'AAAA-MM-JJ'
    if not date_col_name:
        print("   Pas de colonne 'DATE' explicite. Tentative de détection par format (AAAA-MM-JJ)...")
        for col in df.columns:
            try:
                # Vérifier quelques valeurs non vides pour le format
                sample_values = df[col].dropna().unique()[:5] # Prendre jusqu'à 5 échantillons
                if not any(sample_values): # Si la colonne est vide ou que des échantillons sont vides
                    continue

                is_potential_date_col = True
                for val_str in sample_values:
                    if not (isinstance(val_str, str) and len(val_str) == 10 and val_str[4] == '-' and val_str[7] == '-'):
                        is_potential_date_col = False
                        break
                    # Essayer de convertir pour valider
                    pd.to_datetime(val_str, format='%Y-%m-%d', errors='raise')
                
                if is_potential_date_col:
                    potential_date_cols.append(col)
            except Exception:
                continue # Ce n'est pas une colonne de date au format attendu

        if len(potential_date_cols) == 1:
            date_col_name = potential_date_cols[0]
            print(f"   Colonne de dates détectée par format : '{date_col_name}'.")
        elif len(potential_date_cols) > 1:
            print(f"   ATTENTION : Plusieurs colonnes potentielles détectées par format : {potential_date_cols}.")
            print(f"   Utilisation de la première : '{potential_date_cols[0]}'. Modifiez le script si ce n'est pas la bonne.")
            date_col_name = potential_date_cols[0]

    if not date_col_name:
        print("   ERREUR : Impossible de trouver ou de déduire une colonne de dates (nom='DATE' ou format='AAAA-MM-JJ').")
        print("   Colonnes disponibles :", df.columns.tolist())
        print("   Veuillez vérifier votre fichier ou spécifier manuellement la colonne dans le script.")
        return

    # 3. Conversion des dates
    print(f"\n3. Conversion des dates dans la colonne '{date_col_name}'...")
    
    # Copie pour éviter SettingWithCopyWarning si date_col_name est une slice
    df_copy = df.copy()
    
    # Convertir la colonne en objets 'datetime', gérant les erreurs (met NaT si échec)
    # Il est important de spécifier le format si on est sûr, sinon pandas peut mal interpréter
    df_copy[date_col_name + '_datetime'] = pd.to_datetime(df_copy[date_col_name], format='%Y-%m-%d', errors='coerce')

    # Compter les dates qui n'ont pas pu être converties
    invalid_dates_count = df_copy[df_copy[date_col_name + '_datetime'].isna() & df_copy[date_col_name].notna() & (df_copy[date_col_name] != '')].shape[0]
    if invalid_dates_count > 0:
        print(f"   Avertissement : {invalid_dates_count} valeur(s) dans la colonne '{date_col_name}' n'ont pas pu être converties au format date AAAA-MM-JJ.")
        print("   Ces valeurs seront laissées telles quelles ou vides dans la colonne convertie.")

    # Fonction pour formater en français "JJ Mois AAAA"
    def format_date_to_french(date_obj):
        if pd.isna(date_obj):
            return "" # Retourne une chaîne vide si la date est NaT (Not a Time)
        return f"{date_obj.day} {french_months[date_obj.month - 1]} {date_obj.year}"

    # Appliquer la fonction de formatage
    df_copy[date_col_name + '_FR'] = df_copy[date_col_name + '_datetime'].apply(format_date_to_french)
    
    # Remplacer la colonne originale par la version formatée
    # Ou créer une nouvelle colonne si vous préférez garder l'originale
    original_col_index = df.columns.get_loc(date_col_name)
    df.drop(columns=[date_col_name], inplace=True)
    df.insert(original_col_index, date_col_name, df_copy[date_col_name + '_FR'])

    print("   Conversion terminée.")

    # 4. Sauvegarde du fichier modifié
    print(f"\n4. Sauvegarde du fichier modifié sous : {output_file}...")
    try:
        # Déterminer le séparateur utilisé pour la lecture pour le réutiliser
        used_separator = ';'
        try:
            # Petit test pour voir si le point-virgule était bien le séparateur
            # On ne peut pas se fier à df.shape[1] ici car df a été modifié
            # On relit une ligne du fichier original
            with open(input_file, 'r', encoding='utf-8') as f_test:
                first_line = f_test.readline()
                if ',' in first_line and not ';' in first_line:
                    used_separator = ','
                elif ';' in first_line:
                     used_separator = ';'
                # Si aucun des deux n'est dominant, on garde le point-virgule par défaut pour l'Europe
        except:
            pass # Garder le point-virgule par défaut

        df.to_csv(output_file, index=False, sep=used_separator)
        print(f"   Fichier sauvegardé avec succès ('{output_file}') avec le séparateur '{used_separator}'.")
    except Exception as e:
        print(f"   ERREUR lors de la sauvegarde du fichier : {e}")
        return

    print("\n--- ✅ Opération terminée ! ---")
    print("Aperçu des 5 premières lignes du fichier modifié :\n")
    print(df.head().to_string())


# --- Exécution du script ---
if __name__ == "__main__":
    convert_csv_dates(input_filename, output_filename)
