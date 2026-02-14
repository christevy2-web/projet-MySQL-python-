# Notebook_jupyter.py
import sys
import os

# Si vos fichiers sont dans le même dossier, cette ligne n'est souvent pas nécessaire,
# mais on la garde par sécurité si vous utilisez une structure de dossiers spécifique.
sys.path.append('../src')

# Importation des modules
# 1. On garde DatabaseConnector si vous voulez tester la connexion séparément
from database_connector import DatabaseConnector

# 2. CORRECTION ICI : Le fichier s'appelle 'analyses_avancees', pas 'analyses'
from analyses_avancees import AnalysesAvancees

# Test de connexion (optionnel)
print("Test de la connexion...")
db = DatabaseConnector()
db.connect()

# Création de l'instance d'analyse
# 3. CORRECTION ICI : On retire '(db)' car la classe gère sa propre connexion
print("Initialisation des analyses...")
analyses = AnalysesAvancees()

# Exécution d'une analyse (RFM)
print("Lancement de l'analyse RFM...")
resultats_rfm = analyses.analyse_rfm()

print("Terminé !")