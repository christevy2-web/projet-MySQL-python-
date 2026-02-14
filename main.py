# main.py
import sys
import os

# Ajouter le dossier src au path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

# Note : DatabaseConnector n'est techniquement pas nécessaire pour AnalysesAvancees
# car cette classe gère sa propre connexion, mais on peut le laisser pour tester la connexion au début.
from database_connector import DatabaseConnector
from analyses_avancees import AnalysesAvancees

def main():
    print("🚀 Démarrage du projet d'analyse commerciale")
    
    # Test de connexion initial (optionnel mais utile pour vérifier)
    db = DatabaseConnector()
    db.connect()
    
    # Exécuter les analyses (SANS passer 'db' en argument)
    print("📊 Lancement des analyses...")
    analyses = AnalysesAvancees()
    analyses.executer_toutes_analyses()
    
    print("✅ Projet terminé avec succès!")

if __name__ == "__main__":
    main()