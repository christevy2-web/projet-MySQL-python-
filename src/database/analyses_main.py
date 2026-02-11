# analyses_main.py
from analyses_avancees import AnalysesAvancees

def main():
    """Point d'entrée principal pour les analyses avancées"""
    print("=" * 60)
    print("ANALYSES AVANCÉES - PROJET SQL/PYTHON")
    print("=" * 60)
    
    # Créer l'instance
    analyses = AnalysesAvancees()
    
    # Exécuter toutes les analyses
    print("\n🚀 Démarrage des analyses avancées...")
    results = analyses.executer_toutes_analyses()
    
    print("\n" + "=" * 60)
    print("✅ TOUTES LES ANALYSES SONT TERMINÉES !")
    print("=" * 60)
    print("\n📁 Les résultats sont disponibles dans le dossier 'analyses_avancees/'")
    print("📊 Fichiers générés:")
    print("   • Graphiques PNG pour chaque analyse")
    print("   • Fichiers CSV avec les données")
    print("   • Rapports textuels de synthèse")
    print("\n🎯 Vous pouvez maintenant analyser les résultats !")

if __name__ == "__main__":
    main()