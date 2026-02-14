# analyses_avancees.py

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from scipy import stats
import warnings
import sys
import os
from database_visualizer import MySQLVisualizer

# Supprimez l'import problématique et utilisez directement le chemin relatif
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings('ignore')

class AnalysesAvancees:
    def __init__(self):
        """Initialisation avec une connexion directe à la base"""
        from database_visualizer import MySQLVisualizer
        self.visualizer = MySQLVisualizer()
        self.output_dir = "analyses_avancees"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def safe_execute_query(self, query):
        """Exécute une requête avec gestion des erreurs"""
        try:
            df = self.visualizer.execute_query(query)
            if df.empty:
                print(f"⚠️  La requête n'a retourné aucune donnée")
                print(f"   Requête: {query[:100]}...")
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de la requête: {e}")
            print(f"   Requête: {query[:100]}...")
            return pd.DataFrame()
    
    def convert_to_numeric(self, df, columns):
        """Convertit les colonnes en valeurs numériques"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def safe_plot(self, fig, filename):
        """Sauvegarde et affiche un graphique avec interprétation"""
        try:
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/{filename}', 
                       dpi=300, bbox_inches='tight')
            print(f"✅ Graphique sauvegardé: {self.output_dir}/{filename}")
            plt.show()
            plt.close()
        except Exception as e:
            print(f"❌ Erreur lors de la sauvegarde du graphique: {e}")
            plt.close()
    
    def interpreter_rfm(self, df):
        """Interprétation des résultats RFM"""
        print("\n" + "="*80)
        print("📊 INTERPRÉTATION DE L'ANALYSE RFM")
        print("="*80)
        
        if df is None or df.empty:
            print("❌ Pas de données à interpréter")
            return
        
        print("\n🔍 RÉSUMÉ RFM :")
        print("-" * 40)
        print(f"Nombre de clients analysés : {len(df)}")
        
        if 'Segment_RFM' in df.columns:
            segments_counts = df['Segment_RFM'].value_counts()
            print("\n📊 Distribution des segments :")
            for segment, count in segments_counts.items():
                pct = (count / len(df)) * 100
                print(f"  • {segment}: {count} clients ({pct:.1f}%)")
            
            # Interprétation par segment
            print("\n💡 INTERPRÉTATION :")
            if 'VIP' in segments_counts.index:
                vip_pct = (segments_counts.get('VIP', 0) / len(df)) * 100
                print(f"  ✅ Clients VIP ({vip_pct:.1f}%) : Vos meilleurs clients, très fidèles et dépensiers. À fidéliser avec des offres exclusives.")
            
            if 'Fidèle' in segments_counts.index:
                fidele_pct = (segments_counts.get('Fidèle', 0) / len(df)) * 100
                print(f"  👍 Clients Fidèles ({fidele_pct:.1f}%) : Clients réguliers avec un bon potentiel. À encourager vers le statut VIP.")
            
            if 'Régulier' in segments_counts.index:
                regulier_pct = (segments_counts.get('Régulier', 0) / len(df)) * 100
                print(f"  👤 Clients Réguliers ({regulier_pct:.1f}%) : Clients de base. À engager davantage pour augmenter leur fidélité.")
            
            if 'À risque' in segments_counts.index:
                risque_pct = (segments_counts.get('À risque', 0) / len(df)) * 100
                print(f"  ⚠️  Clients à risque ({risque_pct:.1f}%) : Clientes inactifs depuis longtemps. Campagnes de réactivation nécessaires.")
            
            if 'Perdu' in segments_counts.index:
                perdu_pct = (segments_counts.get('Perdu', 0) / len(df)) * 100
                print(f"  ❌ Clients perdus ({perdu_pct:.1f}%) : Très faible probabilité de réachat. Peuvent être retirés des campagnes actives.")
        
        # Statistiques RFM
        if all(col in df.columns for col in ['R_score', 'F_score', 'M_score']):
            print("\n📈 STATISTIQUES RFM MOYENNES :")
            print(f"  • Score Récence moyen : {df['R_score'].mean():.1f}/4")
            print(f"  • Score Fréquence moyen : {df['F_score'].mean():.1f}/4")
            print(f"  • Score Montant moyen : {df['M_score'].mean():.1f}/4")
            
            if 'RFM_Score_Total' in df.columns:
                print(f"  • Score RFM total moyen : {df['RFM_Score_Total'].mean():.1f}/12")
        
        print("="*80)
    
    def interpreter_correlation(self, df, corr_matrix, corr_stats):
        """Interprétation des analyses de corrélation"""
        print("\n" + "="*80)
        print("📊 INTERPRÉTATION DE L'ANALYSE DE CORRÉLATION")
        print("="*80)
        
        if df is None or df.empty:
            print("❌ Pas de données à interpréter")
            return
        
        print("\n🔍 PRINCIPALES CORRÉLATIONS :")
        print("-" * 40)
        
        if corr_matrix is not None and not corr_matrix.empty:
            # Corrélations fortes (abs > 0.5)
            strong_corrs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    col1 = corr_matrix.columns[i]
                    col2 = corr_matrix.columns[j]
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.3:  # Seuil de corrélation significative
                        strong_corrs.append((col1, col2, corr_val))
            
            if strong_corrs:
                print("\n📈 Corrélations significatives détectées :")
                for col1, col2, corr_val in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
                    direction = "positive" if corr_val > 0 else "négative"
                    force = "très forte" if abs(corr_val) > 0.7 else "forte" if abs(corr_val) > 0.5 else "modérée"
                    print(f"  • {col1} ↔ {col2} : {corr_val:.3f} ({force} corrélation {direction})")
            else:
                print("  Aucune corrélation significative détectée")
        
        # Interprétation détaillée
        print("\n💡 INTERPRÉTATION :")
        print("-" * 40)
        
        # Prix vs Quantité
        if 'prix' in df.columns and 'quantite' in df.columns:
            prix_qte_corr = df['prix'].corr(df['quantite'])
            if not np.isnan(prix_qte_corr):
                if prix_qte_corr < -0.3:
                    print("  ✅ Forte relation inverse entre prix et quantité : Plus le prix augmente, moins les clients achètent (élasticité-prix classique).")
                elif prix_qte_corr < -0.1:
                    print("  📉 Faible relation inverse entre prix et quantité : Le prix a un léger impact sur la demande.")
                elif prix_qte_corr > 0.3:
                    print("  📈 Relation positive entre prix et quantité : Les produits chers se vendent en grande quantité (probablement des produits de luxe ou professionnels).")
                else:
                    print("  ➖ Pas de relation claire entre prix et quantité : Le prix n'est pas un facteur déterminant des ventes.")
        
        # Délai de livraison
        if 'delai_livraison' in df.columns:
            if 'score_satisfaction' in df.columns:
                delai_sat_corr = df['delai_livraison'].corr(df['score_satisfaction'])
                if not np.isnan(delai_sat_corr):
                    if delai_sat_corr < -0.3:
                        print("  ⏱️  Impact fort du délai sur la satisfaction : Plus le délai est long, plus la satisfaction diminue.")
                    elif delai_sat_corr < -0.1:
                        print("  ⏱️  Léger impact du délai sur la satisfaction : Les clients sont sensibles aux délais mais tolérants.")
                    else:
                        print("  ⏱️  Pas d'impact significatif des délais : D'autres facteurs influencent plus la satisfaction.")
        
        # Marges
        if all(col in df.columns for col in ['prix', 'prix_achat']):
            marge_moy = (df['prix'] - df['prix_achat']).mean()
            taux_marge_moy = ((df['prix'] - df['prix_achat']) / df['prix'] * 100).mean()
            print(f"  💰 Marge moyenne : ${marge_moy:.2f} ({taux_marge_moy:.1f}%)")
        
        print("="*80)
    
    def interpreter_saisonnalite(self, df_mensuel, stats_saisonnalite):
        """Interprétation de l'analyse de saisonnalité"""
        print("\n" + "="*80)
        print("📊 INTERPRÉTATION DE L'ANALYSE DE SAISONNALITÉ")
        print("="*80)
        
        if df_mensuel is None or df_mensuel.empty:
            print("❌ Pas de données à interpréter")
            return
        
        print("\n🔍 RÉSULTATS DE L'ANALYSE :")
        print("-" * 40)
        
        # Analyse mensuelle
        if 'chiffre_affaires' in df_mensuel.columns:
            # Calcul des moyennes par mois
            mois_stats = df_mensuel.groupby(df_mensuel.index.month)['chiffre_affaires'].mean()
            mois_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                         'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            
            if len(mois_stats) > 0:
                mois_max = mois_stats.idxmax()
                mois_min = mois_stats.idxmin()
                
                print(f"\n📈 PERFORMANCE MENSUELLE :")
                print(f"  • Meilleur mois : {mois_names[mois_max-1]} (CA moyen : ${mois_stats[mois_max]:,.2f})")
                print(f"  • Mois le plus faible : {mois_names[mois_min-1]} (CA moyen : ${mois_stats[mois_min]:,.2f})")
                print(f"  • Écart saisonnier : {mois_stats[mois_max]/mois_stats[mois_min]-1:.1%}")
                
                # Détection des pics saisonniers
                seuil_haut = mois_stats.mean() * 1.2
                seuil_bas = mois_stats.mean() * 0.8
                
                mois_forts = [mois_names[m-1] for m in mois_stats[mois_stats > seuil_haut].index]
                mois_faibles = [mois_names[m-1] for m in mois_stats[mois_stats < seuil_bas].index]
                
                if mois_forts:
                    print(f"  ⬆️  Pics saisonniers : {', '.join(mois_forts)}")
                if mois_faibles:
                    print(f"  ⬇️  Creux saisonniers : {', '.join(mois_faibles)}")
        
        # Analyse par jour de la semaine
        print("\n📅 ANALYSE HEBDOMADAIRE :")
        print("-" * 30)
        
        print("💡 INTERPRÉTATION STRATÉGIQUE :")
        print("-" * 40)
        
        # Recommandations basées sur les pics
        if mois_forts:
            print(f"  🎯 Pour les mois forts ({', '.join(mois_forts)}) :")
            print("     • Augmenter les stocks des produits populaires")
            print("     • Lancer des campagnes marketing ciblées")
            print("     • Préparer du personnel supplémentaire")
        
        if mois_faibles:
            print(f"\n  📉 Pour les mois creux ({', '.join(mois_faibles)}) :")
            print("     • Proposer des promotions pour stimuler les ventes")
            print("     • Utiliser pour les maintenances et formations")
            print("     • Lancer des campagnes de fidélisation")
        
        print("="*80)
    
    def interpreter_panier(self, panier_stats, stats_panier):
        """Interprétation de l'analyse du panier moyen"""
        print("\n" + "="*80)
        print("📊 INTERPRÉTATION DE L'ANALYSE DU PANIER MOYEN")
        print("="*80)
        
        if panier_stats is None or panier_stats.empty:
            print("❌ Pas de données à interpréter")
            return
        
        print("\n🔍 STATISTIQUES DU PANIER :")
        print("-" * 40)
        
        if stats_panier:
            print(f"  • Panier moyen : ${stats_panier.get('panier_moyen', 0):,.2f}")
            print(f"  • Panier médian : ${stats_panier.get('panier_median', 0):,.2f}")
            print(f"  • Produits moyens par panier : {stats_panier.get('produits_moyens', 0):.1f}")
            print(f"  • Paniers analysés : {stats_panier.get('paniers_analyses', 0)}")
        
        # Distribution des paniers
        if 'montant_total' in panier_stats.columns:
            petits_paniers = (panier_stats['montant_total'] < 50).sum()
            moyens_paniers = ((panier_stats['montant_total'] >= 50) & 
                            (panier_stats['montant_total'] < 200)).sum()
            gros_paniers = (panier_stats['montant_total'] >= 200).sum()
            
            total_paniers = len(panier_stats)
            
            print("\n📊 RÉPARTITION DES PANIERS :")
            print(f"  • Petits paniers (< $50) : {petits_paniers} ({petits_paniers/total_paniers*100:.1f}%)")
            print(f"  • Paniers moyens ($50-$200) : {moyens_paniers} ({moyens_paniers/total_paniers*100:.1f}%)")
            print(f"  • Gros paniers (> $200) : {gros_paniers} ({gros_paniers/total_paniers*100:.1f}%)")
        
        print("\n💡 INTERPRÉTATION ET RECOMMANDATIONS :")
        print("-" * 40)
        
        # Analyse du panier moyen
        panier_moy = stats_panier.get('panier_moyen', 0)
        if panier_moy < 50:
            print("  ⚠️  Panier moyen faible :")
            print("     • Mettre en place du cross-selling (\"Les clients ont aussi acheté...\")")
            print("     • Proposer des lots ou packs avantageux")
            print("     • Offrir la livraison gratuite à partir d'un certain montant")
        elif panier_moy < 100:
            print("  👍 Panier moyen correct :")
            print("     • Proposer des produits complémentaires lors du checkout")
            print("     • Mettre en avant les produits populaires")
            print("     • Tester des programmes de fidélité")
        else:
            print("  🎉 Excellent panier moyen :")
            print("     • Maintenir la qualité de service")
            print("     • Développer des programmes VIP")
            print("     • Proposer des produits premium")
        
        # Analyse du nombre de produits
        produits_moy = stats_panier.get('produits_moyens', 0)
        if produits_moy < 2:
            print("\n  📦 Faible diversification :")
            print("     • Proposer des suggestions de produits complémentaires")
            print("     • Créer des bundles thématiques")
        elif produits_moy < 3:
            print("\n  📦 Bonne diversification :")
            print("     • Continuer à suggérer des produits pertinents")
            print("     • Analyser les associations fréquentes")
        else:
            print("\n  📦 Excellente diversification :")
            print("     • Explorer les opportunités de cross-catégorie")
            print("     • Développer des offres personnalisées")
        
        print("="*80)
    
    def interpreter_performance(self, df):
        """Interprétation de l'analyse de performance des employés"""
        print("\n" + "="*80)
        print("📊 INTERPRÉTATION DE L'ANALYSE DE PERFORMANCE DES EMPLOYÉS")
        print("="*80)
        
        if df is None or df.empty:
            print("❌ Pas de données à interpréter")
            return
        
        print("\n🔍 STATISTIQUES DE PERFORMANCE :")
        print("-" * 40)
        
        # Statistiques globales
        print(f"  • Employés analysés : {len(df)}")
        print(f"  • CA total : ${df['chiffre_affaires'].sum():,.2f}")
        print(f"  • CA moyen par employé : ${df['chiffre_affaires'].mean():,.2f}")
        print(f"  • Clients total : {int(df['nombre_clients'].sum())}")
        print(f"  • Clients moyen par employé : {df['nombre_clients'].mean():.1f}")
        
        # Top performeurs
        if len(df) >= 3:
            top3 = df.nlargest(3, 'chiffre_affaires')
            print("\n🏆 TOP 3 PERFORMANCE CA :")
            for i, (_, row) in enumerate(top3.iterrows(), 1):
                print(f"  {i}. {row['nom_employe']} : ${row['chiffre_affaires']:,.2f} " +
                      f"({int(row['nombre_clients'])} clients)")
        
        # Analyse de la distribution
        ca_vals = df['chiffre_affaires']
        q1 = ca_vals.quantile(0.25)
        q3 = ca_vals.quantile(0.75)
        median = ca_vals.median()
        
        print(f"\n📊 DISTRIBUTION DES PERFORMANCES :")
        print(f"  • 25% des employés < ${q1:,.2f}")
        print(f"  • 50% des employés < ${median:,.2f}")
        print(f"  • 75% des employés < ${q3:,.2f}")
        
        # Écart de performance
        if ca_vals.max() > 0:
            ratio_max_min = ca_vals.max() / ca_vals.min() if ca_vals.min() > 0 else float('inf')
            print(f"  • Écart max/min : {ratio_max_min:.1f}x")
        
        print("\n💡 INTERPRÉTATION ET RECOMMANDATIONS :")
        print("-" * 40)
        
        # Analyse de la dispersion
        cv = ca_vals.std() / ca_vals.mean() if ca_vals.mean() > 0 else 0
        if cv > 0.7:
            print("  ⚠️  Forte disparité des performances :")
            print("     • Mettre en place un système de mentorat (top performers → moins performants)")
            print("     • Analyser les meilleures pratiques des top performers")
            print("     • Proposer des formations ciblées")
        elif cv > 0.4:
            print("  📊 Disparité modérée des performances :")
            print("     • Partager les techniques qui fonctionnent")
            print("     • Organiser des ateliers d'échange")
            print("     • Fixer des objectifs progressifs")
        else:
            print("  ✅ Bonne homogénéité des performances :")
            print("     • Maintenir l'esprit d'équipe")
            print("     • Récompenser collectivement")
            print("     • Continuer à développer les compétences")
        
        # Recommandations par segment de performance
        if 'segment_performance' in df.columns:
            print("\n🎯 RECOMMANDATIONS PAR SEGMENT :")
            
            if 'À améliorer' in df['segment_performance'].values:
                print("  • Pour les employés \"À améliorer\" :")
                print("     - Programme de formation intensif")
                print("     - Accompagnement personnalisé")
                print("     - Objectifs progressifs et atteignables")
            
            if 'Bon' in df['segment_performance'].values:
                print("  • Pour les \"Bons\" performeurs :")
                print("     - Encourager à partager leurs méthodes")
                print("     - Donner plus d'autonomie")
                print("     - Préparer à des responsabilités accrues")
            
            if 'Excellent' in df['segment_performance'].values:
                print("  • Pour les \"Excellents\" performeurs :")
                print("     - Reconnaissance et valorisation")
                print("     - Impliquer dans les décisions stratégiques")
                print("     - Envisager des évolutions de carrière")
        
        print("="*80)
    
    # 1. Analyse RFM (modifiée)
    def analyse_rfm(self):
        """Analyse RFM (Recency, Frequency, Monetary)"""
        print("📊 Démarrage de l'analyse RFM...")
        
        query = """
        SELECT 
            c.customerNumber,
            c.customerName,
            MAX(o.orderDate) AS derniere_commande,
            COUNT(DISTINCT o.orderNumber) AS frequence,
            SUM(od.quantityOrdered * od.priceEach) AS montant,
            AVG(od.quantityOrdered * od.priceEach) AS panier_moyen,
            DATEDIFF(CURDATE(), MAX(o.orderDate)) AS jours_inactivite
        FROM customers c
        LEFT JOIN orders o ON c.customerNumber = o.customerNumber
        LEFT JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY c.customerNumber, c.customerName
        HAVING COUNT(DISTINCT o.orderNumber) > 0
        """
        
        df = self.safe_execute_query(query)
        
        if df.empty:
            print("❌ Impossible de réaliser l'analyse RFM: pas de données")
            return None
        
        # Conversion numérique
        numeric_cols = ['frequence', 'montant', 'panier_moyen', 'jours_inactivite']
        df = self.convert_to_numeric(df, numeric_cols)
        df = df.fillna(0)
        
        # Vérifier qu'on a assez de données
        if len(df) < 4:
            print(f"⚠️  Pas assez de données pour l'analyse RFM ({len(df)} clients)")
            return df
        
        # Calcul des scores RFM avec gestion des erreurs
        try:
            df['R_score'] = pd.qcut(df['jours_inactivite'], q=4, 
                                   labels=[4, 3, 2, 1], duplicates='drop').astype(int)
            df['F_score'] = pd.qcut(df['frequence'].rank(method='first'), q=4,
                                   labels=[1, 2, 3, 4], duplicates='drop').astype(int)
            df['M_score'] = pd.qcut(df['montant'], q=4,
                                   labels=[1, 2, 3, 4], duplicates='drop').astype(int)
            
            df['RFM_Score'] = df['R_score'].astype(str) + df['F_score'].astype(str) + df['M_score'].astype(str)
            df['RFM_Score_Total'] = df['R_score'] + df['F_score'] + df['M_score']
        except Exception as e:
            print(f"⚠️  Erreur lors du calcul des scores RFM: {e}")
            print("   Utilisation de scores simplifiés...")
            
            # Scores simplifiés
            df['R_score'] = pd.cut(df['jours_inactivite'], bins=4, labels=[4, 3, 2, 1]).astype(int)
            df['F_score'] = pd.cut(df['frequence'], bins=4, labels=[1, 2, 3, 4]).astype(int)
            df['M_score'] = pd.cut(df['montant'], bins=4, labels=[1, 2, 3, 4]).astype(int)
            df['RFM_Score_Total'] = df['R_score'] + df['F_score'] + df['M_score']
        
        # Segmentation RFM
        def segment_rfm(row):
            total = row['RFM_Score_Total']
            if total >= 10:
                return 'VIP'
            elif total >= 8:
                return 'Fidèle'
            elif total >= 6:
                return 'Régulier'
            elif total >= 4:
                return 'À risque'
            else:
                return 'Perdu'
        
        df['Segment_RFM'] = df.apply(segment_rfm, axis=1)
        
        # Graphique 1: Distribution des segments
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        segment_counts = df['Segment_RFM'].value_counts()
        colors = ['gold', 'lightgreen', 'lightblue', 'orange', 'lightcoral']
        wedges, texts, autotexts = ax1.pie(segment_counts.values, 
                                          labels=segment_counts.index,
                                          autopct='%1.1f%%', 
                                          colors=colors[:len(segment_counts)], 
                                          startangle=90,
                                          textprops={'fontsize': 12})
        ax1.set_title('Distribution des segments RFM', fontsize=14, fontweight='bold')
        
        # Style des pourcentages
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(11)
            autotext.set_fontweight('bold')
        
        self.safe_plot(fig1, 'rfm_distribution_segments.png')
        
        # Graphique 2: Matrice RFM
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        try:
            rfm_matrix = df.pivot_table(index='R_score', columns='F_score',
                                       values='customerNumber', aggfunc='count', fill_value=0)
            sns.heatmap(rfm_matrix, annot=True, fmt='.0f', cmap='YlOrRd', ax=ax2)
            ax2.set_title('Matrice RFM: Récence vs Fréquence', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Score Fréquence (F)', fontsize=12)
            ax2.set_ylabel('Score Récence (R)', fontsize=12)
        except:
            ax2.text(0.5, 0.5, 'Matrice RFM non disponible',
                    ha='center', va='center', fontsize=12)
            ax2.set_title('Matrice RFM', fontsize=14, fontweight='bold')
        
        self.safe_plot(fig2, 'rfm_matrice.png')
        
        # Graphique 3: Boxplot montant par segment
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        segments_order = ['VIP', 'Fidèle', 'Régulier', 'À risque', 'Perdu']
        segments_present = [s for s in segments_order if s in df['Segment_RFM'].unique()]
        
        if len(segments_present) > 0:
            sns.boxplot(x='Segment_RFM', y='montant', data=df,
                       order=segments_present, ax=ax3)
            ax3.set_title('Montant total par segment RFM', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Segment RFM', fontsize=12)
            ax3.set_ylabel('Montant total ($)', fontsize=12)
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3, axis='y')
        else:
            ax3.text(0.5, 0.5, 'Pas de données pour le boxplot',
                    ha='center', va='center', fontsize=12)
            ax3.set_title('Montant par segment', fontsize=14, fontweight='bold')
        
        self.safe_plot(fig3, 'rfm_montant_par_segment.png')
        
        # Graphique 4: Scatter plot Fréquence vs Montant
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        if len(df) > 0:
            scatter = ax4.scatter(df['frequence'], df['montant'],
                                 c=df['jours_inactivite'], 
                                 s=df['panier_moyen']/10 + 20,
                                 cmap='viridis', alpha=0.6,
                                 edgecolors='black', linewidth=0.5)
            ax4.set_xlabel('Fréquence (nombre de commandes)', fontsize=12)
            ax4.set_ylabel('Montant total ($)', fontsize=12)
            ax4.set_title('Analyse Fréquence vs Montant', fontsize=14, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax4, label='Jours depuis dernière commande')
        self.safe_plot(fig4, 'rfm_frequence_vs_montant.png')
        
        # Sauvegarde des résultats
        df.to_csv(f'{self.output_dir}/segmentation_rfm.csv', index=False)
        print(f"✅ Données RFM sauvegardées: {self.output_dir}/segmentation_rfm.csv")
        
        # Interprétation
        self.interpreter_rfm(df)
        
        return df
    
    # 2. Analyse de corrélation (modifiée)
    def analyse_correlation(self):
        """Analyse de corrélation entre variables"""
        print("🔍 Démarrage de l'analyse de corrélation...")
        
        query = """
        SELECT 
            od.priceEach AS prix,
            od.quantityOrdered AS quantite,
            p.buyPrice AS prix_achat,
            p.MSRP AS prix_conseille,
            p.quantityInStock AS stock,
            DATEDIFF(o.shippedDate, o.orderDate) AS delai_livraison,
            DATEDIFF(o.requiredDate, o.orderDate) AS delai_attendu,
            (od.quantityOrdered * od.priceEach) AS montant_ligne
        FROM orderdetails od
        JOIN orders o ON od.orderNumber = o.orderNumber
        JOIN products p ON od.productCode = p.productCode
        WHERE o.status = 'Shipped'
        AND o.shippedDate IS NOT NULL
        LIMIT 5000
        """
        
        df = self.safe_execute_query(query)
        
        if df.empty:
            print("❌ Impossible de réaliser l'analyse de corrélation: pas de données")
            return None, None, None
        
        # Conversion numérique
        numeric_cols = ['prix', 'quantite', 'prix_achat', 'prix_conseille', 
                       'stock', 'delai_livraison', 'delai_attendu', 'montant_ligne']
        df = self.convert_to_numeric(df, numeric_cols)
        df = df.fillna(0)
        
        # Ajouter un score de satisfaction simplifié
        df['score_satisfaction'] = np.where(
            df['delai_livraison'] <= 7, 5,
            np.where(df['delai_livraison'] <= 14, 4,
                    np.where(df['delai_livraison'] <= 30, 3,
                            np.where(df['delai_livraison'] <= 60, 2, 1)))
        )
        
        # Calcul des corrélations
        correlation_matrix = df.corr(method='pearson', numeric_only=True)
        
        # Graphique 1: Heatmap de corrélation
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0,
                   square=True, ax=ax1,
                   annot_kws={'size': 10})
        ax1.set_title('Matrice de corrélation - Vue d\'ensemble', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        self.safe_plot(fig1, 'correlation_heatmap.png')
        
        # Graphique 2: Corrélation prix vs quantité
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        if len(df) > 0:
            ax2.scatter(df['prix'], df['quantite'], alpha=0.3, s=20)
            ax2.set_xlabel('Prix unitaire ($)', fontsize=12)
            ax2.set_ylabel('Quantité commandée', fontsize=12)
            ax2.set_title('Corrélation: Prix vs Quantité', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            # Ajout de la droite de régression
            if len(df) > 1:
                try:
                    z = np.polyfit(df['prix'], df['quantite'], 1)
                    p = np.poly1d(z)
                    ax2.plot(df['prix'], p(df['prix']), "r--", alpha=0.8,
                           linewidth=2, label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
                    ax2.legend(fontsize=10)
                    
                    # Ajout du coefficient de corrélation
                    corr = df['prix'].corr(df['quantite'])
                    ax2.text(0.05, 0.95, f'Corrélation: r = {corr:.3f}', 
                            transform=ax2.transAxes, fontsize=11,
                            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                except:
                    pass
        self.safe_plot(fig2, 'correlation_prix_quantite.png')
        
        # Graphique 3: Délai vs Satisfaction
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        if 'score_satisfaction' in df.columns:
            sns.boxplot(x='score_satisfaction', y='delai_livraison', 
                       data=df, ax=ax3)
            ax3.set_xlabel('Score de satisfaction (1-5)', fontsize=12)
            ax3.set_ylabel('Délai de livraison (jours)', fontsize=12)
            ax3.set_title('Délai de livraison vs Satisfaction client', 
                         fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3, axis='y')
        self.safe_plot(fig3, 'correlation_delai_satisfaction.png')
        
        # Graphique 4: Pair plot
        if len(df) > 0:
            cols_to_plot = ['prix', 'quantite', 'delai_livraison', 'montant_ligne']
            cols_present = [col for col in cols_to_plot if col in df.columns]
            
            if len(cols_present) >= 2:
                try:
                    g = sns.pairplot(df[cols_present].sample(min(1000, len(df))),
                                    diag_kind='kde', plot_kws={'alpha': 0.5})
                    g.fig.suptitle('Pair plot des principales variables', 
                                  y=1.02, fontsize=14, fontweight='bold')
                    plt.savefig(f'{self.output_dir}/correlation_pairplot.png', 
                               dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                except:
                    pass
        
        # Analyse statistique
        corr_stats = {}
        pairs = [('prix', 'quantite'), ('delai_livraison', 'score_satisfaction')]
        
        for col1, col2 in pairs:
            if col1 in df.columns and col2 in df.columns:
                mask = df[col1].notna() & df[col2].notna()
                if mask.sum() > 1:
                    try:
                        corr, p_value = stats.pearsonr(df.loc[mask, col1], 
                                                      df.loc[mask, col2])
                        corr_stats[f'{col1}_vs_{col2}'] = {
                            'correlation': corr,
                            'p_value': p_value,
                            'significatif': p_value < 0.05,
                            'n_observations': mask.sum()
                        }
                    except:
                        pass
        
        # Sauvegarde des résultats
        try:
            correlation_matrix.to_csv(f'{self.output_dir}/matrice_correlation.csv')
        except:
            pass
        
        if corr_stats:
            pd.DataFrame(corr_stats).T.to_csv(f'{self.output_dir}/stats_correlation.csv')
        
        print(f"✅ Données de corrélation sauvegardées dans {self.output_dir}/")
        
        # Interprétation
        self.interpreter_correlation(df, correlation_matrix, corr_stats)
        
        return df, correlation_matrix, corr_stats
    
    # 3. Analyse de saisonnalité (modifiée)
    def analyse_saisonnalite(self):
        """Détection de saisonnalité dans les ventes"""
        print("📅 Démarrage de l'analyse de saisonnalité...")
        
        query = """
        SELECT 
            DATE(o.orderDate) AS date_commande,
            DAYOFWEEK(o.orderDate) AS jour_semaine,
            MONTH(o.orderDate) AS mois,
            QUARTER(o.orderDate) AS trimestre,
            YEAR(o.orderDate) AS annee,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes
        FROM orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY DATE(o.orderDate), DAYOFWEEK(o.orderDate), 
                 MONTH(o.orderDate), QUARTER(o.orderDate), YEAR(o.orderDate)
        ORDER BY date_commande
        """
        
        df = self.safe_execute_query(query)
        
        if df.empty:
            print("❌ Impossible de réaliser l'analyse de saisonnalité: pas de données")
            return None, None
        
        # Conversion des dates et mise en index
        df['date_commande'] = pd.to_datetime(df['date_commande'], errors='coerce')
        df = df.dropna(subset=['date_commande'])
        
        if df.empty:
            print("❌ Pas de dates valides pour l'analyse de saisonnalité")
            return None, None
        
        df.set_index('date_commande', inplace=True)
        
        # Agrégations temporelles
        df_mensuel = df.resample('ME').agg({
            'chiffre_affaires': 'sum',
            'nombre_commandes': 'sum'
        }).fillna(0)
        
        df_trimestriel = df.resample('QE').agg({
            'chiffre_affaires': 'sum',
            'nombre_commandes': 'sum'
        }).fillna(0)
        
        # Analyse de saisonnalité
        df_mensuel['mois'] = df_mensuel.index.month
        df_mensuel['annee'] = df_mensuel.index.year
        
        # Graphique 1: Série temporelle complète
        fig1, ax1 = plt.subplots(figsize=(14, 6))
        if len(df_mensuel) > 0:
            ax1.plot(df_mensuel.index, df_mensuel['chiffre_affaires'], 
                    marker='o', markersize=4, linewidth=2, color='steelblue')
            ax1.set_title('Évolution du chiffre d\'affaires mensuel', 
                         fontsize=14, fontweight='bold')
            ax1.set_xlabel('Date', fontsize=12)
            ax1.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Ajouter les valeurs pour les pics
            max_points = df_mensuel.nlargest(3, 'chiffre_affaires')
            for date, ca in max_points.iterrows():
                ax1.annotate(f'${ca["chiffre_affaires"]:,.0f}', 
                           (date, ca["chiffre_affaires"]),
                           textcoords="offset points", xytext=(0,10), 
                           ha='center', fontsize=9, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        self.safe_plot(fig1, 'saisonnalite_evolution_mensuelle.png')
        
        # Graphique 2: Moyenne mobile
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        if len(df_mensuel) >= 3:
            try:
                df_mensuel['moyenne_mobile_3m'] = df_mensuel['chiffre_affaires'].rolling(
                    window=3, center=True).mean()
                ax2.plot(df_mensuel.index, df_mensuel['chiffre_affaires'], 
                        label='CA réel', alpha=0.7, linewidth=1)
                ax2.plot(df_mensuel.index, df_mensuel['moyenne_mobile_3m'],
                        label='Moyenne mobile 3 mois', linewidth=3, color='red')
                ax2.set_title('Tendance avec moyenne mobile (3 mois)', 
                             fontsize=14, fontweight='bold')
                ax2.set_xlabel('Date', fontsize=12)
                ax2.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
                ax2.legend(fontsize=11)
                ax2.grid(True, alpha=0.3)
                ax2.tick_params(axis='x', rotation=45)
            except:
                ax2.text(0.5, 0.5, 'Données insuffisantes pour moyenne mobile',
                        ha='center', va='center', fontsize=12)
                ax2.set_title('Tendance des ventes', fontsize=14, fontweight='bold')
        self.safe_plot(fig2, 'saisonnalite_moyenne_mobile.png')
        
        # Graphique 3: Boxplot par mois
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        if len(df_mensuel) > 0:
            mois_data = []
            mois_labels = []
            mois_names = ['Jan', 'Fév', 'Mar', 'Avr', 'Mai', 'Juin', 
                         'Juil', 'Août', 'Sep', 'Oct', 'Nov', 'Déc']
            
            for mois in range(1, 13):
                data_mois = df_mensuel[df_mensuel.index.month == mois]['chiffre_affaires'].values
                if len(data_mois) > 0:
                    mois_data.append(data_mois)
                    mois_labels.append(mois_names[mois-1])
            
            if mois_data:
                bp = ax3.boxplot(mois_data, patch_artist=True)
                for patch in bp['boxes']:
                    patch.set_facecolor('lightblue')
                
                ax3.set_xlabel('Mois', fontsize=12)
                ax3.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
                ax3.set_title('Distribution du chiffre d\'affaires par mois', 
                             fontsize=14, fontweight='bold')
                ax3.set_xticklabels(mois_labels)
                ax3.grid(True, alpha=0.3, axis='y')
        self.safe_plot(fig3, 'saisonnalite_distribution_mensuelle.png')
        
        # Graphique 4: CA par jour de la semaine
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        if len(df) > 0:
            # Extraire le nom du jour
            df['jour_semaine_nom'] = df.index.day_name()
            jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                          'Friday', 'Saturday', 'Sunday']
            jour_labels_fr = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 
                             'Vendredi', 'Samedi', 'Dimanche']
            
            ca_par_jour = df.groupby('jour_semaine_nom')['chiffre_affaires'].sum()
            ca_par_jour = ca_par_jour.reindex(jours_ordre)
            
            # Filtrer les jours avec des données
            ca_par_jour = ca_par_jour.dropna()
            
            if len(ca_par_jour) > 0:
                jour_labels = [jour_labels_fr[jours_ordre.index(jour)] 
                              for jour in ca_par_jour.index]
                
                bars = ax4.bar(range(len(ca_par_jour)), ca_par_jour.values,
                              color=plt.cm.Set3(range(len(ca_par_jour))))
                ax4.set_xticks(range(len(ca_par_jour)))
                ax4.set_xticklabels(jour_labels, rotation=45)
                ax4.set_title('Chiffre d\'affaires par jour de la semaine', 
                             fontsize=14, fontweight='bold')
                ax4.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
                ax4.grid(True, alpha=0.3, axis='y')
                
                # Ajouter les valeurs
                for bar, value in zip(bars, ca_par_jour.values):
                    ax4.text(bar.get_x() + bar.get_width()/2., 
                           bar.get_height() + bar.get_height()*0.01,
                           f'${value/1000:.1f}K', 
                           ha='center', va='bottom', fontsize=9)
        self.safe_plot(fig4, 'saisonnalite_par_jour.png')
        
        # Calcul des statistiques de saisonnalité
        stats_saisonnalite = {}
        if len(df_mensuel) > 0:
            try:
                stats_mois = df_mensuel.groupby('mois')['chiffre_affaires'].mean()
                if not stats_mois.empty:
                    stats_saisonnalite['mois_fort'] = int(stats_mois.idxmax())
                    stats_saisonnalite['mois_faible'] = int(stats_mois.idxmin())
                    
                    # Ajouter les noms des mois
                    mois_names = {1:'Janvier', 2:'Février', 3:'Mars', 4:'Avril',
                                 5:'Mai', 6:'Juin', 7:'Juillet', 8:'Août',
                                 9:'Septembre', 10:'Octobre', 11:'Novembre', 12:'Décembre'}
                    
                    stats_saisonnalite['nom_mois_fort'] = mois_names[stats_saisonnalite['mois_fort']]
                    stats_saisonnalite['nom_mois_faible'] = mois_names[stats_saisonnalite['mois_faible']]
            except:
                pass
        
        # Sauvegarde des résultats
        try:
            df_mensuel.to_csv(f'{self.output_dir}/donnees_saisonnalite.csv')
        except:
            pass
        
        if stats_saisonnalite:
            pd.Series(stats_saisonnalite).to_csv(
                f'{self.output_dir}/stats_saisonnalite.csv')
        
        print(f"✅ Données de saisonnalité sauvegardées dans {self.output_dir}/")
        
        # Interprétation
        self.interpreter_saisonnalite(df_mensuel, stats_saisonnalite)
        
        return df_mensuel, stats_saisonnalite
    
    # 4. Analyse du panier moyen et associations (modifiée)
    def analyse_panier_associations(self):
        """Analyse du panier moyen et produits associés"""
        print("🛒 Démarrage de l'analyse du panier moyen...")
        
        # Requête pour les paniers complets
        query_paniers = """
        SELECT 
            o.orderNumber,
            o.customerNumber,
            od.productCode,
            p.productName,
            od.quantityOrdered,
            od.priceEach,
            (od.quantityOrdered * od.priceEach) AS montant_ligne
        FROM orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        JOIN products p ON od.productCode = p.productCode
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        ORDER BY o.orderNumber
        """
        
        df = self.safe_execute_query(query_paniers)
        
        if df.empty:
            print("❌ Impossible de réaliser l'analyse du panier: pas de données")
            return None, None, None
        
        # Analyse du panier moyen
        panier_stats = df.groupby('orderNumber').agg({
            'montant_ligne': 'sum',
            'productCode': 'count',
            'quantityOrdered': 'sum'
        }).rename(columns={
            'montant_ligne': 'montant_total',
            'productCode': 'nombre_produits',
            'quantityOrdered': 'quantite_totale'
        }).fillna(0)
        
        # Graphique 1: Distribution du panier moyen
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        if len(panier_stats) > 0:
            ax1.hist(panier_stats['montant_total'], bins=30,
                    color='skyblue', edgecolor='black', alpha=0.7)
            mean_val = panier_stats['montant_total'].mean()
            median_val = panier_stats['montant_total'].median()
            
            ax1.axvline(mean_val, color='red',
                       linestyle='--', linewidth=2,
                       label=f'Moyenne: ${mean_val:,.2f}')
            ax1.axvline(median_val, color='green',
                       linestyle='--', linewidth=2,
                       label=f'Médiane: ${median_val:,.2f}')
            ax1.set_xlabel('Montant du panier ($)', fontsize=12)
            ax1.set_ylabel('Nombre de commandes', fontsize=12)
            ax1.set_title('Distribution du montant des paniers', 
                         fontsize=14, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Ajouter les statistiques
            ax1.text(0.7, 0.95, f'Écart-type: ${panier_stats["montant_total"].std():,.2f}',
                    transform=ax1.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        self.safe_plot(fig1, 'panier_distribution.png')
        
        # Graphique 2: Relation montant vs nombre de produits
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        if len(panier_stats) > 0:
            scatter = ax2.scatter(panier_stats['nombre_produits'],
                                 panier_stats['montant_total'],
                                 c=panier_stats['quantite_totale'],
                                 alpha=0.6, cmap='viridis', s=50,
                                 edgecolors='black', linewidth=0.5)
            ax2.set_xlabel('Nombre de produits différents', fontsize=12)
            ax2.set_ylabel('Montant total ($)', fontsize=12)
            ax2.set_title('Relation produits vs montant du panier', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='Quantité totale')
            
            # Ajouter une droite de tendance
            if len(panier_stats) > 1:
                try:
                    z = np.polyfit(panier_stats['nombre_produits'], 
                                  panier_stats['montant_total'], 1)
                    p = np.poly1d(z)
                    x_trend = np.linspace(panier_stats['nombre_produits'].min(),
                                         panier_stats['nombre_produits'].max(), 50)
                    ax2.plot(x_trend, p(x_trend), "r--", alpha=0.8,
                           linewidth=2, label='Tendance')
                    ax2.legend()
                except:
                    pass
        self.safe_plot(fig2, 'panier_produits_vs_montant.png')
        
        # Graphique 3: Produits les plus fréquents
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        if 'productName' in df.columns:
            top_produits = df['productName'].value_counts().head(15)
            if len(top_produits) > 0:
                # Tronquer les noms longs
                labels = [name[:25] + '...' if len(name) > 25 else name 
                         for name in top_produits.index]
                
                bars = ax3.barh(range(len(top_produits)), top_produits.values,
                              color=plt.cm.viridis(np.linspace(0, 1, len(top_produits))))
                ax3.set_yticks(range(len(top_produits)))
                ax3.set_yticklabels(labels, fontsize=10)
                ax3.set_xlabel('Nombre d\'occurrences dans les paniers', fontsize=12)
                ax3.set_title('Top 15 produits les plus fréquents dans les paniers', 
                             fontsize=14, fontweight='bold')
                ax3.grid(True, alpha=0.3, axis='x')
                
                # Ajouter les valeurs
                for i, (bar, value) in enumerate(zip(bars, top_produits.values)):
                    ax3.text(value + value*0.01, i, 
                           f'{int(value)}', 
                           va='center', fontsize=9)
        else:
            ax3.text(0.5, 0.5, 'Noms de produits non disponibles',
                    ha='center', va='center', fontsize=12)
        self.safe_plot(fig3, 'panier_top_produits.png')
        
        # Graphique 4: Statistiques du panier
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.axis('off')
        
        if len(panier_stats) > 0:
            # Catégories de paniers
            petit_panier = (panier_stats['montant_total'] < 50).sum()
            moyen_panier = ((panier_stats['montant_total'] >= 50) & 
                           (panier_stats['montant_total'] < 200)).sum()
            grand_panier = (panier_stats['montant_total'] >= 200).sum()
            total = len(panier_stats)
            
            stats_text = f"""
            📊 STATISTIQUES DÉTAILLÉES DU PANIER
            
            📈 Indicateurs principaux:
            • Panier moyen: ${panier_stats['montant_total'].mean():,.2f}
            • Panier médian: ${panier_stats['montant_total'].median():,.2f}
            • Panier minimum: ${panier_stats['montant_total'].min():,.2f}
            • Panier maximum: ${panier_stats['montant_total'].max():,.2f}
            
            📦 Composition des paniers:
            • Produits moyens par panier: {panier_stats['nombre_produits'].mean():.1f}
            • Produits max dans un panier: {int(panier_stats['nombre_produits'].max())}
            • Produits min dans un panier: {int(panier_stats['nombre_produits'].min())}
            
            📊 Répartition par taille:
            • Petits paniers (< $50): {petit_panier} ({petit_panier/total*100:.1f}%)
            • Paniers moyens ($50-$200): {moyen_panier} ({moyen_panier/total*100:.1f}%)
            • Grands paniers (> $200): {grand_panier} ({grand_panier/total*100:.1f}%)
            
            🔍 Analyse des extrêmes:
            • {petit_panier} paniers sous la moyenne
            • {grand_panier} paniers au-dessus de $200
            • {total - grand_panier - moyen_panier - petit_panier} paniers hors catégorie
            """
            
            ax4.text(0.1, 0.5, stats_text, fontsize=11,
                    verticalalignment='center', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        self.safe_plot(fig4, 'panier_statistiques.png')
        
        # Calcul des statistiques synthétiques
        stats_panier = {}
        if len(panier_stats) > 0:
            stats_panier = {
                'panier_moyen': panier_stats['montant_total'].mean(),
                'panier_median': panier_stats['montant_total'].median(),
                'panier_std': panier_stats['montant_total'].std(),
                'produits_moyens': panier_stats['nombre_produits'].mean(),
                'paniers_analyses': len(panier_stats),
                'petits_paniers_pct': (panier_stats['montant_total'] < 50).sum() / len(panier_stats) * 100,
                'grands_paniers_pct': (panier_stats['montant_total'] >= 200).sum() / len(panier_stats) * 100
            }
            
            try:
                pd.Series(stats_panier).to_csv(f'{self.output_dir}/resume_panier.csv')
            except:
                pass
        
        print(f"✅ Données du panier sauvegardées dans {self.output_dir}/")
        
        # Interprétation
        self.interpreter_panier(panier_stats, stats_panier)
        
        return panier_stats, None, stats_panier
    
    # 5. Performance par employé (modifiée)
    def performance_employes(self):
        """Analyse des performances commerciales par employé"""
        print("👥 Démarrage de l'analyse de performance...")
        
        query = """
        SELECT 
            e.employeeNumber,
            CONCAT(e.firstName, ' ', e.lastName) AS nom_employe,
            e.jobTitle,
            o.city AS ville_bureau,
            o.country AS pays_bureau,
            COUNT(DISTINCT c.customerNumber) AS nombre_clients,
            COUNT(DISTINCT ord.orderNumber) AS nombre_commandes,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires
        FROM employees e
        LEFT JOIN customers c ON e.employeeNumber = c.salesRepEmployeeNumber
        LEFT JOIN orders ord ON c.customerNumber = ord.customerNumber
        LEFT JOIN orderdetails od ON ord.orderNumber = od.orderNumber
        LEFT JOIN offices o ON e.officeCode = o.officeCode
        WHERE (e.jobTitle LIKE '%Sales%' OR e.jobTitle LIKE '%Rep%')
        AND ord.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY e.employeeNumber, e.firstName, e.lastName, e.jobTitle,
                 e.officeCode, o.city, o.country
        HAVING COUNT(DISTINCT c.customerNumber) > 0
        ORDER BY chiffre_affaires DESC
        """
        
        df = self.safe_execute_query(query)
        
        if df.empty:
            print("❌ Impossible de réaliser l'analyse de performance: pas de données")
            return None
        
        # Conversion numérique
        numeric_cols = ['nombre_clients', 'nombre_commandes', 'chiffre_affaires']
        df = self.convert_to_numeric(df, numeric_cols)
        df = df.fillna(0)
        
        # Calcul des KPI supplémentaires
        df['CA_par_client'] = np.where(df['nombre_clients'] > 0,
                                      df['chiffre_affaires'] / df['nombre_clients'], 0)
        df['commandes_par_client'] = np.where(df['nombre_clients'] > 0,
                                            df['nombre_commandes'] / df['nombre_clients'], 0)
        
        # Normalisation pour scoring
        metrics = ['chiffre_affaires', 'nombre_clients', 'CA_par_client', 'commandes_par_client']
        metrics_present = [m for m in metrics if m in df.columns]
        
        if len(metrics_present) > 0 and len(df) > 0:
            # Normalisation manuelle
            for metric in metrics_present:
                max_val = df[metric].max()
                min_val = df[metric].min()
                if max_val > min_val:
                    df[f'{metric}_scaled'] = (df[metric] - min_val) / (max_val - min_val) * 100
                else:
                    df[f'{metric}_scaled'] = 50  # Valeur moyenne si pas de variation
            
            # Score composite de performance
            weights = {'chiffre_affaires_scaled': 0.4, 'nombre_clients_scaled': 0.3,
                      'CA_par_client_scaled': 0.2, 'commandes_par_client_scaled': 0.1}
            weights_present = {k: v for k, v in weights.items() if k in df.columns}
            
            if weights_present:
                total_weight = sum(weights_present.values())
                df['score_performance'] = sum(df[col] * (weight/total_weight) 
                                            for col, weight in weights_present.items())
                df['rang_performance'] = df['score_performance'].rank(ascending=False)
                
                # Segmentation des performances
                if len(df) >= 4:
                    try:
                        df['segment_performance'] = pd.qcut(df['score_performance'], q=4,
                                                           labels=['À améliorer', 'Moyen',
                                                                   'Bon', 'Excellent'])
                    except:
                        df['segment_performance'] = 'Non classé'
                else:
                    df['segment_performance'] = 'Non classé'
        
        # Graphique 1: Barres - CA par employé
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        if len(df) > 0:
            df_sorted = df.sort_values('chiffre_affaires', ascending=True)  # Pour barh
            colors = plt.cm.viridis(np.linspace(0, 1, len(df_sorted)))
            bars = ax1.barh(range(len(df_sorted)), df_sorted['chiffre_affaires'],
                          color=colors, edgecolor='black', linewidth=1)
            ax1.set_yticks(range(len(df_sorted)))
            # Tronquer les noms si nécessaire
            labels = [name[:20] + '...' if len(name) > 20 else name 
                     for name in df_sorted['nom_employe']]
            ax1.set_yticklabels(labels, fontsize=10)
            ax1.set_xlabel('Chiffre d\'affaires ($)', fontsize=12)
            ax1.set_title('Chiffre d\'affaires par employé', 
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, (bar, value) in enumerate(zip(bars, df_sorted['chiffre_affaires'])):
                if value > 0:
                    ax1.text(value + value*0.01, i, 
                           f'${value/1000:.1f}K', 
                           va='center', fontsize=9)
        self.safe_plot(fig1, 'performance_ca_employes.png')
        
        # Graphique 2: Scatter - Clients vs CA
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        if len(df) > 0:
            scatter = ax2.scatter(df['nombre_clients'], df['chiffre_affaires'],
                                 alpha=0.6, s=df['nombre_commandes']*5 + 50,
                                 c=df['CA_par_client'], cmap='coolwarm',
                                 edgecolors='black', linewidth=0.5)
            ax2.set_xlabel('Nombre de clients', fontsize=12)
            ax2.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
            ax2.set_title('Relation clients vs chiffre d\'affaires', 
                         fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax2, label='CA par client ($)')
            
            # Annoter les meilleurs performants
            if len(df) >= 5:
                top5 = df.nlargest(5, 'chiffre_affaires')
                for _, row in top5.iterrows():
                    ax2.annotate(row['nom_employe'].split()[0],
                               (row['nombre_clients'], row['chiffre_affaires']),
                               fontsize=9, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        self.safe_plot(fig2, 'performance_clients_vs_ca.png')
        
        # Graphique 3: Performance par bureau
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        if 'ville_bureau' in df.columns and len(df) > 0:
            bureau_stats = df.groupby('ville_bureau').agg({
                'chiffre_affaires': 'sum',
                'nombre_clients': 'sum',
                'employeeNumber': 'count'
            }).rename(columns={'employeeNumber': 'nb_employes'}).sort_values('chiffre_affaires', ascending=False).head(8)
            
            if len(bureau_stats) > 0:
                x = np.arange(len(bureau_stats))
                width = 0.35
                
                bars_ca = ax3.bar(x - width/2, bureau_stats['chiffre_affaires'], width,
                                 label='Chiffre d\'affaires', color='skyblue')
                ax3.set_xlabel('Bureau', fontsize=12)
                ax3.set_ylabel('Chiffre d\'affaires ($)', color='skyblue')
                ax3.tick_params(axis='y', labelcolor='skyblue')
                
                ax3_2 = ax3.twinx()
                bars_clients = ax3_2.bar(x + width/2, bureau_stats['nombre_clients'], width,
                                        label='Nombre de clients', color='lightcoral', alpha=0.7)
                ax3_2.set_ylabel('Nombre de clients', color='lightcoral')
                ax3_2.tick_params(axis='y', labelcolor='lightcoral')
                
                ax3.set_xticks(x)
                ax3.set_xticklabels(bureau_stats.index, rotation=45, ha='right')
                ax3.set_title('Performance par bureau (top 8)', fontsize=14, fontweight='bold')
                
                # Ajouter la légende combinée
                lines1, labels1 = ax3.get_legend_handles_labels()
                lines2, labels2 = ax3_2.get_legend_handles_labels()
                ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
                
                # Ajouter le nombre d'employés
                for i, (_, row) in enumerate(bureau_stats.iterrows()):
                    ax3.text(i - width/2, row['chiffre_affaires'] * 1.02, 
                            f"{int(row['nb_employes'])} emp.",
                            ha='center', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'Données bureau insuffisantes',
                        ha='center', va='center', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'Données de localisation non disponibles',
                    ha='center', va='center', fontsize=12)
        self.safe_plot(fig3, 'performance_par_bureau.png')
        
        # Graphique 4: Distribution des scores de performance
        fig4, ax4 = plt.subplots(figsize=(12, 6))
        if 'score_performance' in df.columns:
            # Histogramme des scores
            ax4.hist(df['score_performance'], bins=20,
                    color='skyblue', edgecolor='black', alpha=0.7)
            ax4.axvline(df['score_performance'].mean(), color='red',
                       linestyle='--', linewidth=2,
                       label=f'Moyenne: {df["score_performance"].mean():.1f}')
            ax4.axvline(df['score_performance'].median(), color='green',
                       linestyle='--', linewidth=2,
                       label=f'Médiane: {df["score_performance"].median():.1f}')
            ax4.set_xlabel('Score de performance', fontsize=12)
            ax4.set_ylabel('Nombre d\'employés', fontsize=12)
            ax4.set_title('Distribution des scores de performance', 
                         fontsize=14, fontweight='bold')
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3)
            
            # Ajouter les segments
            if 'segment_performance' in df.columns:
                segments = df['segment_performance'].value_counts()
                y_pos = 0.8
                for segment, count in segments.items():
                    ax4.text(0.7, y_pos, f'{segment}: {count} employés',
                            transform=ax4.transAxes, fontsize=10,
                            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
                    y_pos -= 0.05
        else:
            ax4.text(0.5, 0.5, 'Scores de performance non disponibles',
                    ha='center', va='center', fontsize=12)
        self.safe_plot(fig4, 'performance_distribution_scores.png')
        
        # Sauvegarde des résultats
        try:
            df.to_csv(f'{self.output_dir}/performance_employes_donnees.csv', index=False)
            print(f"✅ Données de performance sauvegardées: {self.output_dir}/performance_employes_donnees.csv")
        except:
            pass
        
        # Interprétation
        self.interpreter_performance(df)
        
        return df
    
    def generate_all_analyses(self):
        """Génère toutes les analyses avancées une par une avec leurs interprétations"""
        print("="*80)
        print("🚀 GÉNÉRATION DE TOUTES LES ANALYSES AVANCÉES")
        print("="*80)
        
        # 1. Analyse RFM
        print("\n" + "="*80)
        print("📊 ANALYSE RFM")
        print("="*80)
        self.analyse_rfm()
        input("\nAppuyez sur Entrée pour continuer vers l'analyse suivante...")
        
        # 2. Analyse de corrélation
        print("\n" + "="*80)
        print("📊 ANALYSE DE CORRÉLATION")
        print("="*80)
        self.analyse_correlation()
        input("\nAppuyez sur Entrée pour continuer vers l'analyse suivante...")
        
        # 3. Analyse de saisonnalité
        print("\n" + "="*80)
        print("📊 ANALYSE DE SAISONNALITÉ")
        print("="*80)
        self.analyse_saisonnalite()
        input("\nAppuyez sur Entrée pour continuer vers l'analyse suivante...")
        
        # 4. Analyse du panier
        print("\n" + "="*80)
        print("📊 ANALYSE DU PANIER MOYEN")
        print("="*80)
        self.analyse_panier_associations()
        input("\nAppuyez sur Entrée pour continuer vers l'analyse suivante...")
        
        # 5. Performance employés
        print("\n" + "="*80)
        print("📊 ANALYSE DE PERFORMANCE DES EMPLOYÉS")
        print("="*80)
        self.performance_employes()
        
        print("\n" + "="*80)
        print("✅ GÉNÉRATION DE TOUTES LES ANALYSES TERMINÉE")
        print(f"📁 Les analyses sont disponibles dans le dossier: {self.output_dir}")
        print("="*80)


if __name__ == "__main__":
    # Création de l'instance
    analyses = AnalysesAvancees()
    
    # Génération de toutes les analyses une par une
    analyses.generate_all_analyses()
    
    # Alternative : générer une analyse spécifique
    # analyses.analyse_rfm()
    # analyses.analyse_correlation()
    # analyses.analyse_saisonnalite()
    # analyses.analyse_panier_associations()
    # analyses.performance_employes()
    