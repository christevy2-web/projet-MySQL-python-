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
    
    # 1. Analyse RFM
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
        
        # Visualisations RFM
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Analyse RFM des Clients', fontsize=16, fontweight='bold')
        
        # Distribution des segments
        segment_counts = df['Segment_RFM'].value_counts()
        colors = ['gold', 'lightgreen', 'lightblue', 'orange', 'lightcoral']
        axes[0, 0].pie(segment_counts.values, labels=segment_counts.index,
                      autopct='%1.1f%%', colors=colors[:len(segment_counts)], startangle=90)
        axes[0, 0].set_title('Distribution des segments RFM')
        
        # Matrice RFM
        try:
            rfm_matrix = df.pivot_table(index='R_score', columns='F_score',
                                       values='customerNumber', aggfunc='count', fill_value=0)
            sns.heatmap(rfm_matrix, annot=True, fmt='.0f', cmap='YlOrRd',
                       ax=axes[0, 1])
            axes[0, 1].set_title('Matrice RFM: Récence vs Fréquence')
            axes[0, 1].set_xlabel('Score Fréquence')
            axes[0, 1].set_ylabel('Score Récence')
        except:
            axes[0, 1].text(0.5, 0.5, 'Matrice RFM non disponible',
                           ha='center', va='center')
            axes[0, 1].set_title('Matrice RFM')
            axes[0, 1].axis('off')
        
        # Boxplot par segment
        segments_order = ['VIP', 'Fidèle', 'Régulier', 'À risque', 'Perdu']
        segments_present = [s for s in segments_order if s in df['Segment_RFM'].unique()]
        
        if len(segments_present) > 0:
            sns.boxplot(x='Segment_RFM', y='montant', data=df,
                       order=segments_present, ax=axes[1, 0])
            axes[1, 0].set_title('Montant par segment')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].set_ylabel('Montant total ($)')
        else:
            axes[1, 0].text(0.5, 0.5, 'Pas de données pour le boxplot',
                           ha='center', va='center')
            axes[1, 0].set_title('Montant par segment')
            axes[1, 0].axis('off')
        
        # Scatter plot RFM
        if len(df) > 0:
            scatter = axes[1, 1].scatter(df['frequence'], df['montant'],
                                        c=df['jours_inactivite'], 
                                        s=df['panier_moyen']/10 + 20,
                                        cmap='viridis', alpha=0.6)
            axes[1, 1].set_xlabel('Fréquence (nombre de commandes)')
            axes[1, 1].set_ylabel('Montant total ($)')
            axes[1, 1].set_title('Analyse Fréquence vs Montant')
            axes[1, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 1], label='Jours depuis dernière commande')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analyse_rfm.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarde des résultats
        df.to_csv(f'{self.output_dir}/segmentation_rfm.csv', index=False)
        
        print(f"✅ Analyse RFM sauvegardée dans {self.output_dir}/")
        print(f"   • segmentation_rfm.csv")
        print(f"   • analyse_rfm.png")
        
        return df
    
    # 2. Analyse de corrélation
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
        
        # Visualisation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse de Corrélation', fontsize=16, fontweight='bold')
        
        # Heatmap de corrélation
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f',
                   cmap='coolwarm', center=0,
                   square=True, ax=axes[0, 0])
        axes[0, 0].set_title('Matrice de corrélation')
        
        # Corrélation prix vs quantité
        if len(df) > 0:
            axes[0, 1].scatter(df['prix'], df['quantite'], alpha=0.3)
            axes[0, 1].set_xlabel('Prix unitaire ($)')
            axes[0, 1].set_ylabel('Quantité commandée')
            axes[0, 1].set_title('Corrélation: Prix vs Quantité')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Ajout de la droite de régression
            if len(df) > 1:
                try:
                    z = np.polyfit(df['prix'], df['quantite'], 1)
                    p = np.poly1d(z)
                    axes[0, 1].plot(df['prix'], p(df['prix']), "r--", alpha=0.8,
                                   label=f'y = {z[0]:.2f}x + {z[1]:.2f}')
                    axes[0, 1].legend()
                except:
                    pass
        
        # Délai vs satisfaction
        if 'score_satisfaction' in df.columns:
            sns.boxplot(x='score_satisfaction', y='delai_livraison', 
                       data=df, ax=axes[1, 0])
            axes[1, 0].set_xlabel('Score de satisfaction (1-5)')
            axes[1, 0].set_ylabel('Délai de livraison (jours)')
            axes[1, 0].set_title('Délai de livraison vs Satisfaction')
        
        # Pair plot des principales corrélations
        if len(df) > 0:
            cols_to_plot = ['prix', 'quantite', 'delai_livraison', 'montant_ligne']
            cols_present = [col for col in cols_to_plot if col in df.columns]
            
            if len(cols_present) >= 2:
                try:
                    sns.pairplot(df[cols_present].sample(min(1000, len(df))),
                                diag_kind='kde', plot_kws={'alpha': 0.5})
                    plt.savefig(f'{self.output_dir}/pairplot_correlations.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                except:
                    pass
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analyse_correlation.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
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
        
        print(f"✅ Analyse de corrélation sauvegardée dans {self.output_dir}/")
        
        return df, correlation_matrix, corr_stats
    
    # 3. Analyse de saisonnalité
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
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Analyse de Saisonnalité', fontsize=16, fontweight='bold')
        
        # Série temporelle complète
        if len(df_mensuel) > 0:
            axes[0, 0].plot(df_mensuel.index, df_mensuel['chiffre_affaires'], 
                           marker='o', markersize=3, linewidth=1)
            axes[0, 0].set_title('Évolution du CA mensuel', fontsize=12)
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Chiffre d\'affaires ($)')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Moyenne mobile
        if len(df_mensuel) >= 3:
            try:
                df_mensuel['moyenne_mobile_3m'] = df_mensuel['chiffre_affaires'].rolling(
                    window=3, center=True).mean()
                axes[0, 1].plot(df_mensuel.index, df_mensuel['chiffre_affaires'], 
                               label='CA réel', alpha=0.7)
                axes[0, 1].plot(df_mensuel.index, df_mensuel['moyenne_mobile_3m'],
                               label='Moyenne mobile 3 mois', linewidth=2)
                axes[0, 1].set_title('Tendance avec moyenne mobile', fontsize=12)
                axes[0, 1].legend(fontsize=8)
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].tick_params(axis='x', rotation=45)
            except:
                axes[0, 1].text(0.5, 0.5, 'Données insuffisantes\npour moyenne mobile',
                               ha='center', va='center')
                axes[0, 1].set_title('Tendance')
                axes[0, 1].axis('off')
        
        # Boxplot par mois
        if len(df_mensuel) > 0:
            mois_data = []
            mois_labels = []
            for mois in range(1, 13):
                data_mois = df_mensuel[df_mensuel.index.month == mois]['chiffre_affaires'].values
                if len(data_mois) > 0:
                    mois_data.append(data_mois)
                    mois_labels.append(mois)
            
            if mois_data:
                axes[1, 0].boxplot(mois_data)
                axes[1, 0].set_xlabel('Mois')
                axes[1, 0].set_ylabel('Chiffre d\'affaires ($)')
                axes[1, 0].set_title('Distribution du CA par mois', fontsize=12)
                axes[1, 0].set_xticklabels([f'{m}' for m in mois_labels])
        
        # Analyse par jour de la semaine
        if len(df) > 0:
            # Extraire le nom du jour
            df['jour_semaine_nom'] = df.index.day_name()
            jours_ordre = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                          'Friday', 'Saturday', 'Sunday']
            
            ca_par_jour = df.groupby('jour_semaine_nom')['chiffre_affaires'].sum()
            ca_par_jour = ca_par_jour.reindex(jours_ordre)
            
            # Filtrer les jours avec des données
            ca_par_jour = ca_par_jour.dropna()
            
            if len(ca_par_jour) > 0:
                jour_labels_fr = ['Lun', 'Mar', 'Mer', 'Jeu', 'Ven', 'Sam', 'Dim']
                jour_labels = [jour_labels_fr[i] for i, jour in enumerate(jours_ordre) 
                              if jour in ca_par_jour.index]
                
                axes[1, 1].bar(range(len(ca_par_jour)), ca_par_jour.values)
                axes[1, 1].set_xticks(range(len(ca_par_jour)))
                axes[1, 1].set_xticklabels(jour_labels)
                axes[1, 1].set_title('CA par jour de la semaine', fontsize=12)
                axes[1, 1].set_ylabel('Chiffre d\'affaires ($)')
                axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analyse_saisonnalite.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Calcul des statistiques de saisonnalité
        stats_saisonnalite = {}
        if len(df_mensuel) > 0:
            try:
                stats_mois = df_mensuel.groupby('mois')['chiffre_affaires'].mean()
                if not stats_mois.empty:
                    stats_saisonnalite['mois_fort'] = int(stats_mois.idxmax())
                    stats_saisonnalite['mois_faible'] = int(stats_mois.idxmin())
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
        
        print(f"✅ Analyse de saisonnalité sauvegardée dans {self.output_dir}/")
        
        return df_mensuel, stats_saisonnalite
    
    # 4. Analyse du panier moyen et associations
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
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Analyse du Panier Moyen', fontsize=16, fontweight='bold')
        
        # Distribution du panier moyen
        if len(panier_stats) > 0:
            axes[0, 0].hist(panier_stats['montant_total'], bins=30,
                           color='skyblue', edgecolor='black', alpha=0.7)
            mean_val = panier_stats['montant_total'].mean()
            median_val = panier_stats['montant_total'].median()
            
            axes[0, 0].axvline(mean_val, color='red',
                              linestyle='--', label=f'Moyenne: ${mean_val:.2f}')
            axes[0, 0].axvline(median_val, color='green',
                              linestyle='--', label=f'Médiane: ${median_val:.2f}')
            axes[0, 0].set_xlabel('Montant du panier ($)')
            axes[0, 0].set_ylabel('Nombre de commandes')
            axes[0, 0].set_title('Distribution du panier moyen', fontsize=12)
            axes[0, 0].legend(fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Relation montant vs nombre de produits
        if len(panier_stats) > 0:
            scatter = axes[0, 1].scatter(panier_stats['nombre_produits'],
                                        panier_stats['montant_total'],
                                        c=panier_stats['quantite_totale'],
                                        alpha=0.6, cmap='viridis', s=30)
            axes[0, 1].set_xlabel('Nombre de produits différents')
            axes[0, 1].set_ylabel('Montant total ($)')
            axes[0, 1].set_title('Relation produits vs montant du panier', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0, 1], label='Quantité totale')
        
        # Analyse des associations de produits (simplifiée)
        # Groupement par commande
        produits_par_commande = df.groupby('orderNumber')['productCode'].apply(list)
        
        if len(produits_par_commande) > 0:
            # Statistiques simples sur les paniers
            stats_paniers = {
                'paniers_analyses': len(produits_par_commande),
                'panier_moyen_articles': panier_stats['nombre_produits'].mean(),
                'panier_moyen_montant': panier_stats['montant_total'].mean(),
                'panier_median_montant': panier_stats['montant_total'].median()
            }
            
            # Texte informatif
            info_text = f"""
            📊 Statistiques Panier:
            
            Paniers analysés: {stats_paniers['paniers_analyses']}
            Articles moyens par panier: {stats_paniers['panier_moyen_articles']:.1f}
            Montant moyen du panier: ${stats_paniers['panier_moyen_montant']:.2f}
            Montant médian du panier: ${stats_paniers['panier_median_montant']:.2f}
            
            📈 Répartition:
            • Panier < $50: {(panier_stats['montant_total'] < 50).sum()}
            • Panier $50-$200: {((panier_stats['montant_total'] >= 50) & 
                               (panier_stats['montant_total'] < 200)).sum()}
            • Panier > $200: {(panier_stats['montant_total'] >= 200).sum()}
            """
            
            axes[1, 0].text(0.1, 0.5, info_text, fontsize=10,
                           verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            axes[1, 0].axis('off')
            
            # Produits les plus fréquents dans les paniers
            if 'productName' in df.columns:
                top_produits = df['productName'].value_counts().head(10)
                if len(top_produits) > 0:
                    axes[1, 1].barh(range(len(top_produits)), top_produits.values)
                    axes[1, 1].set_yticks(range(len(top_produits)))
                    # Tronquer les noms longs
                    labels = [name[:20] + '...' if len(name) > 20 else name 
                             for name in top_produits.index]
                    axes[1, 1].set_yticklabels(labels)
                    axes[1, 1].set_xlabel('Nombre d\'occurrences')
                    axes[1, 1].set_title('Top 10 produits dans les paniers', fontsize=12)
                    axes[1, 1].grid(True, alpha=0.3, axis='x')
                else:
                    axes[1, 1].text(0.5, 0.5, 'Données produits\ninsuffisantes',
                                   ha='center', va='center')
                    axes[1, 1].axis('off')
            else:
                axes[1, 1].text(0.5, 0.5, 'Noms de produits\nnon disponibles',
                               ha='center', va='center')
                axes[1, 1].axis('off')
        else:
            for i in range(2):
                for j in range(2):
                    if i == 1 and j == 0:
                        axes[i, j].text(0.5, 0.5, 'Pas de données\nde panier',
                                       ha='center', va='center')
                        axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/analyse_panier.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarde des résultats
        try:
            panier_stats.to_csv(f'{self.output_dir}/stats_panier.csv')
        except:
            pass
        
        # Calcul des statistiques synthétiques
        stats_panier = {}
        if len(panier_stats) > 0:
            stats_panier = {
                'panier_moyen': panier_stats['montant_total'].mean(),
                'panier_median': panier_stats['montant_total'].median(),
                'produits_moyens': panier_stats['nombre_produits'].mean(),
                'paniers_analyses': len(panier_stats)
            }
            
            try:
                pd.Series(stats_panier).to_csv(f'{self.output_dir}/resume_panier.csv')
            except:
                pass
        
        print(f"✅ Analyse du panier moyen sauvegardée dans {self.output_dir}/")
        
        return panier_stats, None, stats_panier
    
    # 5. Performance par employé
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
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Analyse de Performance des Employés', fontsize=16, fontweight='bold')
        
        # Barres: CA par employé
        if len(df) > 0:
            df_sorted = df.sort_values('chiffre_affaires', ascending=True)  # Pour barh
            axes[0, 0].barh(range(len(df_sorted)), df_sorted['chiffre_affaires'])
            axes[0, 0].set_yticks(range(len(df_sorted)))
            # Tronquer les noms si nécessaire
            labels = [name[:15] + '...' if len(name) > 15 else name 
                     for name in df_sorted['nom_employe']]
            axes[0, 0].set_yticklabels(labels)
            axes[0, 0].set_xlabel('Chiffre d\'affaires ($)')
            axes[0, 0].set_title('CA par employé', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3, axis='x')
        
        # Scatter: Clients vs CA
        if len(df) > 0:
            scatter = axes[0, 1].scatter(df['nombre_clients'], df['chiffre_affaires'],
                                        alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
            axes[0, 1].set_xlabel('Nombre de clients')
            axes[0, 1].set_ylabel('Chiffre d\'affaires ($)')
            axes[0, 1].set_title('Relation clients vs CA', fontsize=12)
            axes[0, 1].grid(True, alpha=0.3)
            
            # Annoter les meilleurs performants
            if len(df) >= 3:
                top3 = df.nlargest(3, 'chiffre_affaires')
                for _, row in top3.iterrows():
                    axes[0, 1].annotate(row['nom_employe'].split()[-1],
                                       (row['nombre_clients'], row['chiffre_affaires']),
                                       fontsize=8, alpha=0.8)
        
        # Répartition par bureau
        if 'ville_bureau' in df.columns and len(df) > 0:
            bureau_stats = df.groupby('ville_bureau').agg({
                'chiffre_affaires': 'sum',
                'nombre_clients': 'sum'
            }).sort_values('chiffre_affaires', ascending=False).head(5)
            
            if len(bureau_stats) > 0:
                x = np.arange(len(bureau_stats))
                width = 0.35
                
                axes[1, 0].bar(x - width/2, bureau_stats['chiffre_affaires'], width,
                              label='CA', color='skyblue')
                axes2 = axes[1, 0].twinx()
                axes2.bar(x + width/2, bureau_stats['nombre_clients'], width,
                         label='Clients', color='lightcoral', alpha=0.7)
                
                axes[1, 0].set_xlabel('Bureau')
                axes[1, 0].set_ylabel('Chiffre d\'affaires ($)', color='skyblue')
                axes[1, 0].set_xticks(x)
                axes[1, 0].set_xticklabels(bureau_stats.index, rotation=45)
                axes[1, 0].tick_params(axis='y', labelcolor='skyblue')
                
                axes2.set_ylabel('Nombre de clients', color='lightcoral')
                axes2.tick_params(axis='y', labelcolor='lightcoral')
                
                axes[1, 0].set_title('Top 5 bureaux par performance', fontsize=12)
                axes[1, 0].grid(True, alpha=0.3, axis='y')
            else:
                axes[1, 0].text(0.5, 0.5, 'Données bureau\ninsuffisantes',
                               ha='center', va='center')
                axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Données de localisation\nnon disponibles',
                           ha='center', va='center')
            axes[1, 0].axis('off')
        
        # Statistiques globales
        if len(df) > 0:
            stats_text = f"""
            📊 Performance Globale:
            
            Employés analysés: {len(df)}
            CA total: ${df['chiffre_affaires'].sum():,.2f}
            CA moyen: ${df['chiffre_affaires'].mean():,.2f}
            Clients totaux: {int(df['nombre_clients'].sum())}
            Clients moyen/employé: {df['nombre_clients'].mean():.1f}
            
            🏆 Top Performeur:
            {df.loc[df['chiffre_affaires'].idxmax(), 'nom_employe'] if len(df) > 0 else 'N/A'}
            CA: ${df['chiffre_affaires'].max():,.2f}
            """
            
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10,
                           verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_employes.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sauvegarde des résultats
        try:
            df.to_csv(f'{self.output_dir}/performance_employes_detail.csv', index=False)
        except:
            pass
        
        # Rapport de synthèse
        rapport = []
        rapport.append("=" * 60)
        rapport.append("RAPPORT DE PERFORMANCE COMMERCIALE")
        rapport.append("=" * 60)
        rapport.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
        rapport.append(f"Nombre d'employés analysés: {len(df)}")
        rapport.append("")
        
        if len(df) > 0:
            rapport.append("🏆 TOP 3 COMMERCIAUX:")
            for i, (_, row) in enumerate(df.nlargest(3, 'chiffre_affaires').iterrows(), 1):
                rapport.append(f"  {i}. {row['nom_employe']}")
                rapport.append(f"     • CA: ${row['chiffre_affaires']:,.2f}")
                rapport.append(f"     • Clients: {int(row['nombre_clients'])}")
                rapport.append("")
            
            rapport.append("📊 STATISTIQUES D'ENSEMBLE:")
            rapport.append(f"  • CA total: ${df['chiffre_affaires'].sum():,.2f}")
            rapport.append(f"  • CA moyen par employé: ${df['chiffre_affaires'].mean():,.2f}")
            rapport.append(f"  • Clients totaux: {int(df['nombre_clients'].sum())}")
            rapport.append(f"  • Clients moyens par employé: {df['nombre_clients'].mean():.1f}")
        
        with open(f'{self.output_dir}/rapport_performance.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(rapport))
        
        print(f"✅ Analyse de performance sauvegardée dans {self.output_dir}/")
        
        return df
    
    def executer_toutes_analyses(self):
        """Exécute toutes les analyses avancées"""
        print("🚀 LANCEMENT DES ANALYSES AVANCÉES")
        print("=" * 50)
        
        results = {}
        
        analyses = [
            ('RFM', self.analyse_rfm),
            ('Corrélation', self.analyse_correlation),
            ('Saisonnalité', self.analyse_saisonnalite),
            ('Panier moyen', self.analyse_panier_associations),
            ('Performance', self.performance_employes)
        ]
        
        for nom, fonction in analyses:
            print(f"\n📊 Analyse {nom} en cours...")
            try:
                result = fonction()
                if result is not None:
                    results[nom.lower()] = result
                    print(f"   ✅ {nom}: Terminé")
                else:
                    print(f"   ⚠️  {nom}: Aucun résultat")
            except Exception as e:
                print(f"   ❌ {nom}: Erreur - {e}")
        
        print(f"\n" + "=" * 50)
        print(f"✅ ANALYSES TERMINÉES")
        print(f"📁 Résultats dans: {os.path.abspath(self.output_dir)}")
        
        return results

# Script principal simplifié
if __name__ == "__main__":
    print("🔧 INITIALISATION DES ANALYSES AVANCÉES")
    print("=" * 50)
    
    # Créer l'instance
    analyses = AnalysesAvancees()
    
    # Menu interactif
    while True:
        print("\n📋 MENU DES ANALYSES AVANCÉES")
        print("=" * 40)
        print("1. 📊 Exécuter TOUTES les analyses")
        print("2. 👥 Analyse RFM (segmentation clients)")
        print("3. 🔗 Analyse de corrélation")
        print("4. 📅 Analyse de saisonnalité")
        print("5. 🛒 Analyse du panier moyen")
        print("6. 🏆 Analyse de performance")
        print("7. 📁 Ouvrir le dossier des résultats")
        print("0. ❌ Retour")
        print("=" * 40)
        
        choix = input("\nVotre choix (0-7): ").strip()
        
        if choix == '1':
            print("\n⏳ Lancement de toutes les analyses...")
            results = analyses.executer_toutes_analyses()
        
        elif choix == '2':
            print("\n👥 Analyse RFM...")
            results = analyses.analyse_rfm()
        
        elif choix == '3':
            print("\n🔗 Analyse de corrélation...")
            results = analyses.analyse_correlation()
        
        elif choix == '4':
            print("\n📅 Analyse de saisonnalité...")
            results = analyses.analyse_saisonnalite()
        
        elif choix == '5':
            print("\n🛒 Analyse du panier moyen...")
            results = analyses.analyse_panier_associations()
        
        elif choix == '6':
            print("\n🏆 Analyse de performance...")
            results = analyses.performance_employes()
        
        elif choix == '7':
            import subprocess
            import platform
            path = os.path.abspath(analyses.output_dir)
            print(f"\n📁 Ouverture du dossier: {path}")
            
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
        
        elif choix == '0':
            print("\n👋 Retour au menu principal...")
            break
        
        else:
            print("❌ Choix invalide. Veuillez réessayer.")
        
        if choix in ['1', '2', '3', '4', '5', '6']:
            input("\nAppuyez sur Entrée pour continuer...")