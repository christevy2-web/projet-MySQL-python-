# visualisation_finale.py
import mysql.connector
from mysql.connector import pooling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Chargement des variables d'environnement
load_dotenv()

class MySQLVisualizer:
    def __init__(self):
        """Initialisation du pool de connexions MySQL"""
        try:
            # Utilisation de vos paramètres spécifiques
            self.connection_params = {
                'host': 'localhost',
                'user': 'root',
                'password': 'Christevy',
                'database': 'projet_sql_py'
            }
            
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                pool_reset_session=True,
                **self.connection_params
            )
            print("✅ Connexion à la base de données établie")
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            self.pool = None
        
        self.output_dir = "visualizations"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configuration du style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def get_connection(self):
        """Obtient une connexion"""
        if self.pool:
            try:
                return self.pool.get_connection()
            except:
                # Fallback à une connexion directe
                return mysql.connector.connect(**self.connection_params)
        else:
            return mysql.connector.connect(**self.connection_params)
    
    def execute_query(self, query, params=None):
        """Exécute une requête SQL et retourne un DataFrame"""
        connection = None
        try:
            connection = self.get_connection()
            df = pd.read_sql(query, connection, params=params)
            return df
        except Exception as e:
            print(f"❌ Erreur lors de l'exécution de la requête: {e}")
            print(f"Requête: {query[:200]}...")
            return pd.DataFrame()
        finally:
            if connection:
                connection.close()
    
    def convert_to_numeric(self, df, columns):
        """Convertit les colonnes en valeurs numériques"""
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def safe_plot(self, fig, filename):
        """Sauvegarde sécurisée d'un graphique"""
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
    
    # 1. Graphiques d'évolution temporelle des ventes
    def evolution_ventes_temps(self):
        """Graphiques d'évolution temporelle des ventes"""
        print("📈 Génération des graphiques d'évolution temporelle...")
        
        query_mois = """
        SELECT 
            DATE_FORMAT(o.orderDate, '%Y-%m') AS mois,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes,
            AVG(od.quantityOrdered * od.priceEach) AS panier_moyen
        FROM orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY DATE_FORMAT(o.orderDate, '%Y-%m')
        ORDER BY mois
        """
        
        df_mois = self.execute_query(query_mois)
        
        if df_mois.empty:
            print("❌ Aucune donnée temporelle trouvée")
            return None, None
        
        # Conversion numérique
        df_mois = self.convert_to_numeric(df_mois, 
            ['chiffre_affaires', 'nombre_commandes', 'panier_moyen'])
        
        # Remplacer les NaN par 0
        df_mois = df_mois.fillna(0)
        
        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Évolution Temporelle des Ventes', fontsize=16, fontweight='bold')
        
        # Graphique 1: Évolution CA par mois
        if len(df_mois) > 0:
            x = range(len(df_mois))
            axes[0, 0].plot(x, df_mois['chiffre_affaires'], 
                          marker='o', linewidth=2, color='steelblue', markersize=4)
            axes[0, 0].fill_between(x, df_mois['chiffre_affaires'], 
                                   alpha=0.3, color='steelblue')
            axes[0, 0].set_title('Évolution du chiffre d\'affaires mensuel', fontsize=12)
            axes[0, 0].set_xlabel('Mois')
            axes[0, 0].set_ylabel('Chiffre d\'affaires ($)')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(df_mois['mois'], rotation=45, fontsize=8)
            axes[0, 0].grid(True, alpha=0.3)
        
        # Graphique 2: Nombre de commandes par mois
        if len(df_mois) > 0:
            bars = axes[0, 1].bar(range(len(df_mois)), df_mois['nombre_commandes'], 
                                color='lightcoral', alpha=0.7, edgecolor='darkred')
            axes[0, 1].set_title('Nombre de commandes mensuelles', fontsize=12)
            axes[0, 1].set_xlabel('Mois')
            axes[0, 1].set_ylabel('Nombre de commandes')
            axes[0, 1].set_xticks(range(len(df_mois)))
            axes[0, 1].set_xticklabels(df_mois['mois'], rotation=45, fontsize=8)
            axes[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Graphique 3: Panier moyen par mois
        if len(df_mois) > 0:
            axes[1, 0].plot(range(len(df_mois)), df_mois['panier_moyen'], 
                          marker='s', linewidth=2, color='seagreen', markersize=4)
            axes[1, 0].set_title('Évolution du panier moyen mensuel', fontsize=12)
            axes[1, 0].set_xlabel('Mois')
            axes[1, 0].set_ylabel('Panier moyen ($)')
            axes[1, 0].set_xticks(range(len(df_mois)))
            axes[1, 0].set_xticklabels(df_mois['mois'], rotation=45, fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        
        # Graphique 4: Tableau des statistiques
        axes[1, 1].axis('off')
        if len(df_mois) > 0:
            stats_text = f"""
            📊 Statistiques Temporelles:
            
            Période analysée: {len(df_mois)} mois
            CA total: ${df_mois['chiffre_affaires'].sum():,.2f}
            CA moyen mensuel: ${df_mois['chiffre_affaires'].mean():,.2f}
            Commandes totales: {int(df_mois['nombre_commandes'].sum())}
            Panier moyen: ${df_mois['panier_moyen'].mean():,.2f}
            
            Mois avec CA max: {df_mois.loc[df_mois['chiffre_affaires'].idxmax(), 'mois']}
            CA maximum: ${df_mois['chiffre_affaires'].max():,.2f}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, 
                           verticalalignment='center', 
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.safe_plot(fig, 'evolution_ventes_temps.png')
        
        return df_mois, None
    
    # 2. Répartition géographique
    def repartition_geographique(self):
        """Analyse géographique"""
        print("🌍 Génération des graphiques géographiques...")
        
        query = """
        SELECT 
            c.country,
            COUNT(DISTINCT c.customerNumber) AS nombre_clients,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            AVG(od.quantityOrdered * od.priceEach) AS panier_moyen
        FROM customers c
        LEFT JOIN orders o ON c.customerNumber = o.customerNumber
        LEFT JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold') OR o.orderNumber IS NULL
        GROUP BY c.country
        HAVING COUNT(DISTINCT c.customerNumber) > 0
        ORDER BY chiffre_affaires DESC
        """
        
        df = self.execute_query(query)
        
        if df.empty:
            print("❌ Aucune donnée géographique trouvée")
            return None
        
        # Conversion numérique
        df = self.convert_to_numeric(df,
            ['nombre_clients', 'nombre_commandes', 'chiffre_affaires', 'panier_moyen'])
        df = df.fillna(0)
        
        # Graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse Géographique', fontsize=16, fontweight='bold')
        
        # Top 10 pays par CA
        if len(df) > 0:
            df_top10 = df.nlargest(10, 'chiffre_affaires')
            bars1 = axes[0, 0].barh(df_top10['country'], df_top10['chiffre_affaires'], 
                                   color=plt.cm.viridis(np.linspace(0, 1, len(df_top10))))
            axes[0, 0].set_title('Top 10 pays par chiffre d\'affaires', fontsize=12)
            axes[0, 0].set_xlabel('Chiffre d\'affaires ($)')
            axes[0, 0].invert_yaxis()
            axes[0, 0].grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, (country, ca) in enumerate(zip(df_top10['country'], df_top10['chiffre_affaires'])):
                axes[0, 0].text(ca + ca*0.01, i, f'${ca/1000:.1f}K', 
                               va='center', fontsize=8)
        
        # Top 10 pays par nombre de clients
        if len(df) > 0:
            df_top10_clients = df.nlargest(10, 'nombre_clients')
            bars2 = axes[0, 1].barh(df_top10_clients['country'], df_top10_clients['nombre_clients'], 
                                   color=plt.cm.plasma(np.linspace(0, 1, len(df_top10_clients))))
            axes[0, 1].set_title('Top 10 pays par nombre de clients', fontsize=12)
            axes[0, 1].set_xlabel('Nombre de clients')
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(True, alpha=0.3, axis='x')
        
        # Répartition CA par pays
        if len(df) > 0 and df['chiffre_affaires'].sum() > 0:
            ca_total = df['chiffre_affaires'].sum()
            df_top5 = df.nlargest(5, 'chiffre_affaires')
            autres = max(0, ca_total - df_top5['chiffre_affaires'].sum())
            
            valeurs = df_top5['chiffre_affaires'].tolist() + [autres]
            labels = df_top5['country'].tolist() + ['Autres']
            
            # S'assurer qu'on a des valeurs positives
            if sum(v > 0 for v in valeurs) > 0:
                wedges, texts, autotexts = axes[1, 0].pie(
                    [max(v, 0) for v in valeurs], 
                    labels=labels,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 0 else '',
                    startangle=90
                )
                axes[1, 0].set_title('Répartition du CA par pays', fontsize=12)
        
        # Scatter plot: CA vs Nombre de clients
        if len(df) > 0:
            mask = (df['chiffre_affaires'] > 0) & (df['nombre_clients'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = axes[1, 1].scatter(df_plot['nombre_clients'], df_plot['chiffre_affaires'],
                                            c=df_plot['panier_moyen'], 
                                            s=df_plot['nombre_commandes']*5,
                                            alpha=0.6, cmap='coolwarm', 
                                            edgecolors='black', linewidth=0.5)
                
                axes[1, 1].set_xlabel('Nombre de clients')
                axes[1, 1].set_ylabel('Chiffre d\'affaires ($)')
                axes[1, 1].set_title('Relation clients vs CA par pays', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=axes[1, 1], label='Panier moyen ($)')
        
        self.safe_plot(fig, 'repartition_geographique.png')
        
        # Sauvegarde des données
        try:
            df.to_csv(f'{self.output_dir}/donnees_geographiques.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/donnees_geographiques.csv")
        except:
            pass
        
        return df
    
    # 3. Ventes par gamme de produits
    def ventes_par_gamme(self):
        """Analyse des ventes par gamme de produits"""
        print("📊 Génération des graphiques par gamme de produits...")
        
        query = """
        SELECT 
            pl.productLine AS gamme,
            COUNT(DISTINCT od.productCode) AS nombre_produits,
            SUM(od.quantityOrdered) AS quantite_vendue,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            AVG(od.priceEach) AS prix_moyen,
            COUNT(DISTINCT o.customerNumber) AS nombre_clients,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes
        FROM productlines pl
        JOIN products p ON pl.productLine = p.productLine
        JOIN orderdetails od ON p.productCode = od.productCode
        JOIN orders o ON od.orderNumber = o.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY pl.productLine
        ORDER BY chiffre_affaires DESC
        """
        
        df = self.execute_query(query)
        
        if df.empty:
            print("❌ Aucune donnée de gamme trouvée")
            return None
        
        # Conversion numérique
        df = self.convert_to_numeric(df,
            ['nombre_produits', 'quantite_vendue', 'chiffre_affaires', 
             'prix_moyen', 'nombre_clients', 'nombre_commandes'])
        df = df.fillna(0)
        
        # Graphiques comparatifs
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse par Gamme de Produits', fontsize=16, fontweight='bold')
        
        # Barres: CA par gamme
        if len(df) > 0:
            colors_bar = plt.cm.Set3(range(len(df)))
            x_pos = np.arange(len(df))
            bars = axes[0, 0].bar(x_pos, df['chiffre_affaires'], 
                                 color=colors_bar, edgecolor='black', linewidth=1)
            axes[0, 0].set_title('Chiffre d\'affaires par gamme', fontsize=12)
            axes[0, 0].set_xlabel('Gamme de produits')
            axes[0, 0].set_ylabel('Chiffre d\'affaires ($)')
            axes[0, 0].set_xticks(x_pos)
            axes[0, 0].set_xticklabels(df['gamme'], rotation=45, fontsize=9)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars, df['chiffre_affaires']):
                if value > 0:
                    axes[0, 0].text(bar.get_x() + bar.get_width()/2., 
                                   bar.get_height() + bar.get_height()*0.01,
                                   f'${value/1000:.1f}K', 
                                   ha='center', va='bottom', fontsize=8)
        
        # Camembert: Répartition CA
        if len(df) > 0 and df['chiffre_affaires'].sum() > 0:
            wedges, texts, autotexts = axes[0, 1].pie(
                df['chiffre_affaires'], 
                labels=df['gamme'],
                autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                startangle=90,
                colors=colors_bar if 'colors_bar' in locals() else plt.cm.Set3(range(len(df)))
            )
            axes[0, 1].set_title('Répartition du CA par gamme', fontsize=12)
            
            # Style des pourcentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(8)
                autotext.set_fontweight('bold')
        
        # Quantités vendues par gamme
        if len(df) > 0:
            df_sorted_qte = df.sort_values('quantite_vendue', ascending=True)  # Pour avoir la plus grande en bas
            bars_qte = axes[1, 0].barh(range(len(df_sorted_qte)), df_sorted_qte['quantite_vendue'],
                                      color=plt.cm.Pastel1(range(len(df))))
            axes[1, 0].set_title('Quantités vendues par gamme', fontsize=12)
            axes[1, 0].set_xlabel('Quantité vendue')
            axes[1, 0].set_yticks(range(len(df_sorted_qte)))
            axes[1, 0].set_yticklabels(df_sorted_qte['gamme'])
            axes[1, 0].grid(True, alpha=0.3, axis='x')
            
            # Ajout des valeurs
            for i, (bar, value) in enumerate(zip(bars_qte, df_sorted_qte['quantite_vendue'])):
                if value > 0:
                    axes[1, 0].text(value + value*0.01, i, 
                                   f'{int(value):,}', 
                                   va='center', fontsize=8)
        
        # Multi-métriques normalisées
        if len(df) > 0:
            x = np.arange(len(df))
            width = 0.25
            
            # Normalisation en évitant la division par zéro
            ca_max = df['chiffre_affaires'].max()
            qte_max = df['quantite_vendue'].max()
            clients_max = df['nombre_clients'].max()
            
            ca_norm = (df['chiffre_affaires'] / ca_max * 100) if ca_max > 0 else 0
            qte_norm = (df['quantite_vendue'] / qte_max * 100) if qte_max > 0 else 0
            clients_norm = (df['nombre_clients'] / clients_max * 100) if clients_max > 0 else 0
            
            axes[1, 1].bar(x - width, ca_norm, width, label='CA', color='skyblue')
            axes[1, 1].bar(x, qte_norm, width, label='Quantité', color='lightgreen')
            axes[1, 1].bar(x + width, clients_norm, width, label='Clients', color='salmon')
            
            axes[1, 1].set_xlabel('Gamme de produits')
            axes[1, 1].set_ylabel('Valeur normalisée (%)')
            axes[1, 1].set_title('Comparaison multi-métriques', fontsize=12)
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(df['gamme'], rotation=45, fontsize=9)
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        self.safe_plot(fig, 'ventes_par_gamme.png')
        
        # Graphique supplémentaire: Relation prix vs quantité
        if len(df) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            
            mask = (df['prix_moyen'] > 0) & (df['quantite_vendue'] > 0)
            if mask.any():
                df_plot = df[mask]
                
                scatter = ax2.scatter(df_plot['prix_moyen'], df_plot['quantite_vendue'],
                                     s=df_plot['chiffre_affaires']/1000, 
                                     c=df_plot['nombre_clients'], alpha=0.7,
                                     cmap='viridis', edgecolors='black', linewidth=0.5)
                
                for i, row in df_plot.iterrows():
                    ax2.annotate(row['gamme'], (row['prix_moyen'], row['quantite_vendue']),
                                fontsize=9, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                ax2.set_xlabel('Prix moyen ($)')
                ax2.set_ylabel('Quantité vendue')
                ax2.set_title('Relation prix vs quantité par gamme', fontsize=14)
                ax2.grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=ax2, label='Nombre de clients')
                
                self.safe_plot(fig2, 'relation_prix_quantite.png')
        
        return df
    
    # 4. Top produits
    def top_produits(self):
        """Analyse des top produits"""
        print("🏆 Génération des graphiques des top produits...")
        
        query = """
        SELECT 
            p.productCode,
            p.productName,
            p.productLine,
            p.productVendor,
            SUM(od.quantityOrdered) AS quantite_totale,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            AVG(od.priceEach) AS prix_moyen,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes,
            COUNT(DISTINCT o.customerNumber) AS nombre_clients
        FROM products p
        JOIN orderdetails od ON p.productCode = od.productCode
        JOIN orders o ON od.orderNumber = o.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY p.productCode, p.productName, p.productLine, p.productVendor
        ORDER BY quantite_totale DESC
        LIMIT 20
        """
        
        df = self.execute_query(query)
        
        if df.empty:
            print("❌ Aucun produit trouvé")
            return None
        
        # Conversion numérique
        df = self.convert_to_numeric(df,
            ['quantite_totale', 'chiffre_affaires', 'prix_moyen', 
             'nombre_commandes', 'nombre_clients'])
        df = df.fillna(0)
        
        df_top10 = df.head(10)
        
        # Graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Top Produits', fontsize=16, fontweight='bold')
        
        # Top 10 par quantité
        if len(df_top10) > 0:
            noms_tronques = [name[:15] + '...' if len(name) > 15 else name 
                            for name in df_top10['productName']]
            bars1 = axes[0, 0].barh(range(len(df_top10)), df_top10['quantite_totale'],
                                   color=plt.cm.viridis(np.linspace(0, 1, len(df_top10))))
            axes[0, 0].set_title('Top 10 produits par quantité vendue', fontsize=12)
            axes[0, 0].set_xlabel('Quantité vendue')
            axes[0, 0].set_yticks(range(len(df_top10)))
            axes[0, 0].set_yticklabels(noms_tronques)
            axes[0, 0].grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, value in enumerate(df_top10['quantite_totale']):
                axes[0, 0].text(value + value*0.01, i, 
                               f'{int(value):,}', 
                               va='center', fontsize=8)
        
        # Top 10 par CA
        if len(df_top10) > 0:
            bars2 = axes[0, 1].barh(range(len(df_top10)), df_top10['chiffre_affaires'],
                                   color=plt.cm.plasma(np.linspace(0, 1, len(df_top10))))
            axes[0, 1].set_title('Top 10 produits par chiffre d\'affaires', fontsize=12)
            axes[0, 1].set_xlabel('Chiffre d\'affaires ($)')
            axes[0, 1].set_yticks(range(len(df_top10)))
            axes[0, 1].set_yticklabels(noms_tronques)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, value in enumerate(df_top10['chiffre_affaires']):
                axes[0, 1].text(value + value*0.01, i, 
                               f'${value/1000:.1f}K', 
                               va='center', fontsize=8)
        
        # Répartition par gamme
        if len(df) > 0:
            gamme_counts = df.head(20)['productLine'].value_counts()
            if len(gamme_counts) > 0:
                wedges, texts, autotexts = axes[1, 0].pie(
                    gamme_counts.values, 
                    labels=gamme_counts.index,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                    startangle=90
                )
                axes[1, 0].set_title('Répartition des top 20 produits par gamme', fontsize=12)
        
        # Scatter: Prix vs Quantité
        if len(df) > 0:
            mask = (df['prix_moyen'] > 0) & (df['quantite_totale'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = axes[1, 1].scatter(df_plot['prix_moyen'], df_plot['quantite_totale'],
                                            c=df_plot['chiffre_affaires'], 
                                            s=df_plot['nombre_commandes']*5,
                                            cmap='coolwarm', alpha=0.6, 
                                            edgecolors='black', linewidth=0.5)
                
                axes[1, 1].set_xlabel('Prix moyen ($)')
                axes[1, 1].set_ylabel('Quantité vendue')
                axes[1, 1].set_title('Relation prix vs quantité vendue', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=axes[1, 1], label='Chiffre d\'affaires ($)')
        
        self.safe_plot(fig, 'top_produits.png')
        
        # Sauvegarde des données
        try:
            df.to_csv(f'{self.output_dir}/top_produits_donnees.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/top_produits_donnees.csv")
        except:
            pass
        
        return df
    
    # 5. Analyse des stocks
    def analyse_stocks(self):
        """Analyse de l'état des stocks"""
        print("📦 Génération de l'analyse des stocks...")
        
        query = """
        SELECT 
            p.productCode,
            p.productName,
            p.productLine,
            p.quantityInStock,
            p.buyPrice,
            p.MSRP,
            (SELECT SUM(od.quantityOrdered) 
             FROM orderdetails od 
             JOIN orders o ON od.orderNumber = o.orderNumber 
             WHERE od.productCode = p.productCode 
             AND o.status NOT IN ('Cancelled', 'On Hold')) AS quantite_vendue
        FROM products p
        ORDER BY p.quantityInStock
        """
        
        df = self.execute_query(query)
        
        if df.empty:
            print("❌ Aucun produit en stock trouvé")
            return None, None
        
        # Conversion numérique
        df = self.convert_to_numeric(df,
            ['quantityInStock', 'buyPrice', 'MSRP', 'quantite_vendue'])
        df = df.fillna(0)
        
        # Classification des stocks
        conditions = [
            df['quantityInStock'] == 0,
            (df['quantityInStock'] > 0) & (df['quantityInStock'] <= 10),
            (df['quantityInStock'] > 10) & (df['quantityInStock'] <= 50),
            df['quantityInStock'] > 50
        ]
        choices = ['Rupture', 'Faible', 'Normal', 'Élevé']
        df['niveau_stock'] = np.select(conditions, choices, default='Normal')
        
        # Graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse des Stocks', fontsize=16, fontweight='bold')
        
        # Distribution des niveaux de stock
        if len(df) > 0:
            stock_counts = df['niveau_stock'].value_counts()
            colors = ['red', 'orange', 'green', 'blue']
            
            available_colors = []
            for i, niveau in enumerate(stock_counts.index):
                if niveau == 'Rupture':
                    available_colors.append('red')
                elif niveau == 'Faible':
                    available_colors.append('orange')
                elif niveau == 'Normal':
                    available_colors.append('green')
                else:
                    available_colors.append('blue')
            
            bars = axes[0, 0].bar(range(len(stock_counts)), stock_counts.values, 
                                 color=available_colors)
            axes[0, 0].set_title('Distribution des niveaux de stock', fontsize=12)
            axes[0, 0].set_xlabel('Niveau de stock')
            axes[0, 0].set_ylabel('Nombre de produits')
            axes[0, 0].set_xticks(range(len(stock_counts)))
            axes[0, 0].set_xticklabels(stock_counts.index)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars, stock_counts.values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2., 
                               bar.get_height() + 0.5,
                               str(value), 
                               ha='center', va='bottom', fontsize=9)
        
        # Produits à réapprovisionner
        produits_risque = df[df['niveau_stock'].isin(['Rupture', 'Faible'])]
        if not produits_risque.empty:
            produits_risque = produits_risque.head(10)
            noms_tronques = [name[:15] + '...' if len(name) > 15 else name 
                            for name in produits_risque['productName']]
            
            colors_risque = ['red' if x == 'Rupture' else 'orange' 
                            for x in produits_risque['niveau_stock']]
            
            bars_risque = axes[0, 1].barh(range(len(produits_risque)), 
                                         produits_risque['quantityInStock'],
                                         color=colors_risque)
            axes[0, 1].set_title('Produits à réapprovisionner', fontsize=12)
            axes[0, 1].set_xlabel('Quantité en stock')
            axes[0, 1].set_yticks(range(len(produits_risque)))
            axes[0, 1].set_yticklabels(noms_tronques)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Ajouter les ventes historiques si disponibles
            for i, (stock, ventes) in enumerate(zip(produits_risque['quantityInStock'], 
                                                   produits_risque['quantite_vendue'])):
                if ventes > 0:
                    axes[0, 1].text(stock + stock*0.01, i, 
                                   f'Ventes: {int(ventes)}', 
                                   va='center', fontsize=7, color='darkred')
        else:
            axes[0, 1].text(0.5, 0.5, '✅ Stock satisfaisant\nAucun produit critique',
                           ha='center', va='center', fontsize=12,
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            axes[0, 1].set_title('État des stocks', fontsize=12)
            axes[0, 1].axis('off')
        
        # Distribution du stock
        if len(df) > 0:
            hist_values, bins, patches = axes[1, 0].hist(df['quantityInStock'], bins=30,
                                                       color='skyblue', edgecolor='black', alpha=0.7)
            mean_stock = df['quantityInStock'].mean()
            axes[1, 0].axvline(mean_stock, color='red', 
                              linestyle='--', linewidth=2,
                              label=f'Moyenne: {mean_stock:.1f}')
            axes[1, 0].set_xlabel('Quantité en stock')
            axes[1, 0].set_ylabel('Nombre de produits')
            axes[1, 0].set_title('Distribution des quantités en stock', fontsize=12)
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Stock par gamme
        if len(df) > 0:
            stock_par_gamme = df.groupby('productLine')['quantityInStock'].sum().sort_values()
            bars_gamme = axes[1, 1].barh(range(len(stock_par_gamme)), stock_par_gamme.values,
                                        color=plt.cm.Set3(range(len(stock_par_gamme))))
            axes[1, 1].set_title('Stock par gamme (quantité)', fontsize=12)
            axes[1, 1].set_xlabel('Quantité en stock')
            axes[1, 1].set_yticks(range(len(stock_par_gamme)))
            axes[1, 1].set_yticklabels(stock_par_gamme.index)
            axes[1, 1].grid(True, alpha=0.3, axis='x')
            
            # Ajout des valeurs
            for i, value in enumerate(stock_par_gamme.values):
                axes[1, 1].text(value + value*0.01, i, 
                               f'{int(value):,}', 
                               va='center', fontsize=8)
        
        self.safe_plot(fig, 'analyse_stocks.png')
        
        # Alertes
        alertes = produits_risque.copy()
        if not alertes.empty:
            try:
                alertes.to_csv(f'{self.output_dir}/alertes_stock.csv', index=False)
                print(f"⚠️  Alertes stock générées: {len(alertes)} produits")
                print(f"📊 Fichier: {self.output_dir}/alertes_stock.csv")
            except:
                pass
        
        return df, alertes
    
    # 6. Analyse des clients
    def analyse_clients(self):
        """Analyse de la répartition du CA par clients"""
        print("👥 Génération de l'analyse des clients...")
        
        query = """
        SELECT 
            c.customerNumber,
            c.customerName,
            c.country,
            c.city,
            COUNT(DISTINCT o.orderNumber) AS nombre_commandes,
            SUM(od.quantityOrdered * od.priceEach) AS chiffre_affaires,
            AVG(od.quantityOrdered * od.priceEach) AS panier_moyen,
            MAX(o.orderDate) AS derniere_commande
        FROM customers c
        LEFT JOIN orders o ON c.customerNumber = o.customerNumber
        LEFT JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.status NOT IN ('Cancelled', 'On Hold')
        GROUP BY c.customerNumber, c.customerName, c.country, c.city
        HAVING COUNT(DISTINCT o.orderNumber) > 0
        ORDER BY chiffre_affaires DESC
        """
        
        df = self.execute_query(query)
        
        if df.empty:
            print("❌ Aucun client avec commandes trouvé")
            # Essayer une requête plus simple
            query_simple = """
            SELECT 
                c.customerNumber,
                c.customerName,
                c.country,
                c.city
            FROM customers c
            LIMIT 50
            """
            df = self.execute_query(query_simple)
            
            if df.empty:
                print("❌ Aucun client trouvé dans la base")
                return None, None, None
        
        # Conversion numérique si les colonnes existent
        if 'chiffre_affaires' in df.columns:
            df = self.convert_to_numeric(df,
                ['nombre_commandes', 'chiffre_affaires', 'panier_moyen'])
            df = df.fillna(0)
            
            # Pareto
            df = df.sort_values('chiffre_affaires', ascending=False)
            df['cumulative_sum'] = df['chiffre_affaires'].cumsum()
            df['cumulative_percent'] = df['cumulative_sum'] / df['chiffre_affaires'].sum() * 100
            
            pareto_cutoff = 80
            df['segment'] = np.where(df['cumulative_percent'] <= pareto_cutoff, 
                                    'VIP', 'Autres')
            
            top_clients = df[df['segment'] == 'VIP']
        else:
            # Si pas de données de vente
            df['chiffre_affaires'] = 0
            df['nombre_commandes'] = 0
            df['panier_moyen'] = 0
            df['segment'] = 'Nouveau'
            top_clients = pd.DataFrame()
        
        # Graphiques
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Analyse des Clients', fontsize=16, fontweight='bold')
        
        # Courbe de Pareto (si données disponibles)
        if 'chiffre_affaires' in df.columns and df['chiffre_affaires'].sum() > 0:
            bars = axes[0, 0].bar(range(len(df)), df['chiffre_affaires'], 
                                 color=['crimson' if x <= pareto_cutoff else 'lightgray' 
                                        for x in df['cumulative_percent']])
            axes[0, 0].set_xlabel('Clients (triés par CA décroissant)')
            axes[0, 0].set_ylabel('Chiffre d\'affaires ($)', color='black')
            
            ax2 = axes[0, 0].twinx()
            ax2.plot(range(len(df)), df['cumulative_percent'], 
                    color='navy', linewidth=2)
            ax2.set_ylabel('Pourcentage cumulé (%)', color='navy')
            ax2.axhline(y=pareto_cutoff, color='green', linestyle='--', 
                       linewidth=2, label=f'{pareto_cutoff}% du CA')
            ax2.legend(loc='lower right', fontsize=8)
            axes[0, 0].set_title('Analyse Pareto des clients', fontsize=12)
            axes[0, 0].grid(True, alpha=0.3, axis='y')
        else:
            axes[0, 0].text(0.5, 0.5, 'Pas de données de vente\npour l\'analyse Pareto',
                           ha='center', va='center', fontsize=12)
            axes[0, 0].set_title('Analyse Pareto', fontsize=12)
            axes[0, 0].axis('off')
        
        # Top 10 clients
        if len(df) > 0 and 'chiffre_affaires' in df.columns:
            top10 = df.head(10)
            noms_tronques = [name[:15] + '...' if len(name) > 15 else name 
                            for name in top10['customerName']]
            
            bars_top = axes[0, 1].barh(range(len(top10)), top10['chiffre_affaires'],
                                      color=plt.cm.coolwarm(np.linspace(0, 1, len(top10))))
            axes[0, 1].set_title('Top 10 clients par CA', fontsize=12)
            axes[0, 1].set_xlabel('Chiffre d\'affaires ($)')
            axes[0, 1].set_yticks(range(len(top10)))
            axes[0, 1].set_yticklabels(noms_tronques)
            axes[0, 1].grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, value in enumerate(top10['chiffre_affaires']):
                if value > 0:
                    axes[0, 1].text(value + value*0.01, i, 
                                   f'${value/1000:.1f}K', 
                                   va='center', fontsize=8)
        else:
            axes[0, 1].text(0.5, 0.5, 'Liste des clients',
                           ha='center', va='center', fontsize=12)
            axes[0, 1].set_title('Clients', fontsize=12)
            
            # Afficher quelques noms de clients
            if len(df) > 0:
                client_text = "\n".join(df['customerName'].head(5).tolist())
                if len(df) > 5:
                    client_text += "\n..."
                axes[0, 1].text(0.5, 0.3, f"Clients trouvés: {len(df)}\n\n{client_text}",
                               ha='center', va='center', fontsize=10)
            axes[0, 1].axis('off')
        
        # Distribution du CA
        if 'chiffre_affaires' in df.columns and df['chiffre_affaires'].sum() > 0:
            axes[1, 0].hist(df['chiffre_affaires'], bins=30, 
                           color='skyblue', edgecolor='black', alpha=0.7)
            mean_ca = df['chiffre_affaires'].mean()
            axes[1, 0].axvline(mean_ca, color='red', 
                              linestyle='--', linewidth=2,
                              label=f'Moyenne: ${mean_ca:,.0f}')
            axes[1, 0].set_xlabel('Chiffre d\'affaires ($)')
            axes[1, 0].set_ylabel('Nombre de clients')
            axes[1, 0].set_title('Distribution du CA par client', fontsize=12)
            axes[1, 0].legend(fontsize=8)
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'Répartition géographique',
                           ha='center', va='center', fontsize=12)
            axes[1, 0].set_title('Clients par pays', fontsize=12)
            
            # Afficher la répartition par pays
            if 'country' in df.columns and len(df) > 0:
                pays_counts = df['country'].value_counts().head(10)
                if len(pays_counts) > 0:
                    bars_pays = axes[1, 0].bar(range(len(pays_counts)), pays_counts.values,
                                              color=plt.cm.Set3(range(len(pays_counts))))
                    axes[1, 0].set_xlabel('Pays')
                    axes[1, 0].set_ylabel('Nombre de clients')
                    axes[1, 0].set_xticks(range(len(pays_counts)))
                    axes[1, 0].set_xticklabels(pays_counts.index, rotation=45, fontsize=8)
                    axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Scatter: Commandes vs CA ou autre visualisation
        if 'nombre_commandes' in df.columns and 'chiffre_affaires' in df.columns:
            mask = (df['chiffre_affaires'] > 0) & (df['nombre_commandes'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = axes[1, 1].scatter(df_plot['nombre_commandes'], df_plot['chiffre_affaires'],
                                            c=df_plot['panier_moyen'] if 'panier_moyen' in df_plot.columns else 'blue',
                                            alpha=0.6, edgecolors='black', linewidth=0.5)
                axes[1, 1].set_xlabel('Nombre de commandes')
                axes[1, 1].set_ylabel('Chiffre d\'affaires ($)')
                axes[1, 1].set_title('Relation commandes vs CA', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
                
                if 'panier_moyen' in df_plot.columns:
                    plt.colorbar(scatter, ax=axes[1, 1], label='Panier moyen ($)')
            else:
                axes[1, 1].text(0.5, 0.5, 'Informations clients',
                               ha='center', va='center', fontsize=12)
                axes[1, 1].set_title('Détails clients', fontsize=12)
                
                if len(df) > 0:
                    info_text = f"""
                    Total clients: {len(df)}
                    
                    Pays représentés: {df['country'].nunique() if 'country' in df.columns else 'N/A'}
                    Villes: {df['city'].nunique() if 'city' in df.columns else 'N/A'}
                    
                    Données disponibles:
                    • CA: {'Oui' if 'chiffre_affaires' in df.columns else 'Non'}
                    • Commandes: {'Oui' if 'nombre_commandes' in df.columns else 'Non'}
                    """
                    axes[1, 1].text(0.5, 0.5, info_text, 
                                   ha='center', va='center', fontsize=10)
                axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'Pas de données de vente\npour cette analyse',
                           ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Données manquantes', fontsize=12)
            axes[1, 1].axis('off')
        
        self.safe_plot(fig, 'analyse_clients.png')
        
        # Export
        try:
            df.to_csv(f'{self.output_dir}/segmentation_clients.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/segmentation_clients.csv")
        except:
            pass
        
        return df, top_clients, df.shape[0]
    
    def generer_rapport_complet(self):
        """Génère tous les rapports et visualisations"""
        print("=" * 60)
        print("GÉNÉRATION DU RAPPORT COMPLET")
        print("=" * 60)
        print(f"Base de données: projet_sql_py")
        print(f"Utilisateur: root")
        print("=" * 60)
        
        results = {}
        
        try:
            print("\n1. 📈 Analyse de l'évolution temporelle...")
            results['evolution'] = self.evolution_ventes_temps()
            
            print("\n2. 🌍 Analyse géographique...")
            results['geographie'] = self.repartition_geographique()
            
            print("\n3. 📊 Analyse par gamme de produits...")
            results['gammes'] = self.ventes_par_gamme()
            
            print("\n4. 🏆 Analyse des top produits...")
            results['top_produits'] = self.top_produits()
            
            print("\n5. 📦 Analyse des stocks...")
            results['stocks'] = self.analyse_stocks()
            
            print("\n6. 👥 Analyse des clients...")
            results['clients'] = self.analyse_clients()
            
            print("\n" + "=" * 60)
            print("✅ RAPPORT COMPLÈTEMENT GÉNÉRÉ!")
            print("=" * 60)
            
            # Générer un rapport de synthèse
            self.generer_rapport_synthese()
            
        except Exception as e:
            print(f"\n❌ Erreur: {e}")
            import traceback
            traceback.print_exc()
        
        return results
    
    def generer_rapport_synthese(self):
        """Génère un rapport de synthèse"""
        rapport = []
        rapport.append("=" * 70)
        rapport.append("RAPPORT DE SYNTHÈSE - ANALYSE COMMERCIALE")
        rapport.append("=" * 70)
        rapport.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        rapport.append(f"Base de données: projet_sql_py")
        rapport.append("")
        rapport.append("📁 VISUALISATIONS GÉNÉRÉES:")
        rapport.append("  • evolution_ventes_temps.png")
        rapport.append("  • repartition_geographique.png")
        rapport.append("  • ventes_par_gamme.png")
        rapport.append("  • relation_prix_quantite.png")
        rapport.append("  • top_produits.png")
        rapport.append("  • analyse_stocks.png")
        rapport.append("  • analyse_clients.png")
        rapport.append("")
        rapport.append("📊 DONNÉES EXPORTÉES:")
        rapport.append("  • donnees_geographiques.csv")
        rapport.append("  • top_produits_donnees.csv")
        rapport.append("  • alertes_stock.csv (si applicable)")
        rapport.append("  • segmentation_clients.csv")
        rapport.append("")
        rapport.append("📍 EMPLACEMENT:")
        rapport.append(f"  Dossier: {os.path.abspath(self.output_dir)}")
        rapport.append("")
        rapport.append("=" * 70)
        
        with open(f'{self.output_dir}/rapport_synthese.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(rapport))
        
        print(f"📄 Rapport de synthèse généré: {self.output_dir}/rapport_synthese.txt")
        print(f"📁 Tous les fichiers sont dans: {os.path.abspath(self.output_dir)}")

# Menu interactif
def main():
    """Fonction principale"""
    print("🚀 DÉMARRAGE DE L'ANALYSE COMMERCIALE")
    print("=" * 50)
    print("Configuration:")
    print("  • Base: projet_sql_py")
    print("  • Utilisateur: root")
    print("  • Serveur: localhost")
    print("=" * 50)
    
    visualizer = MySQLVisualizer()
    
    while True:
        print("\n📋 MENU PRINCIPAL")
        print("=" * 40)
        print("1. 📊 Générer le rapport COMPLET")
        print("2. 📈 Évolution temporelle")
        print("3. 🌍 Analyse géographique")
        print("4. 🏷️  Ventes par gamme")
        print("5. 🏆 Top produits")
        print("6. 📦 Analyse des stocks")
        print("7. 👥 Analyse des clients")
        print("8. 📁 Ouvrir le dossier des résultats")
        print("0. ❌ Quitter")
        print("=" * 40)
        
        choix = input("\nVotre choix (0-8): ").strip()
        
        if choix == '1':
            print("\n⏳ Lancement de l'analyse complète...")
            visualizer.generer_rapport_complet()
        
        elif choix == '2':
            print("\n📈 Analyse de l'évolution temporelle...")
            visualizer.evolution_ventes_temps()
        
        elif choix == '3':
            print("\n🌍 Analyse géographique...")
            visualizer.repartition_geographique()
        
        elif choix == '4':
            print("\n🏷️  Analyse par gamme de produits...")
            visualizer.ventes_par_gamme()
        
        elif choix == '5':
            print("\n🏆 Analyse des top produits...")
            visualizer.top_produits()
        
        elif choix == '6':
            print("\n📦 Analyse des stocks...")
            visualizer.analyse_stocks()
        
        elif choix == '7':
            print("\n👥 Analyse des clients...")
            visualizer.analyse_clients()
        
        elif choix == '8':
            import subprocess
            import platform
            path = os.path.abspath(visualizer.output_dir)
            print(f"\n📁 Ouverture du dossier: {path}")
            
            if platform.system() == "Windows":
                os.startfile(path)
            elif platform.system() == "Darwin":  # macOS
                subprocess.Popen(["open", path])
            else:  # Linux
                subprocess.Popen(["xdg-open", path])
        
        elif choix == '0':
            print("\n👋 Au revoir!")
            break
        
        else:
            print("❌ Choix invalide. Veuillez réessayer.")
        
        if choix != '0':
            input("\nAppuyez sur Entrée pour continuer...")

if __name__ == "__main__":
    main()