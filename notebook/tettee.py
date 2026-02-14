# visualisation_finale.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
            return None
        
        # Conversion numérique
        df_mois = self.convert_to_numeric(df_mois, 
            ['chiffre_affaires', 'nombre_commandes', 'panier_moyen'])
        
        # Remplacer les NaN par 0
        df_mois = df_mois.fillna(0)
        
        # Graphique 1: Évolution CA par mois
        if len(df_mois) > 0:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            x = range(len(df_mois))
            ax1.plot(x, df_mois['chiffre_affaires'], 
                   marker='o', linewidth=2, color='steelblue', markersize=6)
            ax1.fill_between(x, df_mois['chiffre_affaires'], 
                           alpha=0.3, color='steelblue')
            ax1.set_title('Évolution du chiffre d\'affaires mensuel', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Mois')
            ax1.set_ylabel('Chiffre d\'affaires ($)')
            ax1.set_xticks(x)
            ax1.set_xticklabels(df_mois['mois'], rotation=45, fontsize=10)
            ax1.grid(True, alpha=0.3)
            
            # Ajout des valeurs
            for i, (mois, ca) in enumerate(zip(df_mois['mois'], df_mois['chiffre_affaires'])):
                ax1.annotate(f'${ca:,.0f}', (i, ca), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
            
            self.safe_plot(fig1, 'evolution_ca_mensuel.png')
        
        # Graphique 2: Nombre de commandes par mois
        if len(df_mois) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            bars = ax2.bar(range(len(df_mois)), df_mois['nombre_commandes'], 
                         color='lightcoral', alpha=0.7, edgecolor='darkred', linewidth=1)
            ax2.set_title('Nombre de commandes mensuelles', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Mois')
            ax2.set_ylabel('Nombre de commandes')
            ax2.set_xticks(range(len(df_mois)))
            ax2.set_xticklabels(df_mois['mois'], rotation=45, fontsize=10)
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars, df_mois['nombre_commandes']):
                if value > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                           f'{int(value)}', ha='center', va='bottom', fontsize=9)
            
            self.safe_plot(fig2, 'evolution_commandes_mensuelles.png')
        
        # Graphique 3: Panier moyen par mois
        if len(df_mois) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            ax3.plot(range(len(df_mois)), df_mois['panier_moyen'], 
                   marker='s', linewidth=2, color='seagreen', markersize=6)
            ax3.set_title('Évolution du panier moyen mensuel', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Mois')
            ax3.set_ylabel('Panier moyen ($)')
            ax3.set_xticks(range(len(df_mois)))
            ax3.set_xticklabels(df_mois['mois'], rotation=45, fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            # Ajout des valeurs
            for i, (mois, pm) in enumerate(zip(df_mois['mois'], df_mois['panier_moyen'])):
                ax3.annotate(f'${pm:,.0f}', (i, pm), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
            
            self.safe_plot(fig3, 'evolution_panier_moyen_mensuel.png')
        
        # Graphique 4: Tableau des statistiques
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        ax4.axis('off')
        if len(df_mois) > 0:
            stats_text = f"""
            📊 Statistiques Temporelles:
            
            Période analysée: {len(df_mois)} mois
            CA total: ${df_mois['chiffre_affaires'].sum():,.2f}
            CA moyen mensuel: ${df_mois['chiffre_affaires'].mean():,.2f}
            Commandes totales: {int(df_mois['nombre_commandes'].sum())}
            Commandes moyennes/mois: {df_mois['nombre_commandes'].mean():.1f}
            Panier moyen global: ${df_mois['panier_moyen'].mean():,.2f}
            
            📈 Meilleures performances:
            
            Mois avec CA max: {df_mois.loc[df_mois['chiffre_affaires'].idxmax(), 'mois']}
            CA maximum: ${df_mois['chiffre_affaires'].max():,.2f}
            
            Mois avec plus de commandes: {df_mois.loc[df_mois['nombre_commandes'].idxmax(), 'mois']}
            Commandes max: {int(df_mois['nombre_commandes'].max())}
            
            Mois avec panier max: {df_mois.loc[df_mois['panier_moyen'].idxmax(), 'mois']}
            Panier max: ${df_mois['panier_moyen'].max():,.2f}
            """
            ax4.text(0.1, 0.5, stats_text, fontsize=12, 
                   verticalalignment='center', 
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        self.safe_plot(fig4, 'statistiques_temporelles.png')
        
        # Sauvegarde des données
        try:
            df_mois.to_csv(f'{self.output_dir}/donnees_temporelles.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/donnees_temporelles.csv")
        except:
            pass
        
        return df_mois
    
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
        
        # Graphique 1: top 10 pays par CA
        if len(df) > 0:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            df_top10 = df.nlargest(10, 'chiffre_affaires')
            bars1 = ax1.barh(df_top10['country'], df_top10['chiffre_affaires'], 
                           color=plt.cm.viridis(np.linspace(0, 1, len(df_top10))))
            ax1.set_title('Top 10 pays par chiffre d\'affaires', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Chiffre d\'affaires ($)')
            ax1.invert_yaxis()
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, (country, ca) in enumerate(zip(df_top10['country'], df_top10['chiffre_affaires'])):
                ax1.text(ca + ca*0.01, i, f'${ca/1000:.1f}K', 
                       va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig1, 'top_pays_ca.png')
        
        # Graphique 2: Top 10 pays par nombre de clients
        if len(df) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            df_top10_clients = df.nlargest(10, 'nombre_clients')
            bars2 = ax2.barh(df_top10_clients['country'], df_top10_clients['nombre_clients'], 
                           color=plt.cm.plasma(np.linspace(0, 1, len(df_top10_clients))))
            ax2.set_title('Top 10 pays par nombre de clients', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Nombre de clients')
            ax2.invert_yaxis()
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, (country, clients) in enumerate(zip(df_top10_clients['country'], df_top10_clients['nombre_clients'])):
                ax2.text(clients + clients*0.01, i, f'{int(clients)}', 
                       va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig2, 'top10_pays_clients.png')
        
        # Graphique 3: Répartition CA par pays (camembert)
        if len(df) > 0 and df['chiffre_affaires'].sum() > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            ca_total = df['chiffre_affaires'].sum()
            df_top5 = df.nlargest(5, 'chiffre_affaires')
            autres = max(0, ca_total - df_top5['chiffre_affaires'].sum())
            
            valeurs = df_top5['chiffre_affaires'].tolist() + [autres]
            labels = df_top5['country'].tolist() + ['Autres pays']
            colors = plt.cm.Set3(range(len(valeurs)))
            
            # S'assurer qu'on a des valeurs positives
            if sum(v > 0 for v in valeurs) > 0:
                wedges, texts, autotexts = ax3.pie(
                    [max(v, 0) for v in valeurs], 
                    labels=labels,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                    startangle=90,
                    colors=colors,
                    textprops={'fontsize': 12}
                )
                ax3.set_title('Répartition du chiffre d\'affaires par pays', fontsize=14, fontweight='bold')
                
                # Style des pourcentages
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(10)
                    autotext.set_fontweight('bold')
                
                self.safe_plot(fig3, 'repartition_ca_pays.png')
        
        # Graphique 4: Scatter plot - CA vs Nombre de clients
        if len(df) > 0:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            mask = (df['chiffre_affaires'] > 0) & (df['nombre_clients'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = ax4.scatter(df_plot['nombre_clients'], df_plot['chiffre_affaires'],
                                     c=df_plot['panier_moyen'], 
                                     s=df_plot['nombre_commandes']*5,
                                     alpha=0.6, cmap='coolwarm', 
                                     edgecolors='black', linewidth=0.5)
                
                ax4.set_xlabel('Nombre de clients', fontsize=12)
                ax4.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
                ax4.set_title('Relation clients vs chiffre d\'affaires par pays', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Ajouter les noms des pays
                for i, row in df_plot.iterrows():
                    ax4.annotate(row['country'], 
                               (row['nombre_clients'], row['chiffre_affaires']),
                               fontsize=9, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                
                plt.colorbar(scatter, ax=ax4, label='Panier moyen ($)')
                
                self.safe_plot(fig4, 'relation_clients_ca_pays.png')
        
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
        
        # Graphique 1: Barres - CA par gamme
        if len(df) > 0:
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            colors_bar = plt.cm.Set3(range(len(df)))
            x_pos = np.arange(len(df))
            bars = ax1.bar(x_pos, df['chiffre_affaires'], 
                         color=colors_bar, edgecolor='black', linewidth=1)
            ax1.set_title('Chiffre d\'affaires par gamme de produits', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Gamme de produits')
            ax1.set_ylabel('Chiffre d\'affaires ($)')
            ax1.set_xticks(x_pos)
            ax1.set_xticklabels(df['gamme'], rotation=45, fontsize=10)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars, df['chiffre_affaires']):
                if value > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., 
                           bar.get_height() + bar.get_height()*0.01,
                           f'${value/1000:.1f}K', 
                           ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig1, 'ca_par_gamme.png')
        
        # Graphique 2: Camembert - Répartition CA par gamme
        if len(df) > 0 and df['chiffre_affaires'].sum() > 0:
            fig2, ax2 = plt.subplots(figsize=(10, 8))
            colors_pie = plt.cm.Set3(range(len(df)))
            wedges, texts, autotexts = ax2.pie(
                df['chiffre_affaires'], 
                labels=df['gamme'],
                autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                startangle=90,
                colors=colors_pie,
                textprops={'fontsize': 12}
            )
            ax2.set_title('Répartition du chiffre d\'affaires par gamme', fontsize=14, fontweight='bold')
            
            # Style des pourcentages
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontsize(10)
                autotext.set_fontweight('bold')
            
            self.safe_plot(fig2, 'repartition_ca_gamme.png')
        
        # Graphique 3: Quantités vendues par gamme
        if len(df) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            df_sorted_qte = df.sort_values('quantite_vendue', ascending=True)
            bars_qte = ax3.barh(range(len(df_sorted_qte)), df_sorted_qte['quantite_vendue'],
                              color=plt.cm.Pastel1(range(len(df))))
            ax3.set_title('Quantités vendues par gamme', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Quantité vendue')
            ax3.set_yticks(range(len(df_sorted_qte)))
            ax3.set_yticklabels(df_sorted_qte['gamme'])
            ax3.grid(True, alpha=0.3, axis='x')
            
            # Ajout des valeurs
            for i, (bar, value) in enumerate(zip(bars_qte, df_sorted_qte['quantite_vendue'])):
                if value > 0:
                    ax3.text(value + value*0.01, i, 
                           f'{int(value):,}', 
                           va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig3, 'quantites_par_gamme.png')
        
        # Graphique 4: Comparaison multi-métriques normalisées
        if len(df) > 0:
            fig4, ax4 = plt.subplots(figsize=(12, 6))
            x = np.arange(len(df))
            width = 0.25
            
            # Normalisation en évitant la division par zéro
            ca_max = df['chiffre_affaires'].max()
            qte_max = df['quantite_vendue'].max()
            clients_max = df['nombre_clients'].max()
            
            ca_norm = (df['chiffre_affaires'] / ca_max * 100) if ca_max > 0 else 0
            qte_norm = (df['quantite_vendue'] / qte_max * 100) if qte_max > 0 else 0
            clients_norm = (df['nombre_clients'] / clients_max * 100) if clients_max > 0 else 0
            
            ax4.bar(x - width, ca_norm, width, label='Chiffre d\'affaires', color='skyblue')
            ax4.bar(x, qte_norm, width, label='Quantité vendue', color='lightgreen')
            ax4.bar(x + width, clients_norm, width, label='Nombre de clients', color='salmon')
            
            ax4.set_xlabel('Gamme de produits', fontsize=12)
            ax4.set_ylabel('Valeur normalisée (%)', fontsize=12)
            ax4.set_title('Comparaison multi-métriques par gamme', fontsize=14, fontweight='bold')
            ax4.set_xticks(x)
            ax4.set_xticklabels(df['gamme'], rotation=45, fontsize=10)
            ax4.legend(fontsize=10)
            ax4.grid(True, alpha=0.3, axis='y')
            
            self.safe_plot(fig4, 'comparaison_multimetriques_gamme.png')
        
        # Graphique 5: Relation prix vs quantité
        if len(df) > 0:
            fig5, ax5 = plt.subplots(figsize=(12, 8))
            
            mask = (df['prix_moyen'] > 0) & (df['quantite_vendue'] > 0)
            if mask.any():
                df_plot = df[mask]
                
                scatter = ax5.scatter(df_plot['prix_moyen'], df_plot['quantite_vendue'],
                                     s=df_plot['chiffre_affaires']/1000, 
                                     c=df_plot['nombre_clients'], alpha=0.7,
                                     cmap='viridis', edgecolors='black', linewidth=0.5)
                
                for i, row in df_plot.iterrows():
                    ax5.annotate(row['gamme'], (row['prix_moyen'], row['quantite_vendue']),
                                fontsize=10, ha='center', va='center',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
                
                ax5.set_xlabel('Prix moyen ($)', fontsize=12)
                ax5.set_ylabel('Quantité vendue', fontsize=12)
                ax5.set_title('Relation prix vs quantité par gamme', fontsize=14, fontweight='bold')
                ax5.grid(True, alpha=0.3)
                
                plt.colorbar(scatter, ax=ax5, label='Nombre de clients')
                
                self.safe_plot(fig5, 'relation_prix_quantite_gamme.png')
        
        # Sauvegarde des données
        try:
            df.to_csv(f'{self.output_dir}/donnees_par_gamme.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/donnees_par_gamme.csv")
        except:
            pass
        
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
        
        # Graphique 1: Top 10 produits par quantité
        if len(df_top10) > 0:
            fig1, ax1 = plt.subplots(figsize=(12, 8))
            noms_tronques = [name + '...' if len(name) > 100 else name 
                            for name in df_top10['productName']]
            bars1 = ax1.barh(range(len(df_top10)), df_top10['quantite_totale'],
                           color=plt.cm.viridis(np.linspace(0, 1, len(df_top10))))
            ax1.set_title('Top 10 produits par quantité vendue', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Quantité vendue', fontsize=12)
            ax1.set_yticks(range(len(df_top10)))
            ax1.set_yticklabels(noms_tronques, fontsize=10)
            ax1.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, value in enumerate(df_top10['quantite_totale']):
                ax1.text(value + value*0.01, i, 
                       f'{int(value):,}', 
                       va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig1, 'top10_produits_quantite.png')
        
        # Graphique 2: Top 10 produits par CA
        if len(df_top10) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            df_top10_ca = df.nlargest(10, 'chiffre_affaires')
            noms_tronques_ca = [name[:20] + '...' if len(name) > 20 else name 
                               for name in df_top10_ca['productName']]
            bars2 = ax2.barh(range(len(df_top10_ca)), df_top10_ca['chiffre_affaires'],
                           color=plt.cm.plasma(np.linspace(0, 1, len(df_top10_ca))))
            ax2.set_title('Top 10 produits par chiffre d\'affaires', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Chiffre d\'affaires ($)', fontsize=12)
            ax2.set_yticks(range(len(df_top10_ca)))
            ax2.set_yticklabels(noms_tronques_ca, fontsize=10)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, value in enumerate(df_top10_ca['chiffre_affaires']):
                ax2.text(value + value*0.01, i, 
                       f'${value/1000:.1f}K', 
                       va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig2, 'top10_produits_ca.png')
        
        # Graphique 3: Répartition par gamme des top produits
        if len(df) > 0:
            fig3, ax3 = plt.subplots(figsize=(10, 8))
            gamme_counts = df.head(20)['productLine'].value_counts()
            if len(gamme_counts) > 0:
                colors_pie = plt.cm.Set3(range(len(gamme_counts)))
                wedges, texts, autotexts = ax3.pie(
                    gamme_counts.values, 
                    labels=gamme_counts.index,
                    autopct=lambda pct: f'{pct:.1f}%' if pct > 1 else '',
                    startangle=90,
                    colors=colors_pie,
                    textprops={'fontsize': 12}
                )
                ax3.set_title('Répartition des top produits par gamme', fontsize=14, fontweight='bold')
                
                # Style des pourcentages
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(10)
                    autotext.set_fontweight('bold')
                
                self.safe_plot(fig3, 'repartition_top20_produits_gamme.png')
        
        # Graphique 4: Scatter - Prix vs Quantité
        if len(df) > 0:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            mask = (df['prix_moyen'] > 0) & (df['quantite_totale'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = ax4.scatter(df_plot['prix_moyen'], df_plot['quantite_totale'],
                                     c=df_plot['chiffre_affaires'], 
                                     s=df_plot['nombre_commandes']*5,
                                     cmap='coolwarm', alpha=0.6, 
                                     edgecolors='black', linewidth=0.5)
                
                ax4.set_xlabel('Prix moyen ($)', fontsize=12)
                ax4.set_ylabel('Quantité vendue', fontsize=12)
                ax4.set_title('Relation prix vs quantité vendue', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                # Ajouter les noms des produits (quelques-uns seulement pour éviter la surcharge)
                for i, row in df_plot.head(10).iterrows():
                    nom_court = row['productName'][:15] + '...' if len(row['productName']) > 15 else row['productName']
                    ax4.annotate(nom_court, 
                               (row['prix_moyen'], row['quantite_totale']),
                               fontsize=9, ha='center', va='center',
                               bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
                
                plt.colorbar(scatter, ax=ax4, label='Chiffre d\'affaires ($)')
                
                self.safe_plot(fig4, 'relation_prix_quantite_produits.png')
        
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
        
        # Graphique 1: Distribution des niveaux de stock
        if len(df) > 0:
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            stock_counts = df['niveau_stock'].value_counts()
            
            colors_map = {
                'Rupture': 'red',
                'Faible': 'orange', 
                'Normal': 'green',
                'Élevé': 'blue'
            }
            colors = [colors_map.get(niveau, 'gray') for niveau in stock_counts.index]
            
            bars = ax1.bar(range(len(stock_counts)), stock_counts.values, 
                         color=colors, edgecolor='black', linewidth=1)
            ax1.set_title('Distribution des niveaux de stock', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Niveau de stock')
            ax1.set_ylabel('Nombre de produits')
            ax1.set_xticks(range(len(stock_counts)))
            ax1.set_xticklabels(stock_counts.index)
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars, stock_counts.values):
                ax1.text(bar.get_x() + bar.get_width()/2., 
                       bar.get_height() + 0.5,
                       str(value), 
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            self.safe_plot(fig1, 'distribution_niveaux_stock.png')
        
        # Graphique 2: Produits à réapprovisionner
        produits_risque = df[df['niveau_stock'].isin(['Rupture', 'Faible'])]
        if not produits_risque.empty:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            produits_risque = produits_risque.sort_values('quantityInStock').head(15)
            noms_tronques = [name[:20] + '...' if len(name) > 20 else name 
                            for name in produits_risque['productName']]
            
            colors_risque = ['red' if x == 'Rupture' else 'orange' 
                            for x in produits_risque['niveau_stock']]
            
            bars_risque = ax2.barh(range(len(produits_risque)), 
                                  produits_risque['quantityInStock'],
                                  color=colors_risque, edgecolor='black', linewidth=1)
            ax2.set_title('Produits à réapprovisionner (urgence)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Quantité en stock', fontsize=12)
            ax2.set_ylabel('Produits')
            ax2.set_yticks(range(len(produits_risque)))
            ax2.set_yticklabels(noms_tronques, fontsize=10)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les ventes historiques
            for i, (stock, ventes, niveau) in enumerate(zip(produits_risque['quantityInStock'], 
                                                           produits_risque['quantite_vendue'],
                                                           produits_risque['niveau_stock'])):
                if ventes > 0:
                    ax2.text(stock + stock*0.01, i, 
                           f'Ventes: {int(ventes)} | {niveau}', 
                           va='center', fontsize=9)
            
            self.safe_plot(fig2, 'produits_reapprovisionner.png')
        else:
            # Graphique informatif si pas de produits en risque
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.text(0.5, 0.5, '✅ Stock satisfaisant\nAucun produit critique',
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            ax2.set_title('État des stocks', fontsize=14, fontweight='bold')
            ax2.axis('off')
            self.safe_plot(fig2, 'etat_stocks_satisfaisant.png')
        
        # Graphique 3: Distribution des quantités en stock
        if len(df) > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            # Filtrer les valeurs extrêmes pour une meilleure visualisation
            q99 = df['quantityInStock'].quantile(0.99)
            df_hist = df[df['quantityInStock'] <= q99]
            
            hist_values, bins, patches = ax3.hist(df_hist['quantityInStock'], bins=30,
                                                 color='skyblue', edgecolor='black', alpha=0.7)
            mean_stock = df['quantityInStock'].mean()
            median_stock = df['quantityInStock'].median()
            
            ax3.axvline(mean_stock, color='red', 
                       linestyle='--', linewidth=2,
                       label=f'Moyenne: {mean_stock:.1f}')
            ax3.axvline(median_stock, color='green', 
                       linestyle='--', linewidth=2,
                       label=f'Médiane: {median_stock:.1f}')
            
            ax3.set_xlabel('Quantité en stock', fontsize=12)
            ax3.set_ylabel('Nombre de produits', fontsize=12)
            ax3.set_title('Distribution des quantités en stock', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            self.safe_plot(fig3, 'distribution_quantites_stock.png')
        
        # Graphique 4: Stock par gamme
        if len(df) > 0:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            stock_par_gamme = df.groupby('productLine')['quantityInStock'].sum().sort_values()
            colors_gamme = plt.cm.Set3(range(len(stock_par_gamme)))
            bars_gamme = ax4.barh(range(len(stock_par_gamme)), stock_par_gamme.values,
                                color=colors_gamme, edgecolor='black', linewidth=1)
            ax4.set_title('Stock total par gamme de produits', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Quantité en stock', fontsize=12)
            ax4.set_yticks(range(len(stock_par_gamme)))
            ax4.set_yticklabels(stock_par_gamme.index, fontsize=10)
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Ajout des valeurs
            for i, value in enumerate(stock_par_gamme.values):
                ax4.text(value + value*0.01, i, 
                       f'{int(value):,}', 
                       va='center', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig4, 'stock_par_gamme.png')
        
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
            
            # Graphique 1: Courbe de Pareto
            if df['chiffre_affaires'].sum() > 0:
                fig1, ax1 = plt.subplots(figsize=(12, 6))
                bars = ax1.bar(range(len(df)), df['chiffre_affaires'], 
                             color=['crimson' if x <= pareto_cutoff else 'lightgray' 
                                    for x in df['cumulative_percent']])
                ax1.set_xlabel('Clients (triés par CA décroissant)', fontsize=12)
                ax1.set_ylabel('Chiffre d\'affaires ($)', color='black')
                ax1.tick_params(axis='y', labelcolor='black')
                
                ax2 = ax1.twinx()
                ax2.plot(range(len(df)), df['cumulative_percent'], 
                        color='navy', linewidth=2, marker='o', markersize=3)
                ax2.set_ylabel('Pourcentage cumulé (%)', color='navy')
                ax2.tick_params(axis='y', labelcolor='navy')
                ax2.axhline(y=pareto_cutoff, color='green', linestyle='--', 
                           linewidth=2, label=f'{pareto_cutoff}% du CA')
                ax2.legend(loc='lower right', fontsize=10)
                
                ax1.set_title('Analyse Pareto des clients (80/20)', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3, axis='y')
                
                self.safe_plot(fig1, 'analyse_pareto_clients.png')
        else:
            top_clients = pd.DataFrame()
        
        # Graphique 2: Top 10 clients par CA (si données disponibles)
        if 'chiffre_affaires' in df.columns and len(df) > 0:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            top10 = df.head(10)
            noms_tronques = [name[:20] + '...' if len(name) > 20 else name 
                            for name in top10['customerName']]
            
            bars_top = ax2.barh(range(len(top10)), top10['chiffre_affaires'],
                              color=plt.cm.coolwarm(np.linspace(0, 1, len(top10))),
                              edgecolor='black', linewidth=1)
            ax2.set_title('Top 10 clients par chiffre d\'affaires', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Chiffre d\'affaires ($)', fontsize=12)
            ax2.set_yticks(range(len(top10)))
            ax2.set_yticklabels(noms_tronques, fontsize=10)
            ax2.grid(True, alpha=0.3, axis='x')
            
            # Ajouter les valeurs
            for i, (name, value, pays) in enumerate(zip(top10['customerName'], 
                                                       top10['chiffre_affaires'],
                                                       top10['country'])):
                if value > 0:
                    ax2.text(value + value*0.01, i, 
                           f'${value/1000:.1f}K\n({pays})', 
                           va='center', fontsize=9)
            
            self.safe_plot(fig2, 'top10_clients_ca.png')
        
        # Graphique 3: Distribution du CA par client
        if 'chiffre_affaires' in df.columns and df['chiffre_affaires'].sum() > 0:
            fig3, ax3 = plt.subplots(figsize=(12, 6))
            
            # Filtrer les valeurs extrêmes
            q95 = df['chiffre_affaires'].quantile(0.95)
            df_hist = df[df['chiffre_affaires'] <= q95]
            
            ax3.hist(df_hist['chiffre_affaires'], bins=30, 
                   color='skyblue', edgecolor='black', alpha=0.7)
            mean_ca = df['chiffre_affaires'].mean()
            median_ca = df['chiffre_affaires'].median()
            
            ax3.axvline(mean_ca, color='red', 
                       linestyle='--', linewidth=2,
                       label=f'Moyenne: ${mean_ca:,.0f}')
            ax3.axvline(median_ca, color='green', 
                       linestyle='--', linewidth=2,
                       label=f'Médiane: ${median_ca:,.0f}')
            
            ax3.set_xlabel('Chiffre d\'affaires ($)', fontsize=12)
            ax3.set_ylabel('Nombre de clients', fontsize=12)
            ax3.set_title('Distribution du chiffre d\'affaires par client', fontsize=14, fontweight='bold')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
            
            self.safe_plot(fig3, 'distribution_ca_clients.png')
        
        # Graphique 4: Scatter - Commandes vs CA
        if 'nombre_commandes' in df.columns and 'chiffre_affaires' in df.columns:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            mask = (df['chiffre_affaires'] > 0) & (df['nombre_commandes'] > 0)
            if mask.any():
                df_plot = df[mask]
                scatter = ax4.scatter(df_plot['nombre_commandes'], df_plot['chiffre_affaires'],
                                     c=df_plot['panier_moyen'] if 'panier_moyen' in df_plot.columns else 'blue',
                                     s=100, alpha=0.6, edgecolors='black', linewidth=0.5,
                                     cmap='viridis')
                ax4.set_xlabel('Nombre de commandes', fontsize=12)
                ax4.set_ylabel('Chiffre d\'affaires ($)', fontsize=12)
                ax4.set_title('Relation commandes vs chiffre d\'affaires', fontsize=14, fontweight='bold')
                ax4.grid(True, alpha=0.3)
                
                if 'panier_moyen' in df_plot.columns:
                    plt.colorbar(scatter, ax=ax4, label='Panier moyen ($)')
                
                self.safe_plot(fig4, 'relation_commandes_ca_clients.png')
        
        # Graphique 5: Répartition géographique des clients
        if 'country' in df.columns and len(df) > 0:
            fig5, ax5 = plt.subplots(figsize=(12, 6))
            pays_counts = df['country'].value_counts().head(15)
            colors_pays = plt.cm.Set3(range(len(pays_counts)))
            bars_pays = ax5.bar(range(len(pays_counts)), pays_counts.values,
                              color=colors_pays, edgecolor='black', linewidth=1)
            ax5.set_title('Top 15 pays par nombre de clients', fontsize=14, fontweight='bold')
            ax5.set_xlabel('Pays', fontsize=12)
            ax5.set_ylabel('Nombre de clients', fontsize=12)
            ax5.set_xticks(range(len(pays_counts)))
            ax5.set_xticklabels(pays_counts.index, rotation=45, fontsize=10)
            ax5.grid(True, alpha=0.3, axis='y')
            
            # Ajout des valeurs
            for bar, value in zip(bars_pays, pays_counts.values):
                ax5.text(bar.get_x() + bar.get_width()/2., 
                       bar.get_height() + 0.5,
                       str(value), 
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            self.safe_plot(fig5, 'repartition_clients_pays.png')
        
        # Sauvegarde des données
        try:
            df.to_csv(f'{self.output_dir}/donnees_clients.csv', index=False)
            print(f"📊 Données sauvegardées: {self.output_dir}/donnees_clients.csv")
        except:
            pass
        
        return df, top_clients if 'top_clients' in locals() else None, None
    
    def generate_all_graphs(self):
        """Génère tous les graphiques un par un"""
        print("="*60)
        print("🚀 GÉNÉRATION DE TOUS LES GRAPHIQUES")
        print("="*60)
        
        # Graphiques d'évolution temporelle
        print("\n" + "="*60)
        print("📈 GRAPHIQUES D'ÉVOLUTION TEMPORELLE")
        print("="*60)
        self.evolution_ventes_temps()
        
        # Graphiques géographiques
        print("\n" + "="*60)
        print("🌍 GRAPHIQUES GÉOGRAPHIQUES")
        print("="*60)
        self.repartition_geographique()
        
        # Graphiques par gamme
        print("\n" + "="*60)
        print("📊 GRAPHIQUES PAR GAMME DE PRODUITS")
        print("="*60)
        self.ventes_par_gamme()
        
        # Graphiques top produits
        print("\n" + "="*60)
        print("🏆 GRAPHIQUES TOP PRODUITS")
        print("="*60)
        self.top_produits()
        
        # Graphiques stocks
        print("\n" + "="*60)
        print("📦 GRAPHIQUES ANALYSE DES STOCKS")
        print("="*60)
        self.analyse_stocks()
        
        # Graphiques clients
        print("\n" + "="*60)
        print("👥 GRAPHIQUES ANALYSE DES CLIENTS")
        print("="*60)
        self.analyse_clients()
        
        print("\n" + "="*60)
        print("✅ GÉNÉRATION DE TOUS LES GRAPHIQUES TERMINÉE")
        print(f"📁 Les graphiques sont disponibles dans le dossier: {self.output_dir}")
        print("="*60)


if __name__ == "__main__":
    # Création de l'instance
    visualizer = MySQLVisualizer()
    
    # Génération de tous les graphiques un par un
    visualizer.generate_all_graphs()
    
    # Alternative : générer un type spécifique de graphiques
    # visualizer.evolution_ventes_temps()
    # visualizer.repartition_geographique()
    # visualizer.ventes_par_gamme()
    # visualizer.top_produits()
    # visualizer.analyse_stocks()
    # visualizer.analyse_clients()