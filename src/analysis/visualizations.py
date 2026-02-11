import pandas as pd
import mysql.connector
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration de la connexion
db_config = {
    'port': 3306,
    'host': 'localhost',
    'user': 'root',
    'password': 'Christevy',
    'database': 'projet_sql_py'
}

def run_query(query):
    conn = mysql.connector.connect(**db_config)
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# 1. Répartition par pays
df_country = run_query("SELECT country, COUNT(*) as count FROM customers GROUP BY country ORDER BY count DESC")

# 2. Statut des commandes
df_status = run_query("SELECT status, COUNT(*) as count FROM orders GROUP BY status")

# Création des graphiques
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1 : Pays
sns.barplot(data=df_country, x='count', y='country', ax=ax[0], palette='viridis')
ax[0].set_title('Top 10 Pays des Clients')

# Graphique 2 : Statuts
ax[1].pie(df_status['count'], labels=df_status['status'], autopct='%1.1f%%', startangle=140)
ax[1].set_title('Répartition des Statuts de Commande')

plt.tight_layout()
plt.savefig('visualizations/countries_and_status et Statut des commandes.png')
plt.show()


dt_10_products = run_query("""SELECT productName, SUM(quantityOrdered) as total_quantity
FROM orderdetails od JOIN products p ON od.productCode = p.productCode
GROUP BY productName ORDER BY total_quantity DESC LIMIT 10;""")
plt.figure(figsize=(12,6))
sns.barplot(data=dt_10_products, x='total_quantity', y='productName', palette='magma')
plt.title("Top 10 des Produits les Plus Vendus")
plt.xlabel("Quantité Totale Vendue")
plt.ylabel("Produit")
plt.tight_layout()
plt.savefig('visualizations/top_10_products.png')
plt.show()


 # 3. Répartition ventes et panier moyen
dt__ventes_panier = run_query("""
SELECT 
    SUM(quantityOrdered * priceEach) AS CA_Total,
    SUM(quantityOrdered * priceEach) / COUNT(DISTINCT orderNumber) AS Panier_Moyen
FROM orderdetails;
""")

# Transformer en format long
df_plot = dt__ventes_panier.melt(var_name="Metric", value_name="Amount")

plt.figure(figsize=(8,6))
sns.barplot(data=df_plot, x="Metric", y="Amount", hue="Metric", legend=False)
plt.title("Chiffre d'Affaires Total et Panier Moyen")
plt.ylabel("Montant en USD")
plt.tight_layout()
plt.savefig('visualizations/ca_panier_moyen.png')
plt.show()