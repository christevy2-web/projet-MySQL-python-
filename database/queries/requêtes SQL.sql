
-- ==============================
-- Table : tableau global
-- ==============================
CREATE TABLE projet_sql_py.tableau_global AS
SELECT 'Clients' AS Type, COUNT(*) AS Total FROM customers
UNION
SELECT 'Employés', COUNT(*) FROM employees
UNION
SELECT 'Bureaux', COUNT(*) FROM offices
UNION
SELECT 'Commandes', COUNT(*) FROM orders
UNION
SELECT 'Produits', COUNT(*) FROM products
UNION
SELECT 'Gammes de produits', COUNT(*) FROM productlines;
SELECT * FROM projet_sql_py.tableau_global;




-- ==============================
-- Table : repartition_geographique_des_clients
-- ==============================
CREATE TABLE projet_sql_py.repartition_geographique_des_clients AS
SELECT country,
       COUNT(*) AS nb_clients
FROM customers
GROUP BY country
ORDER BY nb_clients DESC;
SELECT * FROM projet_sql_py.repartition_geographique_des_clients;


-- ==============================
-- Table : statut_des_commandes
-- ==============================
CREATE TABLE projet_sql_py.statut_des_commandes AS
SELECT status, COUNT(*) AS total
FROM orders
GROUP BY status;
SELECT * FROM projet_sql_py.statut_des_commandes;


-- ================================================
-- Table : montant total des ventes et panier moyen
-- =================================================
CREATE TABLE projet_sql_py.montant_total_des_ventes_et_panier_moyen AS
SELECT 
    SUM(quantityOrdered * priceEach) AS CA_Total,
    SUM(quantityOrdered * priceEach) / COUNT(DISTINCT orderNumber) AS Panier_Moyen
FROM orderdetails;
SELECT * FROM projet_sql_py.montant_total_des_ventes_et_panier_moyen;


-- =======================================
-- Table : Les 10 produits les plus vendus
-- =======================================
CREATE TABLE projet_sql_py.Les_10_produits_les_plus_vendus AS
SELECT productName, SUM(quantityOrdered) AS total_vendu
FROM orderdetails od
JOIN products p ON od.productCode = p.productCode
GROUP BY productName
ORDER BY total_vendu DESC
LIMIT 10;
SELECT * FROM projet_sql_py.Les_10_produits_les_plus_vendus;
