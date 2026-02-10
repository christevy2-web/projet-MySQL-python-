drop table projet_sql_py.ventes_complete;


-- =======================================
-- Analyse des ventes avec WINDOW FUNCTIONS
-- =======================================
CREATE TABLE projet_sql_py.ventes_complete AS
WITH ventes_base AS (
    SELECT 
        c.customerNumber,
        c.customerName,
        o.orderNumber,
        o.orderDate,
        o.status,
        od.productCode,
        od.quantityOrdered,
        od.priceEach,
        (od.quantityOrdered * od.priceEach) AS montant_vente,
        EXTRACT(YEAR FROM o.orderDate) AS annee_vente,
        EXTRACT(MONTH FROM o.orderDate) AS mois_vente,
        CONCAT(EXTRACT(YEAR FROM o.orderDate), '-', LPAD(EXTRACT(MONTH FROM o.orderDate), 2, '0')) AS annee_mois
    FROM orders o
    INNER JOIN customers c ON o.customerNumber = c.customerNumber
    INNER JOIN orderdetails od ON o.orderNumber = od.orderNumber
    WHERE o.status NOT IN ('Cancelled', 'On Hold')
),
ventes_par_mois AS (
    SELECT 
        customerNumber,
        annee_mois,
        SUM(montant_vente) AS ventes_mois
    FROM ventes_base
    GROUP BY customerNumber, annee_mois
)
SELECT 
    vb.*,
    -- Rang des ventes par client
    ROW_NUMBER() OVER (PARTITION BY vb.customerNumber ORDER BY vb.montant_vente DESC) AS rang_vente_client,
    RANK() OVER (PARTITION BY vb.customerNumber ORDER BY vb.montant_vente DESC) AS rang_vente_client_avec_egalites,
    
    -- Total des ventes par client
    SUM(vb.montant_vente) OVER (PARTITION BY vb.customerNumber) AS total_ventes_client,
    
    -- Pourcentage de la ligne par rapport au total de la commande
    vb.montant_vente / SUM(vb.montant_vente) OVER (PARTITION BY vb.orderNumber) * 100 AS pourcentage_commande,

    
    -- Moyenne mobile sur 3 mois par client
    AVG(vb.montant_vente) OVER (
        PARTITION BY vb.customerNumber 
        ORDER BY vb.orderDate 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ) AS moyenne_mobile_3mois
    
FROM ventes_base vb;

SELECT * FROM projet_sql_py.ventes_complete;



















-- ========================================================
-- Hiérarchie complète des employés avec niveau et chemin
-- ========================================================
WITH RECURSIVE hierarchie_employes AS (
    -- Employés racines (ceux qui n'ont pas de supérieur)
    SELECT 
        employeeNumber,
        lastName,
        firstName,
        jobTitle,
        reportsTo,
        CONCAT(firstName, ' ', lastName) AS chemin_hierarchique,
        0 AS niveau,
        officeCode
    FROM employees
    WHERE reportsTo IS NULL
    
    UNION ALL
    
    -- Employés subordonnés
    SELECT 
        e.employeeNumber,
        e.lastName,
        e.firstName,
        e.jobTitle,
        e.reportsTo,
        CONCAT(he.chemin_hierarchique, ' → ', e.firstName, ' ', e.lastName) AS chemin_hierarchique,
        he.niveau + 1 AS niveau,
        e.officeCode
    FROM employees e
    INNER JOIN hierarchie_employes he ON e.reportsTo = he.employeeNumber
)
SELECT 
    he.*,
    o.city AS ville_bureau,
    o.country AS pays_bureau,
    COUNT(DISTINCT c.customerNumber) AS nombre_clients_geres,
    COALESCE(SUM(od.quantityOrdered * od.priceEach), 0) AS chiffre_affaires
FROM hierarchie_employes he
LEFT JOIN offices o ON he.officeCode = o.officeCode
LEFT JOIN customers c ON he.employeeNumber = c.salesRepEmployeeNumber
LEFT JOIN orders ord ON c.customerNumber = ord.customerNumber
LEFT JOIN orderdetails od ON ord.orderNumber = od.orderNumber
GROUP BY he.employeeNumber, he.lastName, he.firstName, he.jobTitle, 
         he.reportsTo, he.chemin_hierarchique, he.niveau, he.officeCode, 
         o.city, o.country
ORDER BY he.niveau, he.chemin_hierarchique;

-- ============================================================
-- Segmentation clients VIP avec JOINTURES MULTIPLES
-- ===============================================================
CREATE VIEW segmentation_clients_vip AS
WITH stats_clients AS (
    SELECT 
        c.customerNumber,
        c.customerName,
        c.country,
        c.city,
        c.creditLimit,
        COUNT(DISTINCT o.orderNumber) AS nombre_commandes,
        SUM(od.quantityOrdered * od.priceEach) AS montant_total_achete,
        MAX(o.orderDate) AS derniere_commande,
        COUNT(DISTINCT p.checkNumber) AS nombre_paiements,
        SUM(p.amount) AS montant_total_paye,
        c.salesRepEmployeeNumber,
        CONCAT(e.firstName, ' ', e.lastName) AS commercial_attache,
        pl.productLine AS gamme_preferee,
        COUNT(DISTINCT od.productCode) AS nombre_produits_differents
    FROM customers c
    LEFT JOIN orders o ON c.customerNumber = o.customerNumber
    LEFT JOIN orderdetails od ON o.orderNumber = od.orderNumber
    LEFT JOIN payments p ON c.customerNumber = p.customerNumber
    LEFT JOIN employees e ON c.salesRepEmployeeNumber = e.employeeNumber
    LEFT JOIN products pr ON od.productCode = pr.productCode
    LEFT JOIN productlines pl ON pr.productLine = pl.productLine
    WHERE o.status NOT IN ('Cancelled', 'On Hold')
    GROUP BY c.customerNumber, c.customerName, c.country, c.city, 
             c.creditLimit, c.salesRepEmployeeNumber, e.firstName, e.lastName
),
segmentation AS (
    SELECT 
        *,
        -- Calcul des percentiles pour segmentation
        PERCENT_RANK() OVER (ORDER BY montant_total_achete) AS percentile_montant,
        PERCENT_RANK() OVER (ORDER BY nombre_commandes) AS percentile_frequence,
        DATEDIFF(CURRENT_DATE, derniere_commande) AS jours_depuis_derniere_commande,
        CASE 
            WHEN montant_total_achete > 100000 THEN 'VIP'
            WHEN montant_total_achete BETWEEN 50000 AND 100000 THEN 'Fidèle'
            WHEN montant_total_achete BETWEEN 20000 AND 50000 THEN 'Régulier'
            WHEN montant_total_achete < 20000 THEN 'Occasionnel'
            ELSE 'Nouveau'
        END AS segment_montant,
        CASE 
            WHEN nombre_commandes > 10 THEN 'Très fréquent'
            WHEN nombre_commandes BETWEEN 5 AND 10 THEN 'Fréquent'
            WHEN nombre_commandes BETWEEN 2 AND 4 THEN 'Occasionnel'
            ELSE 'Rare'
        END AS segment_frequence,
        CASE 
            WHEN DATEDIFF(CURRENT_DATE, derniere_commande) < 90 THEN 'Actif'
            WHEN DATEDIFF(CURRENT_DATE, derniere_commande) BETWEEN 90 AND 180 THEN 'À risque'
            WHEN DATEDIFF(CURRENT_DATE, derniere_commande) BETWEEN 181 AND 365 THEN 'Inactif'
            ELSE 'Perdu'
        END AS segment_recence
    FROM stats_clients
)
SELECT 
    *,
    -- Score composite VIP
    CASE 
        WHEN segment_montant = 'VIP' AND segment_frequence IN ('Très fréquent', 'Fréquent') 
             AND segment_recence = 'Actif' THEN 'VIP Premium'
        WHEN segment_montant = 'VIP' THEN 'VIP Standard'
        WHEN segment_montant = 'Fidèle' AND segment_recence = 'Actif' THEN 'Client Fidèle Actif'
        ELSE CONCAT(segment_montant, ' - ', segment_recence)
    END AS segment_global,
    -- Potentiel de vente
    (creditLimit - montant_total_achete) AS marge_credit_restante,
    montant_total_achete / NULLIF(creditLimit, 0) * 100 AS taux_utilisation_credit
FROM segmentation
ORDER BY montant_total_achete DESC;


-- ============================================================
-- Analyse temporelle avec SOUS-REQUÊTES CORRÉLÉES
-- ===============================================================

-- Nouveaux clients par mois (première commande)
SELECT 
    DATE_FORMAT(first_order_date, '%Y-%m') AS mois_acquisition,
    COUNT(*) AS nouveaux_clients,
    SUM(montant_total) AS ca_nouveaux_clients,
    AVG(montant_total) AS panier_moyen_nouveaux
FROM (
    SELECT 
        c.customerNumber,
        MIN(o.orderDate) AS first_order_date,
        (
            SELECT SUM(od2.quantityOrdered * od2.priceEach)
            FROM orders o2
            JOIN orderdetails od2 ON o2.orderNumber = od2.orderNumber
            WHERE o2.customerNumber = c.customerNumber
            AND DATE_FORMAT(o2.orderDate, '%Y-%m') = DATE_FORMAT(MIN(o.orderDate), '%Y-%m')
        ) AS montant_total
    FROM customers c
    JOIN orders o ON c.customerNumber = o.customerNumber
    GROUP BY c.customerNumber
) AS nouveaux
GROUP BY DATE_FORMAT(first_order_date, '%Y-%m')
ORDER BY mois_acquisition;

-- Taux de rétention mois par mois
WITH cohortes AS (
    SELECT 
        customerNumber,
        DATE_FORMAT(MIN(orderDate), '%Y-%m') AS cohorte,
        DATE_FORMAT(orderDate, '%Y-%m') AS mois_activite
    FROM orders
    GROUP BY customerNumber, DATE_FORMAT(orderDate, '%Y-%m')
),
stats_cohorte AS (
    SELECT 
        c1.cohorte,
        c1.mois_activite,
        COUNT(DISTINCT c1.customerNumber) AS clients_actifs,
        (
            SELECT COUNT(DISTINCT customerNumber)
            FROM cohortes c2
            WHERE c2.cohorte = c1.cohorte
            AND c2.mois_activite = c1.cohorte
        ) AS taille_cohorte_initial
    FROM cohortes c1
    GROUP BY c1.cohorte, c1.mois_activite
    HAVING c1.mois_activite >= c1.cohorte
)
SELECT 
    cohorte,
    mois_activite,
    TIMESTAMPDIFF(MONTH, STR_TO_DATE(CONCAT(cohorte, '-01'), '%Y-%m-%d'), 
                  STR_TO_DATE(CONCAT(mois_activite, '-01'), '%Y-%m-%d')) AS mois_depuis_cohorte,
    clients_actifs,
    taille_cohorte_initial,
    ROUND((clients_actifs * 100.0 / NULLIF(taille_cohorte_initial, 0)), 2) AS taux_retention
FROM stats_cohorte
ORDER BY cohorte, mois_activite;


-- ============================================================
-- Rapport PIVOT ventes par gamme et trimestre
-- ===============================================================
CREATE VIEW pivot_ventes_gamme_trimestre AS
SELECT 
    productLine AS gamme_produit,
    SUM(CASE WHEN trimestre = 1 THEN montant_total ELSE 0 END) AS T1,
    SUM(CASE WHEN trimestre = 2 THEN montant_total ELSE 0 END) AS T2,
    SUM(CASE WHEN trimestre = 3 THEN montant_total ELSE 0 END) AS T3,
    SUM(CASE WHEN trimestre = 4 THEN montant_total ELSE 0 END) AS T4,
    SUM(montant_total) AS total_annuel,
    ROUND(AVG(montant_total), 2) AS moyenne_trimestrielle
FROM (
    SELECT 
        pl.productLine,
        QUARTER(o.orderDate) AS trimestre,
        YEAR(o.orderDate) AS annee,
        SUM(od.quantityOrdered * od.priceEach) AS montant_total
    FROM orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    JOIN products p ON od.productCode = p.productCode
    JOIN productlines pl ON p.productLine = pl.productLine
    WHERE o.status NOT IN ('Cancelled', 'On Hold')
    GROUP BY pl.productLine, YEAR(o.orderDate), QUARTER(o.orderDate)
) AS ventes_trimestrielles
GROUP BY productLine
ORDER BY total_annuel DESC;

-- Version dynamique avec GROUP_CONCAT pour tous les trimestres
SET @sql = NULL;
SELECT
  GROUP_CONCAT(DISTINCT
    CONCAT(
      'SUM(CASE WHEN CONCAT(annee, "-T", trimestre) = ''',
      CONCAT(annee, '-T', trimestre),
      ''' THEN montant_total ELSE 0 END) AS `',
      CONCAT(annee, '-T', trimestre), '`'
    )
  ) INTO @sql
FROM (
    SELECT DISTINCT 
        YEAR(orderDate) AS annee,
        QUARTER(orderDate) AS trimestre
    FROM orders
    WHERE orderDate >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
) AS periodes;

SET @sql = CONCAT('CREATE OR REPLACE VIEW pivot_ventes_dynamique AS
SELECT 
    productLine AS gamme_produit, ',
    @sql, ',
    SUM(montant_total) AS total_periode
FROM (
    SELECT 
        pl.productLine,
        YEAR(o.orderDate) AS annee,
        QUARTER(o.orderDate) AS trimestre,
        SUM(od.quantityOrdered * od.priceEach) AS montant_total
    FROM orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    JOIN products p ON od.productCode = p.productCode
    JOIN productlines pl ON p.productLine = pl.productLine
    WHERE o.status NOT IN (''Cancelled'', ''On Hold'')
    GROUP BY pl.productLine, YEAR(o.orderDate), QUARTER(o.orderDate)
) AS ventes
GROUP BY productLine
ORDER BY total_periode DESC');

PREPARE stmt FROM @sql;
EXECUTE stmt;
DEALLOCATE PREPARE stmt;


-- ============================================================
-- Procédure stockée : Calcul commission employé
-- ===============================================================

DELIMITER $$

CREATE PROCEDURE calcul_commission_employe(
    IN p_employeeNumber INT,
    IN p_date_debut DATE,
    IN p_date_fin DATE
)
BEGIN
    DECLARE total_ventes DECIMAL(15,2);
    DECLARE commission DECIMAL(15,2);
    DECLARE taux_commission DECIMAL(5,2);
    
    -- Calcul du total des ventes pour l'employé sur la période
    SELECT COALESCE(SUM(od.quantityOrdered * od.priceEach), 0) 
    INTO total_ventes
    FROM customers c
    JOIN orders o ON c.customerNumber = o.customerNumber
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    WHERE c.salesRepEmployeeNumber = p_employeeNumber
    AND o.orderDate BETWEEN p_date_debut AND p_date_fin
    AND o.status NOT IN ('Cancelled', 'On Hold');
    
    -- Détermination du taux de commission selon le montant
    SET taux_commission = CASE 
        WHEN total_ventes > 100000 THEN 0.10
        WHEN total_ventes > 50000 THEN 0.08
        WHEN total_ventes > 20000 THEN 0.06
        ELSE 0.05
    END;
    
    -- Calcul de la commission
    SET commission = total_ventes * taux_commission;
    
    -- Insertion dans la table des commissions (à créer)
    INSERT INTO commissions_employes (
        employeeNumber,
        periode_debut,
        periode_fin,
        total_ventes,
        taux_commission,
        montant_commission,
        date_calcul
    ) VALUES (
        p_employeeNumber,
        p_date_debut,
        p_date_fin,
        total_ventes,
        taux_commission,
        commission,
        CURDATE()
    );
    
    -- Retour des résultats
    SELECT 
        p_employeeNumber AS employe,
        CONCAT(e.firstName, ' ', e.lastName) AS nom_employe,
        p_date_debut AS periode_debut,
        p_date_fin AS periode_fin,
        total_ventes AS chiffre_affaires,
        taux_commission AS taux,
        commission AS commission_calculee
    FROM employees e
    WHERE e.employeeNumber = p_employeeNumber;
    
END$$

DELIMITER ;


-- ============================================================
-- Procédure stockée : Gestion stocks
-- ==============================================================

DELIMITER $$

CREATE PROCEDURE gestion_stocks_alertes()
BEGIN
    -- Création table temporaire pour les alertes stock
    DROP TEMPORARY TABLE IF EXISTS alertes_stock;
    CREATE TEMPORARY TABLE alertes_stock (
        productCode VARCHAR(15),
        productName VARCHAR(70),
        productLine VARCHAR(50),
        quantityInStock SMALLINT,
        ventes_moyennes_mensuelles DECIMAL(10,2),
        niveau_alerte VARCHAR(20),
        jours_couverture INT,
        recommandation VARCHAR(100)
    );
    
    -- Calcul des ventes moyennes et insertion des alertes
    INSERT INTO alertes_stock
    SELECT 
        p.productCode,
        p.productName,
        p.productLine,
        p.quantityInStock,
        COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) AS ventes_moyennes_mensuelles,
        CASE 
            WHEN p.quantityInStock = 0 THEN 'Rupture'
            WHEN p.quantityInStock <= COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) * 0.5 
                THEN 'Stock critique'
            WHEN p.quantityInStock <= COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) 
                THEN 'Stock faible'
            WHEN p.quantityInStock > COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) * 3 
                THEN 'Surstock'
            ELSE 'Normal'
        END AS niveau_alerte,
        CASE 
            WHEN COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) > 0 
                THEN FLOOR(p.quantityInStock / ventes_moyennes.ventes_moyennes_mensuelles * 30)
            ELSE 999
        END AS jours_couverture,
        CASE 
            WHEN p.quantityInStock = 0 THEN 'Commander urgence'
            WHEN p.quantityInStock <= COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) * 0.5 
                THEN 'Commander rapidement'
            WHEN p.quantityInStock <= COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) 
                THEN 'Surveiller'
            WHEN p.quantityInStock > COALESCE(ventes_moyennes.ventes_moyennes_mensuelles, 0) * 3 
                THEN 'Réduire commandes'
            ELSE 'Maintenir'
        END AS recommandation
    FROM products p
    LEFT JOIN (
        SELECT 
            od.productCode,
            AVG(SUM(od.quantityOrdered)) OVER (
                PARTITION BY od.productCode 
                ORDER BY DATE_FORMAT(o.orderDate, '%Y-%m')
                ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
            ) AS ventes_moyennes_mensuelles
        FROM orders o
        JOIN orderdetails od ON o.orderNumber = od.orderNumber
        WHERE o.orderDate >= DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        GROUP BY od.productCode, DATE_FORMAT(o.orderDate, '%Y-%m')
    ) AS ventes_moyennes ON p.productCode = ventes_moyennes.productCode
    WHERE p.quantityInStock IS NOT NULL;
    
    -- Sélection des résultats
    SELECT * FROM alertes_stock
    WHERE niveau_alerte != 'Normal'
    ORDER BY 
        CASE niveau_alerte
            WHEN 'Rupture' THEN 1
            WHEN 'Stock critique' THEN 2
            WHEN 'Stock faible' THEN 3
            WHEN 'Surstock' THEN 4
            ELSE 5
        END,
        jours_couverture ASC;
    
    -- Mise à jour du statut dans la table products
    UPDATE products p
    JOIN alertes_stock a ON p.productCode = a.productCode
    SET p.productDescription = CONCAT(
        COALESCE(p.productDescription, ''), 
        ' | Dernière alerte: ', a.niveau_alerte, 
        ' (', DATE_FORMAT(CURDATE(), '%Y-%m-%d'), ')'
    )
    WHERE a.niveau_alerte IN ('Rupture', 'Stock critique');
    
END$$

DELIMITER ;



-- ============================================================
-- Fonction : Customer Lifetime Value
-- ===============================================================
DELIMITER $$

CREATE FUNCTION calcul_clv(p_customerNumber INT) 
RETURNS DECIMAL(15,2)
DETERMINISTIC
READS SQL DATA
BEGIN
    DECLARE v_clv DECIMAL(15,2);
    DECLARE v_age_client_mois INT;
    DECLARE v_achat_moyen_mensuel DECIMAL(15,2);
    DECLARE v_marge DECIMAL(5,2) DEFAULT 0.30; -- Marge moyenne de 30%
    
    -- Calcul de l'âge du client en mois
    SELECT TIMESTAMPDIFF(MONTH, MIN(orderDate), CURDATE())
    INTO v_age_client_mois
    FROM orders
    WHERE customerNumber = p_customerNumber;
    
    IF v_age_client_mois = 0 THEN
        SET v_age_client_mois = 1; -- Éviter division par zéro
    END IF;
    
    -- Calcul des achats mensuels moyens
    SELECT COALESCE(SUM(od.quantityOrdered * od.priceEach) / v_age_client_mois, 0)
    INTO v_achat_moyen_mensuel
    FROM orders o
    JOIN orderdetails od ON o.orderNumber = od.orderNumber
    WHERE o.customerNumber = p_customerNumber
    AND o.status NOT IN ('Cancelled', 'On Hold');
    
    -- Calcul CLV simple (3 ans de projection)
    SET v_clv = v_achat_moyen_mensuel * v_marge * 36;
    
    RETURN v_clv;
END$$

DELIMITER ;



-- ============================================================
-- Trigger 2 : Validation stock
-- =============================================================
DELIMITER $$
create trigger validation_stock_before_insert
BEFORE INSERT ON orderdetails
FOR EACH ROW
BEGIN
    DECLARE v_stock_actuel SMALLINT;
    
    -- Vérification du stock disponible
    SELECT quantityInStock INTO v_stock_actuel
    FROM products
    WHERE productCode = NEW.productCode;
    
    IF v_stock_actuel < NEW.quantityOrdered THEN
        SIGNAL SQLSTATE '45000'
        SET MESSAGE_TEXT = CONCAT(
            'Stock insuffisant pour le produit ', 
            NEW.productCode, 
            '. Stock disponible: ', 
            v_stock_actuel, 
            ', Quantité commandée: ', 
            NEW.quantityOrdered
        );
    END IF;
END$$

DELIMITER ;



-- ============================================================
-- Trigger 3 : Mise à jour statut client
-- =============================================================
DELIMITER $$

CREATE TRIGGER update_customer_status
AFTER INSERT ON orders
FOR EACH ROW
BEGIN
    DECLARE v_nombre_commandes INT;
    DECLARE v_total_achats DECIMAL(15,2);
    DECLARE v_days_since_last_order INT;
    DECLARE v_nouveau_statut VARCHAR(20);
    
    -- Calcul des métriques du client
    SELECT 
        COUNT(*),
        COALESCE(SUM(od.quantityOrdered * od.priceEach), 0),
        DATEDIFF(CURDATE(), MAX(orderDate))
    INTO 
        v_nombre_commandes,
        v_total_achats,
        v_days_since_last_order
    FROM orders o
    LEFT JOIN orderdetails od ON o.orderNumber = od.orderNumber
    WHERE o.customerNumber = NEW.customerNumber
    AND o.status NOT IN ('Cancelled', 'On Hold');
    
    -- Détermination du nouveau statut
    SET v_nouveau_statut = CASE 
        WHEN v_total_achats > 100000 THEN 'VIP'
        WHEN v_total_achats > 50000 THEN 'Premium'
        WHEN v_total_achats > 20000 THEN 'Regular'
        WHEN v_days_since_last_order > 365 THEN 'Inactif'
        WHEN v_nombre_commandes = 1 THEN 'Nouveau'
        ELSE 'Standard'
    END;
    
    -- Mise à jour du champ comments avec le statut
    UPDATE customers
    SET comments = CONCAT(
        COALESCE(comments, ''),
        ' | Statut: ', v_nouveau_statut,
        ' (MAJ: ', DATE_FORMAT(CURDATE(), '%Y-%m-%d'), ')'
    )
    WHERE customerNumber = NEW.customerNumber;
    
END$$

DELIMITER ;



-- ============================================================
-- Segmentation clients VIP avec JOINTURES MULTIPLES
-- =============================================================