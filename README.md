powershell 

-------------------------------------------------------------------
# Étape 1 : Préparation du dossier et de l'environnement

Aller dans votre dossier :

Créer l'environnement virtuel (si pas déjà fait): 
python -m venv .venv

 Activer l'environnement virtuel :
& .venv\Scripts\Activate.ps1

------------------------------------------------------------------------
# Étape 2 : Installation des dépendances

Créer le fichier requirements.txt

S'assurer que pip est à jour
python -m pip install --upgrade pip

PowerShell
pip install mysql-connector-python pandas numpy matplotlib seaborn plotly jupyter scipy scikit-learn python-dotenv networkx statsmodels

(Puis installe Plotly dans ce venv :)
pip install plotly==5.18.0

# Installer les packages

pip install plotly==5.18.0
pip install -r requirements.txt
pip show plotly
pip list | findstr plotly


Installer tout d'un coup via le fichier requirements.txt
pip install -r requirements.txt

------------------------------------------------------------------------------
Étape 3 : Structure du projet et Configuration

créer le fichier .env (Indispensable pour vos scripts Python)
# Note : On le place à la racine pour que vos scripts le trouvent facilement
echo "DB_HOST=localhost" > .env
echo "DB_USER=root" >> .env
echo "DB_PASSWORD=" >> .env
echo "DB_NAME=" >> .env

--------------------------------------------------------------------------------
Étape 4 : Vérification et Lancement
# 1. Vérifier que les bibliothèques sont là
pip list | findstr "mysql pandas plotly"

# 3. Ou lancer Jupyter pour la partie analyse
jupyter notebook


# La méthode recommandée est d'utiliser le MySQL Installer.

Étape 1 : Allez à la page de MySQL Workbench sur Academic Software et cliquez sur le bouton 'Télécharger MySQL Workbench' pour télécharger le logiciel.

•   le site officiel https://dev.mysql.com/downloads/workbench/, sélectionnez la version Windows (fichier .msi) et téléchargez-la. MySQL Server, optez pour l’installateur web complet depuis https://dev.mysql.com/downloads/installer/ pour inclure le serveur, Workbench et Shell en une seule fois.

Étape 2 : Ouvrez le fichier d'installation MSI dans votre dossier Téléchargements pour démarrer l'installation. Cliquez sur Next pour continuer.

Étape 3 : Cliquez sur Next pour installer MySQL Workbench à l'emplacement standard. Si vous le souhaitez, vous pouvez également modifier le chemin d'installation.

Étape 4 : Cliquez sur Next pour installer le programme entier ou cochez Custom pour les options avancées.
cochez Custom pour les options avancées: 
•	MySQL Server
•	MySQL Workbench
•	MySQL Shell
•	Documentation
•	etc.
Cochez MySQL Server (essentiel pour la base de données), cliquez sur la flèche droite (→) pour l’ajouter.
Cochez MySQL Workbench (interface graphique), flèche droite.

Étape 5 : Cliquez sur Install pour démarrer l'installation.
Configuration du serveur MySQL
Pendant l’installation du serveur 
•   Laissez le port par défaut 3306.
•   Choisissez Standard Configuration pour une config simple.
•   Sélectionnez Use Strong Password Encryption pour la sécurité.
•   Définissez un mot de passe root robuste (mélange lettres, chiffres, symboles) et notez-le bien !
•   Cochez Configure MySQL as a Windows Service pour un démarrage automatique au boot.
•   Activez Démarrez MySQL Server au démarrage du système.
Cliquez Next, Execute puis Finish.

Étape 6 : Cliquez sur Finish pour terminer l'installation et démarrer MySQL Workbench. Le logiciel est maintenant installé et prêt à l'emploi.
Une fois l’installation terminée, 
•   cliquez Finish. 
•   Lancez MySQL Workbench depuis le menu Démarrer. 
Créez une connexion :
•   Cliquez sur le + près de « MySQL Connections ».
•   Nom : « Localhost », Hôte : localhost, Port : 3306, Utilisateur : root, Mot de passe : celui choisi.
•   Testez la connexion (elle doit réussir), puis OK pour se connecter.





# importer ClassicModels(explique)
1. Prérequis : Télécharger le fichier
Si vous ne l'avez pas encore, téléchargez le fichier mysqlsampledatabase.sql (souvent compressé en .zip) 
sur le site mysqltutorial.org ou un dépôt GitHub. Décompressez-le pour obtenir le fichier .sql.

2. Méthode via MySQL Workbench (Interface Graphique)
Ouvrez MySQL Workbench et connectez-vous à votre serveur.
Allez dans le menu en haut : File > Open SQL Script...
Sélectionnez votre fichier mysqlsampledatabase.sql.
Le code SQL s'affiche dans un onglet. Cliquez sur l'icône de l'éclair (Execute) pour lancer tout le script.
Une fois terminé, faites un clic droit dans la zone "Schemas" à gauche et cliquez sur Refresh All. 
La base classicmodels apparaîtra.