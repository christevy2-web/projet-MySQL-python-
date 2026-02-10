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

Téléchargement : Allez sur le site officiel de MySQL et téléchargez le "Web Installer".

Lancement : Exécutez le fichier .msi.

Choix du type d'installation :

Developer Default : Installe tout (Serveur, Shell, Workbench pour l'interface graphique).

Server only : Uniquement le serveur de base de données.

Configuration : * Laissez le port par défaut (3306).

Important : Choisissez un mot de passe robuste pour l'utilisateur root et notez-le bien !

Service Windows : Laissez l'option "Configure MySQL as a Windows Service" cochée pour qu'il démarre avec l'ordinateur.


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