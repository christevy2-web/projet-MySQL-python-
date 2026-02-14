# src/database_connector.py
import mysql.connector
from mysql.connector import pooling
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnector:
    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'Christevy'),
            'database': os.getenv('DB_NAME', 'projet_sql_py')
        }
        self.pool = None
    
    def connect(self):
        try:
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                **self.connection_params
            )
            print("✅ Connexion établie")
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def execute_query(self, query, params=None):
        connection = mysql.connector.connect(**self.connection_params)
        try:
            df = pd.read_sql(query, connection, params=params)
            return df
        finally:
            connection.close()