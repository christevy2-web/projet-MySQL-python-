import mysql.connector
from mysql.connector import pooling
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

class DatabaseConnector:
    _instance = None

    def __init__(self):
        self.connection_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', 'Christevy'),
            'database': os.getenv('DB_NAME', 'classicmodels') # Nom de ta base
        }
        try:
            self.pool = mysql.connector.pooling.MySQLConnectionPool(
                pool_name="mypool",
                pool_size=5,
                **self.connection_params
            )
        except mysql.connector.Error as err:
            print(f"Erreur de connexion : {err}")

    def get_query_df(self, query, params=None):
        conn = self.pool.get_connection()
        try:
            return pd.read_sql(query, conn, params=params)
        finally:
            conn.close()