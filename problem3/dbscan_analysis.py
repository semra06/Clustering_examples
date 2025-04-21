import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator

# Database connection parameters
DB_CONFIG = {
    "user": "postgres",
    "password": "1234",
    "host": "localhost",
    "port": "5432",
    "database": "gyk1nordwinds"
}

def create_db_connection():
    """
    Creates and returns a database connection using SQLAlchemy.
    
    Returns:
        sqlalchemy.engine.base.Engine: Database engine instance
    """
    connection_string = f'postgresql+psycopg2://{DB_CONFIG["user"]}:{DB_CONFIG["password"]}@{DB_CONFIG["host"]}:{DB_CONFIG["port"]}/{DB_CONFIG["database"]}'
    return create_engine(connection_string)

def main():
    # Create database connection
    engine = create_db_connection()
    
    try:
        # Create connection
        connection = engine.connect()
        print("Successfully connected to the database!")
        
        # TODO: Add your data analysis code here
        
    except Exception as e:
        print(f"Error connecting to database: {e}")

if __name__ == "__main__":
    main() 