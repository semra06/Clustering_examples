import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from dbscan_analysis import create_db_connection

def get_product_features(connection):
    """
    Retrieves product features from the database.
    Returns a DataFrame with product features for clustering.
    """
    query = """
    WITH product_stats AS (
        SELECT 
            p.product_id,
            p.product_name,
            AVG(od.unit_price) as avg_price,
            COUNT(DISTINCT od.order_id) as order_count,
            AVG(od.quantity) as avg_quantity_per_order,
            COUNT(DISTINCT o.customer_id) as unique_customers
        FROM products p
        LEFT JOIN order_details od ON p.product_id = od.product_id
        LEFT JOIN orders o ON od.order_id = o.order_id
        GROUP BY p.product_id, p.product_name
    )
    SELECT * FROM product_stats;
    """
    return pd.read_sql(query, connection)

def find_optimal_eps(X, min_samples=5):
    """
    Finds optimal epsilon value for DBSCAN using the elbow method.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(X)
    distances, _ = neighbors_fit.kneighbors(X)
    distances = np.sort(distances[:, min_samples-1], axis=0)
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    return kneedle.knee_y

def main():
    # Create database connection
    engine = create_db_connection()
    
    try:
        # Create connection
        connection = engine.connect()
        print("Successfully connected to the database!")
        
        # Get product features
        df = get_product_features(connection)
        print(f"Retrieved data for {len(df)} products")
        
        # Prepare features for clustering
        features = ['avg_price', 'order_count', 'avg_quantity_per_order', 'unique_customers']
        X = df[features].fillna(0)
        
        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Find optimal epsilon
        optimal_eps = find_optimal_eps(X_scaled)
        print(f"Optimal epsilon value: {optimal_eps:.2f}")
        
        # Apply DBSCAN
        dbscan = DBSCAN(eps=optimal_eps, min_samples=5)
        df['cluster'] = dbscan.fit_predict(X_scaled)
        
        # Analyze results
        cluster_stats = df.groupby('cluster').agg({
            'product_name': 'count',
            'avg_price': 'mean',
            'order_count': 'mean',
            'avg_quantity_per_order': 'mean',
            'unique_customers': 'mean'
        }).round(2)
        
        print("\nCluster Statistics:")
        print(cluster_stats)
        
        # Print outliers (cluster -1)
        outliers = df[df['cluster'] == -1]
        print(f"\nFound {len(outliers)} outlier products:")
        print(outliers[['product_id', 'product_name', 'avg_price', 'order_count']].to_string())
        
        # Visualize clusters (first two features)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df['cluster'], cmap='viridis')
        plt.title('Product Clusters (First Two Features)')
        plt.xlabel('Scaled Average Price')
        plt.ylabel('Scaled Order Count')
        plt.colorbar(label='Cluster')
        plt.savefig('product_clusters.png')
        plt.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 