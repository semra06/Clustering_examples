import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from dbscan_analysis import create_db_connection

def get_country_sales_data():
    """
    Retrieves sales data aggregated by country from the database.
    Returns a DataFrame with features for each country.
    """
    query = """
    SELECT 
        c.country,
        COUNT(DISTINCT o.order_id) as total_orders,
        AVG(od.unit_price * od.quantity) as avg_order_amount,
        AVG(od.quantity) as avg_products_per_order
    FROM 
        customers c
        JOIN orders o ON c.customer_id = o.customer_id
        JOIN order_details od ON o.order_id = od.order_id
    GROUP BY 
        c.country
    """
    
    engine = create_db_connection()
    df = pd.read_sql(query, engine)
    return df

def find_optimal_eps(data, min_samples=3):
    """
    Finds the optimal epsilon value for DBSCAN using the knee method.
    """
    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(data)
    distances, _ = neighbors_fit.kneighbors(data)
    distances = np.sort(distances[:, -1])
    
    kneedle = KneeLocator(range(len(distances)), distances, curve='convex', direction='increasing')
    return kneedle.knee_y

def analyze_country_sales_patterns():
    """
    Main function to analyze country sales patterns using DBSCAN.
    """
    # Get data
    df = get_country_sales_data()
    
    # Prepare features for clustering
    features = ['total_orders', 'avg_order_amount', 'avg_products_per_order']
    X = df[features]
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Find optimal epsilon
    eps = find_optimal_eps(X_scaled)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=3)
    df['cluster'] = dbscan.fit_predict(X_scaled)
    
    # Analyze results
    print("\nCluster Analysis Results:")
    print("------------------------")
    print(f"Number of clusters: {len(df['cluster'].unique()) - 1}")  # -1 for noise points
    
    # Print countries in each cluster
    for cluster in sorted(df['cluster'].unique()):
        if cluster == -1:
            print("\nOutliers (Countries with unusual patterns):")
        else:
            print(f"\nCluster {cluster}:")
        countries = df[df['cluster'] == cluster]['country'].tolist()
        print(f"Number of countries: {len(countries)}")
        print(f"Countries: {', '.join(countries)}")
    
    # Visualize the clusters
    plt.figure(figsize=(12, 8))
    
    # Create a 3D plot
    ax = plt.axes(projection='3d')
    
    # Plot each cluster
    for cluster in df['cluster'].unique():
        if cluster == -1:
            label = 'Outliers'
            marker = 'x'
        else:
            label = f'Cluster {cluster}'
            marker = 'o'
        
        cluster_data = df[df['cluster'] == cluster]
        ax.scatter(cluster_data['total_orders'],
                  cluster_data['avg_order_amount'],
                  cluster_data['avg_products_per_order'],
                  label=label,
                  marker=marker)
    
    ax.set_xlabel('Total Orders')
    ax.set_ylabel('Average Order Amount')
    ax.set_zlabel('Average Products per Order')
    ax.set_title('Country Sales Pattern Clusters')
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_country_sales_patterns() 