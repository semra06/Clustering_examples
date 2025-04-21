import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator
from dbscan_analysis import create_db_connection, DB_CONFIG

class SupplierDBSCANAnalyzer:
    def __init__(self, connection):
        self.connection = connection
        self.features = None
        self.labels = None
        self.data = None
        self.feature_columns = ['product_count', 'total_sales', 'avg_price', 'customer_count']
        
    def fetch_data(self):
        query = """
        SELECT 
            s.supplier_id,
            s.company_name,
            COUNT(DISTINCT p.product_id) as product_count,
            COALESCE(SUM(od.quantity), 0) as total_sales,
            COALESCE(AVG(p.unit_price), 0) as avg_price,
            COALESCE(COUNT(DISTINCT od.order_id), 0) as customer_count
        FROM suppliers s
        LEFT JOIN products p ON s.supplier_id = p.supplier_id
        LEFT JOIN order_details od ON p.product_id = od.product_id
        GROUP BY s.supplier_id, s.company_name
        ORDER BY s.supplier_id;
        """
        try:
            self.data = pd.read_sql(query, self.connection)
            print(f"Toplam {len(self.data)} tedarikçi verisi çekildi.")
            return self.data
        except Exception as e:
            raise Exception(f"Veri çekme hatası: {str(e)}")

    def preprocess_data(self):
        if self.data is None:
            raise Exception("Önce fetch_data() metodunu çalıştırın!")
            
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.data[self.feature_columns])
        return self.features

    def find_optimal_eps(self, min_samples=2):
        if self.features is None:
            raise Exception("Önce preprocess_data() metodunu çalıştırın!")
            
        nbrs = NearestNeighbors(n_neighbors=min_samples).fit(self.features)
        distances, _ = nbrs.kneighbors(self.features)
        distances = np.sort(distances[:, min_samples-1])
        
        kneedle = KneeLocator(
            range(len(distances)),
            distances,
            S=1.0,
            curve='convex',
            direction='increasing'
        )
        
        eps = distances[kneedle.knee] if kneedle.knee is not None else np.mean(distances)
        print(f"Optimal epsilon değeri: {eps:.4f}")
        return eps

    def perform_clustering(self, eps=None, min_samples=2):
        if eps is None:
            eps = self.find_optimal_eps(min_samples)
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.features)
        self.data['cluster'] = self.labels
        
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        n_noise = list(self.labels).count(-1)
        print(f"Bulunan küme sayısı: {n_clusters}")
        print(f"Aykırı değer sayısı: {n_noise}")
        
        return self.data

    def analyze_clusters(self):
        if self.labels is None:
            raise Exception("Önce perform_clustering() metodunu çalıştırın!")
            
        analysis = {
            'total_suppliers': len(self.data),
            'num_clusters': len(set(self.labels[self.labels != -1])),
            'num_outliers': sum(self.labels == -1)
        }
        
        analysis['cluster_stats'] = self.data.groupby('cluster').agg({
            col: ['mean', 'min', 'max', 'count'] for col in self.feature_columns
        }).round(2)
        
        return analysis

    def visualize_clusters(self):
        if self.labels is None:
            raise Exception("Önce perform_clustering() metodunu çalıştırın!")
            
        plt.style.use('classic')
        fig = plt.figure(figsize=(15, 10))
        
        # Satış vs Fiyat grafiği
        self._plot_scatter(2, 2, 1, 'total_sales', 'avg_price', 
                         'Toplam Satış', 'Ortalama Fiyat',
                         'Tedarikçi Kümeleri: Satış vs Fiyat')
        
        # Ürün vs Müşteri grafiği
        self._plot_scatter(2, 2, 2, 'product_count', 'customer_count',
                         'Ürün Sayısı', 'Müşteri Sayısı',
                         'Tedarikçi Kümeleri: Ürün vs Müşteri')
        
        # Küme dağılımı pasta grafiği
        self._plot_pie(2, 2, 3)
        
        plt.tight_layout()
        return fig
    
    def _plot_scatter(self, rows, cols, pos, x_col, y_col, x_label, y_label, title):
        plt.subplot(rows, cols, pos)
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(set(self.labels))))
        
        for i, label in enumerate(sorted(set(self.labels))):
            mask = self.labels == label
            if label == -1:
                color = 'red'
                marker = 'x'
                label_text = 'Aykırı Değerler'
            else:
                color = colors[i]
                marker = 'o'
                label_text = f'Küme {label}'
            
            plt.scatter(self.data[mask][x_col], 
                       self.data[mask][y_col],
                       c=[color], marker=marker, s=100,
                       label=label_text)
        
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)
        plt.legend()
        plt.grid(True)
    
    def _plot_pie(self, rows, cols, pos):
        plt.subplot(rows, cols, pos)
        cluster_sizes = pd.Series(self.labels).value_counts()
        
        colors = ['red' if i == -1 else plt.cm.viridis(idx/max(1, len(set(self.labels))-1))
                 for idx, i in enumerate(cluster_sizes.index)]
        
        plt.pie(cluster_sizes, 
                labels=[f'Küme {i}' if i != -1 else 'Aykırı Değerler' 
                       for i in cluster_sizes.index],
                autopct='%1.1f%%',
                colors=colors)
        plt.title('Küme Dağılımı')

    def generate_report(self):
        analysis = self.analyze_clusters()
        
        report = [
            "=== Tedarikçi Segmentasyon Raporu ===\n",
            f"Toplam Tedarikçi Sayısı: {analysis['total_suppliers']}",
            f"Küme Sayısı: {analysis['num_clusters']}",
            f"Aykırı Değer Sayısı: {analysis['num_outliers']}\n"
        ]
        
        for cluster in sorted(set(self.labels)):
            cluster_data = self.data[self.data['cluster'] == cluster]
            cluster_type = "Aykırı Değerler (Sıra Dışı Tedarikçiler)" if cluster == -1 else f"Küme {cluster}"
            
            report.extend([
                f"\n=== {cluster_type} ===",
                f"Tedarikçi Sayısı: {len(cluster_data)}"
            ])
            
            avg_stats = cluster_data[self.feature_columns].mean().round(2)
            report.extend([
                "\nKüme Özellikleri:",
                f"  * Ortalama Ürün Sayısı: {avg_stats['product_count']}",
                f"  * Ortalama Toplam Satış: {avg_stats['total_sales']}",
                f"  * Ortalama Fiyat: {avg_stats['avg_price']}",
                f"  * Ortalama Müşteri Sayısı: {avg_stats['customer_count']}"
            ])
            
            report.append("\nTedarikçiler:")
            for _, supplier in cluster_data.iterrows():
                report.extend([
                    f"- {supplier['company_name']}",
                    f"  * Ürün Sayısı: {supplier['product_count']}",
                    f"  * Toplam Satış: {supplier['total_sales']}",
                    f"  * Ort. Fiyat: {supplier['avg_price']:.2f}",
                    f"  * Müşteri Sayısı: {supplier['customer_count']}"
                ])
            
        return "\n".join(report)

def main():
    try:
        engine = create_db_connection()
        connection = engine.connect()
        print("Veritabanına başarıyla bağlanıldı!")
        
        analyzer = SupplierDBSCANAnalyzer(connection)
        
        print("\nVeriler çekiliyor...")
        analyzer.fetch_data()
        
        print("\nVeriler hazırlanıyor...")
        analyzer.preprocess_data()
        
        print("\nKümeleme yapılıyor...")
        analyzer.perform_clustering()
        
        print("\nGörselleştirmeler oluşturuluyor...")
        fig = analyzer.visualize_clusters()
        plt.show()
        
        print("\nRapor oluşturuluyor...")
        report = analyzer.generate_report()
        print(report)
        
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")
    finally:
        if 'connection' in locals():
            connection.close()
            print("\nVeritabanı bağlantısı kapatıldı.")

if __name__ == "__main__":
    main()

