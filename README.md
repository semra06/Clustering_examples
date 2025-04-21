Problem 2: Ürün Kümeleme (Benzer Ürünler)
Veritabanı tabloları: Products, OrderDetails

Soru:

“Benzer sipariş geçmişine sahip ürünleri DBSCAN ile gruplandırın. Az satılan ya da alışılmadık kombinasyonlarda geçen ürünleri belirleyin.”

Özellik vektörleri:

Ortalama satış fiyatı

Satış sıklığı

Sipariş başına ortalama miktar

Kaç farklı müşteriye satıldı

Amaç:

Benzer ürünlerin segmentasyonu

-1 olan ürünler → belki özel ürünler veya niş ihtiyaçlar

Problem 3: Tedarikçi Segmentasyonu
Veritabanı tabloları: Suppliers, Products, OrderDetails

Soru:

“Tedarikçileri sağladıkları ürünlerin satış performansına göre gruplandırın. Az katkı sağlayan veya sıra dışı tedarikçileri bulun.”

Özellik vektörleri:

Tedarik ettiği ürün sayısı

Bu ürünlerin toplam satış miktarı

Ortalama satış fiyatı

Ortalama müşteri sayısı

Problem 4: Ülkelere Göre Satış Deseni Analizi
Veritabanı tabloları: Customers, Orders, OrderDetails

Soru:

“Farklı ülkelerden gelen siparişleri DBSCAN ile kümeleyin. Sıra dışı sipariş alışkanlığı olan ülkeleri tespit edin.”

Özellikler:

Toplam sipariş

Ortalama sipariş tutarı

Sipariş başına ürün sayısı
