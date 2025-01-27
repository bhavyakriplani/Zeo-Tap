import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Load the datasets
customers = pd.read_csv('Datasets/Customers.csv')
products = pd.read_csv('Datasets/Products.csv')
transactions = pd.read_csv('Datasets/Transactions.csv')

# Data Cleaning
customers.fillna('Unknown', inplace=True)
transactions.dropna(inplace=True)
customers.drop_duplicates(inplace=True)

# Convert dates to datetime
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])

# Merge data
data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID')

# Feature Engineering
data['TotalAmount'] = data['Quantity'] * data['Price']
customer_data = data.groupby('CustomerID').agg({
    'TotalAmount': 'sum',
    'TransactionID': 'count', 
    'TransactionDate': lambda x: (x.max() - x.min()).days, 
}).reset_index()

customer_data.rename(columns={
    'TransactionID': 'Frequency',
    'TransactionDate': 'Recency'
}, inplace=True)

# Normalize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data[['TotalAmount', 'Frequency', 'Recency']])

# Clustering
db_scores = []
for k in range(2, 11): 
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    db_index = davies_bouldin_score(scaled_data, labels)
    db_scores.append((k, db_index))
    silhouette = silhouette_score(scaled_data, labels)
    print(f"Clusters: {k}, DB Index: {db_index:.3f}, Silhouette Score: {silhouette:.3f}")

# Plot DB Index values
db_scores = pd.DataFrame(db_scores, columns=['Clusters', 'DB_Index'])
plt.plot(db_scores['Clusters'], db_scores['DB_Index'], marker='o')
plt.title('DB Index vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('DB Index')
plt.show()

# Optimal Clustering
optimal_k = db_scores.loc[db_scores['DB_Index'].idxmin(), 'Clusters']
kmeans = KMeans(n_clusters=int(optimal_k), random_state=42)
labels = kmeans.fit_predict(scaled_data)
customer_data['Cluster'] = labels

# Visualize clusters using PCA
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)
pca_df = pd.DataFrame(pca_data, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = labels

sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='viridis', s=100)
plt.title('Clusters Visualization')
plt.show()