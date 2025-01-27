import pandas as pd
customers = pd.read_csv(r'Datasets\Customers.csv')
products = pd.read_csv(r'Datasets\Products.csv')
transactions = pd.read_csv(r'Datasets\Transactions.csv')


print(customers.head())
print(products.head())
print(transactions.head())

print(customers.isnull().sum())
print(products.isnull().sum())
print(transactions.isnull().sum())


customers.fillna('Unknown', inplace=True)  
transactions.dropna(inplace=True)        
print(customers.duplicated().sum())
customers.drop_duplicates(inplace=True)

print(customers.isnull().sum()) 
print(products.isnull().sum()) 
print(transactions.isnull().sum())
customers.fillna('Unknown', inplace=True)  
transactions.dropna(inplace=True)         
print(customers.duplicated().sum())
customers.drop_duplicates(inplace=True)
customers['SignupDate'] = pd.to_datetime(customers['SignupDate'])
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
print(customers.describe()) 
print(products['Category'].value_counts())
data = transactions.merge(customers, on='CustomerID').merge(products, on='ProductID') 
print(data.head())
