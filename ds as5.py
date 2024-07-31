import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine, text
import pickle
import os

# Load dataset
df = pd.read_csv(r"C:\Users\manas\Downloads\Data Science 360Dgtmg\apriori,asccociation rules\Data Set (1)\book.csv", sep=';', header=None)

# Check for null values
print(df.isnull().sum())

# Database credentials
user = 'root'
pw = '2170'
db = 'bookstore_db'
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

# Load data into MySQL DB
df.to_sql('books', con=engine, if_exists='replace', chunksize=1000, index=False)

# Read data from database
sql = text('SELECT * FROM books;')
groceries = pd.read_sql_query(sql, con=engine.connect())

# Convert data to list of transactions
books = groceries.iloc[:, 0].to_list()
books_list = [i.split(",") for i in books]

# Remove null values from list
books_list = [list(filter(None, i)) for i in books_list]

# Encode transactions data into a NumPy array
from mlxtend.preprocessing import TransactionEncoder

TE = TransactionEncoder()
X_1hot = TE.fit_transform(books_list)

# Transform encoded data into a DataFrame
transf_df = pd.DataFrame(X_1hot, columns=TE.columns_)

# Elementary analysis: most popular items
count = transf_df.loc[:, :].sum()
pop_item = count.sort_values(ascending=False).head(10).to_frame().reset_index()
pop_item.columns = ["items", "count"]

# Data visualization: most popular items
plt.rcParams['figure.figsize'] = (10, 6)
plt.style.use('dark_background')
pop_item.plot.barh(x='items', y='count', legend=False)
plt.title('Most popular items')
plt.gca().invert_yaxis()
plt.show()

# Apply the apriori algorithm from mlxtend
from mlxtend.frequent_patterns import apriori, association_rules

frequent_itemsets = apriori(transf_df, min_support=0.0075, max_len=4, use_colnames=True)
frequent_itemsets.sort_values('support', ascending=False, inplace=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('lift', ascending=False, inplace=True)

# Print support and confidence of the rules
print("Support and Confidence of the Association Rules:")
for index, rule in rules.iterrows():
    print(f"Rule: {rule['antecedents']} -> {rule['consequents']}")
    print(f"Support: {rule['support']}")
    print(f"Confidence: {rule['confidence']}\n")

# Handle duplication of rules
def to_list(i):
    return sorted(list(i))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
ma_X = ma_X.apply(sorted)
rules_sets = list(ma_X)
unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = [rules_sets.index(i) for i in unique_rules_sets]

# Get unique rules without redundancy
rules_no_redundancy = rules.iloc[index_rules, :]
rules10 = rules_no_redundancy.sort_values('lift', ascending=False).head(10)

# Plot the top 10 rules
rules10.plot(x="support", y="confidence", kind="scatter", s=rules10['lift']*100, cmap=plt.cm.coolwarm, legend=True)
plt.title('Top 10 Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()

# Store rules into SQL database
rules10['antecedents'] = rules10['antecedents'].apply(lambda x: ', '.join(list(x)))
rules10['consequents'] = rules10['consequents'].apply(lambda x: ', '.join(list(x)))
rules10.to_sql('books_ar', con=engine, if_exists='replace', chunksize=1000, index=False)
rules10
import matplotlib.pyplot as plt

#support and confidence visualisation
plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='coolwarm', s=100)
plt.colorbar(label='Lift')
plt.title('Support vs Confidence of Association Rules')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.show()
