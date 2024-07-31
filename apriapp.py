from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
from mlxtend.frequent_patterns import apriori, association_rules
from sqlalchemy import create_engine


X_1hot_fit1 = pickle.load(open('TE.pkl','rb'))

# Connecting to sql by creating sqlachemy engine
engine = create_engine("mysql+pymysql://{user}:{pw}@localhost/{db}"
                       .format(user = "root", # user
                               pw = "2170", # password
                               db = "bookstore_db")) # database
# Define flask
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        
        data = pd.read_csv(f, sep=';', header=None)
        
        data = data.iloc[:, 0].to_list()
        
        books_list = []
        for i in data:
           books_list.append(i.split(","))
            
        # removing null values from list
        books_list_new = []
        for i in  books_list:
            books_list_new.append(list(filter(None, i)))
            
        transf_df = pd.DataFrame(X_1hot_fit1.transform(books_list_new), 
                                 columns = X_1hot_fit1.columns_)
        
        transf_df = transf_df.drop(transf_df.columns[0], axis = 1)
        
        # Itemsets
        frequent_itemsets = apriori(transf_df, min_support = 0.0075, max_len = 4, use_colnames = True)
        frequent_itemsets
        
        # Most frequent itemsets based on support 
        frequent_itemsets.sort_values('support', ascending = False, inplace = True)
        frequent_itemsets
        
        rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
        rules.head()
        
        def to_list(i):
            return (sorted(list(i)))

        ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)
        ma_X = ma_X.apply(sorted)
        

        rules_sets = list(ma_X)

        unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

        index_rules = []
        for i in unique_rules_sets:
            index_rules.append(rules_sets.index(i))
            

        # Rules without any redudancy 
        rules_no_redundancy = rules.iloc[index_rules, :]
        rules_no_redundancy

        # Sorted list and top 10 rules 
        rules_new = rules_no_redundancy.sort_values('lift', ascending = False).head(15)
        rules_new = rules_new.replace([np.inf, -np.inf], np.nan)
        
        rules_new['antecedents'] = rules_new['antecedents'].astype('string')
        rules_new['consequents'] = rules_new['consequents'].astype('string')
        
        rules_new['antecedents'] = rules_new['antecedents'].str.removeprefix("frozenset({")
        rules_new['antecedents'] = rules_new['antecedents'].str.removesuffix("})")
        
        rules_new['consequents'] = rules_new['consequents'].str.removeprefix("frozenset({")
        rules_new['consequents'] = rules_new['consequents'].str.removesuffix("})")
        
        rules_new.to_sql('books_ar', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        html_table = rules_new.to_html(classes = 'table table-striped')
       
        return render_template("new.html", Y =   f"<style>\
                    .table {{\
                        width: 50%;\
                        margin: 0 auto;\
                        border-collapse: collapse;\
                    }}\
                    .table thead {{\
                        background-color: #39648f;\
                    }}\
                    .table th, .table td {{\
                        border: 1px solid #ddd;\
                        padding: 8px;\
                        text-align: center;\
                    }}\
                        .table td {{\
                        background-color: #5e617d;\
                    }}\
                            .table tbody th {{\
                            background-color: #ab2c3f;\
                        }}\
                </style>\
                {html_table}") 
                

if __name__=='__main__':
    app.run(debug = True)
