import duckdb
from pyarrow import csv
import pandas as pd
import os

db_name = 'genie.duckdb'
if os.path.exists(db_name):
  os.remove(db_name) 

con = duckdb.connect(database=db_name, read_only=False)
metadata = csv.read_csv('src/data/genie.metadata.csv')
con.execute("CREATE TABLE meta_table AS SELECT * FROM metadata")  # FROM <variable name>
con.execute("INSERT INTO meta_table SELECT * FROM metadata")

genie = pd.read_csv('src/data/genie.mutated.csv')
# genie['aa_mutated'].fillna('na', inplace=True)
genie_pivot = genie.pivot(
    index='tissuename',
    columns='symbol',
    values='aa_mutated'
).reset_index()
genie_pivot.fillna('na', inplace=True)
con.execute("CREATE TABLE mutated_table AS SELECT * FROM genie_pivot")
con.execute("INSERT INTO mutated_table SELECT * FROM genie_pivot")