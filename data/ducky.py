import duckdb
import pandas as pd
import os

db_name = './genie.duckdb'
con = duckdb.connect(database=db_name, read_only=True)
query = """
SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE table_name = 'meta_table'
"""
df = con.execute(query).df()
print(df.head())
all_columns = df['column_name']
print(all_columns.size)
exclude_features = ['gender', 'tumortype']
exclude_onehot_features = all_columns.str.startswith(tuple(exclude_features))
print('exclude_onehot_features', exclude_onehot_features.sum()/all_columns.size)
select = all_columns[~exclude_onehot_features]
exclude = all_columns[exclude_onehot_features]
select_text = ', '.join(f'"{w}"' for w in select)

#print('#select\n', select)
#print('#exclude\n', exclude)

print('\n\n\n')


query = f"SELECT {select_text} FROM meta_table LIMIT 10"
#print('#query\n', query)
df = con.execute(query).df()
#print(df.head())
print('lb')
