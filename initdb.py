import duckdb
from pyarrow import csv
import pandas as pd
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


db_name = 'data/genie.duckdb'
if os.path.exists(db_name):
  print(f"Database {db_name} already exists. Deleting it.")
  os.remove(db_name)


con = duckdb.connect(database=db_name, read_only=False)

def create_meta():
  print('--- META DATA ---')
  print('Reading data from csv file...')
  meta = pd.read_csv('data/db_exports/genie.metadata.csv', engine='pyarrow')
  meta.drop(columns=['tdpid'], inplace=True) # drop columns that are never going to get used

  print('Imputing missing values...')
  num_cols = meta.select_dtypes(include=['number']).columns.tolist()
  meta_num = pd.DataFrame()
  if num_cols.__len__() > 0:
    num_enc = SimpleImputer(strategy='constant', fill_value=-1) # set all missing values to -1
    meta_num = pd.DataFrame(
      num_enc.fit_transform(meta[num_cols]),
      columns=num_cols,
      index=meta.index)

  print('Encoding categorical values...')  
  cat_cols = meta.select_dtypes(include=['object']).columns.tolist()
  meta_cat_in = meta[cat_cols].copy()
  nunique = meta_cat_in.nunique()
  cols_to_drop = nunique[nunique <= 1].index # 0 if all are missing, 1 if ther is only one catgeory/value
  meta_cat_in.drop(cols_to_drop, axis='columns', inplace=True) # drop the columns with only one category
  meta_cat_in.drop(columns=['tissuename'], inplace=True) # drop tissuename column because that does not need to be one hot encoded
  # impute missing values
  # there are already some samples labelled 'unknown', having both makes no sense (otherwise treated as separate category)
  meta_cat_in.fillna('unkown', inplace=True)
  meta_cat_in.replace({
    'NULL': 'unknown',
    'null':'unknown'
  }, inplace=True)

  meta_cat_out = pd.DataFrame()
  if cat_cols.__len__() > 0:
    # One hot encode the categorical data
    enc = OneHotEncoder(
      handle_unknown='ignore', # we fit and transform in one step, so the encoder wont see unknown values
      # sparse=True # saves space, does not work with follow up code
      sparse=False 
    )
    meta_cat_out = pd.DataFrame(
      enc.fit_transform(meta_cat_in),
      columns=enc.get_feature_names_out(),
      index=meta.index)

  meta_processed = pd.concat([   # variable is used in SQL statements below
      meta['tissuename'],
     # meta[cols_to_drop], # untouched categorical data
      meta_num, # imputed numerical data
      # TODO numerical data that was not touched by imputer
      meta_cat_out # one hot encoded categorical data
    ], axis=1)

  print('Creating table...')
  con.execute("CREATE TABLE meta_table AS SELECT * FROM meta_processed")  # FROM <variable name>
  con.execute("INSERT INTO meta_table SELECT * FROM meta_processed")


def create_mutated():
  print('--- MUTATED DATA ---')
  print('Reading data from csv file...')
  mutated = pd.read_csv('data/db_exports/genie.mutated.csv', engine='pyarrow')
  print('Pivot the table...')
  mutated_pivot = mutated.pivot(
      index='tissuename',
      columns='symbol',
      values='aa_mutated'
  )
  print('Fill and replace values...')
  mutated_pivot.fillna('na', inplace=True) # does not handle all cases, see replace below for empty values
  processed_mutated_pivot = mutated_pivot.replace({
      'f': -1,
      't': 1,
      'na': 0,
      '': 0  # must not be na, could also be empty, handle like na
    }) # variable is used in SQL statements below

  print(processed_mutated_pivot.loc['GENIE-VICC-176620-unk-1'], 'ABL1')
  processed_mutated_pivot.reset_index(inplace=True)
  print('Creating table...')
  con.execute("CREATE TABLE mutated_table AS SELECT * FROM processed_mutated_pivot")
  con.execute("INSERT INTO mutated_table SELECT * FROM processed_mutated_pivot")

create_meta()
create_mutated()

print('Done 🎉')