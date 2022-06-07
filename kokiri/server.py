# TODO patch sklearn if necessary https://intel.github.io/scikit-learn-intelex/
from .settings import KokiriSettings

import uvicorn # For debugging
from typing import Dict, Optional

from fastapi import FastAPI
from fastapi import __version__ as fastapi_version
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, conlist
import asyncio
import json

import pandas as pd
import duckdb

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# h2o.init()
# https://medium.com/tech-vision/random-forest-classification-with-h2o-python-for-beginners-b31f6e4ccf3c
# https://www.kaggle.com/code/nanomathias/h2o-distributed-random-forest-starter/script

import logging
import time

__author__ = 'Klaus Eckelt'

config = KokiriSettings()
separator = ', '

app = FastAPI()
#print(f"fastapi version: {fastapi_version}")

origins = [
  "http://localhost",
  "http://localhost:5500",
  "http://localhost:8080",
  "http://localhost:9000",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)


class CmpData(BaseModel):
  exclude: Optional[list[str]] = None
  ids: conlist( # list of cohorts # TODO replace with list of cohort IDs
    conlist(Dict, min_items=1), # list of item (must not be empty)
    min_items=2 # at least two cohorts are needed to compare
    )

# E.g.
# {
#   exclude: ['age', 'gender'],
#   ids: [ # list of cohorts
#     ['homer', 'marge', 'bart', 'lisa', 'maggie'], # 1st cohort's ids
#     ['bart', 'lisa', 'maggie'], # 2nd cohort's ids
#   ]
# }

# TODO replace with websocket: @app.websocket("/ws/{client_id}")
@app.post("/cmp_meta/")
def cmp_meta(cmp_data: CmpData):
  con = duckdb.connect(database=config.dbName, read_only=True) # TODO check if this is the way to go for multi thread access (cursors were mentioned in a blog psot)
  frames = []
  for i, cht_ids in enumerate(cmp_data.ids):
    query = create_query(i, cht_ids, ['tissuename', 'tdpid'] + cmp_data.exclude, 'meta')
    df = con.execute(query).df()
    frames.append(df)

  df = pd.concat(frames)

  y = df['cht']
  X = df.drop(columns=['cht']) # drop the target column

  # find features with same value for all samples and drop them
  nunique = X.nunique()
  cols_to_drop = nunique[nunique <= 1].index # 0 if all are missing, 1 if ther is only one catgeory/value
  X_train = X.drop(cols_to_drop, axis='columns')
  # X_train= X_train.rename(columns={"tumortype": "Tumor Type"}) to test exclusion

  num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
  cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

  X_train_cat = pd.DataFrame()
  one_hot_column_names = []
  if cat_cols.__len__() > 0:
    # One hot encode the categorical data
    enc = OneHotEncoder(
      handle_unknown='ignore', # we fit and transform in one step, so the encoder wont see unknown values
      sparse=False
      )
    X_train_cat = pd.DataFrame(
                      enc.fit_transform(X_train[cat_cols]),
                      columns=enc.get_feature_names_out(),
                      index=df.index)
    one_hot_column_names = enc.get_feature_names_out().tolist()

  X_train_num = pd.DataFrame()
  if num_cols.__len__() > 0:
    num_enc = SimpleImputer(strategy='constant', fill_value=-1) # set all missing values to -1
    X_train_num = pd.DataFrame(
      num_enc.fit_transform(X_train[num_cols]),
      columns=num_cols,
      index=df.index)

  X_train_coded = pd.concat([X_train_cat, X_train_num], axis=1)

  columns = one_hot_column_names + num_cols
  # train the model
  results = rf(X_train_coded, y, columns)

  return StreamingResponse(encode_results(results))
  
  
@app.post("/cmp_mutated/")
def cmp_mutated(cmp_data: CmpData):
  con = duckdb.connect(database=config.dbName, read_only=True)
  frames = []
  for i, cht_ids in enumerate(cmp_data.ids):
    query = create_query(i, cht_ids, ['tissuename'] + cmp_data.exclude, 'mutated')
    df = con.execute(query).df()
    frames.append(df)

  df = pd.concat(frames)

  y = df['cht']
  X = df.drop(columns=['cht']) # drop the target column

  # find features with same value for all samples and drop them
  nunique = X.nunique()
  cols_to_drop = nunique[nunique <= 1].index # 0 if all are missing, 1 if ther is only one catgeory/value
  X_train = X.drop(cols_to_drop, axis='columns')

  X_train_coded = X_train.replace({
    'f': -1,
    't': 1,
    'na': 0
  })

  # train the model
  results = rf(X_train_coded, y, X_train_coded.columns.tolist())

  return StreamingResponse(encode_results(results))

# never ending generator for our streaming response
def rf(X, y, feature_names, batch_size=25, total_forest_size=500):
  params = {
    "class_weight": 'balanced',
    "random_state":  42,
    "warm_start": True
  }
  forest = RandomForestClassifier(**params)
  for i in range(batch_size, total_forest_size+1, batch_size):
    forest.set_params(n_estimators=i)
    forest.fit(X, y)
    importances = [
      {
        'attribute': name[:name.rindex('_')] if '_' in name else name,
        'category': name[name.rindex('_')+1:] if '_' in name else None,
        'importance': round(importance, 3)
      } for name,importance in zip(feature_names, forest.feature_importances_)
    ]
    #importances.sort(reverse=True)
    response = {
      "trees": i,
      "importances": importances, 
    }
    print('done', i)
    yield response


async def encode_results(data):
  try:
    for feature_list in data: # data is inside another array
      #for feature_importance in feature_list:
      yield json.dumps(feature_list).encode('utf-8')
      await asyncio.sleep(0.25) # necessary to catch cancellation
  except asyncio.CancelledError:
    print("caught cancelled error")


def create_query(cht: int, ids: list[str], exclude: list[str], table_name: str):
  # exlucde = https://github.com/duckdb/duckdb/pull/2276
  tissue_list = map(lambda d: d['tissuename'], ids)
  id_string_list = separator.join(f"'{id}'" for id in tissue_list)
  query =  "SELECT * " + \
    ("" if not exclude else f"EXCLUDE ({separator.join(exclude)}) ") + \
    f", {cht} AS cht " + \
    f"FROM {table_name}_table " + \
    f"WHERE {table_name}_table.tissuename IN ({id_string_list})"
  # logging.warning(f'Query of cohort {cht}: {query}');
  return query

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=8444)