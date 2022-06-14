# TODO patch sklearn if necessary https://intel.github.io/scikit-learn-intelex/ (or use skranger)
from .settings import KokiriSettings
# from settings import KokiriSettings # REPLACE IMPORT FOR DEBUGGING

import uvicorn # For debugging
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket
from fastapi import __version__ as fastapi_version
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import asyncio
import json

import pandas as pd
import duckdb

from sklearn.ensemble import RandomForestClassifier

import warnings
if __name__ != "__main__":
  # Hides sklearn warning logs when deployed
  # We are using warm_start=True and class_weight='balanced'
  # These should only be used together, if you train on the whole dataset (which you don't have to with warm_statrt)
  # We are currently training on the whole dataset, so we can ignore the error
  warnings.simplefilter('ignore', category=UserWarning)

# h2o.init()
# https://medium.com/tech-vision/random-forest-classification-with-h2o-python-for-beginners-b31f6e4ccf3c
# https://www.kaggle.com/code/nanomathias/h2o-distributed-random-forest-starter/script

import logging
_log = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('{levelname:<9s} {message}', style='{')
handler.setFormatter(formatter)
_log.addHandler(handler)
_log.setLevel(logging.DEBUG)

__author__ = 'Klaus Eckelt'

config = KokiriSettings()
separator = ', '

app = FastAPI()
_log.debug(f"fastapi version: {fastapi_version}")

origins = [
  "http://localhost",
  "http://127.0.0.1",
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

@app.get("/kokiri")
def root():
  return {"message": "Hello World"}

# TODO replace with custom websocket? i.e.: @app.websocket("/ws/{client_id}")
@app.websocket("/kokiri/cmp_meta/")
async def cmp_meta(websocket: WebSocket):
  await websocket.accept()
  cmp_data = await websocket.receive_json()

  X_train, y = load_data(cmp_data, 'meta_table')
  results = rf(X_train, y, X_train.columns.tolist())

  return await encode_results(websocket, results)


@app.websocket("/kokiri/cmp_mutated/")
async def cmp_mutated(websocket: WebSocket):
  await websocket.accept()
  cmp_data = await websocket.receive_json()

  X_train, y = load_data(cmp_data, 'mutated_table')
  results = rf(X_train, y, X_train.columns.tolist())

  return await encode_results(websocket, results)


def load_data(cmp_data: CmpData, table_name):
  con = duckdb.connect(database=config.dbName, read_only=True) # TODO check if this is the way to go for multi thread access (cursors were mentioned in a blog psot)
  frames = []
  
  _log.debug(f'Fetching data for {len(cmp_data["ids"])} cohorts')
  for i, cht_ids in enumerate(cmp_data["ids"]):
    query = create_query(con, i, cht_ids, ['tissuename', 'tdpid'] + cmp_data["exclude"], table_name)
    df = con.execute(query).df()
    frames.append(df)
    _log.debug(f'Size of {i}. cohort: {df.shape}')

  _log.debug(f'Concat cohort dataframes')
  df = pd.concat(frames)

  y = df['cht']
  X = df.drop(columns=['cht']) # drop the target column

  _log.debug(f'Drop columns with constant data')
  # find features with same value for all samples and drop them
  nunique = X.nunique()
  cols_to_drop = nunique[nunique <= 1].index # 0 if all are missing, 1 if ther is only one catgeory/value
  X_train = X.drop(cols_to_drop, axis='columns')
  # X_train= X_train.rename(columns={"tumortype": "Tumor Type"}) to test exclusion
  return X_train, y

# never ending generator for our streaming response
def rf(X, y, feature_names, batch_size=25, total_forest_size=500):
  params = {
    "class_weight": 'balanced',
    "random_state":  42,
    "warm_start": True
  }

  _log.info('Starting RF with features: '+', '.join(feature_names))
  forest = RandomForestClassifier(**params)
  for i in range(batch_size, total_forest_size+1, batch_size):
    _log.debug(f'{i}/{total_forest_size} estimators')
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
    yield response


async def encode_results(ws: WebSocket, data):
  try:
    for feature_list in data: # data is inside another array
      await ws.send_json(feature_list, mode='text')
      await asyncio.sleep(0.1) # necessary so that the json is actually sent
  except asyncio.CancelledError:
    _log.info("Request was cancelled")


def create_query(con: duckdb.DuckDBPyConnection, cht: int, ids: list[str], exclude: list[str], table_name: str):
  # get onehot encoded column names
  column_query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = '{table_name}'"
  table_info = con.execute(column_query).df()
  all_columns = table_info['column_name']
  excluded_db_columns_mask = all_columns.str.startswith(tuple(exclude)) # matches single feature columns and one hot encoded columns (e.g. 'age' and 'tumortype_*)
  
  tissue_list = map(lambda d: d['tissuename'], ids)
  id_string_list = separator.join(f"'{id}'" for id in tissue_list)
  
  if excluded_db_columns_mask.sum()/all_columns.size > 0.5:
    # more than half of the columns are excluded
    # --> select remaining columns
    select_db_columns = all_columns[~excluded_db_columns_mask].tolist()
    select_sql = separator.join(f'"{col}"' for col in select_db_columns) # column names need to be quoted
    _log.debug(f'select: {select_sql}')
    query =  f"SELECT {select_sql} " + \
      f", {cht} AS cht " + \
      f"FROM {table_name} " + \
      f"WHERE {table_name}.tissuename IN ({id_string_list})"
  else:
    # less than half of the columns are excluded
    # --> exclude specified columns from the query
    # exlucde = https://github.com/duckdb/duckdb/pull/2276
    exclude_db_columns = all_columns[excluded_db_columns_mask].tolist()
    exclude_sql = separator.join(f'"{col}"' for col in exclude_db_columns) # column names need to be quoted
    _log.debug(f'exclude: {exclude_sql}')  
    query =  "SELECT * " + \
      ("" if not exclude_sql else f"EXCLUDE ({exclude_sql}) ") + \
      f", {cht} AS cht " + \
      f"FROM {table_name} " + \
      f"WHERE {table_name}.tissuename IN ({id_string_list})"

  _log.debug(f'Query of cohort {cht}: {query}')
  return query

if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=9666)
