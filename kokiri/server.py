# TODO patch sklearn if necessary https://intel.github.io/scikit-learn-intelex/ (or use skranger)
#from .settings import KokiriSettings
from settings import KokiriSettings # IMPORT FOR DEBUGGING

import uvicorn # For debugging
from typing import Dict, Optional

from fastapi import FastAPI, WebSocket
from fastapi import __version__ as fastapi_version
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, conlist
import asyncio

import numpy as np
import pandas as pd
import duckdb

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import confusion_matrix
from sklearn.impute import KNNImputer

from umap import UMAP

# from coral.coral.sql_query_mapper import QueryElements
# from flask import Flask, abort, jsonify, request

import warnings
if __name__ != "__main__":
  # Hides sklearn warning logs when deployed
  # We are using warm_start=True and class_weight='balanced'
  # These should only be used together, if you train on the whole dataset (which you don't have to with warm_statrt)
  # We are currently training on the whole dataset, so we can ignore the error
  warnings.simplefilter('ignore', category=UserWarning)


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
  n_estimators: Optional[int] = None
  max_depth: Optional[int] = None
  min_samples_leaf: Optional[int] = None
  impute: bool = False
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
  return {"message": "Hello World2"}


# @app.websocket("/kokiri/cmp_meta/")
# async def cmp_meta(websocket: WebSocket):
#   await websocket.accept()
#   cmp_data = await websocket.receive_json()

#   X_train, y, meta = load_data(cmp_data, 'meta_table')
#   results = rf(X_train, y, meta, X_train.columns.tolist(), 25,
#     cmp_data["n_estimators"],
#     cmp_data["max_depth"],
#     cmp_data["min_samples_leaf"], # minimum size of a leaf (min_samples_split is similar, but can split, e.g., 10 patients into groups of 9 and 1)
#     False
#   )
#   final_model = await encode_results(websocket, results)
#   await embed(websocket, X_train, y, meta, final_model, 'prediction', 'euclidean')
#   return


# @app.get("/kokiri/recommendSplit")
# def recommendSplit():
#     error_msg = """Paramerter missing or wrong!
#     For the {route} query the following parameters are needed:
#     - name: name of the cohort
#     - isInitial: 0 if it has a parent cohort, 1 if it is the initial table
#     - previous: id of the previous cohort, -1 for the initial cohort
#     - database: database of the entitiy tale
#     - schema: schema of the entity table
#     - table: table of the entitiy""".format(
#         route="create"
#     )

#     try:
#         query = QueryElements()
#         cohort = query.get_cohort_from_db(
#             request.values, error_msg
#         )  # get parent cohort
#         new_cohort = query.create_cohort_num_filtered(
#             request.values, cohort, error_msg
#         )  # get filtered cohort from args and cohort
#         return query.add_cohort_to_db(new_cohort)  # save new cohort into DB
#     except RuntimeError as error:
#         abort(400, error)


# @app.route("/recommendSplit", methods=["GET", "POST"])
# # @login_required
# def recommend_split():
#   return "asdf"

@app.websocket("/kokiri/cmp_meta/")
async def cmp_meta(websocket: WebSocket):
  await websocket.accept()
  cmp_data = await websocket.receive_json()

  X_train, y, meta = load_data(cmp_data, 'meta_table')
  results = rf(X_train, y, meta, X_train.columns.tolist(), 25,
    cmp_data["n_estimators"],
    cmp_data["max_depth"],
    cmp_data["min_samples_leaf"], # minimum size of a leaf (min_samples_split is similar, but can split, e.g., 10 patients into groups of 9 and 1)
    False
  )
  final_model = await encode_results(websocket, results)
  await embed(websocket, X_train, y, meta, final_model, 'prediction', 'euclidean')
  return


@app.websocket("/kokiri/cmp_mutated/")
async def cmp_mutated(websocket: WebSocket):
  await websocket.accept()
  cmp_data = await websocket.receive_json()

  X_train, y, meta = load_data(cmp_data, 'mutated_table')

  if cmp_data["impute"]:
    # group X_train by cohort label (y value)
    # check ratio  of zeros in each set
    zero_values_count = X_train.groupby(y).apply(lambda x: (x == 0).sum() / len(x))

    # print the ratio of zeros for column SMAD4
    # print("number of zeros for column SMAD4: ", zero_values_count['SMAD4'])
    # ones = X_train.groupby(y).apply(lambda x: (x == 1).sum() / len(x))
    # print("number of 1 for column SMAD4: ", ones['SMAD4'])
    # minues_ones = X_train.groupby(y).apply(lambda x: (x == -1).sum() / len(x))
    # print("number of -1 for column SMAD4: ", minues_ones['SMAD4'])

    # drop all columns where more than 50% of the values in any group of y are zero
    X_train_filtered = X_train.drop(zero_values_count.columns[zero_values_count.gt(0.5, axis=0).any()], axis=1)
    # print all droppped columns sorted alphabetically
    print(f"dropped {len(X_train.columns)-len(X_train_filtered.columns)} of {len(X_train.columns)} columns. \n Dropped:", sorted(set(X_train.columns) - set(X_train_filtered.columns)))
    # print number of columns in x train

    # fill up missing values with sklearns knn imputer
    # use just one neighbor, because it calculates the mean which does not make sense with binary data (-1,1)
    # TODO implement custom function that uses the mode, see https://datascience.stackexchange.com/q/92308
    imputer = KNNImputer(missing_values=0, n_neighbors=1, weights="distance")
    # TODO impute within each cohort, possibly using a custom distance metric (different cohort, high distance, same cohort, low distance)
    X_train_filtered = pd.DataFrame(imputer.fit_transform(X_train_filtered), columns=X_train_filtered.columns)

    X_train = X_train_filtered
    # ones = X_train.groupby(y).apply(lambda x: (x == 1).sum() / len(x))
    # print("number of 1 for column SMAD4 - after imputation: ", ones['SMAD4'])
    # minues_ones = X_train.groupby(y).apply(lambda x: (x == -1).sum() / len(x))
    # print("number of -1 for column SMAD4 - after imputation: ", minues_ones['SMAD4'])
  else:
    print("skip imputation")

  results = rf(X_train, y, meta, X_train.columns.tolist(), 25,
    cmp_data["n_estimators"],
    cmp_data["max_depth"],
    cmp_data["min_samples_leaf"], # minimum size of a leaf (min_samples_split is similar, but can split, e.g., 10 patients into groups of 9 and 1)
    True
  )
  final_model = await encode_results(websocket, results)
  await embed(websocket, X_train, y, meta, final_model, 'prediction', 'euclidean')
  return


def load_data(cmp_data: CmpData, table_name):
  con = duckdb.connect(database=config.dbName, read_only=True) # TODO check if this is the way to go for multi thread access (cursors were mentioned in a blog psot)
  frames = []

  _log.debug(f'Fetching data for {len(cmp_data["ids"])} cohorts')
  for i, cht_ids in enumerate(cmp_data["ids"]):
    query = create_query(con, i, cht_ids, ['tdpid'] + cmp_data["exclude"], table_name)
    df = con.execute(query).df()
    _log.debug(f'Size of {i}. cohort: {df.shape}')
    frames.append(df)

  _log.debug(f'Concat cohort dataframes')
  df = pd.concat(frames, ignore_index=True)

  y = df['cht']
  X = df.drop(columns=['tissuename', 'cht']) # drop the target column
  meta = df[['tissuename', 'cht']]

  _log.debug(f'Drop columns with constant data')
  # find features with same value for all samples and drop them
  nunique = X.nunique()
  cols_to_drop = nunique[nunique <= 1].index # 0 if all are missing, 1 if ther is only one catgeory/value
  X_train = X.drop(cols_to_drop, axis='columns')
  # X_train= X_train.rename(columns={"tumortype": "Tumor Type"}) to test exclusion
  return X_train, y, meta


# never ending generator for our streaming response
def rf(X, y, meta, feature_names, batch_size=25, total_forest_size=500, max_depth=40, min_samples_leaf=5, remove_unknown=False):
  params = {
    "class_weight": 'balanced',
    "n_jobs": -1,
    "max_depth": max_depth if max_depth < 100 else None,
    "max_features": 0.8,
    "min_samples_leaf": min_samples_leaf,
    "oob_score": True,
    "bootstrap": True, # necessary for oob_score --> default: True for Random Forest, False for Extremely Random Forest
    "random_state":  42,
    "warm_start": True
  }

  _log.info('Starting RF with features: '+', '.join(feature_names))
  forest = RandomForestClassifier(**params)
  for i in range(batch_size, total_forest_size+1, batch_size):
    forest = forest.set_params(n_estimators=i)
    forest = forest.fit(X, y)
    score = forest.score(X, y)
    oobError = forest.oob_score_
    _log.debug(f'{len(forest.estimators_)}/{total_forest_size} estimators. Score: {oobError}')

    conf = confusion_matrix(y, forest.predict(X), normalize='pred') # normalize='pred' to get percentages per row/cohort

    prediction = forest.predict_proba(X)
    max_prediction = np.max(prediction, axis=1)
    probabilities = {"probs": prediction.tolist(), "prob_max": max_prediction}
    df_probs = pd.DataFrame(probabilities, columns=["probs", "prob_max"], index=X.index.copy())

    df_prediction = pd.concat([df_probs, meta],axis=1)
    tissue_chts = df_prediction.groupby('tissuename').agg({'cht': lambda cht_no: list(cht_no.map(str))})['cht'] # aggregate to list an parse to string - without turning into string, .agg(list) would also work
    df_prediction = df_prediction.drop(columns=['cht']).drop_duplicates('tissuename') # remove duplicates and columns already aggregated
    df_pred_grp = pd.merge(df_prediction, tissue_chts, how='left', on='tissuename')
    prediction_list = list(df_pred_grp.T.to_dict().values()) # convert to list of dicts, because using JSON becomes a string in frontend

    importance_threshold = 0.005
    estimators_threshold = 20
    importances = [
      {
        'attribute': name[:name.rindex('_')] if '_' in name else name,
        'category': name[name.rindex('_')+1:] if '_' in name else None,
        'importance': round(importance, 3),
        'distribution': [
          {
            'cht': '#'+str(cht),
             # sum of 1s in column divided by cohort size
            'value': round(X[name][(y == cht) &(X[name]>=0)].sum()/(X[name].abs()[y == cht].sum() if remove_unknown else (y==cht).sum()), 3) if i > estimators_threshold and round(importance, 3) >= importance_threshold else 1 # todo handle meta and mutated cases
          } for cht in np.unique(y).tolist()
        ],
        'random': True if i <= estimators_threshold or round(importance, 3) < importance_threshold else False,
        'type': 'cat' if X[name].isin([-1,0,1]).all() else 'num'
      } for name,importance in zip(feature_names, forest.feature_importances_)
    ]
    response = {
      "trees": i,
      "accuracy": score,
      "oobError": oobError,
      "confusionMatrix": conf.tolist(),
      "importances": importances,
      "probabilities": prediction_list
    }
    yield response, forest


async def encode_results(ws: WebSocket, data):
  final_model = None
  try:
    for feature_list, model in data: # data is inside another array
      await ws.send_json(feature_list, mode='text')
      final_model = model
      await asyncio.sleep(0.1) # necessary so that the json is actually sent
  except asyncio.CancelledError:
    _log.info("Request was cancelled")
  return final_model


# kudos https://github.com/gdmarmerola/forest-embeddings
async def embed(ws, X_train, y, meta, final_model, data='prediction', metric="euclidean"):
    await asyncio.sleep(0.1)

    np.random.seed(42)
    if data == 'prediction':
      prediction = final_model.predict_proba(X_train)
      embedding = UMAP(metric=metric, init='random', verbose=True).fit_transform(prediction, y)
      max_prediction = np.max(prediction, axis=1).reshape(-1, 1)
      predicted = np.argmax(prediction, axis=1).reshape(-1, 1)
    elif data == 'leaves':
      leaves = final_model.apply(X_train)
      embedding = UMAP(metric=metric, init='random', verbose=True).fit_transform(leaves, y)
    elif data == 'data':
      embedding = UMAP(metric=metric, init='random', verbose=True).fit_transform(X_train, y)

    df_xy = pd.DataFrame(
            np.concatenate((embedding, max_prediction, predicted), axis=1),
            columns=['x','y', 'max_prob', 'predicted'],
            index=X_train.index.copy())
    df_plot = pd.concat([df_xy, meta],axis=1)
    df_plot_list = list(df_plot.T.to_dict().values()) # convert to list of dicts, because using JSON becomes a string in frontend
    await ws.send_json({
      "data": data,
      "embedding": df_plot_list
    }, mode='text')


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

  #_log.debug(f'Query of cohort {cht}: {query}')
  return query


if __name__ == "__main__":
  uvicorn.run(app, host="0.0.0.0", port=9666)
