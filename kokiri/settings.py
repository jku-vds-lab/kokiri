from typing import Dict

from pydantic import BaseSettings


class KokiriSettings(BaseSettings):
    dbName: str = "./kokiri/genie.duckdb"
    logging: Dict = {
      "version": 1,
      "disable_existing_loggers": False,
      "loggers": {"kokiri": {"level": "DEBUG"}}
    }
