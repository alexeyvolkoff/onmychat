import configparser
import os

_config = configparser.ConfigParser()
_config.read(os.path.join(os.path.dirname(__file__), "config.ini"), encoding="utf-8")

SETTINGS = _config["settings"]
USER_DATA_DIR = "user_data"
BASE_INDEX_DIR = "memory_index"
