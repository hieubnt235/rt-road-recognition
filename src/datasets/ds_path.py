import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
ds_path_env = os.getenv("DATASET_PATH")
assert ds_path_env is not None

datasets_path = Path(ds_path_env).resolve()


