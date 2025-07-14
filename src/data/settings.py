from pathlib import Path
from typing import Self
from urllib.parse import urlparse

from dotenv import find_dotenv
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from loguru import logger

class DatasetSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=find_dotenv(".env"), validate_default=True, extra="ignore")
    dataset_path: str

    aws_endpoint_url_s3: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    dataset_zip_path_s3: str = "s3://"
    """Raw zip path get from environment, start with s3://"""
    
    _dataset_bucket_name: str|None = None
    _dataset_path_s3: str|None = None

    @model_validator(mode="after")
    def check_exist_and_resolve(self) -> Self:
        assert (path := Path(self.dataset_path).resolve()).is_dir() and path.exists()
        assert (s3_path := self.dataset_zip_path_s3).startswith(
            "s3://"
        ) and s3_path.endswith("/")

        self.dataset_path = path.as_posix()
        parsed_url = urlparse(s3_path)
        self._dataset_bucket_name = parsed_url.netloc
        self._dataset_path_s3 = parsed_url.path.lstrip("/")

        return self
    
    @property
    def bucket_name(self):
        assert isinstance(p:=self._dataset_bucket_name, str)
        return p
    
    @property
    def dataset_path_s3(self):
        assert isinstance(p:=self._dataset_path_s3, str)
        return p
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        logger.info(self.model_dump_json(indent=4))