from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
import azure.ai.ml._artifacts._artifact_utilities as artifact_utils
import os
from typing import Union

class Web_handler:
    def __init__(self, config: str) -> None:
        credential = DefaultAzureCredential()
        self.client = MLClient.from_config(credential, path=config)

    def upload_to_datastore(self, filepath: str, name: str, version: Union[str, None] = None, description: str='') -> None:
        file = Data(
            name=name,
            description=description,
            version=version,
            path=filepath,
            type=AssetTypes.URI_FILE,
        )
        self.client.data.create_or_update(file)
        print(f"Data asset created. Name: {file.name}, version: {file.version}")

    def download_from_datastore(self, name: str, save_dir: str, version: Union[str, None] = None) -> None:
        #get version if specified, else get latest
        if version is not None:
            data = self.client.data.get(name, version=version)
        else:
            data = self.client.data.get(name, label='latest')

        # Download the dataset
        artifact_utils.download_artifact_from_aml_uri(
            uri = data.path,
            destination = save_dir, 
            datastore_operation=self.client.datastores
        )

        # Verify it is downloaded
        file_path = os.path.basename(data.path[10:])
        assert os.path.exists(os.path.join(save_dir, file_path))
        print(f'Data download [{name}] successful. Local Path: {os.path.normpath(os.path.abspath(save_dir))}')