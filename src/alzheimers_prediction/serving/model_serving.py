import mlflow
from loguru import logger

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput
)

from loguru import logger


class ModelServing:
    def __init__(self, model_name:str, endopoint_name: str):
        """
        It manages the model serving configuration and deployment.
        """
        self.model_name = model_name
        self.endpoint_name = endopoint_name
        self.workspace_client = WorkspaceClient()
        logger.info(f"Databricks Workspace ID: {self.workspace_client.get_workspace_id()}")

    def get_latest_model_version(self):
        """
        Get the latest version of the model from MLflow.
        """
        client = mlflow.MlflowClient()
        logger.info(f"MLFLOW Registry uri {client._registry_uri}")
        latest_version = client.get_model_version_by_alias(self.model_name, alias="latest-model").version
        logger.info(f"Latest model version: {latest_version}")
        return latest_version
    
    def deploy_or_update_serving_endpoint(self, 
                                          version: str = "latest",
                                          workload_size: str = "Small",
                                          scale_to_zero: bool = True):
        """
        Deploy or update the serving endpoint.
        : param version: str. Version of the model to deploy
        : param workload_size: str. Number of concurrent requests (Small is 4) https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints?language=MLflow%C2%A0Deployments%C2%A0SDK
        : param scale_to_zero: bool. If True, the endpoint will shut down when there are no requests. It will take longer to start up again.
        """
        endpoint_exists = any(item.name == self.endpoint_name for item in self.workspace_client.serving_endpoints.list())
        logger.info(f"Endpoint exists: {endpoint_exists}")

        if version == "latest":
            logger.info("Getting latest model version")
            entity_version = self.get_latest_model_version()
        else:
            logger.info(f"Using model version {version}")
            entity_version = version

        logger.info(f"Model version: {entity_version}")
        served_entities = [
            ServedEntityInput(
                entity_name=self.model_name,
                scale_to_zero_enabled=scale_to_zero,
                workload_size=workload_size,
                entity_version=entity_version
            )
        ]

        if not endpoint_exists:
            logger.info(f"Creating endpoint {self.endpoint_name}")
            self.workspace_client.serving_endpoints.create(
                name=self.endpoint_name,
                config=EndpointCoreConfigInput(
                    served_entities=served_entities
                )
            )
            logger.info(f"Endpoint {self.endpoint_name} created")
        else:
            logger.info(f"Updating endpoint {self.endpoint_name}")
            self.workspace_client.serving_endpoints.update_config(
                name=self.endpoint_name,
                served_entities=served_entities
                )
            