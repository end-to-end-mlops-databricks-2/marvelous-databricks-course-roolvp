from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel


class ProjectConfig(BaseModel):
    num_features: List[str]
    cat_features: List[str]
    target: str
    catalog_name: str
    schema_name: str
    parameters: Dict[str, Any]  # Dictionary to hold model-related parameters
    experiment_name_basic: Optional[str]
    experiment_name_custom: Optional[str]
    experiment_name_fe: Optional[str]

    @classmethod
    def from_yaml(cls, config_path: str, env: Optional[str] = None): 
        """Load configuration from a YAML file."""
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        if env is not None:
            config_dict["catalog_name"] = config_dict[env]["catalog_name"]
            config_dict["schema_name"] = config_dict[env]["schema_name"]

        else:
            config_dict["catalog_name"] = config_dict["catalog_name"]
            config_dict["schema_name"] = config_dict["schema_name"]

        
        return cls(**config_dict)


class Tags(BaseModel):
    git_sha: str
    branch: str
