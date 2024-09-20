import os
import wandb
import torch
from .segmentation_model import SegmentationModel
from dotenv import load_dotenv

load_dotenv()

MODELS_DIR = "artifacts"
MODEL_FILE_NAME = "best_model.pth"


def download_artifact():
    assert "WANDB_API_KEY" in os.environ, "Please enter the required environment variables."

    wandb.login()
    wandb_org = os.environ.get("WANDB_ORG")
    wandb_project = os.environ.get("WANDB_PROJECT")
    wandb_model_name = os.environ.get("WANDB_MODEL_NAME")
    wandb_model_version = os.environ.get("WANDB_MODEL_VERSION")

    artifact_path = f"{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}"
    artifact = wandb.Api().artifact(artifact_path, type="model")
    artifact.download(root=MODELS_DIR)


def get_raw_model() -> SegmentationModel:
    model = SegmentationModel()
    return model


def load_model() -> SegmentationModel:
    download_artifact()

    model = get_raw_model()
    model_state_dict_path = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    model_state_dict = torch.load(model_state_dict_path, map_location="cpu")
    model.load_state_dict(model_state_dict, strict=False)
    model.eval()

    return model
