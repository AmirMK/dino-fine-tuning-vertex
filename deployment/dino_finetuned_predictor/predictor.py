import os
import io
import json
import math
import tempfile
from typing import List, Dict, Any

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from transformers import (
    AutoImageProcessor,
    Dinov2Model,
    Dinov2Config,
    PreTrainedModel,
    PretrainedConfig,
)

from safetensors.torch import load_file
from google.cloud import storage
from google.cloud.aiplatform.prediction.predictor import Predictor


# ------------------------
# Helpers
# ------------------------
def _download_selected_from_gcs(prefix: str, filenames: List[str]) -> str:
    """Download only specific files from a gs://bucket/prefix into a temp dir. Return local dir."""
    if not prefix.startswith("gs://"):
        # Already local path; assume files are present there.
        return prefix

    bucket_name, key_prefix = prefix[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    tmpdir = tempfile.mkdtemp()

    for fname in filenames:
        blob = bucket.blob(f"{key_prefix.rstrip('/')}/{fname}")
        if not blob.exists(client):
            raise FileNotFoundError(f"Missing required file in GCS: gs://{bucket_name}/{key_prefix}/{fname}")
        dst = os.path.join(tmpdir, fname)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        blob.download_to_filename(dst)

    return tmpdir


def _prepare_dir_with_patched_image_size(model_dir: str) -> str:
    """
    Read positional embedding length from the checkpoint, infer grid size,
    set backbone_config.image_size accordingly, and write a patched config.json.
    Return a new temp dir containing the patched files.
    """
    cfg_path = os.path.join(model_dir, "config.json")
    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    bcfg = cfg.get("backbone_config", {})
    patch_size = int(bcfg.get("patch_size", 14))  # dinov2-base default: 14

    # Find position embeddings tensor to infer grid
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    pos_key = None
    for k in (
        "backbone.embeddings.position_embeddings",
        "embeddings.position_embeddings",
    ):
        if k in sd:
            pos_key = k
            break
    if pos_key is None:
        # fallback heuristic
        for k in sd.keys():
            if k.endswith("embeddings.position_embeddings"):
                pos_key = k
                break
    if pos_key is None:
        raise RuntimeError("Could not find position_embeddings in state dict.")

    L = sd[pos_key].shape[1]  # includes class token
    num_patches = L - 1
    grid = int(round(math.sqrt(num_patches)))
    if grid * grid != num_patches:
        raise RuntimeError(
            f"Pos-emb length {L} -> {num_patches} patches is not a perfect square."
        )
    image_size = int(grid * patch_size)

    # Patch config.json into a temp dir
    patched = tempfile.mkdtemp()
    # copy the three model files
    for fname in ("config.json", "preprocessor_config.json", "model.safetensors"):
        src = os.path.join(model_dir, fname)
        dst = os.path.join(patched, fname)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(src, "rb") as s, open(dst, "wb") as d:
            d.write(s.read())

    cfg.setdefault("backbone_config", {})
    cfg["backbone_config"]["image_size"] = image_size
    with open(os.path.join(patched, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    return patched


def _load_image_from_gcs(gcs_uri: str) -> Image.Image:
    """Load an image from gs:// URI into a PIL Image without writing to disk."""
    assert gcs_uri.startswith("gs://")
    bucket_name, blob_path = gcs_uri[5:].split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    data = blob.download_as_bytes()
    return Image.open(io.BytesIO(data)).convert("RGB")


def _load_image_from_b64(b64_str: str) -> Image.Image:
    import base64
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


# ------------------------
# Custom Config & Model
# ------------------------
class DinoV2SSLConfig(PretrainedConfig):
    model_type = "dinov2_ssl_head"

    def __init__(self, backbone_config=None, out_dim=16384, hidden_dim=2048, bottleneck_dim=256, use_pooler=True, **kwargs):
        super().__init__(**kwargs)
        self.backbone_config = backbone_config or {}
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.use_pooler = use_pooler


class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2, eps=1e-6)
        return self.last_layer(x)


class DinoV2SSLModel(PreTrainedModel):
    config_class = DinoV2SSLConfig

    def __init__(self, config: DinoV2SSLConfig):
        super().__init__(config)
        dcfg = Dinov2Config(**config.backbone_config)
        self.backbone = Dinov2Model(dcfg)
        self.head = DinoHead(
            in_dim=self.backbone.config.hidden_size,
            out_dim=config.out_dim,
            hidden_dim=config.hidden_dim,
            bottleneck_dim=config.bottleneck_dim,
        )

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values)
        feats = outputs.pooler_output if getattr(outputs, "pooler_output", None) is not None else outputs.last_hidden_state.mean(dim=1)
        # Return normalized backbone features (embedding)
        return nn.functional.normalize(feats, dim=-1, p=2)


# ------------------------
# Predictor
# ------------------------
class DinoPredictor(Predictor):
    """
    Vertex AI Custom Python Predictor.
    Loads model once at startup in `load()` and serves embeddings in `predict()`.
    """

    def __init__(self):
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = None
        self._model = None
        self._ready = False

    def load(self, artifacts_uri: str):
        """
        Called exactly once when the server starts.
        `artifacts_uri` is the GCS folder you point Vertex AI to (package root).
        By default, we expect an environment variable MODEL_GCS_URI that
        points to your fine-tuned checkpoint folder containing:
           - config.json
           - preprocessor_config.json
           - model.safetensors

        If MODEL_GCS_URI is not set, we fallback to `artifacts_uri`.
        """
        # Prefer explicit model path via env var so the package can live separate from weights
        model_uri = os.environ.get("MODEL_GCS_URI", "").strip() or artifacts_uri

        # Download just the files we need
        local_model_dir = _download_selected_from_gcs(
            model_uri,
            ["config.json", "preprocessor_config.json", "model.safetensors"],
        )

        # Patch image size based on learned pos-emb grid
        patched_dir = _prepare_dir_with_patched_image_size(local_model_dir)

        # Load processor and model once
        self._processor = AutoImageProcessor.from_pretrained(patched_dir)
        self._model = DinoV2SSLModel.from_pretrained(patched_dir).to(self._device).eval()

        self._ready = True
        print("[DinoPredictor] Model loaded and ready.")

    def predict(self, instances: List[Dict[str, Any]], parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Request schema:
          {
            "instances": [
              {"gcs_uri": "gs://bucket/path/image1.jpg"},
              {"gcs_uri": "gs://bucket/path/image2.png"},
              {"bytes_base64": "<...>"},
              ...
            ]
          }

        Response schema:
          {
            "predictions": [
               {"embedding": [float, ...], "shape": [D]},
               ...
            ]
          }
        """
        if not self._ready:
            raise RuntimeError("Model not loaded yet.")

        images: List[Image.Image] = []        
        for inst in instances['instances']:            
            if "gcs_uri" in inst:
                images.append(_load_image_from_gcs(inst["gcs_uri"]))
            elif "bytes_base64" in inst:
                images.append(_load_image_from_b64(inst["bytes_base64"]))
            else:
                raise ValueError("Each instance must include either 'gcs_uri' or 'bytes_base64'.")

        inputs = self._processor(images=images, return_tensors="pt")
        with torch.no_grad():
            embs = self._model(inputs["pixel_values"].to(self._device))  # [N, D]
        embs_np = embs.cpu().numpy()

        predictions = [{"embedding": emb.astype(np.float32).tolist(), "shape": [int(emb.shape[0])]} for emb in embs_np]
        return {"predictions": predictions}

    # Optional (used by Vertex AI for health checks)
    def health(self) -> Dict[str, Any]:
        return {"status": "ok" if self._ready else "loading"}
