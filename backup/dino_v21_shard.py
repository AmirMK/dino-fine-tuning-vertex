import os
import copy
import logging
import shutil
import tempfile
import random
import math
import argparse
import sys
from typing import Dict, Any, List

from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (
    Dinov2Model,
    Dinov2Config,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    PreTrainedModel,
    PretrainedConfig,
)
from datasets import Dataset, Image as HFImage
from google.cloud import storage
try:
    from hypertune import HyperTune
except ImportError:
    HyperTune = None

import webdataset as wds

# --- 1. Configuration & Seeding ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True

def load_data_config() -> Dict[str, Any]:
    """Loads GCS path configuration from environment variables."""
    return {
        "MODEL_GCS_BUCKET_NAME": os.environ.get("MODEL_GCS_BUCKET_NAME", "dino-model"),
        "MODEL_GCS_PREFIX": os.environ.get("MODEL_GCS_PREFIX", "dino_v2_serving_package/model_artifacts"),
        "IMAGES_GCS_BUCKET_NAME": os.environ.get("IMAGES_GCS_BUCKET_NAME", "my_dino_test"),
        "IMAGES_GCS_PREFIX": os.environ.get("IMAGES_GCS_PREFIX", "my_dino_shard"),
        "OUTPUT_GCS_BUCKET_NAME": os.environ.get("OUTPUT_GCS_BUCKET_NAME", "dino-model"),
        "OUTPUT_GCS_PREFIX": os.environ.get("OUTPUT_GCS_PREFIX", "dino_v21_shard_finetuned_on_custom_data_full"),
    }

def parse_args() -> argparse.Namespace:
    """Parse CLI hyperparameters."""

    p = argparse.ArgumentParser(description="DINOv2 fine-tuning-test")
    p.add_argument("--projection_head_out_dim", type=int, default=16384)
    p.add_argument("--training_epochs", type=int, default=10)
    p.add_argument("--per_device_batch_size", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--num_global_crops", type=int, default=2)
    p.add_argument("--num_local_crops", type=int, default=0)
    p.add_argument("--student_temp", type=float, default=0.1)
    p.add_argument("--center_momentum", type=float, default=0.9)

    return p.parse_args()

def load_hyperparameters(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads model hyperparameters from CLI args."""
    return {
        "PROJECTION_HEAD_OUT_DIM": args.projection_head_out_dim,
        "TRAINING_EPOCHS": args.training_epochs,
        "PER_DEVICE_BATCH_SIZE": args.per_device_batch_size,
        "LEARNING_RATE": args.learning_rate,
        "WEIGHT_DECAY": args.weight_decay,
        "GRADIENT_ACCUMULATION_STEPS": args.gradient_accumulation_steps,
        "NUM_GLOBAL_CROPS": args.num_global_crops,
        "NUM_LOCAL_CROPS": args.num_local_crops,
        "STUDENT_TEMP": args.student_temp,
        "CENTER_MOMENTUM": args.center_momentum,
    }

# --- 1.2 Load configs and define paths ---
args = parse_args()
data_config = load_data_config()
hyperparameters = load_hyperparameters(args)

MODEL_GCS_BUCKET_NAME = data_config["MODEL_GCS_BUCKET_NAME"]
MODEL_GCS_PREFIX = data_config["MODEL_GCS_PREFIX"]
IMAGES_GCS_BUCKET_NAME = data_config["IMAGES_GCS_BUCKET_NAME"]
IMAGES_GCS_PREFIX = data_config["IMAGES_GCS_PREFIX"]
OUTPUT_GCS_BUCKET_NAME = data_config["OUTPUT_GCS_BUCKET_NAME"]
OUTPUT_GCS_PREFIX = data_config["OUTPUT_GCS_PREFIX"]

LOCAL_MODEL_PATH = tempfile.mkdtemp()
LOCAL_OUTPUT_DIR = tempfile.mkdtemp()

PROJECTION_HEAD_OUT_DIM = hyperparameters["PROJECTION_HEAD_OUT_DIM"]
TRAINING_EPOCHS = hyperparameters["TRAINING_EPOCHS"]
PER_DEVICE_BATCH_SIZE = hyperparameters["PER_DEVICE_BATCH_SIZE"]
LEARNING_RATE = hyperparameters["LEARNING_RATE"]
WEIGHT_DECAY = hyperparameters["WEIGHT_DECAY"]
GRADIENT_ACCUMULATION_STEPS = hyperparameters["GRADIENT_ACCUMULATION_STEPS"]
NUM_GLOBAL_CROPS = hyperparameters["NUM_GLOBAL_CROPS"]
NUM_LOCAL_CROPS = hyperparameters["NUM_LOCAL_CROPS"]
STUDENT_TEMP = hyperparameters["STUDENT_TEMP"]
CENTER_MOMENTUM = hyperparameters["CENTER_MOMENTUM"]

MOMENTUM_TEACHER_SCHEDULE = np.array([])
TEACHER_TEMP_SCHEDULE = np.array([])
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- 2. GCS & File Helper Functions ---
def download_gcs_folder(bucket_name, prefix, local_path):
    logger.info(f"Downloading gs://{bucket_name}/{prefix} to {local_path}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for blob in bucket.list_blobs(prefix=prefix):
        if blob.name.endswith("/"):
            continue
        rel_path = os.path.relpath(blob.name, prefix or "")
        file_path = os.path.join(local_path, rel_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        blob.download_to_filename(file_path)

def upload_gcs_folder(bucket_name, prefix, local_path):
    logger.info(f"Uploading from {local_path} to gs://{bucket_name}/{prefix}...")
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    for root, _, files in os.walk(local_path):
        for file in files:
            local_file = os.path.join(root, file)
            gcs_path = os.path.join(prefix, os.path.relpath(local_file, local_path))
            bucket.blob(gcs_path).upload_from_filename(local_file)

def list_gcs_images_as_fuse_paths(bucket_name: str, prefix: str) -> List[str]:
    logger.info(f"Building file manifest from gs://{bucket_name}/{prefix}...")
    storage_client = storage.Client()
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)
    
    image_paths = []
    for blob in blobs:
        # FIX 3: Skip "directory placeholder" blobs which end with a "/"
        if blob.name.endswith('/'):
            continue
        if blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
            fuse_path = f"/gcs/{bucket_name}/{blob.name}"
            image_paths.append(fuse_path)
            
    return image_paths

class VertexHPTCallback(TrainerCallback):
    def __init__(self, tag: str = "train_dino_loss"):
        self.tag = tag
        # FIX 4: Simplified HyperTune check
        self.ht = HyperTune() if HyperTune else None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.ht and logs and "loss" in logs:
            self.ht.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=self.tag,
                metric_value=float(logs["loss"]),
                global_step=int(state.global_step),
            )

# --- 3. DINOv2 Multi-Crop Augmentation (No changes) ---
class DinoMultiCropTransform:
    def __init__(self, processor):
        size_cfg = getattr(processor, "crop_size", None) or processor.size
        H = int(size_cfg['height']) if isinstance(size_cfg, dict) else int(size_cfg)
        
        self.normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        self.global_transform = transforms.Compose([
            transforms.RandomResizedCrop(H, scale=(0.4, 1.0), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            self.normalize,
        ])
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(H, scale=(0.05, 0.4), interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            self.normalize,
        ])

    def __call__(self, image):
        crops = [self.global_transform(image) for _ in range(NUM_GLOBAL_CROPS)]
        crops += [self.local_transform(image) for _ in range(NUM_LOCAL_CROPS)]
        return crops

# --- 4. DINO Architecture (No changes) ---
class DinoHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, bottleneck_dim))
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False)); self.last_layer.weight_g.data.fill_(1); self.last_layer.weight_g.requires_grad = False
    def forward(self, x):
        x = self.mlp(x); x = nn.functional.normalize(x, dim=-1, p=2); return self.last_layer(x)

class DinoV2SSLConfig(PretrainedConfig):
    model_type = "dinov2_ssl_head"
    def __init__(self, backbone_config=None, out_dim=16384, hidden_dim=2048, bottleneck_dim=256, use_pooler=True, **kwargs):
        super().__init__(**kwargs); self.backbone_config = backbone_config or {}; self.out_dim = out_dim; self.hidden_dim = hidden_dim; self.bottleneck_dim = bottleneck_dim; self.use_pooler = use_pooler

class DinoV2SSLModel(PreTrainedModel):
    config_class = DinoV2SSLConfig
    def __init__(self, config: DinoV2SSLConfig):
        super().__init__(config); dcfg = Dinov2Config(**config.backbone_config); self.backbone = Dinov2Model(dcfg)
        self.head = DinoHead(in_dim=self.backbone.config.hidden_size, out_dim=config.out_dim, hidden_dim=config.hidden_dim, bottleneck_dim=config.bottleneck_dim)
    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values); feats = outputs.pooler_output if self.config.use_pooler and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None else outputs.last_hidden_state.mean(dim=1); return self.head(feats)

# --- 5. Custom Loss, Trainer, and Callback (No changes) ---
class DinoLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__(); self.student_temp = student_temp; self.center_momentum = center_momentum; self.register_buffer("center", torch.zeros(1, out_dim))
    def forward(self, student_output, teacher_output, teacher_temp):
        student_out = student_output / self.student_temp; teacher_out = nn.functional.softmax((teacher_output - self.center) / teacher_temp, dim=-1).detach()
        student_chunks, teacher_chunks = student_out.chunk(NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS), teacher_out.chunk(NUM_GLOBAL_CROPS); total_loss, n_loss_terms = 0, 0
        for iq, q in enumerate(teacher_chunks):
            for v in range(len(student_chunks)):
                if v != iq: loss = torch.sum(-q * nn.functional.log_softmax(student_chunks[v], dim=-1), dim=-1); total_loss += loss.mean(); n_loss_terms += 1
        total_loss /= n_loss_terms; self.update_center(teacher_output); return total_loss
    
    @torch.no_grad()
    def update_center(self, teacher_output): self.center.mul_(self.center_momentum).add_(teacher_output.mean(dim=0, keepdim=True), alpha=(1 - self.center_momentum))

class TeacherEmaCallback(TrainerCallback):
    def __init__(self, trainer): self.trainer = trainer
    def on_optimizer_step(self, args, state, control, **kwargs):
        step = max(0, min(state.global_step, len(MOMENTUM_TEACHER_SCHEDULE) - 1)); momentum = MOMENTUM_TEACHER_SCHEDULE[step]; student, teacher = self.trainer.model, self.trainer.teacher; student_device = next(student.parameters()).device
        if next(teacher.parameters()).device != student_device: teacher.to(student_device)
        with torch.no_grad():
            for sp, tp in zip(student.parameters(), teacher.parameters()): tp.data.mul_(momentum).add_(sp.data, alpha=1.0 - momentum)

class DinoTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.teacher = DinoV2SSLModel(copy.deepcopy(self.model.config)); self.teacher.load_state_dict(self.model.state_dict());
        for p in self.teacher.parameters(): p.requires_grad = False
        self.teacher.eval().to(self.args.device); self.loss_fn = DinoLoss(self.model.config.out_dim, STUDENT_TEMP, CENTER_MOMENTUM).to(self.args.device)
   
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        pixel_values = inputs["pixel_values"]
        V = NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS
        batch_size = pixel_values.shape[0] // V  # infer instead of reading "batch_size"

        # forward pass
        student_output = model(pixel_values)
        with torch.no_grad():
            teacher_output = self.teacher(pixel_values[:NUM_GLOBAL_CROPS * batch_size])

        # loss
        step = self.state.global_step
        temp_idx = min(step, len(TEACHER_TEMP_SCHEDULE) - 1)
        current_teacher_temp = TEACHER_TEMP_SCHEDULE[temp_idx]
        loss = self.loss_fn(student_output, teacher_output, current_teacher_temp)

        # logging
        with torch.no_grad():
            momentum = MOMENTUM_TEACHER_SCHEDULE[min(step, len(MOMENTUM_TEACHER_SCHEDULE) - 1)]
            q = nn.functional.softmax(
                (teacher_output - self.loss_fn.center) / current_teacher_temp, dim=-1
            )
            entropy = -(q * (q.clamp_min(1e-12).log())).sum(-1).mean()
            self.log({
                "teacher_entropy": float(entropy.item()),
                "teacher_temp": float(current_teacher_temp),
                "ema_momentum": float(momentum),
            })

        return loss


# --- 6. Data Collator & Schedule (No changes) ---
def collate_fn(examples):
    V = NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS
    pixel_values_by_view = [[] for _ in range(V)]
    for ex in examples:
        for i, crop in enumerate(ex["pixel_values"]):
            pixel_values_by_view[i].append(crop)
    collated_views = [torch.stack(view_tensors) for view_tensors in pixel_values_by_view]
    return {"pixel_values": torch.cat(collated_views, dim=0)}  # ← drop "batch_size"

def cosine_schedule(start, end, steps): return end + 0.5 * (start - end) * (1 + np.cos(np.pi * np.arange(steps) / steps))

# Build a list of shard paths via the API, but open via FUSE
def list_gcs_tar_shards_as_fuse_paths(bucket: str, prefix: str) -> List[str]:
    client = storage.Client()
    shards = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if name.endswith("/") or not name.lower().endswith(".tar"):
            continue
        shards.append(f"/gcs/{bucket}/{name}")
    return shards

def _map_img_to_crops(img):
    return {"pixel_values": mc_transform(img.convert("RGB"))}

MC_TRANSFORM = None  # global, visible to workers

def init_mc_transform(processor):
    global MC_TRANSFORM
    MC_TRANSFORM = DinoMultiCropTransform(processor)

def map_record(tup):
    # tup is (PIL.Image,) from .to_tuple("jpg;png;jpeg")
    img = tup[0]
    return {"pixel_values": MC_TRANSFORM(img.convert("RGB"))}

# --- 7. Main Execution Logic ---
def main():
    try:
        download_gcs_folder(MODEL_GCS_BUCKET_NAME, MODEL_GCS_PREFIX, LOCAL_MODEL_PATH)
        
        # FIX 2: Use a safer FUSE path check on the bucket root
        fuse_bucket_root = f"/gcs/{IMAGES_GCS_BUCKET_NAME}"
        gcs_uri = f"gs://{IMAGES_GCS_BUCKET_NAME}/{IMAGES_GCS_PREFIX}"
        logger.info(f"Checking for GCS FUSE mount at: {fuse_bucket_root}")
        if not os.path.exists(fuse_bucket_root):
            raise FileNotFoundError(
                f"GCS FUSE bucket mount not found: {fuse_bucket_root}. "
                "Verify Vertex AI FUSE mount and service account access."
            )
        
        
        shard_paths = list_gcs_tar_shards_as_fuse_paths(IMAGES_GCS_BUCKET_NAME, IMAGES_GCS_PREFIX)
        if not shard_paths:
            raise FileNotFoundError(f"No .tar shards found in gs://{IMAGES_GCS_BUCKET_NAME}/{IMAGES_GCS_PREFIX}")

        logger.info(f"Using {len(shard_paths)} shards from gs://{IMAGES_GCS_BUCKET_NAME}/{IMAGES_GCS_PREFIX}")
        image_processor = AutoImageProcessor.from_pretrained(LOCAL_MODEL_PATH, use_fast=False)
        mc_transform = DinoMultiCropTransform(image_processor)
        
        dataset = (
            wds.WebDataset(shard_paths, handler=wds.handlers.warn_and_continue)
            .shuffle(1000)             # optional; safe to keep
            .decode("pil")
            .to_tuple("jpg;jpeg;png")
            #.map(lambda tup: _map_img_to_crops(tup[0]))
            #.map(map_record)
            .map(lambda tup: {"pixel_values": mc_transform(tup[0].convert("RGB"))})
        )

        base_backbone = Dinov2Model.from_pretrained(LOCAL_MODEL_PATH)
        ssl_cfg = DinoV2SSLConfig(backbone_config=base_backbone.config.to_dict(), out_dim=PROJECTION_HEAD_OUT_DIM)
        student_model = DinoV2SSLModel(ssl_cfg)
        student_model.backbone.load_state_dict(base_backbone.state_dict())
        student_model.backbone.gradient_checkpointing_enable()
        student_model.backbone.config.use_cache = False

        # dataset size (if known)
        DATASET_SIZE = 3500  # for now, or count your WebDataset samples dynamically

        # effective batch size
        effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

        # steps per epoch
        steps_per_epoch = math.ceil(DATASET_SIZE / effective_batch)

        
        # FIX 1: Use correct `dataloader_*` argument names for TrainingArguments
        training_args = TrainingArguments(
            output_dir=LOCAL_OUTPUT_DIR,
            per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
            num_train_epochs=TRAINING_EPOCHS,
            max_steps=steps_per_epoch * TRAINING_EPOCHS,
            fp16=False,
            bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
            torch_compile=True,
            optim="adamw_torch_fused", #new
            save_strategy="epoch",
            save_total_limit=1, #new
            logging_steps=100,#25
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            dataloader_drop_last=True,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            seed=SEED,
            # Arguments for data loading performance
            dataloader_num_workers=min(int(os.cpu_count() * 0.75), len(shard_paths)),#max(1, (os.cpu_count() or 2) // 2),
            dataloader_pin_memory=True if torch.cuda.is_available() else False,
            dataloader_persistent_workers=True if torch.cuda.is_available() else False,
            dataloader_prefetch_factor=6, #2
        )
        
        trainer = DinoTrainer(
            model=student_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collate_fn
        )

        num_optimizer_steps = training_args.max_steps
        if not num_optimizer_steps or num_optimizer_steps <= 0:
            raise ValueError("Set a positive max_steps when using WebDataset/IterableDataset.")

        
        global MOMENTUM_TEACHER_SCHEDULE, TEACHER_TEMP_SCHEDULE
        MOMENTUM_TEACHER_SCHEDULE = cosine_schedule(0.996, 1.0, num_optimizer_steps).tolist()
        TEACHER_TEMP_SCHEDULE = np.concatenate([np.linspace(0.04, 0.04, 100), cosine_schedule(0.04, 0.07, num_optimizer_steps - 100)]).tolist()

        trainer.add_callback(VertexHPTCallback("train_dino_loss"))
        trainer.add_callback(TeacherEmaCallback(trainer))

        logger.info("Starting DINOv2 self-supervised fine-tuning...")
        trainer.train()

        logger.info(f"Saving final model to {LOCAL_OUTPUT_DIR}...")
        trainer.save_model(LOCAL_OUTPUT_DIR)
        image_processor.save_pretrained(LOCAL_OUTPUT_DIR)

        upload_gcs_folder(OUTPUT_GCS_BUCKET_NAME, OUTPUT_GCS_PREFIX, LOCAL_OUTPUT_DIR)
    
    finally:
        logger.info("Cleaning up temporary local directories.")
        shutil.rmtree(LOCAL_MODEL_PATH, ignore_errors=True)
        shutil.rmtree(LOCAL_OUTPUT_DIR, ignore_errors=True)

if __name__ == "__main__":
    main()
