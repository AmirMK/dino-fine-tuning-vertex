from google.cloud import aiplatform
import os
import base64
from google.cloud.aiplatform.prediction import LocalModel
from dino_finetuned_predictor.predictor import DinoPredictor


# --- Your Configuration ---
PROJECT_ID = "vertex-ai-search-v2"
REGION = "us-central1"
REPOSITORY = "dino-docker-repo" # The Artifact Registry repo 

# --- Configuration for the custom container image ---
MODEL_DISPLAY_NAME = "dino-full_v0-built-with-cpr"
IMAGE_NAME = "dino-full_v1-cpu-server" # Name for our the image

# The full URI for the custom image in Artifact Registry
CUSTOM_IMAGE_URI = f"{REGION}-docker.pkg.dev/{PROJECT_ID}/{REPOSITORY}/{IMAGE_NAME}:latest"

# The GCS path to the model weights, which will be loaded by the container
GCS_ARTIFACT_URI = f"gs://dino-model/dino_v2_finetuned_on_custom_data_full"

# The LOCAL path to the source code for building the container
LOCAL_MODEL_SOURCE_DIR = "./dino_fullfinetune_registry"


print("Configuration is set.")

# ==============================================================================
# Step 2: Initialize the Vertex AI SDK
# ==============================================================================
aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=GCS_BUCKET_NAME)
print("Vertex AI SDK Initialized.")

# ==============================================================================
# Step 3: Build a custom prediction routine (CPR) model image
# ==============================================================================
# It builds a new Docker image locally.
# It takes the base image, copies your predictor.py and other source files into it,
# and runs 'pip install' on your requirements.txt.

print(f"Building a new custom container image: {CUSTOM_IMAGE_URI}")
# This command can take several minutes to run.
local_model = LocalModel.build_cpr_model(
    src_dir=LOCAL_MODEL_SOURCE_DIR,
    output_image_uri=CUSTOM_IMAGE_URI,
    predictor=DinoPredictor,
    requirements_path=os.path.join(LOCAL_MODEL_SOURCE_DIR, "requirements.txt"),
)
print("✅ Custom CPR image built locally.")

# ==============================================================================
# Step 4: Push the custom image to Artifact Registry
# ==============================================================================
# The 'local_model' object you just created holds a reference to the image.
# The .push_image() method uploads the local image to the cloud repository.

print(f"Pushing the custom image to Artifact Registry...")
# This requires Docker permissions and gcloud authentication, which you have
# already configured in the terminal.
local_model.push_image()
print("✅ Custom image pushed successfully.")

# ==============================================================================
# Step 5: Register the Model using the Custom Image URI (The Correct Way)
# ==============================================================================
# This is the final step. We are creating a Model resource in Vertex AI
# by pointing directly to the custom container image you pushed to Artifact Registry
# and the model artifacts in GCS.

print("Registering the model using the custom container image you pushed...")

# We use the standard Model.upload() function.
model = aiplatform.Model.upload(
    display_name=MODEL_DISPLAY_NAME,
    
    serving_container_image_uri=CUSTOM_IMAGE_URI,
    
    artifact_uri=GCS_ARTIFACT_URI,
    serving_container_environment_variables={
        "MODEL_GCS_URI": GCS_ARTIFACT_URI 
    },
)

print("\n✅ Model registered successfully to the Vertex AI Model Registry!")
print(f"  - Model Resource Name: {model.resource_name}")
print(f"  - You can now deploy this 'model' object to an endpoint.")

# ==============================================================================
# Step 6: Deploy the registered model to a live Endpoint
# ==============================================================================
# This uses the 'model' object that was returned by the Model.upload() command.

print("✅ Starting deployment of the model to a new CPU-only Vertex AI Endpoint.")
print("⏳ This can take 15-20 minutes. Please be patient...")

# --- Deployment Configuration ---
# Specify a standard machine type for inference
MACHINE_TYPE = "n2d-standard-4"

# The .deploy() method creates the live prediction server.
endpoint = model.deploy(
    deployed_model_display_name=MODEL_DISPLAY_NAME,
    machine_type=MACHINE_TYPE,
    # We do NOT include accelerator_type, so it will be a CPU-only endpoint.
    sync=True # This makes the script wait until the deployment is complete.
)

print(f"\n🎉🎉🎉 DEPLOYMENT COMPLETE! 🎉🎉🎉")
print(f"Model is now live and ready for predictions at Endpoint:")
print(f"  - Endpoint Resource Name: {endpoint.resource_name}")

