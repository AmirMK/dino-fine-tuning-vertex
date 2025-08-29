# Fine-tuning DINO Image Embedding Model

This repository demonstrates the end-to-end process of fine-tuning a DINO image embedding model using unlabeled images. It provides the necessary steps to perform the fine-tuning and then deploy the resulting fine-tuned model as a managed endpoint on Google Cloud's Vertex AI. The included `.py` files are sample implementations designed to illustrate this workflow and are not intended for production environments.

## Fine-tunnng Process

### 1. Clone the Repository

First, clone this repository to your GCP project:

```bash
git clone https://github.com/AmirMK/dino-fine-tuning-vertex.git
cd dino-fine-tuning-vertex
```


### 2. Build the Trainer Wheel

Navigate to the root of the repository (where your `setup.py` or equivalent build files are located) and build the Python wheel for your trainer application.

```bash
# Build the wheel (All these commands need to be run from the repo-dir where all the files are in)
python -m pip install --upgrade build wheel setuptools
python -m build
ls -lh dist/
# Expected output similar to: trainer-0.1.0-py3-none-any.whl
```
### 3. Define Variables

Set the following environment variables. Replace the placeholder values with your actual project details and Google Cloud Storage (GCS) paths.

```bash
# Define variables
export PROJECT_ID=YOUR_PROJECT_ID
export REGION=us-central1
export WHEEL_BUCKET=gs://your-wheel-bucket/ # e.g., gs://my-dino-finetune-bucket/wheels/

# This has to be compatible with the wheel generated in previous step
export TRAINER_WHL=trainer-0.1.0-py3-none-any.whl 
export WHEEL_URI=${WHEEL_BUCKET}/${TRAINER_WHL}

# You only need to define these if you want to pass them as env variables to the custom job.
export BASE_MODEL_URI=gs://your-base-model-uri/
export TRAIN_IMAGES_URI=gs://your-training-images-uri/
export OUTPUT_URI=gs://your-output-uri/

# Example prebuilt PyTorch CPU image for 2.4 (adjust region if needed).
# Here is the list of pre-built containers: https://cloud.google.com/vertex-ai/docs/training/create-custom-container#pre-built
export EXECUTOR_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-cpu.2-4:latest"

# Optional dedicated service account
export SERVICE_ACCOUNT="vertex-training@${PROJECT_ID}.iam.gserviceaccount.com"
```
### 4. Upload the Wheel to GCS

Upload the generated trainer wheel to your specified GCS bucket.

```bash
gsutil cp dist/${TRAINER_WHL} ${WHEEL_BUCKET}/
gsutil ls ${WHEEL_BUCKET}/
```

### 5. Submit the Job to Vertex AI

Use the following `gcloud` command to submit your custom training job to Vertex AI. This setup is designed for fast iteration and minimal ceremony.

```bash
# Submit the job
# Use when you want fast iteration and minimal ceremony.
gcloud ai custom-jobs create \
    --project=${PROJECT_ID} \
    --region=${REGION} \
    --display-name=dino-finetune-v3 \
    --service-account=${SERVICE_ACCOUNT} \
    --worker-pool-spec=replica-count=1,machine-type=e2-standard-4,executor-image-uri=${EXECUTOR_IMAGE},python-module=trainer.task,package-uris=${WHEEL_URI}
```

## Deployment Process

### 1. Create an Artifact Registry Repository

Before building and pushing your custom inference container image, you need a Docker repository in Google Cloud's Artifact Registry. This command creates a new repository to store your Docker images.

```bash
gcloud artifacts repositories create {REPOSITORY} \
    --repository-format=docker \
    --location=us-central1 \
    --description="Docker repository for DINO model serving images
```

*   **`gcloud artifacts repositories create {REPOSITORY}`**: This command initiates the creation of an Artifact Registry repository.
*   **`{REPOSITORY}`**: This is a placeholder for the name you choose for your Docker repository. It should be a unique, lowercase name (e.g., `dino-model-serving`). This name will be referenced in subsequent steps.
*   **`--repository-format=docker`**: Specifies that this repository will store Docker container images.
*   **`--location=us-central1`**: Sets the Google Cloud region where your repository will be located. Ensure this matches your project's region.
*   **`--description="..."`**: An optional, descriptive text for your repository.

### 2. Configure `deployment/deployment.py`

Navigate to the `deployment/deployment.py` file. You need to modify the following variables within this script to match your project and model details.


*   **`PROJECT_ID`**: Your Google Cloud Project ID.
*   **`REGION`**: Your Google Cloud Region (must match the Artifact Registry location you chose in Step 1).
*   **`REPOSITORY`**: The name of the Artifact Registry repository created in Step 1 (e.g., `dino-docker-repo`).

*   **`MODEL_DISPLAY_NAME`**: A user-friendly display name for your model in Vertex AI.
*   **`IMAGE_NAME`**: A chosen name for your custom Docker image.
*   **`CUSTOM_IMAGE_URI`**: The full URI where your custom image will be pushed/pulled from in Artifact Registry. This is typically constructed using the `REGION`, `PROJECT_ID`, `REPOSITORY`, and `IMAGE_NAME`.
*   **`GCS_ARTIFACT_URI`**: The Google Cloud Storage path to the fine-tuned model weights. This should be the output location from your training job, as your custom container will load these weights for inference.
*   **`LOCAL_MODEL_SOURCE_DIR`**: The local path to the source code directory required by the deployment script. This directory should contain your model's `predictor.py` file (which loads the fine-tuned model and handles inference) and your `requirements.txt` file.

**Note on Machine Type:** The `deployment/deployment.py` script is currently configured for CPU inference. You can adjust the machine type and potentially enable GPU acceleration by modifying the `machine_type` and `accelerator_type`/`accelerator_count` parameters around line 99 in the `deployment/deployment.py` file if your inference requires different compute resources.

### 3. Run the Deployment Script

Execute the `deployment/deployment.py` script. This script will handle building your custom container, pushing it to Artifact Registry, uploading your model to Vertex AI Model Registry, and deploying it to a Vertex AI Endpoint.

**Note:** The deployment process within the `deployment.py` file is currently configured for CPU-based inference without any accelerators. To run inference with accelerators (e.g., GPUs), you can find relevant configuration options and examples in the Vertex AI documentation on [Endpoints](https://cloud.google.com/python/docs/reference/aiplatform/latest?utm_source=chatgpt.com#endpoints). For the full syntax and arguments available for the `deploy` method, refer to the [Vertex AI Python SDK `Model.deploy` documentation](https://cloud.google.com/python/docs/reference/aiplatform/latest/google.cloud.aiplatform.Model#google_cloud_aiplatform_Model_deploy).

```bash
python deployment/deployment.py
```

This process may take approximately 25-30 minutes to complete. Upon successful completion, the script will output the unique endpoint ID or name. This endpoint number is crucial as it's used to send inference requests to your deployed model.

A sample notebook, `deployment/call_endpoint.ipynb`, is provided to demonstrate how to interact with your newly created Vertex AI endpoint for inference.
