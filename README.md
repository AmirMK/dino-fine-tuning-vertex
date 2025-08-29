# Fine-tuning DINO Image Embedding Model

This repository provides a sample implementation for fine-tuning the DINO image embedding model using unlabeled images. It's designed to demonstrate the fine-tuning process and includes all the necessary steps to run this job as a custom training job on Google Cloud's Vertex AI. Please note, this code is for demonstration purposes and not intended for production environments.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-repository-name>
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
