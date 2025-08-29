from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "transformers==4.55.2",
    "datasets==4.0.0",
    "google-cloud-storage==2.19.0",
    "Pillow==11.3.0",
    "accelerate>=0.26.0",
]

setup(
    name="trainer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    description="DINOv2 self-supervised fine-tuning.",
)
