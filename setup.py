from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    "transformers==4.55.2",
    "datasets>=2.19.0,<3",
    "google-cloud-storage==2.19.0",
    #"Pillow==11.3.0",
    "pillow-simd==9.5.0.post1",
    "accelerate>=0.26.0",
    "fsspec>=2024.6.0",
    "gcsfs>=2024.6.0",
    "cloudml-hypertune",
    "webdataset"
]

setup(
    name="trainer",
    version="0.0.2",
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
    description="DINOv2 self-supervised fine-tuning.",
)
