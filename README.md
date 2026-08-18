# mlops-pytorch-pipeline

Deploying a PyTorch image classification model (ResNet-18 / CNN on CIFAR-10) through the full MLOps lifecycle: local development -> containerized training (Docker) -> orchestrated training and serving (Kubernetes).

## Dataset

CIFAR-10 (10 classes, 32x32 RGB images), loaded via torchvision.datasets.CIFAR10 with download=True.

## Setup

1. Local: pip install -r requirements/train.txt && python src/train.py
2. Docker: docker build -f docker/Dockerfile.train -t mlops-train:v1 .
3. Kubernetes: kubectl apply -f k8s/