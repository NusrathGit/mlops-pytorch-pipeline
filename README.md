# mlops-pytorch-pipeline

Deploying a PyTorch image classification model (ResNet-18 / CNN on CIFAR-10) through the full MLOps lifecycle: local development -> containerized training (Docker) -> orchestrated training and serving (Kubernetes).

## Dataset

CIFAR-10 (10 classes, 32x32 RGB images), loaded via torchvision.datasets.CIFAR10 with download=True.

## Setup

1. Local: pip install -r requirements/train.txt && python src/train.py
2. Docker: docker build -f docker/Dockerfile.train -t mlops-train:v1 .
3. Kubernetes: kubectl apply -f k8s/

# mlops-pytorch-pipeline

Deploying a PyTorch image classification model (ResNet-18 / CNN on CIFAR-10)
through the full MLOps lifecycle: local development -> containerized
training (Docker) -> orchestrated training and serving (Kubernetes).

## Architecture

```
                    +----------------------+
                    |   configs/           |
                    | training_config.yaml |
                    +----------+-----------+
                               | mounted via ConfigMap
                               v
 +----------------+   +-----------------+    +------------------+
 | docker/        |   |  Kubernetes Job  |    | PersistentVolume |
 | Dockerfile.train|-->|  (training)      |--->| /app/checkpoints |
 +----------------+   +-----------------+    +--------+---------+
                                                        | read-only mount
                                                        v
 +----------------+   +------------------+    +------------------+
 | docker/        |   |  Kubernetes       |    |  Service          |
 | Dockerfile.serve|-->|  Deployment (x2)  |<-->|  (ClusterIP)       |
 +----------------+   |  Flask /predict    |    +------------------+
                       |  Flask /health     |
                       +------------------+
```

Training runs once as a Kubernetes **Job**, writing a checkpoint to a shared
PersistentVolumeClaim (`checkpoints-pvc`). The serving **Deployment** mounts
that same PVC read-only and loads the checkpoint at startup to serve
predictions. An optional **HorizontalPodAutoscaler** scales the serving
Deployment between 2 and 5 replicas based on CPU utilization.

## Project structure

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml       # lint + unit tests on every push/PR
├── src/
│   ├── train.py                   # training loop (epochs, early stopping, JSON logs)
│   ├── model.py                   # get_model(): ResNet-18 or SimpleCNN
│   ├── dataset.py                 # CIFAR-10 dataloaders + augmentation
│   └── serve.py                   # Flask API: GET /health, POST /predict
├── configs/training_config.yaml
├── docker/
│   ├── Dockerfile.train           # multi-stage build for the training image
│   └── Dockerfile.serve           # non-root, HEALTHCHECK, port 8080
├── k8s/
│   ├── namespace.yaml
│   ├── training-job.yaml          # Job + data-pvc + checkpoints-pvc
│   ├── serving-deployment.yaml    # 2 replicas, probes, rolling update
│   ├── serving-service.yaml       # ClusterIP, port 80 -> 8080
│   ├── configmap.yaml
│   └── hpa.yaml                   # bonus: CPU-based autoscaling
├── requirements/
│   ├── train.txt
│   └── serve.txt
├── tests/test_model.py
├── data/            # dataset cache (gitignored, created automatically)
└── checkpoints/     # trained model output (gitignored, created automatically)
```

## Dataset

CIFAR-10 (10 classes, 32x32 RGB images), loaded via `torchvision.datasets.CIFAR10`
with `download=True` - no manual download needed.

---

## 1. Run locally (no Docker)

Requires Python 3.10+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements/train.txt
python src/train.py
```

Trains for up to 10 epochs (per `configs/training_config.yaml`), with early
stopping if validation loss doesn't improve for 3 epochs. Writes
`checkpoints/classifier_v1.pt` and prints JSON-lines metrics per epoch.

Run the API locally:

```bash
pip install -r requirements/serve.txt
set MODEL_PATH=checkpoints\classifier_v1.pt   # Windows
python src/serve.py
```

Test it:

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict -F "image=@some_image.png"
```

## 2. Build and run with Docker

```bash
# Training image
docker build -f docker/Dockerfile.train -t mlops-train:v1 .

docker run --rm ^
  -v %cd%/data:/app/data ^
  -v %cd%/checkpoints:/app/checkpoints ^
  mlops-train:v1

# Serving image
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .

docker run --rm -p 8080:8080 ^
  -v %cd%/checkpoints:/app/checkpoints ^
  mlops-serve:v1
```

Check the built-in Docker healthcheck: `docker ps` (STATUS column shows
"healthy" once `/health` responds).

## 3. Deploy to Kubernetes

Using `kind` (Kubernetes-in-Docker):

```bash
kind create cluster --name mlops-pipeline
kind load docker-image mlops-train:v1 --name mlops-pipeline
kind load docker-image mlops-serve:v1 --name mlops-pipeline
```

Apply manifests (namespace and ConfigMap first, since the Job references them):

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml
```

Watch training complete:

```bash
kubectl get pods -n ml-training -w
kubectl logs -n ml-training job/pytorch-training -f
```

Once the Job shows `Completed`, deploy serving:

```bash
kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml
```

Check status:

```bash
kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training
```

Access the API:

```bash
kubectl port-forward svc/model-serving 8080:80 -n ml-training

curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "image=@some_image.png"
```

### Cleanup

```bash
kubectl delete -f k8s/
kind delete cluster --name mlops-pipeline
```

## Development workflow

This repo follows a `main` / `develop` / `feature/*` branching model:

- `main` - always deployable
- `develop` - integration branch, created from `main`
- `feature/<name>` - one branch per unit of work, merged into `develop` via PR

## Secrets management

`k8s/secret.yaml` in this repo contains a **placeholder** value
(`API_KEY: change-me-in-production`) - it is committed only to demonstrate
the Secret manifest structure required by the assignment. No real
credential is present.

In a real deployment, secrets should never be committed to Git, even as
placeholders in production overlays. Instead:

- Create the Secret directly against the cluster, without a file:
```bash
  kubectl create secret generic serving-secret \
    --from-literal=API_KEY=<real-value> \
    -n ml-training
```
- Or use a tool designed for GitOps-safe secrets, such as
  [Sealed Secrets](https://github.com/bitnami-labs/sealed-secrets) or
  [External Secrets Operator](https://external-secrets.io/), which let the
  *encrypted* form live in Git while the plaintext only ever exists inside
  the cluster.
- Keep `k8s/secret.yaml` (or an equivalent local-only file) out of version
  control via `.gitignore` once it holds a real value.