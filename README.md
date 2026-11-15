# Dry Eye Disease Screening Model

A small ML service that estimates the probability that a person has dry eye disease from three
self-reported eye symptoms. Midterm project for the DataTalks.Club Machine Learning Zoomcamp 2026.

> **Disclaimer:** this is a course project, not a medical device. The model is a weak screening
> signal, not a diagnosis.

## Problem description

Dry eye disease (DED) is a common eye condition. A questionnaire-style tool could tell people
whether they should book an eye exam. Given yes/no answers about eye strain, redness and
itchiness, the service returns the probability of dry eye disease (binary classification).

- **Who uses it:** a clinic or a health app showing a quick pre-screening result.
- **Target:** `Dry Eye Disease` (Y/N).
- **Metric:** ROC AUC. The classes are imbalanced (65% positive).

## Dataset

Kaggle: [Dry Eye Disease](https://www.kaggle.com/datasets/dakshnagra/dry-eye-disease)
(20,000 rows, 26 columns, ages 18 to 45). Check the license on the Kaggle page.
The CSV is **not committed** to this repository.

Download it, either:

1. **Manually (recommended):** download the zip from the Kaggle page and unzip it so that the file
   is at `data/Dry_Eye_Dataset.csv`.
2. **With a script** (needs a Kaggle API token in `~/.kaggle/access_token`):
   `uv run python download_data.py`

## Key findings (see `notebook.ipynb`)

- No missing values, no duplicates, no impossible values. 65.2% of people have the disease.
- Almost all columns are uniformly spread between round limits (for example every Y/N column is
  50/50), which suggests the data may be synthetic.
- **Only the three symptoms matter.** Their correlation with the target is 0.10 to 0.11; every
  other feature is at 0.02 or below. Mutual information tells the same story.
- Disease rate: about **30%** with no symptom, **67 to 72%** with at least one symptom.
- Models compared with 5-fold CV ROC AUC (best setting of each, after tuning):

| Features | Logistic reg. | Decision tree | Random forest | XGBoost |
| --- | ---: | ---: | ---: | ---: |
| All features | 0.601 | 0.596 | 0.601 | 0.600 |
| Symptoms only | 0.598 | 0.592 | 0.593 | 0.599 |
| Lifestyle only (no symptoms) | 0.500 | 0.507 | 0.503 | 0.503 |

- Lifestyle features (sleep, stress, screen time, ...).
- All models tie within noise, so the simplest one was chosen:
  **logistic regression on one feature, `any_symptom`**

## Project structure

```
.
├── notebook.ipynb         # cleaning, EDA, feature importance, model comparison and tuning
├── train.py               # trains the final model and writes model.bin
├── predict.py             # FastAPI service (/health, /predict)
├── download_data.py       # optional Kaggle download helper
├── model.bin              # trained model
├── Dockerfile
├── k8s/                   # Kubernetes deployment and service
├── docs/deployment.png    # screenshot of the deployed service
├── pyproject.toml, uv.lock, .python-version
```

## Run it

### Install dependencies ([uv](https://docs.astral.sh/uv/))
```bash
uv sync
```

### Notebook
```bash
uv run jupyter notebook     # open notebook.ipynb (kernel: the project's .venv)
```

### Train
```bash
uv run python train.py      # prints the test ROC AUC and writes model.bin
```

### Run the service locally
```bash
uv run uvicorn predict:app --host 0.0.0.0 --port 9696
```
Interactive docs: http://localhost:9696/docs

```bash
curl -s localhost:9696/health
curl -s -X POST localhost:9696/predict -H 'Content-Type: application/json' \
  -d '{"discomfort_eye_strain":1,"redness_in_eye":1,"itchiness_irritation_in_eye":0}'
# {"dry_eye_probability":0.70...,"dry_eye":true}
```
Inputs are 0/1 integers; unknown fields are rejected.

### Docker
```bash
docker build -t dry-eye-prediction .
docker run -it --rm -p 9696:9696 dry-eye-prediction
```

### Kubernetes (local kind cluster)
```bash
kind create cluster --name dry-eye          # or use an existing kind cluster
kind load docker-image dry-eye-prediction:latest --name dry-eye
kubectl apply -f k8s/
kubectl rollout status deployment/dry-eye
kubectl port-forward service/dry-eye 9696:80
```
Then use the same `curl` commands as above. Screenshot of the working deployment:
[docs/deployment.png](docs/deployment.png).
