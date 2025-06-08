# DVC Demo

This repository provides a small demonstration of using [DVC](https://dvc.org/) to track data files alongside source code. The project contains a single dataset `train.csv` stored under `data/raw/` and tracked with DVC via `train.csv.dvc`. The actual file lives in an S3 bucket so the repository remains lightweight.

## Data

The dataset is stored in the S3 bucket configured in `.dvc/config` (`s3://mldvc`, region `cn-north-1`). To retrieve the data locally you need DVC installed and access to that bucket (AWS profile `cn-lz-prod` by default).

```bash
# Install dependencies
poetry install

# Pull the dataset from the remote storage
poetry run dvc pull data/raw/train.csv.dvc
```

## Running the example

After pulling the data you can launch Jupyter to run the included notebook:

```bash
poetry run jupyter notebook notebooks/training.ipynb
```

The notebook loads `data/raw/train.csv`, performs a simple preprocessing step and trains a small model using `scikit-learn`. Feel free to modify the notebook or use it as a starting point for your own experiments.
