# R2D2: Rethinking Table Retrieval from Data Lakes

## Setup

Make sure you have [Conda](https://docs.conda.io/projects/miniconda/en/latest) installed.

Create a new Conda environment, activate it, and add the project to the Python path:

```bash
eval "$(conda shell.bash hook)"
conda env create -f environment-cuda.yml  # use environment.yml if you don't have CUDA
conda activate r2d2
export PYTHONPATH=${PYTHONPATH}:./
```

## Datasets

### Open-WikiTable

To prepare the Open-WikiTable dataset, run:

```bash
python scripts/lakes/download_openwikitable.py
python scripts/lakes/create_openwikitable_lakes.py
```

### Spider

To prepare the Spider dataset, **you first have to manually obtain the dataset**.

Download the ZIP archive
from [here](https://drive.google.com/u/0/uc?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download) and unpack its
contents into `data/spider/download/`.

Then run:

```bash
python scripts/lakes/create_spider_lakes.py  # runs for several hours
```

## Experiments

### Dataset Exploration

The exploration scripts generate histograms about the dataset statistics (e.g., tables per lake, cells per table...) and
can also plot the ground truths.

To explore the datasets, run:

```bash
bash scripts/explore/explore.sh
```

The results will be placed in `data/<dataset>/explore`.

### Slicing

To evaluate the slicing, run:

```bash
bash scripts/slicing/slicing.sh  # runs multiple hours
```

The results will be placed in `data/<dataset>/slicing`.

### Table Question Answering

To evaluate the table question answering, run:

```bash
bash scripts/tqa/tqa.sh  # runs multiple hours
```

The results will be placed in `data/<dataset>/tqa`.

To replicate our results you must obtain the `data/openai_cache` directory from us.
