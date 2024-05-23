# Rethinking Table Retrieval from Data Lakes

*Table retrieval from data lakes has recently become important for many downstream tasks, including data discovery and
table question answering. Existing table retrieval approaches estimate each table's relevance to a particular
information need and return a ranking of the most relevant tables. This approach is not ideal since (1) the returned
tables often include irrelevant data and (2) the required information may be scattered across multiple tables. To
address these issues, we propose the idea of fine-grained structured table retrieval and present our vision of R2D2, a
system which slices tables into small tiles that are later composed into a structured result that is tailored to the
user-provided information need. An initial evaluation of our approach demonstrates how our idea can improve table
retrieval and relevant downstream tasks such as table question answering.*

Please check out our [paper](https://doi.org/10.1145/3663742.3663972) and cite our work:

```bibtex
@inproceedings{DBLP:conf/aidm/BodensohnB24,
    author       = {Jan{-}Micha Bodensohn and
                  Carsten Binnig},
    title        = {Rethinking Table Retrieval from Data Lakes},
    booktitle    = {Proceedings of the Seventh International Workshop on Exploiting Artificial
                  Intelligence Techniques for Data Management, aiDM 2024, Santiago,
                  Chile, 14 June 2024},
    pages        = {2:1--2:5},
    publisher    = {{ACM}},
    year         = {2024},
    url          = {https://doi.org/10.1145/3663742.3663972},
    doi          = {10.1145/3663742.3663972}
}
```

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
