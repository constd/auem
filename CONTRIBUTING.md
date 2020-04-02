## Installing `auem` for development

We use [Poetry](https://python-poetry.org/) to install the package. We strongly recommend that you use pipx to install poetry in it's own global environment,
thereby not interfering with the environment used to run this package.

### Install pipx in your base conda/virtualenv
```bash
(base)$ pip install pipx
```

### Install Poetry
This will install poetry in it's own environment:
```bash
(base)$ pipx install poetry
```

```bash
# install Poetry
pip install poetry
```

### Use poetry to install auem
```bash
First create a new environment with the conda base environment:
$ conda create -f base37.yaml
$ conda activate auem37dev
# Will install from the poetry lockfile
(auem37dev)$ poetry install
```

## Installing pre-commit hooks
We request that you use the provided pre-commit hooks if you plan to submit any code.

```bash
$ pipx install pre-commit
# From the root directory of this repo
$ pre-commit install
```

## Running Tests

```bash
# Run all tests
(auem37dev)$ poetry run pytest
```
