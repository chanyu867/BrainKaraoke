# BrainKaraoke tests

This folder contains pytest-based tests for the 4 main modules:
- `models.py`
- `pipeline.py`
- `dataset.py`
- `main.py`

## Setup (development)
From the project root (the folder that contains `BrainKaraoke/`):

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

## Run all tests in this folder
```bash
pytest BrainKaraoke/src/tests -v
```

## Run a single test file
```bash
pytest BrainKaraoke/src/tests/test_models.py -v
pytest BrainKaraoke/src/tests/test_pipeline.py -v
pytest BrainKaraoke/src/tests/test_dataset.py -v
pytest BrainKaraoke/src/tests/test_main.py -v
```