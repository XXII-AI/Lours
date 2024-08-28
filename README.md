# README LibIA

A lib to help R&D team with its experiments.

## DOC API

Sphinx docs is available here:

- [stable](UPDATE-ME)
- [latest](UPDATE-ME)

## Installation

### From Pypi repository

### With poetry

```bash
poetry add xxii-libia
```

You can also install the pre-release by modifying the last line

```bash
poetry add xxii-libia --alow-prereleases
```

### With pip

```bash
pip install xxii-libia
```

You can also install the pre-release by adding the `--pre` option

```bash
pip install xxii-libia --pre
```

### From source

- `pip`

```bash
pip install -e . # Only for pip > 21.3
```

- `poetry >= 1.2`

Assuming both libia and your project are in the same folder, adapt the relative path of the repo if needed

```bash
poetry add --editable ../libia/
poetry add --editable NEW_GIT_ADDRESS/libia.git
```

## The dataset object

See <https://UPDATE-ME>

## Usage

```python
from libia.dataset import from_caipy, from_coco
dataset1 = from_caipy("path/to/caipy")
print(dataset1)
dataset2 = from_coco("path/to/coco", images_root="/path/to/coco_images")
dataset2 = dataset2.remap_from_preset("coco", "pascalvoc")
print(dataset2)
```

## Tutorials

See some notebooks in folder `docs/notebooks`

Alternatively, See <UPDATE-ME/stable/tutorials>
