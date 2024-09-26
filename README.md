<!-- markdownlint-disable MD041 -->

## Lours, the pandas companion

| | | |
|-|-|-|
| 📚 [Docs](https://lours.readthedocs.io)| 📊 [Slides](https://docs.google.com/presentation/d/1crfwQmDnzkMmekznpZZiX0E0XPEhCAOkFPWa1NmflXU/pub) | 📝 Blogpost (coming soon) |

[![codecov](https://img.shields.io/codecov/c/gh/XXII-AI/lours?logo=codecov&color=cyan)](https://codecov.io/gh/XXII-AI/Lours) ![GitHub License](https://img.shields.io/github/license/XXII-AI/Lours?color=violet&logo=license) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/lours?logo=python)
 ![PyPI - Version](https://img.shields.io/pypi/v/lours?logo=pypi) ![PyPI - Downloads](https://img.shields.io/pypi/dm/lours?color=yellow) ![Read the Docs](https://img.shields.io/readthedocs/lours?logo=readthedocs&color=orange)

![logo lours](https://github.com/XXII-AI/Lours/raw/main/docs/_static/logo.png)

## DOC API

Sphinx docs is available here:

- [stable](https://lours.readthedocs.io/stable/)
- [latest](https://lours.readthedocs.io/latest/)

## Installation

```bash
pip install lours
```

You can also install the pre-release by adding the `--pre` option

```bash
pip install lours --pre
```

### From source

- `pip`

```bash
pip install -e . # Only for pip > 21.3
```

- `poetry >= 1.2`

Assuming both lours and your project are in the same folder, adapt the relative path of the repo if needed

```bash
poetry add --editable ../lours/
poetry add --editable https://github.com/XXII-AI/lours.git
```

## The dataset object

See <https://UPDATE-ME>

## Usage

```python
from lours.dataset import from_caipy, from_coco
dataset1 = from_caipy("path/to/caipy")
print(dataset1)
dataset2 = from_coco("path/to/coco", images_root="/path/to/coco_images")
dataset2 = dataset2.remap_from_preset("coco", "pascalvoc")
print(dataset2)
```

## Tutorials

See some notebooks in folder `docs/notebooks`

Alternatively, See <https://lours.readthedocs.io/stable/tutorials.html>
