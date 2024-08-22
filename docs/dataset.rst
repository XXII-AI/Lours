Dataset
=======

.. currentmodule:: libia.dataset

The Dataset object
~~~~~~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :caption: The Dataset Object

    Dataset


Input Output
~~~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :caption: I/O

    Dataset.to_parquet
    Dataset.to_caipy
    Dataset.to_darknet
    Dataset.to_coco
    Dataset.to_fiftyone
    from_parquet
    from_caipy
    from_caipy_generic
    from_coco
    from_folder
    from_mot
    from_crowd_human
    from_darknet
    from_darknet_yolov5
    from_darknet_generic
    from_darknet_json
    from_pascalVOC_generic
    from_pascalVOC_detection

Remapping
~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :recursive:
    :caption: Remapping

    Dataset.remap_classes
    Dataset.remap_from_preset
    Dataset.remap_from_csv
    Dataset.remap_from_dataframe
    Dataset.remap_from_other
    Dataset.remove_classes

Merging
~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :recursive:
    :caption: Merging

    Dataset.merge
    Dataset.__add__

Splitting
~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :recursive:
    :caption: Splitting

    Dataset.split
    Dataset.simple_split


Indexing
~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :caption: Indexing

    Dataset.loc
    Dataset.iloc
    Dataset.loc_annot
    Dataset.iloc_annot
    Dataset.filter_images
    Dataset.filter_annotations


Re-Indexing
~~~~~~~~~~~

.. autosummary::
    :nosignatures:
    :toctree: generated
    :caption: Re-Indexing

    Dataset.match_index
    Dataset.reset_index
    Dataset.reset_index_from_mapping


Internal API
~~~~~~~~~~~~

.. autosummary::
   :nosignatures:
   :toctree: generated
   :recursive:
   :caption: Internal API

   io
   remap_presets
   split
   merge
   indexing
