# MPDD - ALIGNN Calculator

This tool is a modified version of the **NIST-JARVIS** [**`ALIGNN`**](https://github.com/usnistgov/alignn) optimized in terms of model performance and to some extent reliability, for large-scale deployments over the [**`MPDD`**](https://phaseslab.org/mpdd) infrastructure by Phases Research Lab.

Critical modifications that were made here:
- A set of models of interest has been selected and defined in [**`config.yaml`**](config.yaml) for consistency, readability, and easy tracking. These are the models which will be populating MPDD.
- The process of model fetching was far too slow using `pretrained.get_figshare_model()`; thus, we reimplemented it similar to [`pySIPFENN`](https://pysipfenn.org) by multi-threading connection to Figshare via `pysmartdl2` we maintain, and parallelizing the process on per model-basis. **It now downloads all 7 default models in 6.1 vs 41.4 seconds.**