# MPDD - ALIGNN Calculator

This tool is a modified version of the **NIST-JARVIS** [**`ALIGNN`**](https://github.com/usnistgov/alignn) optimized in terms of model performance and to some extent reliability, for large-scale deployments over the [**`MPDD`**](https://phaseslab.org/mpdd) infrastructure by Phases Research Lab.

Critical modifications that were made here:
- A set of models of interest has been selected and defined in [**`config.yaml`**](alignn/config.yaml) for consistency, readability, and easy tracking. These are the models which will be populating MPDD.
- The process of model fetching was far too slow using `pretrained.get_figshare_model()`; thus, we reimplemented it similar to [`pySIPFENN`](https://pysipfenn.org) by multi-threading connection to Figshare via `pysmartdl2` we maintain, and parallelize the process on per-model basis. **Model download is now 7 times faster**, fetching all 7 default models in 6.1 vs 41.4 seconds.
- Optimized what is included in the built package. Now, its **size is reduced 33.5 times**, from 21.7MB to 0.65MB.