# MPDD - ALIGNN Calculator

This tool is a modified version of the **NIST-JARVIS** [**`ALIGNN`**](https://github.com/usnistgov/alignn) optimized in terms of model performance and to some extent reliability, for large-scale deployments over the [**`MPDD`**](https://phaseslab.org/mpdd) infrastructure by Phases Research Lab.

## Critical Changes

Key modifications that were made here:
- A set of models of interest has been selected and defined in [**`config.yaml`**](alignn/config.yaml) for consistency, readability, and easy tracking. These are the models which will be populating MPDD.
- **Dependency optimizations for running models**, skipping by default installation of several packages needed only for training and auxiliary tasks. Full
set can still be installed by `pip install "mpdd-alignn[full]"`.
- The process of model fetching was far too slow using `pretrained.get_figshare_model()`; thus, we reimplemented it similar to [`pySIPFENN`](https://pysipfenn.org) by multi-threading connection to Figshare via `pysmartdl2` we maintain, and parallelize the process on per-model basis. **Model download is now 7 times faster**, fetching all 7 default models in 6.1 vs 41.4 seconds.
- Optimized what is included in the built package. Now, its **package size is reduced 33.5 times**, from 21.7MB to 0.65MB.
- Streamlined operation, where we can get results for a directory of POSCARS for all default models in just 3 quick lines
    ```python
    from alignn import pretrained
    pretrained.download_default_models()
    result = pretrained.run_models_from_directory('example.SigmaPhase', mode='serial')
    ```

    Which give us neat:

    ```
    [{
        'ALIGNN-JARVIS Bulk Modulus [GPa]': 98.06883239746094,
        'ALIGNN-JARVIS Exfoliation Energy [meV/atom]': 101.71208190917969,
        'ALIGNN-JARVIS Formation Energy [eV/atom]': -1.1146986484527588,
        'ALIGNN-JARVIS MBJ Bandgap [eV]': 0.5845542550086975,
        'ALIGNN-JARVIS Shear Modulus [GPa]': 39.18968963623047,
        'ALIGNN-MP Formation Energy [eV/atom]': -1.4002774953842163,
        'ALIGNN-MP PBE Bandgap [eV]': 1.074204921722412,
        'name': '9-Pb8O12.POSCAR'
    },
    {
        'ALIGNN-JARVIS Bulk Modulus [GPa]': 194.2947540283203,
        'ALIGNN-JARVIS Exfoliation Energy [meV/atom]': 362.1310729980469,
        'ALIGNN-JARVIS Formation Energy [eV/atom]': 0.010236039757728577,
        'ALIGNN-JARVIS MBJ Bandgap [eV]': 0.0064897798001766205,
        'ALIGNN-JARVIS Shear Modulus [GPa]': 85.74588775634766,
        'ALIGNN-MP Formation Energy [eV/atom]': -0.018119990825653076,
        'ALIGNN-MP PBE Bandgap [eV]': -0.00551827996969223,
        'name': '19-Fe4Ni26.POSCAR'
    },
    {
        'ALIGNN-JARVIS Bulk Modulus [GPa]': 185.35687255859375,
        'ALIGNN-JARVIS Exfoliation Energy [meV/atom]': 379.46417236328125,
        'ALIGNN-JARVIS Formation Energy [eV/atom]': 0.10529126971960068,
    ...
    ```

## ALIGNN Compatibility and Install

In general, we tried to retain full compatibility with the original `ALIGNN`, so this should be a drop-in replacement. You have to simply:

    pip install mpdd-alignn

or (as recommended) clone this repository and

    pip install -e .

## Contributions

Please direct all contributions to [the ALIGNN repository](https://github.com/usnistgov/alignn). We will be synching our fork with them every once in a while and can do it quickly upon reasonable request. 

The only contributions we will accept here are:
- Expanding the list of default models.
- Performance improvements to our section of the code.