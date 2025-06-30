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
    [
        {
            'name': '22-Fe10Ni20.POSCAR',
            'ALIGNN-Alexandria Bandgap [eV]': 0.001391,
            'ALIGNN-Alexandria Formation Energy [eV/atom]': 0.095294,
            'ALIGNN-Alexandria Volume Per Atom [A^3]': 11.140231,
            'ALIGNN-JARVIS Bulk Modulus [GPa]': 183.945847,
            'ALIGNN-JARVIS Exfoliation Energy [meV/atom]': 350.855591,
            'ALIGNN-JARVIS Formation Energy [eV/atom]': 0.027578,
            'ALIGNN-JARVIS MBJ Bandgap [eV]': 0.017667,
            'ALIGNN-JARVIS Shear Modulus [GPa]': 74.540077,
            'ALIGNN-MP Formation Energy [eV/atom]': -0.045874,
            'ALIGNN-MP PBE Bandgap [eV]': 0.01164,
        },
        {
            'name': '2-Fe8Ni22.POSCAR',
            'ALIGNN-Alexandria Bandgap [eV]': 0.001679,
            'ALIGNN-Alexandria Formation Energy [eV/atom]': 0.25086,
            'ALIGNN-Alexandria Volume Per Atom [A^3]': 10.656669,
            'ALIGNN-JARVIS Bulk Modulus [GPa]': 187.983017,
            'ALIGNN-JARVIS Exfoliation Energy [meV/atom]': 352.69455,
            'ALIGNN-JARVIS Formation Energy [eV/atom]': 0.025119,
            'ALIGNN-JARVIS MBJ Bandgap [eV]': 0.010531,
            'ALIGNN-JARVIS Shear Modulus [GPa]': 80.09848,
            'ALIGNN-MP Formation Energy [eV/atom]': -0.042081,
            'ALIGNN-MP PBE Bandgap [eV]': 0.019553,
        },
        {
            'name': '11-Fe10Ni20.POSCAR',
            'ALIGNN-Alexandria Bandgap [eV]': 0.001165,
            'ALIGNN-Alexandria Formation Energy [eV/atom]': 0.217117,
            'ALIGNN-Alexandria Volume Per Atom [A^3]': 10.583747,
        ...
    ```

## ALIGNN Compatibility and Install

In general, we tried to retain full compatibility with the original `ALIGNN`, so this should be a drop-in replacement. You have to simply:

    pip install mpdd-alignn

or (as recommended) clone this repository and

    pip install -e .

## `DGL` Issues

On some platforms, `pip` may struggle to get the `dgl` from PyPI that is compatible with the `torch` version you need for other dependencies and so on. In such case, we recommend to install lightweight (CPU) version of `torch` and matching `dgl` that should work well together by:
```shell
pip install torch==2.3.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```
and
```shell
pip install  dgl -f https://data.dgl.ai/wheels/torch-2.3/repo.html
```
adjusting the torch version in both to your needs.

## Contributions

Please direct all contributions to [the ALIGNN repository](https://github.com/usnistgov/alignn). We will be synching our fork with them every once in a while and can do it quickly upon reasonable request. 

The only contributions we will accept here are:
- Expanding the list of default models.
- Performance improvements to our section of the code.