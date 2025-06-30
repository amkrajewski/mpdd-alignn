"""Module to download and load pre-trained ALIGNN models."""
# Standard imports
import requests
import os
import sys
import json
import time
from typing import List, Dict, Union
from importlib import resources

# Extra utility imports
import zipfile
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from ruamel.yaml import YAML
import tempfile
import pandas as pd
tqdm.pandas()
from pysmartdl2 import SmartDL
from dgl.data.utils import save_graphs, load_graphs

# ML imports
import torch
device = "cpu"
from torch.utils.data import DataLoader

# ALIGNN imports
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
from alignn.data import get_torch_dataset
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson

"""Default models for MPDD from ALIGNN. Stored in `config.yaml` in the root directory of the package."""
yaml = YAML(typ="safe")
with resources.files('alignn').joinpath("config.yaml").open("r") as f:
    config = yaml.load(f)
    default_models = config["defaultModels"]

def get_default_models() -> Dict[str, List[Dict[str, str]]]:
    """Return the default models for MPDD from ALIGNN."""
    return default_models

def download_model(model, verbose: bool = True) -> None:
    modelPath = str(resources.files('alignn').joinpath(model['model']))
    if not os.path.exists(modelPath):
        if verbose:
            print(f"Downloading {model['name']} from {model['url']} to {model['model']}", flush=True)
        obj = SmartDL(model["url"], modelPath, threads=4, progress_bar=False)
        obj.start()
        if obj.isSuccessful():
            if verbose:
                print(f"--> {model['model']} download complete!", flush=True)
        else:
            print(f"xxx {model['model']} download failed!", flush=True)
    else:
        if verbose:
            print(f"Model {model['model']} already exists at {model['model']}\n", flush=True)

def download_default_models(verbose: bool = True, parallel: bool = True) -> None:
    """Download the default models for MPDD from ALIGNN."""
    t0 = time.time()
    if parallel:
        process_map(
            download_model,
            default_models,
            max_workers=7,
        )
    else:
        for model in default_models:
            download_model(model, verbose=verbose)
    if verbose:
        print(f"All models downloaded in {time.time() - t0:.2f} seconds", flush=True)

def unzip_default_models() -> None:
    # Check if all are downloaded
    for model in default_models:
        if not os.path.exists(model['model']):
            raise FileNotFoundError(f"Model {model['name']} not found at {model['model']}!")
    # Unzip all models
    for model in default_models:
        with zipfile.ZipFile(model['model'], 'r') as zip_ref:
            zip_ref.extractall(model['model'].replace('.zip', ''))

def run_models_from_directory(
        directory: str, 
        mode: str = "serial",
        saveGraphs: bool = False):
    """Run all default models on all structures in a directory that are either in POSCAR or CIF format."""
    # Parse all structures into Atoms objects
    atoms_array = []
    outputs: List[Dict[str, Union[float, str]]] = []
    for file in os.listdir(directory):
        if file.endswith(("poscar", "POSCAR", "vasp", "VASP")):
            outputs.append({"name": file})
            atoms_array.append(Atoms.from_poscar(os.path.join(directory, file)))
        elif file.endswith(("cif", "CIF")):
            outputs.append({"name": file})
            atoms_array.append(Atoms.from_cif(os.path.join(directory, file)))
        else:
            print(f"Skipping file {file} as it is not a POSCAR or CIF file!", flush=True)
    # Convert all Atoms to Graphs
    print(f"Converting {len(atoms_array)} structures to graphs...", flush=True)
    if mode == "serial":
        graph_array = []
        for atoms in tqdm(atoms_array):
            graph_array.append(Graph.atom_dgl_multigraph(atoms))
    elif mode == "parallel":
        graph_array = process_map(
            Graph.atom_dgl_multigraph,
            atoms_array,
            chunksize=10,
            max_workers=4
        )
    else:
        raise ValueError(f"Mode {mode} not implemented!")
    
    if saveGraphs:
        print("Saving graphs to disk...", flush=True)
        input_files = [i["name"] for i in outputs]
        for g, name in zip(graph_array, input_files):
            save_graphs(f"graphs/{name}.graph.bin", list(g), formats='coo')
        print("Graphs saved!", flush=True)

    modelArray = []
    for model in default_models:
        modelPath = str(resources.files('alignn').joinpath(model['model']))
        zp = zipfile.ZipFile(modelPath, 'r')
        # Get the full path of checkpoint_300.pt or best_model.pt in the zip. Pick the first one found.
        # This is a workaround for the fact that some models (new ones) have a different naming convention
        # for the ALIGNN checkpoint files.
        modelCheckpoint = [
            i for i in zp.namelist() 
            if ("checkpoint_" in i and "pt" in i) or "best_model.pt" in i
            ]
        if len(modelCheckpoint) == 0:
            raise ValueError(f"No model identifier found in {modelPath} for model {model['name']}!", flush=True)
        if len(modelCheckpoint) > 1:
            print(f"Multiple model identifiers ({len(modelCheckpoint)}) found in {modelPath} for model {model['name']}: {modelCheckpoint}. Using the first one in the list.", flush=True)
            modelCheckpoint = modelCheckpoint[0]
        else:
            modelCheckpoint = modelCheckpoint[0]
        config = json.loads(zp.read([i for i in zp.namelist() if "config.json" in i][0]))
        data = zipfile.ZipFile(modelPath).read(modelCheckpoint)
        
        # Create model based on the model name in config
        model_type = config["model"]["name"]
        if model_type == "alignn":
            loadedModel = ALIGNN(ALIGNNConfig(**config["model"]))
        elif model_type == "alignn_atomwise":
            loadedModel = ALIGNNAtomWise(ALIGNNAtomWiseConfig(**config["model"]))
        else:
            raise ValueError(f"Unknown model type: {model_type} for model {model['name']}")

        _, filename = tempfile.mkstemp()
        with open(filename, "wb") as f:
            f.write(data)
        
        # Load checkpoint and handle different checkpoint formats
        checkpoint = torch.load(filename, map_location='cpu')
        if "model" in checkpoint:
            loadedModel.load_state_dict(checkpoint["model"])
        else:
            # Some checkpoints store the model state dict directly
            loadedModel.load_state_dict(checkpoint)
            
        loadedModel.to('cpu')
        loadedModel.eval()
        modelArray.append(loadedModel)
        
        print(f"Model {model['name']} loaded!", flush=True)
    
    print(f"Running {len(default_models)} models on {len(graph_array)} structures...", flush=True)
    # Run all models on all graphs
    for model, loaded_model in zip(default_models, modelArray):
        for g, out in zip(graph_array, outputs):
            model_output = loaded_model([g[0], g[1]])
            
            # Handle different output formats
            if isinstance(model_output, dict):
                # For atomwise models that return dict with 'out' key
                if 'out' in model_output:
                    out_data = model_output['out'].item()
                else:
                    # Take the first value if dict has other keys
                    out_data = list(model_output.values())[0].item()
            else:
                # For regular models that return tensor directly
                out_data = model_output.item()
                
            out[model['name']] = round(out_data, 6)
    
    print("All models runs complete!", flush=True)

    return outputs

# ******* Old method to download models *******
"""
Name of the model, figshare link, number of outputs,
extra config params (optional)
"""
# For ALIGNN-FF pretrained models see, alignn/ff/ff.py
all_models = {
    "jv_formation_energy_peratom_alignn": [
        "https://figshare.com/ndownloader/files/31458679",
        1,
    ],
    "jv_optb88vdw_total_energy_alignn": [
        "https://figshare.com/ndownloader/files/31459642",
        1,
    ],
    "jv_optb88vdw_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31459636",
        1,
    ],
    "jv_mbj_bandgap_alignn": [
        "https://figshare.com/ndownloader/files/31458694",
        1,
    ],
    "jv_spillage_alignn": [
        "https://figshare.com/ndownloader/files/31458736",
        1,
    ],
    "jv_slme_alignn": ["https://figshare.com/ndownloader/files/31458727", 1],
    "jv_bulk_modulus_kv_alignn": [
        "https://figshare.com/ndownloader/files/31458649",
        1,
    ],
    "jv_shear_modulus_gv_alignn": [
        "https://figshare.com/ndownloader/files/31458724",
        1,
    ],
    "jv_n-Seebeck_alignn": [
        "https://figshare.com/ndownloader/files/31458718",
        1,
    ],
    "jv_n-powerfact_alignn": [
        "https://figshare.com/ndownloader/files/31458712",
        1,
    ],
    "jv_magmom_oszicar_alignn": [
        "https://figshare.com/ndownloader/files/31458685",
        1,
    ],
    "jv_kpoint_length_unit_alignn": [
        "https://figshare.com/ndownloader/files/31458682",
        1,
    ],
    "jv_avg_elec_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458643",
        1,
    ],
    "jv_avg_hole_mass_alignn": [
        "https://figshare.com/ndownloader/files/31458646",
        1,
    ],
    "jv_epsx_alignn": ["https://figshare.com/ndownloader/files/31458667", 1],
    "jv_mepsx_alignn": ["https://figshare.com/ndownloader/files/31458703", 1],
    "jv_max_efg_alignn": [
        "https://figshare.com/ndownloader/files/31458691",
        1,
    ],
    "jv_ehull_alignn": ["https://figshare.com/ndownloader/files/31458658", 1],
    "jv_dfpt_piezo_max_dielectric_alignn": [
        "https://figshare.com/ndownloader/files/31458652",
        1,
    ],
    "jv_dfpt_piezo_max_dij_alignn": [
        "https://figshare.com/ndownloader/files/31458655",
        1,
    ],
    "jv_exfoliation_energy_alignn": [
        "https://figshare.com/ndownloader/files/31458676",
        1,
    ],
    "jv_supercon_tc_alignn": [
        "https://figshare.com/ndownloader/files/38789199",
        1,
    ],
    "jv_supercon_edos_alignn": [
        "https://figshare.com/ndownloader/files/39946300",
        1,
    ],
    "jv_supercon_debye_alignn": [
        "https://figshare.com/ndownloader/files/39946297",
        1,
    ],
    "jv_supercon_a2F_alignn": [
        "https://figshare.com/ndownloader/files/38801886",
        100,
    ],
    "mp_e_form_alignn": [
        "https://figshare.com/ndownloader/files/31458811",
        1,
    ],
    "mp_gappbe_alignn": [
        "https://figshare.com/ndownloader/files/31458814",
        1,
    ],
    "tinnet_O_alignn": ["https://figshare.com/ndownloader/files/41962800", 1],
    "tinnet_N_alignn": ["https://figshare.com/ndownloader/files/41962797", 1],
    "tinnet_OH_alignn": ["https://figshare.com/ndownloader/files/41962803", 1],
    "AGRA_O_alignn": ["https://figshare.com/ndownloader/files/41966619", 1],
    "AGRA_OH_alignn": ["https://figshare.com/ndownloader/files/41966610", 1],
    "AGRA_CHO_alignn": ["https://figshare.com/ndownloader/files/41966643", 1],
    "AGRA_CO_alignn": ["https://figshare.com/ndownloader/files/41966634", 1],
    "AGRA_COOH_alignn": ["https://figshare.com/ndownloader/41966646", 1],
    "qm9_U0_alignn": ["https://figshare.com/ndownloader/files/31459054", 1],
    "qm9_U_alignn": ["https://figshare.com/ndownloader/files/31459051", 1],
    "qm9_alpha_alignn": ["https://figshare.com/ndownloader/files/31459027", 1],
    "qm9_gap_alignn": ["https://figshare.com/ndownloader/files/31459036", 1],
    "qm9_G_alignn": ["https://figshare.com/ndownloader/files/31459033", 1],
    "qm9_HOMO_alignn": ["https://figshare.com/ndownloader/files/31459042", 1],
    "qm9_LUMO_alignn": ["https://figshare.com/ndownloader/files/31459045", 1],
    "qm9_ZPVE_alignn": ["https://figshare.com/ndownloader/files/31459057", 1],
    "hmof_co2_absp_alignn": [
        "https://figshare.com/ndownloader/files/31459198",
        5,
    ],
    "hmof_max_co2_adsp_alignn": [
        "https://figshare.com/ndownloader/files/31459207",
        1,
    ],
    "hmof_surface_area_m2g_alignn": [
        "https://figshare.com/ndownloader/files/31459222",
        1,
    ],
    "hmof_surface_area_m2cm3_alignn": [
        "https://figshare.com/ndownloader/files/31459219",
        1,
    ],
    "hmof_pld_alignn": ["https://figshare.com/ndownloader/files/31459216", 1],
    "hmof_lcd_alignn": ["https://figshare.com/ndownloader/files/31459201", 1],
    "hmof_void_fraction_alignn": [
        "https://figshare.com/ndownloader/files/31459228",
        1,
    ],
    "ocp2020_all": ["https://figshare.com/ndownloader/files/41411025", 1],
    "ocp2020_100k": ["https://figshare.com/ndownloader/files/41967303", 1],
    "ocp2020_10k": ["https://figshare.com/ndownloader/files/41967330", 1],
    "jv_pdos_alignn": [
        "https://figshare.com/ndownloader/files/36757005",
        66,
        {"alignn_layers": 6, "gcn_layers": 6},
    ],
}

def get_all_models() -> Dict[str, List[Union[str, int, Dict[str, int]]]]:
    """Return the figshare links for models."""
    return all_models


def get_figshare_model(
    model_name: str = "jv_formation_energy_peratom_alignn"
) -> ALIGNN:
    """Get ALIGNN torch models from figshare."""
    tmp = all_models[model_name]
    url = tmp[0]
    zfile = model_name + ".zip"
    path = str(os.path.join(os.path.dirname(__file__), zfile))
    if not os.path.isfile(path):
        response = requests.get(url, stream=True)
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte
        progress_bar = tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True
        )
        with open(path, "wb") as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
    zp = zipfile.ZipFile(path)
    names = zp.namelist()
    chks = []
    cfg = []
    for i in names:
        if "checkpoint_" in i and "pt" in i:
            tmp = i
            chks.append(i)
        if "config.json" in i:
            cfg = i
        if "best_model.pt" in i:
            tmp = i
            chks.append(i)
    print("Using chk file", tmp, "from ", chks)
    print("Path", os.path.abspath(path))
    print("Config", os.path.abspath(cfg))
    config = json.loads(zipfile.ZipFile(path).read(cfg))
    data = zipfile.ZipFile(path).read(tmp)
    model = ALIGNN(ALIGNNConfig(**config["model"]))

    new_file, filename = tempfile.mkstemp()
    with open(filename, "wb") as f:
        f.write(data)
    model.load_state_dict(torch.load(filename, map_location=device)["model"])
    model.to(device)
    model.eval()
    if os.path.exists(filename):
        os.remove(filename)
    return model


def get_prediction(
    model_name: str = "jv_formation_energy_peratom_alignn",
    atoms: Atoms = None,
    cutoff: float = 8,
    max_neighbors: int = 12,
) -> List[float]:
    """Get model prediction on a single structure."""
    model = get_figshare_model(model_name)
    g, lg = Graph.atom_dgl_multigraph(
        atoms,
        cutoff=float(cutoff),
        max_neighbors=max_neighbors,
    )
    out_data = (
        model([g.to(device), lg.to(device)])
        .detach()
        .cpu()
        .numpy()
        .flatten()
        .tolist()
    )
    return out_data


def get_multiple_predictions(
    atoms_array: List[Atoms] = [],
    cutoff: float = 8,
    neighbor_strategy: str = "k-nearest",
    max_neighbors: int = 12,
    use_canonize: bool = True,
    target: str = "prop",
    atom_features: str = "cgcnn",
    line_graph: bool = True,
    workers: str = 0,
    filename: str = "pred_data.json",
    include_atoms: bool = True,
    pin_memory: bool = False,
    output_features=1,
    batch_size: int = 1,
    model: ALIGNN = None,
    model_name: str = "jv_formation_energy_peratom_alignn",
    print_freq: int = 100,
):
    """Use pretrained model on a number of structures."""
    # import glob
    # atoms_array=[]
    # for i in glob.glob("alignn/examples/sample_data/*.vasp"):
    #      atoms=Atoms.from_poscar(i)
    #      atoms_array.append(atoms)
    # get_multiple_predictions(atoms_array=atoms_array)

    mem = []
    for i, ii in enumerate(atoms_array):
        info = {}
        info["atoms"] = ii.to_dict()
        info["prop"] = -9999  # place-holder only
        info["jid"] = str(i)
        mem.append(info)

    if model is None:
        try:
            model = get_figshare_model(model_name)
        except Exception as exp:
            raise ValueError(
                'Check is the model name exists using "pretrained.py -h"', exp
            )
            pass

    # Note cut-off is usually 8 for solids and 5 for molecules
    def atoms_to_graph(atoms):
        """Convert structure dict to DGLGraph."""
        structure = Atoms.from_dict(atoms)
        return Graph.atom_dgl_multigraph(
            structure,
            cutoff=cutoff,
            atom_features="atomic_number",
            max_neighbors=max_neighbors,
            compute_line_graph=True,
            use_canonize=use_canonize,
        )

    test_data = get_torch_dataset(
        dataset=mem,
        target="prop",
        neighbor_strategy=neighbor_strategy,
        atom_features=atom_features,
        use_canonize=use_canonize,
        line_graph=line_graph,
    )

    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=workers,
        pin_memory=pin_memory,
    )

    results = []
    with torch.no_grad():
        ids = test_loader.dataset.ids
        for dat, id in zip(test_loader, ids):
            g, lg, target = dat
            out_data = model([g.to(device), lg.to(device)])
            out_data = out_data.cpu().numpy().tolist()
            target = target.cpu().numpy().flatten().tolist()
            info = {}
            info["id"] = id
            info["pred"] = out_data
            results.append(info)
            print_freq = int(print_freq)
            if len(results) % print_freq == 0:
                print(len(results))
    df1 = pd.DataFrame(mem)
    df2 = pd.DataFrame(results)
    df2["jid"] = df2["id"]
    df3 = pd.merge(df1, df2, on="jid")
    save = []
    for i, ii in df3.iterrows():
        info = {}
        info["id"] = ii["id"]
        info["atoms"] = ii["atoms"]
        info["pred"] = ii["pred"]
        save.append(info)

    dumpjson(data=save, filename=filename)


if __name__ == "__main__":
    #print(get_default_models())
    #download_default_models()
    #runModels_fromDirectory('example.SigmaPhase')

    if False:
        args = parser.parse_args(sys.argv[1:])
        model_name = args.model_name
        file_path = args.file_path
        file_format = args.file_format
        cutoff = args.cutoff
        max_neighbors = args.max_neighbors
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)
        else:
            raise NotImplementedError("File format not implemented", file_format)

    model_name='jv_formation_energy_peratom_alignn'

    out_data = get_prediction(
        model_name=model_name,
        cutoff=8,
        max_neighbors=12,
        atoms=Atoms.from_poscar('alignn/examples/sample_data/POSCAR-JVASP-10.vasp'),
    )

    print("Predicted value:", model_name, out_data)
