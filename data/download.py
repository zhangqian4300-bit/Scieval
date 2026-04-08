"""Download MoleculeNet datasets (CSV format) for the SciEval demo."""
import os
import requests
import pandas as pd

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")

# MoleculeNet datasets available as CSV
DATASET_URLS = {
    "qm9": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/qm9.csv",
    "tox21": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz",
    "bace": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv",
    "esol": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv",
    "freesolv": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv",
}

# Column that contains SMILES for each dataset
SMILES_COLUMNS = {
    "qm9": "smiles",
    "tox21": "smiles",
    "bace": "mol",
    "esol": "smiles",
    "freesolv": "smiles",
}


def download_dataset(name: str, force: bool = False) -> str:
    """Download a single dataset. Returns path to saved CSV."""
    os.makedirs(DATASETS_DIR, exist_ok=True)
    save_path = os.path.join(DATASETS_DIR, f"{name}.csv")

    if os.path.exists(save_path) and not force:
        print(f"  {name}: already exists, skipping")
        return save_path

    url = DATASET_URLS[name]
    print(f"  {name}: downloading from {url}...")

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()

    # Handle gzipped files
    if url.endswith(".gz"):
        import gzip
        import io
        content = gzip.decompress(resp.content).decode("utf-8")
        df = pd.read_csv(io.StringIO(content))
    else:
        import io
        df = pd.read_csv(io.StringIO(resp.text))

    # Normalize SMILES column name
    smiles_col = SMILES_COLUMNS[name]
    if smiles_col != "smiles":
        df = df.rename(columns={smiles_col: "smiles"})

    df.to_csv(save_path, index=False)
    print(f"  {name}: saved {len(df)} rows to {save_path}")
    return save_path


def download_all(force: bool = False):
    """Download all datasets."""
    print("Downloading MoleculeNet datasets...")
    for name in DATASET_URLS:
        try:
            download_dataset(name, force=force)
        except Exception as e:
            print(f"  {name}: FAILED - {e}")
    print("Done.")


if __name__ == "__main__":
    download_all()
