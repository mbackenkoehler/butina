import sys
import functools

from typing import Iterable

import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator
from tqdm.auto import tqdm
from multiprocessing import Pool


@functools.cache
def compute_fp(smi: str, radius: int = 3, fp_dim: int = 2048, target: str = "numpy"):
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_dim)
    try:
        match target:
            case "numpy":
                return mfpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(smi))
            case "native":
                return mfpgen.GetFingerprint(Chem.MolFromSmiles(smi))
            case _:
                raise ValueError(f"unkown fingerpritn target: '{target}'")
    except TypeError:
        print(f"No fp for SMILES={smi}", file=sys.stderr)
        return np.nan


def par_compute_fp(smiles: Iterable[str], n_jobs=16, target: str = "numpy"):
    with Pool(n_jobs) as p:
        return p.map(functools.partial(compute_fp, target=target), tqdm(smiles))


def main():
    input_csv = sys.argv[1]
    df = pd.read_csv(input_csv)
    smiles_list = np.unique(df["canonical_smiles"].tolist())

    fps = par_compute_fp(smiles_list)
    valid_smiles = [
        smi for (smi, fp) in zip(smiles_list, fps) if not np.any(np.isnan(fp))
    ]
    fps = [fp for fp in fps if not np.any(np.isnan(fp))]

    np.save("fingerprints.npy", np.array(fps, dtype=np.uint8))
    with open("smiles.txt", "w") as f:
        for smi in valid_smiles:
            f.write(smi + "\n")

    print(f"Saved {len(valid_smiles)} fingerprints to fingerprints.npy")
    print(f"Saved SMILES to smiles.txt")


if __name__ == "__main__":
    main()
