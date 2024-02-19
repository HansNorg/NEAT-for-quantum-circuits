from time import time
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore", "Couldn't import `kahypar` - skipping from default hyper optimizer and using basic `labels` method instead.")

import numpy as np
import pandas as pd
import quimb as q
from quantumneat.problems.chemistry import GroundStateEnergy

def load_data(molecule:str):
    data:pd.DataFrame = pd.read_pickle(f"{molecule}_hamiltonian.pkl")
    new_index = []
    for index in data.index:
        new_index.append(np.round(index, 2))
    data.insert(0, "R", new_index)
    data.reset_index()
    data.set_index("R", inplace=True)
    return data

def exact_diagonalisation(H):
    el, ev = q.eigh(H, k=1)
    return el[0]

def get_solution(data:pd.DataFrame):
    solutions = []
    for _, instance in data.iterrows():
        H = GroundStateEnergy.hamiltonian(instance)
        solutions.append(exact_diagonalisation(H)+instance["repulsion"])
    return solutions

def add_solution(molecule:str, save = True):
    starttime = time()
    data = load_data(molecule)
    print(f"Load time {molecule} = {time()-starttime}")
    solutiontime = time()
    solutions = get_solution(data)
    print(f"Solution time {molecule} = {time()-solutiontime}")
    # data.insert(len(data.columns), "solution", solutions)
    data["solution"] = solutions
    print(f"Total time {molecule} = {time()-starttime}")
    print(data.head())
    if save:
        data.to_pickle(f"{molecule}_hamiltonian.pkl")

if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument("molecules", type=str, nargs="+")
    argparser.add_argument("-d", "--dry_run", action="store_false")
    args = argparser.parse_args()

    for molecule in args.molecules:
        # print(molecule, args.dry_run)
        add_solution(molecule, args.dry_run)