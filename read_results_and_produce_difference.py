import pickle
import argparse
import numpy as np
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, TypeVar, Generic
from pathlib import Path

T = TypeVar('T')

@dataclass
class Sims(Generic[T]):
    Source: T
    Target: T
    def __str__(self):
        return f"Source: {self.Source}\nTarget: {self.Target}"

@dataclass
class Method:
    original:  Sims[List[float]]
    naive:     Sims[List[float]]
    anchored:  Sims[List[float]]
    def __str__(self):
        return f"Original:\n{self.original}\n\nNaive:\n{self.naive}\n\nAnchored:\n{self.anchored}"

@dataclass
class Result:
    naive:     Sims[Tuple[float, float]]
    anchored:  Sims[Tuple[float, float]]
    def __str__(self):
        return f"Naive:\n{self.naive}\n\nAnchored:\n{self.anchored}"

def load_pickle(file_path: Path) -> Dict:
    try:
        with file_path.open('rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}

def seed_from_path(path: Path) -> int:
    return int(path.parts[path.parts.index('seeds') + 1])

def find_seed(path: Path) -> int:
    hypers_path = (Path(*path.parts[:path.parts.index('seeds')]) / 'hypers.json')
    with hypers_path.open() as f:
        hypers = json.load(f)
        return seed_from_path(Path(hypers["prev_folder"])) if hypers["prev_folder"] else seed_from_path(path)

def process(data: Dict) -> List[float]:
    return [np.mean(data[key]) for key in sorted(data.keys(), key=find_seed)]

def load_sims(method_path: Path) -> Sims[List[float]]:
    return Sims(
        Source=process(load_pickle(method_path / 'Source.pkl')),
        Target=process(load_pickle(method_path / 'Target.pkl'))
    )

def load_pickles_from_folder(folder_path: Path) -> Method:
    return Method(
        original=load_sims(folder_path / 'original'),
        naive=load_sims(folder_path / 'naive'),
        anchored=load_sims(folder_path / 'anchored')
    )

def diff_mean_std(from_list: List[float], to_list: List[float]) -> Tuple[float, float]:
    diff = np.array(to_list) - np.array(from_list)
    return np.mean(diff), np.std(diff)

def calculate_results(method: Method) -> Result:
    return Result(
        naive=Sims(
            Source=diff_mean_std(method.original.Source, method.naive.Source),
            Target=diff_mean_std(method.original.Target, method.naive.Target)
        ),
        anchored=Sims(
            Source=diff_mean_std(method.original.Source, method.anchored.Source),
            Target=diff_mean_std(method.original.Target, method.anchored.Target)
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load pickle files from a folder structure.')
    parser.add_argument('folder', type=str, help='folder containing the pickle files')
    method = load_pickles_from_folder(Path(parser.parse_args().folder))
    print("Method:")
    print(method)
    result = calculate_results(method)
    print("\nResult:")
    print(result)
