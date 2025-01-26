"""This module provides functionality to load 3D model data from a file.

The `load_data` function reads a file containing 3D model data, extracts the vertex and face
information, and returns them as numpy arrays. The file is expected to be in a specific format
where each line represents either a vertex or a face. Vertices are denoted by lines starting
with 'v' and faces by lines starting with 'f'.

The function uses pandas to read the file and process the data. The vertices and faces are
extracted based on their respective identifiers and returned as numpy arrays.
"""

import pandas as pd


def load_data(file_path):
    """
    Load 3D model data from a file.

    Parameters
    ----------
    file_path : str
        Path to the file containing the 3D model data.

    Returns
    -------
    vertices : numpy.ndarray
        Array of vertex coordinates.
    faces : numpy.ndarray
        Array of face indices.
    """
    comet = pd.read_csv(file_path, sep=r"\s+", names=["TYPE", "X1", "X2", "X3"])
    vertices = comet.loc[comet["TYPE"] == "v"][["X1", "X2", "X3"]].values
    faces = comet.loc[comet["TYPE"] == "f"][["X1", "X2", "X3"]].values
    faces = faces - 1
    faces = faces.astype(int)
    return vertices, faces
