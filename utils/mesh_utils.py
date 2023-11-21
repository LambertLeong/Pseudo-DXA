"""
File Name: mesh_utils.py
Author: Lambert T Leong
Description: Contains 3D mesh utility functions
"""

import numpy as np
import random, os
from plyfile import PlyData, PlyElement
from typing import Generator, Tuple
import tensorflow as tf
from typing import List, Tuple

def convert_mesh_ply_to_npy(ply_dir: str, npy_dir: str) -> int:
    """
    Converts PLY files in a directory to NumPy arrays and saves them.
    
    Args:
        ply_dir (str): Directory containing PLY files.
        npy_dir (str): Directory where NumPy arrays will be saved.

    Returns:
        int: Number of PLY files converted.
    """
    c=1
    for i in os.listdir(ply_dir):
        if not '.ply' in i:
            continue
        ply = PlyData.read(ply_dir+i)
        xyz = np.vstack((np.array(ply['vertex']['x']), np.array(ply['vertex']['y']), np.array(ply['vertex']['z']))).T
        np.save(npy_dir+i+'.npy',xyz)
        c+=1
        return c

def ply_to_npy(path: str) -> np.ndarray:
    """
    Convert a PLY file to a NumPy array.

    Args:
        path (str): Path to the PLY file.

    Returns:
        np.ndarray: NumPy array containing vertex coordinates.
    """
    ply = PlyData.read(path)
    xyz = np.vstack((np.array(ply['vertex']['x']), np.array(ply['vertex']['y']), np.array(ply['vertex']['z']))).T
    return xyz

def load_full_mesh(path: List[str]) -> np.ndarray:
    """
    Load multiple full meshes from a list of file paths.

    Args:
        path (List[str]): List of file paths.

    Returns:
        np.ndarray: NumPy array containing loaded meshes.
    """
    out = []
    count=1
    for name in path:
        #print(name)#,end='\r')
        out+=[np.load(name)]
    out = np.stack(out)
    #print(low_pix, hi_pix)
    return out


def sample_points_uniform(meshes: List[np.ndarray], tot_pts: int = 110306, perc: float = 0.2) -> np.ndarray:
    """
    Sample points uniformly from multiple meshes.

    Args:
        meshes (List[np.ndarray]): List of mesh arrays.
        tot_pts (int): Total number of points to sample.
        perc (float): Percentage of points to sample from each mesh.

    Returns:
        np.ndarray: NumPy array containing sampled points.
    """
    out = []
    uniform = []
    num_pts=round(meshes[0].shape[0]*perc)
    for n in range(tot_pts):
        if n%(tot_pts//num_pts)==0:
            uniform+=[n]
    diff = abs(len(uniform)-num_pts)
    if diff != 0:
        uniform=uniform[:-(diff)]

    for m in meshes:
        out+=[m[uniform]]
    return np.stack(out)

def sample_points_random(mesh,perc=0.2):
    num_pts=round(mesh.shape[0]*perc)
    sample_ind = random.sample(range(0, mesh.shape[0]), num_pts)
    return mesh[sample_ind]#[:-1]

def sample_points_random_tf(mesh,perc=0.2):
    num_pts = round(mesh.shape[0] * perc)
    sample_ind = tf.random.shuffle(tf.range(mesh.shape[0]))[:num_pts]
    return tf.gather(mesh, sample_ind)

def mesh2dxa_gen(mesh: np.ndarray, dxa: np.ndarray, batch_size: int = 32) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    Generator function to yield batches of mesh and corresponding DXA data.

    Args:
        mesh (np.ndarray): The mesh data, expected to be a multi-dimensional NumPy array.
        dxa (np.ndarray): The DXA data corresponding to the mesh, expected to be a multi-dimensional NumPy array.
        batch_size (int, optional): The size of the batch to yield. Defaults to 32.

    Yields:
        Generator[Tuple[np.ndarray, np.ndarray], None, None]: 
        A tuple containing a batch of mesh data and the corresponding DXA data.
    """
    while True:
        idx = np.random.choice(range(mesh.shape[0]), size=batch_size, replace=True)
        batch_mesh = [sample_points_random(mesh[i]) for i in idx]
        batch_dxa = [dxa[i] for i in idx]
        yield np.stack(batch_mesh), np.stack(batch_dxa)