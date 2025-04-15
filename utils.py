import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json
from scipy import ndimage


W = 19

"""
this script contains a variety of utility functions with different purposes
they are used to load the dataset, and preprocess the data for training
"""


def load_image(image_path):
    """
    load the image and its parameters

    Args:
        image_path (string): A link to the NifTI image
        
    Return:
        ct_scan (numpy.ndarray): A 3D array representing the image (CTA)
        origin (numpy.ndarray): A (1, 3) array representing the coordinates of the image origin
        affine (numpy.ndarray): A (4, 4) array representing the affine matrix of the image with new voxel spacing
    """
    voxel_spacing = 1.0
    
    nibimage = nib.load(image_path)

    ct_scan = nibimage.get_fdata()
    affine = nibimage.affine 
    origin = affine[:3, 3]
    spacing = nib.affines.voxel_sizes(affine)
    
    # Resample the images to achieve a voxel size of voxel_spacing
    x_zoom, y_zoom, z_zoom = spacing / voxel_spacing
    ct_scan = ndimage.zoom(ct_scan, (x_zoom, y_zoom, z_zoom))
    
    
    # Update the affine
    a, b, c = np.sign(affine[0, 0]), np.sign(affine[1, 1]), np.sign(affine[2, 2])
    new_affine = affine.copy()
    new_affine[:3, :3] = np.diag((a * voxel_spacing, b * voxel_spacing, c * voxel_spacing))
    
    return ct_scan, origin, new_affine


def load_seeds(image_path, graph_path):
    """
    load the image and ground-truth root nodes using load_image

    Args:
        image_path (string): A link to the NifTI image
        graph_path (string): A link to the graph.json

    Return:
        image (numpy.ndarray): A 3D array representing the image (CTA)
        ostia (numpy.ndarray): A (2, 3) array representing the coordinates of the root nodes
    """    
    image, _, affine = load_image(image_path)
    
    with open(graph_path, "r") as file:
        data = json.load(file)
    graph = json_graph.node_link_graph(data, edges="edges")
    
    node_points = np.array(list(nx.get_node_attributes(graph, 'pos').values()))
    
    # Get the voxel coordinates from the real-world coordinates
    node_points_h = np.hstack((node_points, np.ones((node_points.shape[0], 1))))
    node_points_voxel = (node_points_h@np.linalg.inv(affine).T)[:,:3]
    
    # Identify ostia
    is_root = np.array(list(nx.get_node_attributes(graph, 'is_root').values()))
    ostia = node_points_voxel[np.where(is_root)]
    
    return image, ostia


def calculate_distance(point1, point2):
    
    if (type(point1) != np.ndarray) or (type(point2) != np.ndarray):
        point1, point2 = np.array(point1), np.array(point2)
    distance = (point1 - point2) ** 2
    distance = np.sqrt(np.sum(distance))
    return distance


def segment_image(image, point, w=W):
    """
    extract a cube from a given image that centers around a given point
    returns None if the cube extends outside of the given image
    
    Args:
        image (numpy.ndarray): A 3D array representing the image (CTA) to extract cube from
        point (numpy.ndarray): An array of shape (1, 3) that is the voxel coordinates of the point at the center of the returned cube
        w (int): length of the sides of the cube
    
    Return:
        (array): the extracted cube (3D array) or None
        (list): A list of shape (3, 3) containing the lower and upper bounds of the cube in the image
    """
    m = int((w - 1) / 2)
    x, y, z = point
    x, y, z = int(x), int(y), int(z)
    
    try:
        return image[x - m:x + m + 1, y - m:y + m + 1, z - m:z + m + 1].copy(), [[x - m, y - m, z - m],
                                                                                 [x + m + 1, y + m + 1, z + m + 1]]
    except:
        return np.empty(0), None
