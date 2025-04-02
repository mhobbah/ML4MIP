import numpy as np
import torch
import networkx as nx
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.morphology import skeletonize_3d, binary_closing, binary_opening, ball
from scipy.ndimage import label
from skan.csr import Skeleton, summarize, skeleton_to_nx
from OstiaDetector import OstiaDetector
from utils import load_image, segment_image

"""
this script contains functions to obtain a directed acyclic graph from a segmentation mask
"""


def skeletonization(mask): 
    """
    Compute a smoothed skeleton of the mask

    Args:
        mask (nibabel.nifti1.Nifti1Image): A 3D image representing the mask of the coronary arteries 

    Return:
        skeleton (numpy.ndarray): A 3D array representing the skeleton of the coronary arteries 
    """
    segmentation = mask.get_fdata().astype(np.uint8)

    # Binary opening followed by a closing with a small kernel
    segmentation = binary_opening(segmentation, ball(2.5))
    segmentation = binary_closing(segmentation, ball(1))

    # Remove small components
    labeled, _ = label(segmentation)
    sizes = np.bincount(labeled.ravel())
    mask_sizes = sizes > 10  # Keep components larger than 10 voxels
    mask_sizes[0] = 0  # Exclude background
    cleaned_segmentation = mask_sizes[labeled]

    # Normalize cleaned segmentation to binary (0 and 1)
    binary_cleaned_segmentation = (cleaned_segmentation > 0).astype(np.uint8)

    # Apply skeletonization
    skeleton = skeletonize_3d(binary_cleaned_segmentation)
    
    return skeleton


def graph_initialization(skeleton):
    """
    Returns a connected graph of the skeleton where edges have a given minimum length and a summary of edges properties

    Args:
        skeleton (numpy.ndarray): A 3D array representing the skeleton of the coronary arteries 

    Return:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are bijection or end points of the skeleton
        summary(pandas.DataFrame): A summary of the skeleton branches properties / the graph edges properties 
    """
    # Use skan functions to get a networkx graph and a summary of edges properties
    skeleton_skan = Skeleton(skeleton)
    G = nx.Graph(skeleton_to_nx(skeleton_skan)) # create a simple graph (skeleton_to_nx computes a multi graph)
    summary = summarize(skeleton_skan)

    # Remove edges with a length below a given threshold (considered as skeletonization noise), remove disconnected nodes
    threshold = 5
    short_branches = summary.loc[summary['branch-distance'] < threshold][['node-id-src', 'node-id-dst']]
    short_branches = list(zip(short_branches['node-id-src'], short_branches['node-id-dst']))
    G.remove_edges_from(short_branches)
    G.remove_nodes_from(list(n for n in G.nodes if G.degree(n) == 0))
    
    return G, summary


def add_node_pos(G, summary, mask):
    """
    Add positions of the nodes in real-world coordinates

    Args:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are intersection or end points of the skeleton
        summary(pandas.DataFrame): A summary of the skeleton branches properties / the graph edges properties 
        mask (nibabel.nifti1.Nifti1Image): A 3D image representing the mask of the coronary arteries 

    Return:
    """
    pos = {}
    
    for node in G.nodes:
        if not summary.loc[summary['node-id-src'] == node].empty:
            coords_df = summary.loc[summary['node-id-src'] == node][['image-coord-src-0', 'image-coord-src-1', 'image-coord-src-2']]
            coords_voxel = np.array(coords_df.iloc[0])
            coords_h = np.hstack((coords_voxel, 1))
            coords = (coords_h@mask.affine.T)[:3]
            pos[node] = coords.tolist()

        else:
            coords_df = summary.loc[summary['node-id-dst'] == node][['image-coord-dst-0', 'image-coord-dst-1', 'image-coord-dst-2']]
            coords_voxel = np.array(coords_df.iloc[0])
            coords_h = np.hstack((coords_voxel, 1))
            coords = (coords_h@mask.affine.T)[:3]
            pos[node] = coords.tolist()

    nx.set_node_attributes(G, pos, "pos")
    

def add_edge_attributes(G, mask):
    """
    Add length in mm and skeletons in real world coordinates

    Args:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are intersection or end points of the skeleton
        mask (nibabel.nifti1.Nifti1Image): A 3D image representing the mask of the coronary arteries 

    Return:
    """
    length, skeletons = {}, {}
    
    for edge in G.edges:
        path_voxel = G.edges[edge]['path']
        path_h = np.hstack((path_voxel, np.ones((path_voxel.shape[0], 1))))
        path_rw = (path_h@mask.affine.T)[:,:3]
        path_rw = path_rw[::5]
        skeletons[edge] = path_rw.tolist()

        distances = np.linalg.norm(path_rw[1:] - path_rw[:-1], axis=1)
        length[edge] = float(distances.sum())

    nx.set_edge_attributes(G, length, "length")     
    nx.set_edge_attributes(G, skeletons, "skeletons")   
    
    # Remove the useless edges attributes
    for _, _, data in G.edges(data=True):
        if 'path' in data:
            del data['path']
        if 'indices' in data:
            del data['indices']
        if 'values' in data:
            del data['values']
            
            
def graph_processing(G):    
    """
    Make the graph into two trees, where no node has degree 2

    Args:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are intersection or end points of the skeleton

    Return:
    """
    # Delete all cycles by removing the smallest edge of each cycle
    while list(nx.simple_cycles(G)): # while G has a cycle
        cycles = sorted(nx.simple_cycles(G), key=len, reverse=True)
        cycle = cycles[0] # take the largest cycle of the list 
        edges_cycle = [(u, v) for u, v in G.edges(cycle) if u in cycle and v in cycle]
        smallest_edge = edges_cycle[0]
        
        # Determine the smallest edge
        for edge in edges_cycle[1:]:
            if G.edges[edge]["length"] < G.edges[smallest_edge]["length"]:
                smallest_edge = edge
            
        G.remove_edge(*smallest_edge)
        
        
    # Delete 2-degree nodes and rebuild edges with their attributes
    for node in list(n for n in G.nodes if G.degree(n) == 2):
        neighbor0, neighbor1 = list(G.neighbors(node))
        
        length0, length1 = G.edges[node, neighbor0]['length'], G.edges[node, neighbor1]['length']
        skeletons0, skeletons1 = np.array(G.edges[node, neighbor0]['skeletons']), np.array(G.edges[node, neighbor1]['skeletons'])

        new_edge_attr = {'length': float(length0 + length1),
                         'skeletons': np.vstack((skeletons0, skeletons1)).tolist()}

        G.remove_node(node)
        G.add_edge(neighbor0, neighbor1, **new_edge_attr)
        
        
    # Remove components that are not the two longest components
    for nodes in sorted(nx.connected_components(G), key=len, reverse=True)[2:]:
        G.remove_nodes_from(nodes)
        

def ostia_proximity_map(image_path, ostia_model, device="cpu"):
    """
    Compute the proximity map of the ostia points

    Args:
        image_path (string): A link to the NifTI image
        ostia_model (OstiaDetector): A trained OstiaDetector model
        device (string): The device to use to perform the model application ("cpu" or "cuda")

    Return:
        proximity_map (numpy.array): A map of the image where highest values are the most probable locations of the ostia points
        affine (numpy.array): A 4x4 array representing the affine matrix of the resized image (1mm * 1mm * 1mm)
    """
    # Make patches of an image
    image, _, affine = load_image(image_path) # image has a 1mm * 1mm * 1mm voxel spacing

    W = 19 
    M = int((W - 1) / 2) 
    stride = 5

    inputs = []

    for i in range(M, image.shape[0] - M + 1, stride):
        for j in range(M, image.shape[1] - M + 1, stride):
            for k in range(M, image.shape[2] - M + 1, stride):
                point = [i, j, k] # center of the patch
                patch, _ = segment_image(image, point)

                if patch.shape == (19, 19, 19):
                    patch = torch.tensor(np.array(patch), dtype=torch.float32).reshape(1, 1, 19, 19, 19)
                    inputs.append([point, patch])
                    
    # Apply model on the patches
    proximity_map = np.zeros(image.shape)
    ostia_model.to(device)

    for point, patch_i in inputs:
        patch_i = patch_i.to(device)
        proximity_value = ostia_model(patch_i)
        proximity_value = proximity_value.item()

        proximity_map[point[0], point[1], point[2]] = proximity_value # assign the value to the point coordinates on the proximity map
        
    return proximity_map, affine


def retrieve_ostia_voxel(proximity_map):
    """
    Returns the ostia points predicted by the model in real-world coordinates 

    Args:
        proximity_map (numpy.array): A map of the image where highest values are the most probable locations of the ostia points
        
    Return:
        ostia_pred_voxel (numpy.ndarray): A (2, 3) array representing the voxel coordinates of the ostia points
    """
    # Select the 10 max values of the proximity map
    max_values = np.sort(proximity_map, axis=None)[::-1][:10]
    
    clusters, id_clusters = [], []

    # Make clusters of points 
    while len(max_values) > 0:
        id_cluster = 10 - len(max_values) # ensures unicity of cluster value if new cluster
        value = max_values[0]
        coords = np.argwhere(proximity_map == value) # coordinates of value in the proximity map
        
        for i in range(len(clusters)):
            center = clusters[i]
            _, coords_c, _ =  center

            if np.all((coords <= coords_c+5) & (coords >= coords_c-5)): # add the point to the same cluster if all 
                id_cluster = i                                            # its coordinates are within -5 +5 of the center ones
                break
          
        clusters.append((id_cluster, coords, value))   
        id_clusters.append(id_cluster)
        max_values = np.delete(max_values, 0) # delete the value just added to clusters
        
    # Get the first two clusters with highest scores (sums of proximity map values)
    scores = np.empty((0, 2))
    
    for id_cluster in set(id_clusters):
        score_cluster = 0

        for elem in clusters:
            id_cluster_e, _, value_e = elem

            if id_cluster_e == id_cluster:
                score_cluster += value_e

        scores = np.vstack((scores, [id_cluster, score_cluster]))
    
    idx_max = np.argsort(scores[:, 1], axis=0)[::-1][:2] # get the indexes of the two clusters with highest scores in the scores array
   
    id_clusters_max = scores[idx_max, 0] # get the indexes of the two clusters with highest scores

    # Determine the ostia points as the center of mass of clusters
    ostia_pred_voxel = np.empty((0, 3))
    
    for id_cluster in id_clusters_max:
        points = np.empty((0, 3))

        for elem in clusters:
            id_cluster_e, coords_e, _ = elem

            if id_cluster_e == id_cluster:
                coords_e = np.array(coords_e).reshape(1, 3)
                points = np.vstack((points, coords_e))

        ostium_pred_voxel = np.mean(points, axis=0)
        ostia_pred_voxel = np.vstack((ostia_pred_voxel, ostium_pred_voxel))
    
        
    return ostia_pred_voxel
    

def root_nodes_identification(G, proximity_map, affine):
    """
    Identify the root nodes

    Args:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are intersection or end points of the skeleton
        proximity_map (numpy.array): A map of the image where highest values are the most probable locations of the ostia points
        affine (numpy.array): A 4x4 array representing the affine matrix of the resized image (1mm * 1mm * 1mm)
        
    Return:
    """
    # Get the real-world coordinates of the predicted ostia from their voxel coordinates
    ostia_pred_voxel = retrieve_ostia_voxel(proximity_map)
    
    ostia_pred_h = np.hstack((ostia_pred_voxel, np.ones((ostia_pred_voxel.shape[0], 1))))
    ostia_pred = (ostia_pred_h@affine.T)[:, :3]
    
    is_root = {}
    
    trees = list(nx.connected_components(G))
    
    # Iterate on each artery tree
    for tree in trees:
        list_tree = list(tree)
        
        root_node = list_tree[0]

        d = np.min([np.linalg.norm(ostia_pred[0] - G.nodes[root_node]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[root_node]['pos'])]) 
        
        # Determine the closest node to one of the predicted ostia
        for node in list_tree:
            if (np.min([np.linalg.norm(ostia_pred[0] - G.nodes[node]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[node]['pos'])]) < d):

                d = np.min([np.linalg.norm(ostia_pred[0] - G.nodes[node]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[node]['pos'])])

                root_node = node

            is_root[node] = bool(False)

        is_root[root_node] = bool(True)
        
    # In case there is less than 2 artery trees detected
    if len(trees) < 2:
        tree = trees[0]
        list_tree = list(tree)
        list_tree.remove(root_node)
        
        root_node2 = list_tree[0]
        
        d = np.min([np.linalg.norm(ostia_pred[0] - G.nodes[root_node2]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[root_node2]['pos'])]) 
        
        # Determine the second closest node to one of the predicted ostia
        for node in list_tree:    
            if (np.min([np.linalg.norm(ostia_pred[0] - G.nodes[node]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[node]['pos'])]) < d):

                d = np.min([np.linalg.norm(ostia_pred[0] - G.nodes[node]['pos']), 
                    np.linalg.norm(ostia_pred[1] - G.nodes[node]['pos'])])

                root_node2 = node

            is_root[node] = bool(False)

        is_root[root_node2] = bool(True)
        
    nx.set_node_attributes(G, is_root, "is_root")
    
    
def directed_graph(G):
    """
    Relabels the nodes with lower ids and compute a directed graph from the simple graph with ostia points as root nodes
    
    Args:
        G (networkx.Graph): A graph extracted from the skeleton, nodes are intersection or end points of the skeleton

    Return:
        DiG (networkx.Graph): A directed graph computed from G
    """
    # Relabel the nodes
    i = 1
    mapping = {}
    
    for node in G.nodes:
        mapping[node] = i
        i += 1 
        
    nx.relabel_nodes(G, mapping, copy=False)
    
    DiG = nx.DiGraph(G)
    
    # Breadth-first search
    root_nodes = [node for node in DiG.nodes if DiG.nodes[node]['is_root']]

    def affectation(node):
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited: 
                DiG.remove_edge(neighbor, node)
                affectation(neighbor)

    visited = set()
    
    for root_node in root_nodes:
        affectation(root_node)
    
    return DiG