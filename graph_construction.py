from pathlib import Path
from argparse import ArgumentParser
import os
import torch
from skeleton_to_digraph import *
import json
import warnings
warnings.filterwarnings("ignore")


def main():
    # Add arguments
    parser = ArgumentParser()
    parser.add_argument("--image_folder", type=Path)
    parser.add_argument("--label_folder", type=Path)
    parser.add_argument("--graph_folder", type=Path)
    args = parser.parse_args()
    
    if not os.path.isdir(args.image_folder):
        raise FileNotFoundError("Image folder {} does not exist".format(args.image_folder))
        
    if not os.path.isdir(args.label_folder):
        raise FileNotFoundError("Label folder {} does not exist".format(args.label_folder))
        
    if not os.path.isdir(args.graph_folder):
        raise FileNotFoundError("Graph folder {} does not exist".format(args.graph_folder))
    
    # List all the NIfTI files in the directory
    image_files = [os.path.join(args.image_folder, f) for f in os.listdir(args.image_folder) if f.endswith('img.nii.gz')]
    label_files = [os.path.join(args.label_folder, f) for f in os.listdir(args.label_folder) if f.endswith('label.nii.gz')]
    
    # Create a list of dictionaries to store data
    data = []
    
    for i in image_files:
        for l in label_files:
            if i.split("{}/".format(args.image_folder))[1][:6] == l.split("{}/".format(args.label_folder))[1][:6]: # identify the sample
                data += [{'image': i, 'label': l}]
                break
                
    # Choose a trained model (weights)
    model_path = "./ostia_model_trained"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model for root nodes detection
    ostia_model = OstiaDetector()
    ostia_model.load_state_dict(torch.load(model_path))
    ostia_model.eval()
    
    # Create .graph.json files from predicted masks
    i = 0

    for sample in data:    
        mask_pred = nib.load(sample['label'])

        skeleton = skeletonization(mask_pred)
        G, summary = graph_initialization(skeleton)
        add_node_pos(G, summary, mask_pred)
        add_edge_attributes(G, mask_pred)
        graph_processing(G)
        proximity_map, affine = ostia_proximity_map(sample['image'], ostia_model, device)
        root_nodes_identification(G, proximity_map, affine)
        DiG = directed_graph(G)

        # Convert the graph to node-link data
        graph_data = json_graph.node_link_data(DiG, edges='edges')
        graph_data['graph'] = {'coordinateSystem': 'RAS'}
        name = sample['image'].split("{}/".format(args.image_folder))[1][:6] 
 
        # Save the data as a JSON file
        with open("{}/{}.graph.json".format(args.graph_folder, name), "w") as f:
            json.dump(graph_data, f, indent=4)
        
        i += 1
        
        if (i % 10 == 0) or (i == len(data)):
            print("{}/{}".format(i, len(data)))
            
            if i == len(data):
                print("Execution is done. All graphs are stored in {}.".format(args.graph_folder))

                  
if __name__ == "__main__":
    main()
