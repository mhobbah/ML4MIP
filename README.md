# ML4MIP
Machine Learning for Medical Image Processing project

## Introduction
Coronary artery disease (CAD) is a leading cause of mortality worldwide, necessitating accurate and efficient diagnostic tools. Coronary computed tomography angiography (CCTA) has become a pivotal non-invasive imaging modality for visualizing coronary artery anatomy and assessing luminal stenosis. However, manual segmentation of coronary arteries from CTA images is labor-intensive, time-consuming, and subject to inter-observer variability, underscoring the need for automated approaches.

## Objectives
The goals of this project are to develop a method to produce a segmentation mask for the coronary arteries from 3D CTA images and extract a graph from the CTA image to represent the structure of the coronary tree. This repository presents the method I implemented to solve the second task.

## Files
- **`graph_construction.py`**: python script to create graphs from CTA images and their masks (labels) in nibabel.nifti1.Nifti1Image format
- **`utils.py`**: contains various functions useful across all the implementation
- **`skeleton_to_digraph.py`**: contains the functions to compute the graph used in `graph_construction.py`
- **`graph_construction-Notebook.ipynb`**: allows to visualize the actions of the `skeleton_to_digraph.py` functions
- **`Ostia Detector.py`**: contains the CNN model that retrieves the coronary osia
- **`Ostia Detector-Training.ipynb`**: trains the CNN model
- **`models`**: folder containing one trained model for every epoch
- **`losses.npz`**: arr1=training losses and arr2=validation losses 
- **`Ostia Detector-Processing.ipynb`**: selects the best CNN model out of the last 5 epochs of training and shows visualization of the performances of the selected model
- **`ostia_model_trained`**: model (weights) selected by `Ostia Detector-Processing.py`

## Installation and execution
### Instructions
1. Clone this repository:
   
   ```bash
   git clone https://github.com/mhobbah/ML4MIP.git
   ```
2. Install depedencies:
   
   ```bash
   pip install -r requirements.txt
   ```
3. Execute main script:
   
   ```bash
   python3 graph_construction.py --image_folder /path/to/your/image/folder --label_folder /path/to/your/label/folder --graph_folder /path/to/your/graph/folder
   ```
