Prompt-Guided Refinement: A Novel Technique for Improving Intervertebral Disc Semantic Labeling
Author: Mohammed N. Alharbi and Mohammad D. Alahmadi
Overview
Accurate detection and semantic labeling of intervertebral discs (IVDs) in magnetic resonance imaging (MRI) is critical for diagnosing and managing spine-related disorders. Traditional convolutional neural network (CNN)–based approaches extract contextual information from MRI scans but often overlook the inherent geometric structure of the vertebral column. This omission can lead to anatomically inconsistent predictions, especially in challenging regions of the spine.
To address these limitations, we propose a novel prompt encoder framework that integrates explicit geometric priors into the learning process. Our method:
Learns a global spinal skeleton representation from the training dataset
Encodes this representation as a geometric prompt
Aligns predicted features with anatomical constraints via prompt consistency loss
Enforces vertebral structural relationships through skeleton structural loss
Evaluations on multi-center datasets show that our model surpasses existing state-of-the-art techniques and achieves robust performance across both T1w and T2w MRI modalities. The incorporation of geometric guidance significantly enhances the accuracy and consistency of IVD localization and semantic labeling.
Repository Structure
src/
│   main.py                 # Train and evaluate the model
│   create_skeleton.py      # Generate skeleton and Gaussian prompt maps
│
└───pgrnet/                 # Prompt-guided regression network implementation
prepared_data/
│   (Place processed dataset files here)
Setup Instructions
Step 0 — Download Processed Dataset
The preprocessed train, validation, and test sets are provided in the repository under prepared_data:
the original dataset can be found here:
https://github.com/spine-generic/data-multi-subject
After downloading the file:
Unzip the archive
Move all contents into the folder:
prepared_data/

Your folder should contain files similar to:
prepared_trainset_t1
prepared_valset_t1
prepared_testset_t1
prepared_trainset_t2
...
Step 1 — Generate the Skeleton Prompt
Before training, you must compute the average spinal skeleton and its Gaussian-encoded prompt map.
Run:
python src/create_skeleton.py

This script automatically:
Extracts vertebral skeletons from training labels
Computes normalized joint coordinates
Generates a 2D Gaussian heatmap representation
Saves results in:
prepared_data/t1_Skelet.npy
prepared_data/t1_Skelet_gaussian_heatmap.png
Step 2 — Train and Evaluate the Model
To train the prompt-guided IVD labeling model:
python src/main.py

The script will:
Train on the preprocessed dataset
Evaluate performance
Save logs, metrics, and checkpoints based on configuration settings
Citation
If you use this repository or build upon this work, please cite:
@article{alahmadi2024promptivd,
title={Prompt-Guided Refinement: A Novel Technique for Improving Intervertebral Disc Semantic Labeling},
author={Mohammed N. Alharbi, Mohammad D. Alahmadi},
journal={Mathematics 2025},
year={2025}
}
