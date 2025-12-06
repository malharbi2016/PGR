# Prompt-Guided Refinement: A Novel Technique for Improving Intervertebral Disc Semantic Labeling

**Authors:** Mohammed N. Alharbi & Mohammad D. Alahmadi  

---

## Overview

Accurate detection and semantic labeling of intervertebral discs (IVDs) in magnetic resonance imaging (MRI) is essential for diagnosing and managing spine-related disorders. While convolutional neural network (CNN)–based methods extract contextual information from MRI scans, they often fail to incorporate the geometric structure of the vertebral column—leading to anatomically inconsistent predictions, particularly in challenging regions of the spine.

To address this limitation, we introduce a **Prompt-Guided Refinement (PGR)** framework that integrates explicit geometric priors into the learning process. The proposed approach:

- Learns a **global spinal skeleton representation** from the training dataset  
- Encodes this representation as a **geometric prompt**  
- Aligns predicted features with anatomical constraints via **prompt consistency loss**  
- Enforces vertebral structural relationships through a **skeleton structural loss**

Experiments conducted on multi-center datasets demonstrate that our model **outperforms state-of-the-art techniques** across both **T1w** and **T2w** MRI modalities. Incorporating geometric guidance significantly improves the accuracy and structural consistency of IVD localization and semantic labeling.

---

## Repository Structure

```text
src/
│   main.py                 # Train and evaluate the model
│   create_skeleton.py      # Generate skeleton and Gaussian prompt maps
│
└── pgrnet/                 # Prompt-guided regression network implementation

prepared_data/
│   (Place processed dataset files here)
