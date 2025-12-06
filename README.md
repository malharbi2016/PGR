\# Prompt-Guided Refinement: A Novel Technique for Improving Intervertebral Disc Semantic Labeling

\*\*Authors:\*\* Mohammed N. Alharbi \& Mohammad D. Alahmadi  

---

\## Overview

Accurate detection and semantic labeling of intervertebral discs (IVDs) in magnetic resonance imaging (MRI) is essential for diagnosing and managing spine-related disorders. While convolutional neural network (CNN)–based methods extract contextual information from MRI scans, they often fail to incorporate the geometric structure of the vertebral column—leading to anatomically inconsistent predictions, particularly in challenging regions of the spine.



To address this limitation, we introduce a \*\*Prompt-Guided Refinement (PGR)\*\* framework that integrates explicit geometric priors into the learning process. The proposed approach:

\- Learns a \*\*global spinal skeleton representation\*\* from the training dataset  

\- Encodes this representation as a \*\*geometric prompt\*\*  

\- Aligns predicted features with anatomical constraints via a \*\*prompt consistency loss\*\*  

\- Enforces vertebral structural relationships through a \*\*skeleton structural loss\*\*



Experiments conducted on multi-center datasets demonstrate that our model \*\*outperforms state-of-the-art techniques\*\* across both \*\*T1w\*\* and \*\*T2w\*\* MRI modalities. Incorporating geometric guidance significantly enhances the accuracy and structural consistency of IVD localization and semantic labeling.



---

\## Repository Structure

\\`\\`\\`

src/

│   main.py                 # Train and evaluate the model

│   create\_skeleton.py      # Generate skeleton and Gaussian prompt maps

│

└── pgrnet/                 # Prompt-guided regression network implementation



prepared\_data/

│   (Place processed dataset files here)

\\`\\`\\`



---

\## Setup Instructions



\### Step 0 — Download the Processed Dataset

Preprocessed training, validation, and test sets should be placed under:

\\`\\`\\`

prepared\_data/

\\`\\`\\`



The original dataset can be downloaded from:  

🔗 https://github.com/spine-generic/data-multi-subject



After downloading:

1\. Unzip the archive  

2\. Move all extracted contents into the `prepared\_data/` folder  



Your folder should include files such as:

\\`\\`\\`

prepared\_trainset\_t1

prepared\_valset\_t1

prepared\_testset\_t1

prepared\_trainset\_t2

...

\\`\\`\\`



---

\### Step 1 — Generate the Skeleton Prompt

Before training, compute the average spinal skeleton and Gaussian-encoded prompt map by running:

\\`\\`\\`

python src/create\_skeleton.py

\\`\\`\\`



This script automatically:

\- Extracts vertebral skeletons from training labels  

\- Computes normalized joint coordinates  

\- Generates a 2D Gaussian heatmap representation  

\- Saves results to:

\\`\\`\\`

prepared\_data/t1\_Skelet.npy

prepared\_data/t1\_Skelet\_gaussian\_heatmap.png

\\`\\`\\`



---

\### Step 2 — Train and Evaluate the Model

To train the prompt-guided IVD labeling model:

\\`\\`\\`

python src/main.py

\\`\\`\\`



The script will:

\- Train on the preprocessed dataset  

\- Evaluate performance  

\- Save logs, metrics, and model checkpoints based on configuration settings  



---

\## Citation

If you use this repository or build upon this work, please cite:



\\`\\`\\`bibtex

@article{alahmadi2024promptivd,

&nbsp; title={Prompt-Guided Refinement: A Novel Technique for Improving Intervertebral Disc Semantic Labeling},

&nbsp; author={Mohammed N. Alharbi and Mohammad D. Alahmadi},

&nbsp; journal={Mathematics},

&nbsp; year={2025}

}

\\`\\`\\`



