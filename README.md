# multimodal_active_learning_ccss
Multi-modal cold start active learning for compositionally complex solid solutions
This work addresses the cold-start problem in active learning (AL) for experimental materials science by leveraging multimodal priors—numerical (EDX), visual (wafer images), and textual (literature-derived embeddings)—to select diverse, informative initial measurement points. The goal is to train accurate surrogate models (Gaussian Processes) for predicting functional properties (e.g., resistance) with minimal costly measurements.


## Raw Data Description

###  `resistance_raw` folder has subfolders for each materials library used in the study and in inside the subfolders are:
#### 1. `Resistance` Data

- **Source**: Experimental electrical resistance measurements.
- **Format**: CSV file

  - Each measurement point (e.g., physical location on a wafer) has 30 repeated resistance measurements for robustness.
  - There are 342 measurement points, resulting in a total of 30 × 342 = 10,260 rows in the raw file.
  - The mean resistance for each measurement point is later computed and used as the prediction target in machine learning using the Resistance_mean.ipynb from the notebook. folder


#### 2. `EDX` Data

- **Source**: Energy Dispersive X-ray Spectroscopy for elemental composition.
- **Format**: CSV file
- **Contents**:
  - Elemental weight percentages (`Ag`, `Cu`, `Pt`, etc.)
  - Mapping to stage coordinates (`x`, `y`) or index

#### 3. `Wafer Photographs`

- **Source**:  images of wafer samples.
- **Format**: `.JPG` images

#### 4. `EDX_Similarity`
- ** Source: The similarity calculated using Matnexus for the composition and the word "resistance" using the Iterative_Corpus_Refinement.ipynb from the folder "matnexus/Example/Iterative_Corpus_Refinement.ipynb"
- **Format**: CSV file
---
Beside the subfolders for each materials library, there are the EDX_min_max_summary, which has the index of the measurement points of the wafer for the maximum and minimum compositions, Top5_Similarity_Summary which has the top 5 similarity indices for each material library; and the centroids_mapped_with_indices, which has the centroids resulting from clustering the photographs of the wafer for each materials library.

## clean Data Description
###  `resistance_cleaned_files` folder has all the cleaned resistance files for all the materials libraries. 
- **Format**: CSV file
- **Contents**:
  - Elemental weight percentages (`Ag`, `Cu`, `Pt`, etc.)
  - Mapping to stage coordinates (`x`, `y`) and ID
   - Resistance values
###  `init_choices` folder has all the initial selection indices files for all the materials libraries. 


## Running the Code from the Notebook
### Image-Clustering
This notebook performs clustering on wafer images (e.g. K-means) after preprocessing (contrast/saturation).
It extracts centroids from the clustered regions and maps them back to measurement point indices on the wafer grid using stage coordinates.


### initial_points
This notebook generates initial index JSON files per material library using:

Top-5 similarity indices (Top5_Similarity_Summary)

Max/Min composition points (EDX_min_max_summary)

Cluster centroids from wafer images (centroids_mapped_with_indices)

It also adds computed strategies:

Random, LHS, K-Means, FPS, ODAL, K-Center

All combined and saved to DATA_CLEAN_InIT_CHOICES as <folder>_indices.json.

### analysis 
This notebook runs the active learning for the two acquisition functions, 
Uncertainty-based acquisition using a standard GP model (GPBasic) and SAWEI (Similarity-Aware Expected Improvement) acquisition using a similarity-weighted GP model (GPSawei)

### Ploting
It plots the mean absolut error of the paper and the selected initial points (marked with black Xs) overlaid on the wafer grid.

### compare
This notebook calculates and plots all the heatmaps of the paper and the summary of the acquisition functions.



