# Face Project for Issa Lab

This repository contains code and notebooks for a face processing project conducted in the Issa Lab. The project involves analyzing face representations using various computational models and comparing them with neural data.

## Getting Started

1. Download the data and pre-trained models:

   - Access the data and saved models from [Google Drive](https://drive.google.com/drive/folders/1_xLVSTu8r6WDh5yp5LYIVN67J_vRYmZV?usp=drive_link)

2. Data Processing (`MyDataProcessing.ipynb`):

   - Crop images: Remove extra empty pixels from images generated using mkturk, based on alpha values.
   - Split data into training and validation sets.

3. Neural Data Analysis (`MyDorisNeuralData.ipynb`):

   - Compare model representations with neural data from the paper: ["Explaining face representation in the primate brain using different computational models"](<https://www.cell.com/current-biology/fulltext/S0960-9822(21)00527-3>)
   - Utilizes regression and Representational Similarity Analysis (RSA).

4. Scene File Editing (`MyEditScenefile.ipynb`):
   Edit the scene file template to support:

   - Basel face meshes, each with 50k samples
   - Basel face meshes with naturalistic distribution (50k samples each)
   - 8 identities for computing i2
   - Out-of-distribution (OOD) Basel meshes test
   - Rendered images of the ideal 3D observer

5. Depth Map Generation (`MyGetDepthMap.ipynb`):

   - Generate depth map images from meshes

6. Ideal 3D Observer Analysis (`MyIdeal3d.ipynb`):

   - Use images generated from the ideal 3D renderer to compute performance and i1
   - Compare results with biological data

7. Main Analysis (`MyMain.ipynb`):

   - Extract model features
   - Compute model i1
   - Compare with biological i1

8. Regression Analysis (`MyRegAnalysis.ipynb`):

   - Use latent variables (vbsli\*) to predict model or biological i1

9. RDM Analysis (`MyRDMAnalysis.ipynb`)

   - Generate representational dissimilarity matrices (RDMs)
   - Run model RDM comparison with the face space distance matrix (`scaledCoordsMax_all_updated.mat`)

10. PNG rdm files
      - Folder of RDMs of various models computed across 84 face categories, 50 images per category
      - `rdm/vbsl50_registered_avgpool` contains .npy files
      - `rdm/vbsl50_registered_figure_20250528` contains .png figures
      
