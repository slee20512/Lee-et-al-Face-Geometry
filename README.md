# Lee-et-al-Face-Geometry

This document provides a high-level overview of the combined codebase supporting the Lee et al. face project. The repository is organized into two main sub-projects: **`proj-face`** (model training, testing, and analysis) and **`apple_facestim_generation`** (stimulus generation from face capture). Pre-trained models, data, and rendered stimuli are available via Google Drive (see links below).

---

## Repository Structure
```
.
├── proj-face/                         # Model training, testing, RDM analysis
├── apple_facestim_generation/         # Stimulus generation via Apple ARKit
├── scenefiles/                        # JSON scene files for model test stimuli generation
├── human_behav_scenefiles/            # JSON scene files for human behavioral test stimuli generation
└── apple_facestim_generation/
    └── scaledCoordsMax_all_updated.mat  KEY FILE: all face identity & expression vertex matrices
```

---

## Data & Stimuli (Google Drive)

**[Google Drive — Models, Data & Stimuli](https://drive.google.com/drive/folders/1FgQFqIWUn6qi4sW2iZhHRol0H-tS6EmI?usp=sharing)**

This Drive contains:
- `test_img/` — Canonical testing images for each model task
- `human_behav_test_img/` — All stimuli used in behavioral testing

Additional model checkpoints and data (used in `proj-face`):
**[Google Drive — proj-face data & saved models](https://drive.google.com/drive/folders/1_xLVSTu8r6WDh5yp5LYIVN67J_vRYmZV?usp=drive_link)**

---

## Sub-Project 1: `proj-face` — Model Training, Testing & Analysis

This sub-project contains all notebooks and scripts for training and evaluating face processing models, generating representational dissimilarity matrices (RDMs), and comparing model representations against neural and human behavioral data.

### Key Notebooks

| Notebook | Description |
|---|---|
| `MyDataProcessing.ipynb` | Crop images (remove empty pixels from mkturk-generated images), split into train/val sets |
| `MyDorisNeuralData.ipynb` | Compare model representations to neural data from [Lee et al. 2021](https://www.cell.com/current-biology/fulltext/S0960-9822(21)00527-3) via regression and RSA |
| `MyEditScenefile.ipynb` | Edit scene file templates for Basel face meshes, naturalistic distributions, OOD test sets, and ideal 3D observer renders |
| `MyGetDepthMap.ipynb` | Generate depth map images from 3D meshes |
| `MyIdeal3d.ipynb` | Compute performance and i1 from ideal 3D observer images; compare with biological data |
| `MyMain.ipynb` | Extract model features, compute model i1, compare with biological i1 |
| `MyRegAnalysis.ipynb` | Use latent variables (`vbsli*`) to predict model or biological i1 via regression |
| `MyRDMAnalysis.ipynb` | Generate RDMs and compare model RDMs to the face space distance matrix (`scaledCoordsMax_all_updated.mat`) |

### RDM Output Files (`rdm/`)

Pre-computed RDMs are available for various models computed across 84 face categories × 50 images per category:

- `rdm/vbsl50_registered_avgpool/` — `.npy` RDM files
- `rdm/vbsl50_registered_figure_20250528/` — `.png` RDM figures

---

## Sub-Project 2: `apple_facestim_generation` — Stimulus Generation via Apple ARKit

This sub-project contains MATLAB code (and one Swift iPhone app) for capturing 3D face meshes and generating face stimuli by performing mesh "surgery" on a blank head model.

### Key Data File

**`apple_facestim_generation/scaledCoordsMax_all_updated.mat`**

> This is the central data file for the entire stimulus set. It contains the vertex matrices summarizing **all face identities and expressions** used in the project. Both model and behavioral stimuli are derived from this file. Start here if you need to understand or reproduce any stimuli.

### Pipeline Overview (5 Steps)

**Step 1 — Capture face mesh (iPhone + ARKit)**
Use the `FaceDataRecorder.xcodeproj` Swift app on an iPhone X to capture a 3D face mesh: 1220 vertices (x,y,z), 2304 triangular faces, and 52 blendshapes. Record ~1 minute of varied expressions. Parse the output with `faceData_readLog.m`.

**Step 2 — Fit linear blendshape-to-vertex mapping**
Run `LinRegressBeta.m` to compute a ridge regression (Y = Xb) mapping 52 blendshapes to vertex coordinates. Pre-computed betas are available as `BETA.mat`.

**Step 3 — Segment face parts**
Identify vertex indices for eyes, nose, and mouth (using blendshape beta magnitudes or manual selection). Reference data provided in `FaceParts_example.mat`.

**Step 4 — Mesh surgery: stitch parts to blank head**
Place segmented face parts onto a blank head model using 3D rigid transforms, boundary optimization, Delaunay triangulation stitching, and Laplacian smoothing.

**Step 5 — Animate faces**
Apply betas from Step 2 to generate faces with arbitrary expressions. See `StimGenExample.m` for a full worked example. Texture mapping from captured photos is also supported via `Texturemapping.m`.

---

## Scene Files

| Folder | Contents |
|---|---|
| `scenefiles/` | JSON scene files used to render **model test stimuli** |
| `human_behav_scenefiles/` | JSON scene files used to render **human behavioral test stimuli** |

These scene files drive the rendering pipeline and reference face meshes built from `scaledCoordsMax_all_updated.mat`.

---

## Dependencies

- **MATLAB** — all stimulus generation code (Steps 2–5)
- **Swift / Xcode** — iPhone face capture app (Step 1)
- **Python** — model training and analysis notebooks in `proj-face`
- **Apple ARKit** — required for face mesh capture on iPhone X or later
