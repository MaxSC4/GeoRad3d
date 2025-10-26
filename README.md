# 🧭 GeoRad3D
**3D Modelling and Visualization of Natural Radioactivity in Geological Outcrops**

---

## 🧪 Project Overview

**GeoRad3D** is a scientific research project dedicated to the **three-dimensional modelling of natural radioactivity** within geological outcrops.
It aims to bridge the gap between **surface radiometric measurements** and the **internal distribution of radionuclides** in the subsurface, using geostatistical interpolation and 3D visualization techniques.

The approach combines **field data, geostatistics, and 3D graphics** to generate volumetric models of radioactivity, offering a new way to explore, validate, and visualize radiometric signatures within geological structures.

Under the supervision of **B. Saint-Bezar** and **A. Benedicto** ([GEOPS](https://geops.geol.u-psud.fr/), [Université Paris-Saclay](https://www.universite-paris-saclay.fr/)).

---

## 🎯 Scientific Objectives

- **Model the 3D distribution of radioactivity** in rock outcrops using surface data.
- **Extrapolate in depth** (beneath the outcrop surface) using **3D ordinary kriging**, accounting for spatial correlations and variograms fitted to the data.
- **Quantify uncertainty** through kriging variance and provide validation metrics (ME, RMSE, MSSE, VSE). (*WIP*)
- **Visualize and explore** volumetric radioactivity in 3D (isosurfaces, slices, point data overlay).
- **Integrate geological meshes** (.obj models) to map radioactivity directly on 3D geometry via vertex coloring or texture baking.

The resulting models can be explored interactively, exported, and compared with geological interpretations, providing a physically consistent and reproducible framework for studying spatial heterogeneities of natural radioelements within rocks.

---

## ⚙️ Methodology Overview

1. **Data Preparation**
   - Import and cleaning of georeferenced radiometric points (`X, Y, Z, R`), where `R` is the measured radioactivity (in counts per minute).
   - Automatic computation of a bounding box surrounding the dataset.
   - Creation of a **regular 3D grid** inside this domain.

2. **3D Interpolation (Kriging)**
   - Ordinary Kriging in three dimensions using *PyKrige* (`OrdinaryKriging3D`).
   - User-selectable variogram model (`spherical`, `exponential`, `gaussian`, or `linear`).
   - Computation of both **estimated values** and **kriging variance** over the full 3D grid.

3. **Cross-Validation (optional)**
   - Leave-One-Out (LOO) validation to evaluate predictive performance.
   - Statistical indicators automatically computed:
     - Mean Error (ME)
     - Root Mean Square Error (RMSE)
     - Mean Standardized Squared Error (MSSE)
     - Variance of Standardized Errors (VSE)

4. **Visualization and Exploration**
   - **3D interactive visualization** of the interpolated volume:
     - Isosurfaces of radioactivity levels (semi-transparent solids)
     - Neon-colored measurement points
     - Mouse-over tooltips and click-labels with values
   - **2D depth slices** linked to a slider for exploring different Z-levels.
   - Gaussian smoothing and percentile-based color scaling for perceptual readability.

   *(see [CLI – `view`](docs/cli.md#command-view) for command usage)*

5. **Integration with 3D Meshes**  
   - Ability to load external `.obj` geological models (with `.mtl` if available).
   - “Painting” of the mesh surface according to interpolated radioactivity values:
     - **Vertex coloring** (RGB per vertex)
     - or **Texture baking** (UV → PNG + updated MTL).
   - The resulting mesh can be directly visualized in any 3D viewer or GIS.

   *(see [CLI – `paint`](docs/cli.md#command-paint))*

---

## 📘 Documentation

For a detailed guide on how to use the GeoRad3D command-line interface (CLI),  
see the [**CLI Documentation**](docs/cli.md).

This includes explanations and usage examples for:
- **3D interpolation (`fit`)**
- **Volume visualization (`view`)**
- **Mesh painting (`paint`)**
- **Z-slice animation (`gif`)**
- *(soon)* **Model alignment (`align-obj`)**

---

## 💡 Expected Outcomes

GeoRad3D provides a workflow to:
- Build **spatially coherent 3D models** of natural radioactivity;
- Evaluate the **internal structure of anomalies** within rock masses;
- Facilitate **visual and quantitative comparison** between radiometric data and geological features;
- Produce **publication-ready figures and 3D exports** for further interpretation or integration in GIS or modeling platforms.

By grounding surface radioactivity measurements into a volumetric and geostatistical context, the project contributes to a better understanding of radionuclide distribution processes in fractured and altered rocks, and to more realistic radiological mapping at the outcrop scale.

---

## 🏛️ Institutional Context

**GeoRad3D** is part of a research program conducted at **GEOPS (Géosciences Paris-Saclay, Université Paris-Saclay)**.

- **Supervisors:**
  - *[B. Saint-Bezar](https://orcid.org/0000-0003-0326-3976) (GEOPS)*
  - *[A. Benedicto](https://orcid.org/0000-0002-8222-2744) (GEOPS)*

- **Keywords:**
  `geostatistics`, `kriging`, `radioactivity`, `3D modeling`, `geological outcrop`, `visualization`
