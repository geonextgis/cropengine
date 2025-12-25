# Welcome to cropengine

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/geonextgis/cropengine/blob/main)
[![Open in Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geonextgis/cropengine/main?labpath=notebooks%2Fintro.ipynb)
[![Open In Studio Lab](https://studiolab.sagemaker.aws/studiolab.svg)](https://studiolab.sagemaker.aws/import/github/geonextgis/cropengine/blob/main/notebooks/intro.ipynb)
[![PyPI Version](https://img.shields.io/pypi/v/cropengine.svg)](https://pypi.org/project/cropengine)
[![Downloads](https://static.pepy.tech/badge/cropengine)](https://pepy.tech/project/cropengine)
[![Documentation Status](https://github.com/geonextgis/cropengine/workflows/docs/badge.svg)](https://geonextgis.github.io/cropengine)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">
  <a href="https://geonextgis.github.io/cropengine">
    <img src="https://raw.githubusercontent.com/geonextgis/cropengine/main/docs/assets/logo.png" alt="logo" width="250"/>
  </a>
</div>

**A Python package for streamlining process-based crop modeling and simulation**

- GitHub repo: <https://github.com/geonextgis/cropengine>
- Documentation: <https://geonextgis.github.io/cropengine>
- PyPI: <https://pypi.org/project/cropengine>
- Notebooks: <https://github.com/geonextgis/cropengine/tree/main/docs/examples>
- License: [MIT](https://opensource.org/licenses/MIT)

---

## Introduction

**cropengine** is a Python package designed to bridge the gap between geospatial data and process-based crop modeling. It streamlines the complex workflows involved in preparing input data, configuring simulation parameters, and executing crop models for yield prediction and agricultural research.

While traditional crop modeling often requires extensive manual data preparation and file manipulation, **cropengine** automates these tasks. It is built to integrate seamlessly with geospatial workflows (such as those using `geeagri`), allowing users to easily drive simulations with site-specific weather, soil, and management data.

**cropengine** is ideal for:

- Agronomists and researchers running point-based or spatial crop simulations.
- Data scientists integrating biophysical models with machine learning pipelines.
- Developers building agricultural decision support systems.

For a complete list of examples and use cases, visit the [notebooks](https://github.com/geonextgis/cropengine/tree/main/docs/examples) section.

---

## Key Features

* **Automated Data Preparation** — Streamline the formatting of weather, soil, and management data into model-ready structures.
* **Simulation Management** — Easily configure and run process-based crop simulations with a Pythonic API.
* **Geospatial Integration** — Connect directly with satellite and climate data sources to drive simulations for specific locations (lat/lon) or regions.
* **Scalable Workflows** — specialized tools for running batch simulations across multiple sites or growing seasons efficiently.
* **Result Analysis** — Built-in utilities to parse simulation outputs, calculate yield gaps, and visualize crop growth dynamics over time.
* **Model Agnostic Design** — Designed to support various crop modeling engines and frameworks through a unified interface.

---

## Installation

```bash
conda create -n cropengine python=3.10
conda activate cropengine
pip install cropengine
# (Optional) Upgrade to the latest version if already installed
pip install --upgrade cropengine
```

---

## Model

| Model ID | Model Name | Description | Production Level | Water Balance | Nutrient Balance |
|---------|-----------|-------------|------------------|---------------|------------------|
| Wofost72_Phenology | WOFOST 7.2 (Phenology Only) | Simulates only the phenological development stages of the crop, ignoring biomass growth. | Phenology | N/A | N/A |
| Wofost72_PP | WOFOST 7.2 (Potential Production) | Simulates crop growth under potential production conditions (no water or nutrient stress). | Potential | N/A | N/A |
| Wofost72_WLP_CWB | WOFOST 7.2 (Water-Limited) | Simulates crop growth limited by water availability using the Classic Water Balance (free drainage). | Water-limited | Classic | N/A |
| Wofost73_PP | WOFOST 7.3 (Potential Production) | Includes atmospheric CO₂ response and biomass reallocation under potential conditions. | Potential | N/A | N/A |
| Wofost73_WLP_CWB | WOFOST 7.3 (Water-Limited, Classic) | Includes CO₂ response and biomass reallocation under water-limited conditions using the Classic Water Balance. | Water-limited | Classic | N/A |

---