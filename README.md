# Rasterized Experimental Data: Vapor Advection Plumes for Open Research

This is the source code repository used in preprocessing the open access dataset
**Rasterized Experimental Data: Vapor Advection Plumes for Open Research**.  

In this repository you find the scripts used in the preprocessing steps, as well as some Jupyter notebooks for visualizing the recorded data.

### About the Dataset
This dataset contains timeseries and voxelmap reconstructions of the response of various gas sensors to a synthetic gas plume.
This plume was systematically scanned in a dense 3D grid pattern, complemented by additional sampling trajectories.  
The dataset allows direct comparison between low-cost metal oxide semiconductor gas sensors (MiCS-5524, MiCS-6814) and advanced photoionization detectors (PID-AH2), examining both their static and dynamic responses to the plume. 
Additionally, the industrial-like model landscape utilized enables the evaluation of gas dispersion models in complex settings under controlled wind conditions.
Finally, the data support the comparison and evaluation of novel sampling strategies for mobile robotic sensor systems for gas distribution mapping and source localization, providing more realistic experimental data compared to computational fluid dynamics simulations.
The experiments have been conducted in March 2023 in the DNW Low-Speed Tunnel (DNW LST, Marknesse, NL).


### Have a look at the interactive examples!  

| Jupyter Notebook Name          | View online                                                                                                                                                                                   |
|:-------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `notebooks/timeseries.ipynb`   | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/DLR-KN/red-vapor/blob/master/notebooks/timeseries.ipynb)   |
| `notebooks/spatial_data.ipynb` | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/DLR-KN/red-vapor/blob/master/notebooks/spatial_data.ipynb) |



### Access the full dataset on Zenodo!   
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16414472.svg)](https://doi.org/10.5281/zenodo.16414472)


### [▶ Watch our Timelapse Video of the Experiments](https://www.youtube.com/watch?v=L7GP5C-VqhM)
[![Red:Vapor - Rasterized Experimental Data:Vapor Advection Plumes for Open Research - YouTube](https://img.youtube.com/vi/L7GP5C-VqhM/maxresdefault.jpg)](https://www.youtube.com/watch?v=L7GP5C-VqhM)

## Usage
For running the interactive examples yourself, modifying the preprocessing code and exploring the dataset in its entirety,
please clone this repository.
You need Python set up on your machine, including a setup to execute Jupyter notebooks.

The scripts expect the dataset be placed and fully unzipped at `./dataset`. You should have the following directory structure:
```
.
├── dataset
│   ├── DNW_Data
│   ├── Fly-Through_Experiments
│   ├── Purging_Runs
│   ├── Sampling_Experiments
│   ├── experiments.pkl
│   ├── overview_table.csv
│   └── voxel_maps.pkl
└── ...
```

A good starting point are the interactive notebooks `timeseries.ipynb` and `spatial_data.ipynb`, 
as well as the notebook for exploring the entire set of timeseries, `all_timeseries.ipynb`.

To execute the full preprocessing chain, run the individual `.py` scripts in sequence, starting with `p00_*.py`. 

## Citing the Software or the Dataset
If you find this software or the dataset useful in an academic context, please consider citing us!  
A journal paper describing the dataset is currently in the review process. In the meantime, please cite the dataset directly:
```
@dataset{RedVapor,
  author       = {Hinsen, Patrick and
                  Wiedemann, Thomas and
                  Shutin, Dmitriy and
                  Lilienthal, Achim J.},
  title        = {Rasterized Experimental Data: Vapor Advection
                   Plumes for Open Research
                  },
  month        = jul,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.16414472},
  url          = {https://doi.org/10.5281/zenodo.16414472},
}
```
Modify the DOI if you want to refer to a specific version of the dataset.
