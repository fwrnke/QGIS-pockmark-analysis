# Pockmark analysis

QGIS processing scripts for pockmark analysis using

1. **Pockmark delineation**
2. **Pockmark characterization**

## 1) Pockmark delineation
Delineate pockmarks (morphological depressions) on DEM (digital elevation model) using Bathymetric Position Index (BPI) and user-specified thresholds.

Input:
- DEM (e.g., bathymetry, seismic horizon)
- calculates BPI (Bathymetric Position Index)

Output:
- pockmark polygons

## 2) Pockmark characterization
Characterize pockmark polygons using computed statistical attributes and (optionally) extract centroids and local minima within pockmark polygons.

Input:
- DEM
- pockmark polygons

Output:
- pockmark polygons (with statistical attributes)
- pockmark centroids
- pockmark local minima

## Acknowledgements
These scripts are inspired by the [`CoMMa`](https://github.com/ricarosio/CoMMa) ArcGIS Pro Toolbox provided semi-automatic mapping of morphological features.

## Reference
Supplementary information for the following publication:

```
to be added
```
