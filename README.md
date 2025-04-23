spock: Volcano Plot Fitting Tool
==============================================

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rlaplaza/spock/HEAD?labpath=App.ipynb)
[![DOI](https://zenodo.org/badge/764582532.svg)](https://zenodo.org/doi/10.5281/zenodo.12804607)

![spock logo](./images/spock_logo.png)
[![PyPI version](https://badge.fury.io/py/navicat-spock.svg)](https://badge.fury.io/py/navicat-spock)

## Contents
* [About](#about-)
* [Install](#install-)
* [Examples](#examples-)
* [Citation](#citation-)

## About [↑](#about)

The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `statsmodels`

## Install [↑](#install)

You can install spock using pip:

```python
pip install navicat_spock
```

Afterwards, you can call spock as:

```python
python -m navicat_spock [-h] [-version] -i [FILENAMES] [-wp WP] [-v VERB] [-pm PLOTMODE] [-rng SEED] [-fa FA] [-is IMPUTER_STRAT] [--plot_all PREFIT] [--save_fig SAVE_FIG] [--save_csv SAVE_CSV]
```
or simply
```python
navicat_spock [-h] [-version] -i [FILENAMES] [-wp WP] [-v VERB] [-pm PLOTMODE] [-rng SEED] [-fa FA] [-is IMPUTER_STRAT] [--plot_all PREFIT] [--save_fig SAVE_FIG] [--save_csv SAVE_CSV]
```

Alternatively, you can download the package and execute:

```python 
python setup.py install
```

Afterwards, you can call spock as above. Options can be consulted using the `-h` flag in either case. The help menu is quite detailed. 

Note that the volcano plot generation function is directly exposed in case you want to use it in your code.

## Examples [↑](#examples)

The `manuscript_examples` subdirectory contains several examples used in the manuscript. Any of the data files can be run as:

```python
python -m navicat_spock -i [FILENAME]
```

We also provide a jupyter notebook where the plots are generated one by one.

The input of spock is a `pandas` compatible dataframe, which includes plain .csv and .xls files. 

Regarding format, spock expects headers for all columns. The first column must contain names/identifiers. The catalytic performance metric must include the word "target" in its header. The rest of the columns will be interpreted as potential descriptors.

High verbosity levels (`-v 1`, `-v 2`, etc.) will force spock to print out information about every step. This can be useful for understanding what the code is doing and what can have possibly gone wrong. To be as automated as possible, reasonable default values are set for most choices. 

The plotmode (`-pm 1`, `-pm 2`) option can be used to modify the default look of the generated pngs, including more detail as the plotmode level increases. 


## Citation [↑](#citation)

Please cite our work with the repository DOI and the manuscript that introduced the code. You can find it [here](https://doi.org/10.1021/acscatal.5c00412) and in the reference:

```
Suvarna, M.; Laplaza, R.; Graux, R.; López, N.; Corminboeuf, C.; Jorner, K.; Pérez-Ramírez, J. SPOCK Tool for Constructing Empirical Volcano Diagrams from Catalytic Data. ACS Catal. 2025, 7296–7307. https://doi.org/10.1021/acscatal.5c00412
```


---




