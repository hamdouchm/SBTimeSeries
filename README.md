
Generative Modeling for Time Series Via Schrödinger Bridge 
==========================================================

This repo is the official code for our paper *Generative Modeling for Time Series Via Schrödinger Bridge*, available at https://papers.ssrn.com/abstract_id=4412434

## Quickstart
### Python
Create a conda environement and install python packages required for this project
```
conda create -name SBTimeSeries --file requirements.txt python=3.8.16
```
### Build C++ Code
#### Windows
The code can be build by opening visual studio solution `SBTimeSeries.sln` and compile it.
#### Linux
For linux users, you can build the solution by running the batch file `Build.sh`:
```
bash Build.sh
```
## Structure

This repo is organized as follows:
- `src` contains C++ code to diffuse SBTS.
- `deepHedging` contains tensorflow model for deep hedging and its data generator.
- `notebook` contains two Jupyter Notebooks files: `SBTSNumericalExperiments.ipynb` where you can reproduce all the numerical experiments displayed in the paper, and it generates samples stored in a folder `data` (created automatically) for deep hedging. `DeepHedging.ipynb` uses generated samples to run deep hedging as presented in the paper.
