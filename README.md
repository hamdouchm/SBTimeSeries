
Generative Modeling for Time Series Via Schrödinger Bridge 
==========================================================

This repo is the official code for our paper *Generative Modeling for Time Series Via Schrödinger Bridge*, available at https://papers.ssrn.com/abstract_id=4412434

## Quickstart
### Python
To get started, create a conda environment and install the required Python packages for this project using the following command:
```
conda create -name SBTimeSeries --file requirements.txt python=3.8.16
```
### Build C++ Code
#### Windows
To build the code on Windows, open the Visual Studio solution `SBTimeSeries.sln` and compile it.
#### Linux
For Linux users, you can build the solution by running the batch file `Build.sh` using the following command:
```
bash Build.sh
```
## Repository Structure
The repository is organized as follows:
- `src` directory contains the C++ code for SBTS diffusion..
- `deepHedging` directory contains the TensorFlow model for deep hedging and its data generator.
- `notebook` contains two Jupyter Notebooks files: `SBTSNumericalExperiments.ipynb` which allows you to reproduce all the numerical experiments presented in the paper and generates samples stored in a folder named data (created automatically) for deep hedging. `DeepHedging.ipynb` uses the generated samples to run deep hedging as described in the paper.
