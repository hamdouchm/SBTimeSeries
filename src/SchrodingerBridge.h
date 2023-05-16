#pragma once
#include "StdAfx.h"
#include "RandomGenerator.h"

class SchrodingerBridge
{
public:
	SchrodingerBridge(long distSize_, long nbpaths_, vector<vector<double>> timeSeriesData_);
	SchrodingerBridge(long distSize_, long nbpaths_, long dimension_, vector<vector<vector<double>>> timeSeriesDataVector_);
	~SchrodingerBridge();

	//SBTS for dimension 1
	vector<double> SimulateKernel(long nbStepsPerDeltati, double H, double deltati);

	//SBTS for dimension > 1
	vector<vector<double>> SimulateKernelVectorized(long nbStepsPerDeltati, double H, double deltati);

private:

	//Kernel function used for kernel regression
	double kernel(double x, double H);

	//Method to generate diffusion calendar
	void schedule(vector<double>& timeEuler, double maturity, double timestep);

	//Number of time steps of the time series (denoted N in the paper)
	long distSize;

	//Number of samples (denoted M in the paper)
	long nbpaths;

	//Time series samples
	vector<vector<double>> timeSeriesData; //dimension 1
	vector<vector<vector<double>>> timeSeriesDataVector; //dimension > 1

	//Generated time series via SB
	vector<double> timeSeries; //dimension 1
	vector<vector<double>> timeSeriesVector; //dimension > 1

	//Brownian increments
	vector<double> Brownian;

	//Diffusion Calendar (t_i,t_{i+1})
	vector<double> vtimestepEuler;

	//Kernel regression weights
	vector<double> weights;
	vector<double> weights_tilde;
	
	//Time series dimension
	long dimension;

	//Drift, numerator and denominator
	double drift;
	double expecY;
	double expecX;
};

