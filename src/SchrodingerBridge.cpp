#include "SchrodingerBridge.h"

SchrodingerBridge::SchrodingerBridge(long distSize_, long nbpaths_, vector<vector<double>> timeSeriesData_) : distSize(distSize_), nbpaths(nbpaths_)
{
	//Clear and resize vectors

	timeSeriesData.clear();
	timeSeriesData.resize(nbpaths_);
	for (long particle = 0; particle < nbpaths_; ++particle) {
		timeSeriesData[particle].resize(distSize_ + 1);
		for (long interval = 0; interval <= distSize_; ++interval) {
			timeSeriesData[particle][interval] = timeSeriesData_[particle][interval];
		}
	}

	timeSeries.clear();
	timeSeries.resize(distSize_+1);
	timeSeries[0] = timeSeriesData[0][0];

	weights.clear();
	weights.resize(nbpaths_);

	weights_tilde.clear();
	weights_tilde.resize(nbpaths_);
}

SchrodingerBridge::SchrodingerBridge(long distSize_, long nbpaths_, long dimension_, vector<vector<vector<double>>> timeSeriesDataVector_) : distSize(distSize_), nbpaths(nbpaths_), dimension(dimension_)
{
	//Clear and resize vectors
	timeSeriesDataVector.clear();
	timeSeriesDataVector.resize(nbpaths_);
	for (long particle = 0; particle < nbpaths_; ++particle) {
		timeSeriesDataVector[particle].resize(distSize_ + 1);
		for (long interval = 0; interval <= distSize_; ++interval) {
			timeSeriesDataVector[particle][interval].resize(dimension_);
			for (long i = 0; i < dimension_; ++i) {
				timeSeriesDataVector[particle][interval][i] = timeSeriesDataVector_[particle][interval][i];
			}
		}
	}

	timeSeriesVector.clear();
	timeSeriesVector.resize(distSize_ + 1);
	for (long interval = 0; interval <= distSize_; ++interval) {
		timeSeriesVector[interval].resize(dimension_);
	}

	for (long i = 0; i < dimension_; ++i) {
		timeSeriesVector[0][i] = 0.;
	}
	
	weights.clear();
	weights.resize(nbpaths_);

	weights_tilde.clear();
	weights_tilde.resize(nbpaths_);
}

SchrodingerBridge::~SchrodingerBridge() {

}

vector<double> SchrodingerBridge::SimulateKernel(long nbStepsPerDeltati, double H, double deltati) {

	//Diffusion calendar
	vtimestepEuler.clear();
	double timestepEuler = deltati / (double)(nbStepsPerDeltati);
	schedule(vtimestepEuler, deltati, timestepEuler);

	//Generate Brownian Increments
	Brownian.clear();
	Brownian.resize(distSize * (vtimestepEuler.size() - 1));
	RandomGenerator Rnd;
	Rnd.GaussianMT(Brownian);

	//Simulation
	double X_ = timeSeries[0];
	double timeprev;
	double timestep;
	long index_ = 0;
	double termtoadd;

	for (long interval = 0; interval < distSize; ++interval) {

		for (long particle = 0; particle < timeSeriesData.size(); ++particle) {

			if (interval == 0){
				weights[particle] = (1. / static_cast<double>(nbpaths));
			}
			else {
				weights[particle] *= kernel(timeSeriesData[particle][interval] - X_, H);
			}

			weights_tilde[particle] = weights[particle] * exp((timeSeriesData[particle][interval + 1] - X_) * (timeSeriesData[particle][interval + 1] - X_) / (2. * deltati));
		}

		for (long nbtime = 0; nbtime < vtimestepEuler.size() - 1; ++nbtime) {

			expecY = 0.0;
			expecX = 0.0;
			timeprev = vtimestepEuler[nbtime];
			timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime];

			for (long particle = 0; particle < timeSeriesData.size(); ++particle) {
				if (nbtime == 0) {
					expecX += weights[particle];
					expecY += weights[particle] * (timeSeriesData[particle][interval+1] - X_);
				}
				else {
					termtoadd = -(timeSeriesData[particle][interval + 1] - X_) * (timeSeriesData[particle][interval + 1] - X_) / (2. * (deltati - timeprev));
					termtoadd = weights_tilde[particle] * exp(termtoadd);
					expecX += termtoadd;
					termtoadd *= (timeSeriesData[particle][interval + 1] - X_);
					expecY += termtoadd;
				}
			}

			drift = (expecX > 0.0) ? (1. / (deltati - timeprev)) * (expecY / expecX) : 0.0;
			X_ += drift * timestep + Brownian[index_] * sqrt(timestep);
			index_++;
		}

		timeSeries[interval + 1] = X_;
	}
	return timeSeries;
}

vector<vector<double>> SchrodingerBridge::SimulateKernelVectorized(long nbStepsPerDeltati, double H, double deltati) {

	//Diffusion calendar
	vtimestepEuler.clear();
	double timestepEuler = deltati / nbStepsPerDeltati;
	schedule(vtimestepEuler, deltati, timestepEuler);

	//Generate BrownianIncrement
	Brownian.clear();
	Brownian.resize(distSize * dimension * (vtimestepEuler.size() - 1));
	RandomGenerator Rnd;
	Rnd.GaussianMT(Brownian);

	//Simulation
	vector<double> X_;
	X_.clear();
	X_.resize(dimension);
	for (long i = 0; i < dimension; ++i) {
		X_[i] = 0.;
	}

	double timeprev;
	double timestep;
	double timestepsqrt;
	long index_ = 0;
	double termtoadd;
	vector<double> numerator;
	numerator.clear();
	numerator.resize(dimension);

	for (long interval = 0; interval < distSize; ++interval) {

		#pragma omp parallel for
		for (long particle = 0; particle < timeSeriesDataVector.size(); ++particle) {
			if (interval == 0) {
				weights[particle] = (1. / static_cast<double>(nbpaths));
			}
			else {
				for (long i = 0; i < dimension; ++i) {
					weights[particle] *= kernel(timeSeriesDataVector[particle][interval][i] - X_[i], H);
				}
			}
			weights_tilde[particle] = weights[particle];
			for (long i = 0; i < dimension; ++i) {
				weights_tilde[particle] *= exp((timeSeriesDataVector[particle][interval + 1][i] - X_[i]) * (timeSeriesDataVector[particle][interval + 1][i] - X_[i]) / (2. * deltati));
			}		
		}

		for (long nbtime = 0; nbtime < vtimestepEuler.size() - 1; ++nbtime) {

			for (long i = 0; i < dimension; ++i) {
				numerator[i] = 0.;
			}
			expecX = 0.0;
			timeprev = vtimestepEuler[nbtime];
			timestep = vtimestepEuler[nbtime + 1] - vtimestepEuler[nbtime];

			#pragma omp parallel for
			for (long particle = 0; particle < timeSeriesDataVector.size(); ++particle) {
				if (nbtime == 0) {
					expecX += weights[particle];
					for (long i = 0; i < dimension; ++i) {
						numerator[i] += weights[particle] * (timeSeriesDataVector[particle][interval + 1][i] - X_[i]);
					}
				}
				else {
					termtoadd = 0.;
					for (long i = 0; i < dimension; ++i) {
						termtoadd += (timeSeriesDataVector[particle][interval + 1][i] - X_[i]) * (timeSeriesDataVector[particle][interval + 1][i] - X_[i]);
					}
					
					termtoadd = weights_tilde[particle] * exp(-termtoadd / (2. * (deltati - timeprev)));
					expecX += termtoadd;

					for (long i = 0; i < dimension; ++i) {
						numerator[i] += termtoadd * (timeSeriesDataVector[particle][interval + 1][i] - X_[i]);
					}
				}
			}
			timestepsqrt = sqrt(timestep);
			for (long i = 0; i < dimension; ++i) {
				drift = (expecX > 0.0) ? (1. / (deltati - timeprev)) * (numerator[i] / expecX) : 0.0;
				X_[i] += drift * timestep + Brownian[index_] * timestepsqrt;
				index_++;
			}
		}
		for (long i = 0; i < dimension; ++i) {
			timeSeriesVector[interval + 1][i] = X_[i];
		}
	}
	return timeSeriesVector;
}

double SchrodingerBridge::kernel(double x, double H) {
	if (abs(x) < H) {
		return (H*H - x * x) * (H*H - x * x);
	}
	else {
		return 0.0;
	}
}


void SchrodingerBridge::schedule(vector<double>& timeEuler, double maturity, double timestep) {

	double time_ = 0.0;
	while (time_ < maturity) {
		timeEuler.push_back(time_);
		time_ += timestep;
	}
	timeEuler.push_back(maturity);
}