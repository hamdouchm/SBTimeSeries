#include "StdAfx.h"
#include "RandomGenerator.h"

RandomGenerator::RandomGenerator()
{
	gen = new mt19937{random_device()()};
}

RandomGenerator::~RandomGenerator()
{

}

void RandomGenerator::GaussianMT(vector<double>& Gaussian) {
	normal_distribution<double>* Gaussian_;
	Gaussian_ = new normal_distribution<>(0, 1);
	for (int i = 0; i < Gaussian.size(); ++i)
	{
		Gaussian[i] = (*Gaussian_)(*gen);
	}
}


void RandomGenerator::UniformMT(vector<double>& Uniform) {
	uniform_real_distribution<double>* Uniform_;
	Uniform_ = new uniform_real_distribution<>(0, 1);
	for (int i = 0; i < Uniform.size(); ++i)
	{
		Uniform[i] = (*Uniform_)(*gen);
	}
}