#pragma once
#include <random>

class RandomGenerator
{
public:
	RandomGenerator();
	~RandomGenerator();

	void  GaussianMT(vector<double>& Gaussian);
	void  UniformMT(vector<double>& Uniform);

private:

	mt19937* gen;
};

