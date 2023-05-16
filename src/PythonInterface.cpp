#include "StdAfx.h"
#include "pybind11/pybind11.h"
#include <pybind11/stl.h>
#include "SchrodingerBridge.h"

namespace py = pybind11;

PYBIND11_MODULE(SBTimeSeries, m) {

	py::class_<SchrodingerBridge>(m, "SchrodingerBridge")
		.def(py::init<long, long, vector<vector<double>>>())
		.def(py::init<long, long, long, vector<vector<vector<double>>>>())
		.def("SimulateKernel", &SchrodingerBridge::SimulateKernel)
		.def("SimulateKernelVectorized", &SchrodingerBridge::SimulateKernelVectorized);

}
