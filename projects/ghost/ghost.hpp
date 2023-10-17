/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
//#define GHOST
#define REACTIVE
#pragma once
#include "core.hpp"
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <deque>
#include <queue>
#include <ctime>
#include <cmath>
#include <typeinfo>
#include <limits>
#include <stdio.h>
#include <stddef.h>
#include <signal.h>
#include "boost/thread.hpp"
#include "Matrix.hpp"
#include "MatrixOperations.hpp"

#define USE_GL

#define MATERIALS 2
#define DIMENSIONS 2
#define GC 2

#include "grid.hpp"

template<Processor P>
struct Mesh {
  typedef Mesh2D<real, P, NUMBER_VARIABLES> type;
};

template<Processor P>
struct LevelSet {
  typedef Mesh2D<real, P, 2> type;
};

StridedCell<real, NUMBER_VARIABLES> typedef Cell;

// Fluid properties /////////////////////////////////////////////////////

// Reynodls number
const real RE_NUM = 10000.0;

// Prandlt number 
const real PR_NUM = 1.0;

// Schmidt number 
const real SC_NUM = 1.0;

// Specific heat ratio
const real GAMMA = 1.32;

// Temperature relation
const real s = 0.7;

// Reference viscosity at initial temperature
const real mu0 = 1 / RE_NUM;


/////////////////////////////////////////////////////////////////////////

// Reaction model constants /////////////////////////////////////////////
/*
// Activation energy
const real Ea = 20.0;

// Pre-exponential factor
const real KR = 16.45;

// Dimensionless energy release
const real Q = 50.0;

// Chapman-Jouguet VoD
const real V_CJ = 6.8095;
*/

const real Q = 21.365;
const real TS = 5.0373;
const real EI = 5.414 * TS;
const real ER = 1.0 * TS;
const real KI = 1.0022;
const real KR = 4.0;

const real specific_heat_ratio = 1.32;

/////////////////////////////////////////////////////////////////////////

const real RHO_0 = 1.0;
const real P_0 = 1.0;
const real P_1 = 10*21.54;
const real T_wall = 1.0;
const real YCORNER = 20;
const real Wall_thickness = 1; 
const real PCJ = 21.54;
const real rho_CJ = 1.795;
const real pShock = 2.0 * PCJ;
const real rhoShock = 1.0 * rho_CJ;
const real pAir = 1.0;
const real rhoAir = 1.0;

const real X1 = 10;
const real X2 = 30;
const real RS = 26;

//const real X1 = 15.0;
//const real X2 = 17.5;
//const real RS = 20.0;
const real PI = 3.14159265359;
const real AMP = 5.0;           // Amplitude
const real LAMBDA = 100.0;       // Wavelength

const real Tol = 1.0e-8;

const real X_MID = 0.5;
real Check = 1.0;
const int Skip_lines = 5;
/////////////////////////////////////////////////////////////////
const real length_x = 0; //horizontal length of the cubes
const real length_y = 0; //vertical length of the cubes

const real number_x = 0;
const real number_y = 0;

const real start_x = 50;
const real start_y = 25;

const real space_x = 20;
const real space_y = 10;
/////////////////////////////////////////////////////////////////
__device__ __host__ __forceinline__ real mu() {
  return 1.0 / RE_NUM;
}

__device__ __host__ __forceinline__ real gamma(const Cell u) {
  return specific_heat_ratio;
}

__device__ __host__ __forceinline__ real gamma() {
  return specific_heat_ratio;
}

__device__ __host__ __forceinline__ real p0(const Cell u) {
  return 0.0;
  //return 0.87e8;
}
__device__ __host__ __forceinline__ real p0() {
  return 0.0;
  //return 0.87e8;
}

#include "boundaryconditions.hpp"
#include "flux.hpp"
#include "diffusionflux.hpp"
#include "wavespeed.hpp"
#include "find_minimum.hpp"
#include "HLLC.hpp"
#include "HLLE.hpp"
#include "Solver.hpp"
#include "initialconditions.hpp"
#include "render.hpp"
#include "opengl.hpp"
#ifdef GHOST
#include "ghostfluid.hpp"
#endif
#include "source.hpp"

#ifdef REACTIVE
#include "source.hpp"
#include "shockdetect.hpp"
#endif

struct ImageOutputs {
  std::string prefix;
  int plotVariable;
  ColourMode colourMode;
  real min;
  real max;
};

#include "SDF/BoundaryMesh.hpp"
#include "SDF/Polyhedron.cu"
#include "SDF/ConnectedEdge.cu"
#include "SDF/Edge.cu"
#include "SDF/Face.cu"
#include "SDF/Vertex.cu"
#include "SDF/ConnectedFace.cu"
#include "SDF/ConnectedVertex.cu"
#include "SDF/ScanConvertiblePolygon.cu"
#include "SDF/ScanConvertiblePolyhedron.cu"
#include "SDF/BoundaryMesh.cu"

#include "kernels/boundaryconditions.ipp"
#include "kernels/flux.ipp"
#include "kernels/diffusionflux.ipp"
#ifdef GHOST
#include "kernels/ghostfluid.ipp"
#endif
/*
#ifdef REACTIVE
#include "kernels/source.ipp"
#include "kernels/shockdetect.ipp"
#endif
*/
#include "kernels/source.ipp"
#include "kernels/HLLC.ipp"
#include "kernels/HLLE.ipp"
#include "kernels/wavespeed.ipp"
#include "kernels/find_minimum.ipp"
