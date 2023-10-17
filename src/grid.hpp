#pragma once
#include <algorithm>
#include <iostream>
#include <ctime>
#include <cstdio>
#include <string.h>
#include <string>
#include "Vector.hpp"

double typedef real;

Vector<real, 2> typedef Vec;

const int NUMBER_VARIABLES = 8;
const int GHOST_CELLS = 2;
const int CONSERVATIVE_VARIABLES = 6;
const int NONCONSERVATIVE_VARIABLES = 0;
const int DUMMY_VARIABLES = 0;
enum ConservedVariables {
  DENSITY, XMOMENTUM, YMOMENTUM, ENERGY,  LAMBDA0, LAMBDA1, ISSHOCK, PMAX
/*
#ifdef GHOST
  PHI, FRAC,
#endif
#ifdef REACTIVE
  LAMBDA0, LAMBDA1, ISSHOCK,
#endif
  PMAX
  */
};
enum FluxVariables {DENSITYFLUX, XMOMENTUMFLUX, YMOMENTUMFLUX, ENERGYFLUX};
enum PrimitiveVariables {DENSITYPRIM, XVELOCITY, YVELOCITY, PRESSURE};
enum BoundaryConditions {REFLECTIVE, TRANSMISSIVE};

#include "Timer.hpp"
#include "StridedArray.hpp"
#include "Grid2.hpp"
#include "Mesh2.hpp"

