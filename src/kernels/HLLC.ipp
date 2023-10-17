/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#include "HLLC.hpp"

template<bool X>
__device__ __host__ void starState(const Cell p, const real SS, const real SD, Cell star) {
  const int MOMENTUM = X ? XMOMENTUMFLUX : YMOMENTUMFLUX;
  const int TMOMENTUM = X ? YMOMENTUMFLUX : XMOMENTUMFLUX;
  const int VELOCITY = X ? XVELOCITY : YVELOCITY;
  const int TVELOCITY = X ? YVELOCITY : XVELOCITY;
  const real chi = (SD - p[VELOCITY]) / (SD - SS);
  star[DENSITY] = chi * p[DENSITY];
  star[MOMENTUM] = chi * p[DENSITY] * SS;
  star[TMOMENTUM] = chi * p[DENSITY] * p[TVELOCITY];
  star[ENERGY] = chi * (((p[PRESSURE] + gamma(p) * p0(p)) / (gamma(p) - 1.0) + 0.5 * p[DENSITY] * (p[XVELOCITY] * p[XVELOCITY] + p[YVELOCITY] * p[YVELOCITY])) + (SS - p[VELOCITY]) * (p[DENSITY] * SS + p[PRESSURE] / (SD - p[VELOCITY])));
#ifdef REACTIVE
  star[LAMBDA0] = chi * p[LAMBDA0];
  star[LAMBDA1] = chi * p[LAMBDA1];
#endif
  for (int i = CONSERVATIVE_VARIABLES; i < CONSERVATIVE_VARIABLES + NONCONSERVATIVE_VARIABLES; i++) {
    star[i] = p[i];
  }
}

template<bool X>
__device__ __host__ void riemannSolverHLLC(const Cell p_L, const Cell p_R, const real S, Cell solution, Cell temp) {
  const int VELOCITY = X ? XVELOCITY : YVELOCITY;
  //const int TVELOCITY = X ? YVELOCITY : XVELOCITY;

  const real cL = soundSpeedPrimitive(p_L), cR = soundSpeedPrimitive(p_R);
  const real uROE = (sqrt(p_L[DENSITY]) * p_L[VELOCITY] + sqrt(p_R[DENSITY]) * p_R[VELOCITY]) / (sqrt(p_L[DENSITY]) + sqrt(p_R[DENSITY]));
  const real cROE = (sqrt(p_L[DENSITY]) * cL + sqrt(p_R[DENSITY]) * cR) / (sqrt(p_L[DENSITY]) + sqrt(p_R[DENSITY]));

  const real SL = min(uROE - cROE, p_L[VELOCITY] - cL);
  const real SR = max(uROE + cROE, p_R[VELOCITY] + cR);

  const real SS = (p_R[PRESSURE] - p_L[PRESSURE] + p_L[DENSITY] * p_L[VELOCITY] * (SL - p_L[VELOCITY]) - p_R[DENSITY] * p_R[VELOCITY] * (SR - p_R[VELOCITY])) / (p_L[DENSITY] * (SL - p_L[VELOCITY]) - p_R[DENSITY] * (SR - p_R[VELOCITY]));
  if (SL >= S) {
    primitiveToFlux<X>(p_L, solution);
  } else if (SS >= S) {
    primitiveToFlux<X>(p_L, solution);
    starState<X>(p_L, SS, SL, temp);
    for (int k = 0; k < p_L.length(); k++) {
      solution[k] += SL * temp[k];
    }
    primitiveToConservative(p_L, temp);
    for (int k = 0; k < p_L.length(); k++) {
      solution[k] -= SL * temp[k];
    }
  } else if (SR >= S) {
    primitiveToFlux<X>(p_R, solution);
    starState<X>(p_R, SS, SR, temp);
    for (int k = 0; k < p_R.length(); k++) {
      solution[k] += SR * temp[k];
    }
    primitiveToConservative(p_R, temp);
    for (int k = 0; k < p_R.length(); k++) {
      solution[k] -= SR * temp[k];
    }
  } else {
    primitiveToFlux<X>(p_R, solution);
  }
}

template<bool X>
__device__ __host__ void riemannStateHLLC(const Cell p_L, const Cell p_R, const real S, Cell solution) {
  const int VELOCITY = X ? XVELOCITY : YVELOCITY;
  //const int TVELOCITY = X ? YVELOCITY : XVELOCITY;

  const real cL = soundSpeedPrimitive(p_L), cR = soundSpeedPrimitive(p_R);
  const real uROE = (sqrt(p_L[DENSITY]) * p_L[VELOCITY] + sqrt(p_R[DENSITY]) * p_R[VELOCITY]) / (sqrt(p_L[DENSITY]) + sqrt(p_R[DENSITY]));
  const real cROE = (sqrt(p_L[DENSITY]) * cL + sqrt(p_R[DENSITY]) * cR) / (sqrt(p_L[DENSITY]) + sqrt(p_R[DENSITY]));

  const real SL = min(uROE - cROE, p_L[VELOCITY] - cL);
  const real SR = max(uROE + cROE, p_R[VELOCITY] + cR);

  const real SS = (p_R[PRESSURE] - p_L[PRESSURE] + p_L[DENSITY] * p_L[VELOCITY] * (SL - p_L[VELOCITY]) - p_R[DENSITY] * p_R[VELOCITY] * (SR - p_R[VELOCITY])) / (p_L[DENSITY] * (SL - p_L[VELOCITY]) - p_R[DENSITY] * (SR - p_R[VELOCITY]));
  if (SL >= S) {
    solution = p_L;
  } else if (SS >= S) {
    starState<X>(p_L, SS, SL, solution);
    conservativeToPrimitiveInPlace(solution);
  } else if (SR >= S) {
    starState<X>(p_R, SS, SR, solution);
    conservativeToPrimitiveInPlace(solution);
  } else {
    solution = p_R;
  }
}

