/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#include "source.hpp"

template<int blockDimX, int blockDimY>
__global__ void sources(Mesh<GPU>::type u, const real dt) {
	const int dimX = blockDimX,
	          dimY = blockDimY;
	const int i = dimX * blockIdx.x + threadIdx.x;
	const int j = dimY * blockIdx.y + threadIdx.y;
        const real x = u.x(i);
        const real y = u.y(j);

   if (u.active(i, j)) {
	real c[NUMBER_VARIABLES];
	real p[NUMBER_VARIABLES];
	for (int k = 0; k < NUMBER_VARIABLES; k++) {
	    c[k] = u(i, j, k);
	}
	
    conservativeToPrimitive(c, p);

    const real R = 1.0;
    const real T = p[PRESSURE] / (p[DENSITY] * R);
    const real LA0 = c[LAMBDA0] / p[DENSITY];
    const real LA1 = c[LAMBDA1] / p[DENSITY];

    if (p[PRESSURE] > 1.0 + Tol && LA1 < 1.0-Tol) {
        
        // 2-step induciton-reaction kinetics
        if (LA0 < 1.0-Tol) {
          const real Omega_I = p[DENSITY] * KI * exp(EI * (1.0/TS - 1.0/T));
          c[LAMBDA0] += dt * Omega_I;
        } else {
          const real Omega_R = p[DENSITY] * KR * (1.0 - LA1) * exp(-ER / T);
          c[LAMBDA1] += dt * Omega_R;
          c[ENERGY] += Q * dt * Omega_R;
        }
      /*
        // Single-step Arrhenius kinetics
          const real Omega_R = p[DENSITY] * KR * (1.0 - LA1) * exp(-Ea / T);
          c[LAMBDA1] += dt * Omega_R;
          c[ENERGY] += Q * dt * Omega_R;
		  */
    }

    for (int k = 0; k < NUMBER_VARIABLES; k++) {
	u(i, j, k) = c[k];
    }
  }
}

