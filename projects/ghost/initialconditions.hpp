/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#pragma once

__global__ void setInitialConditions(Mesh<GPU>::type u) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x,
            j = blockIdx.y * blockDim.y + threadIdx.y;
 
if (u.active(i, j)) {
    const real x = u.x(i), y = u.y(j);
    real density = 0.0, v_x = 0.0, v_y = 0.0, p = 0.0, lambda0 = 0, lambda1 = 0;

   // v_x -= V_CJ;
    if (x > X1 && x < AMP * sin (2.0*PI*y/LAMBDA) + X2) {
  //     if (x > X1 && x < X2) {
       p = pShock;
       density = rhoShock;
    } 
	
	else {
		p = 1.0;
		density = 1.0;			
	}			

    u(i, j, DENSITY)   = density;
    u(i, j, XMOMENTUM) = v_x * density;
    u(i, j, YMOMENTUM) = v_y * density;
    u(i, j, ENERGY)    = p / (GAMMA - 1.0) + 0.5 * density * (v_x * v_x + v_y * v_y);
    u(i, j, LAMBDA0)   = lambda0 * density;
    u(i, j, LAMBDA1)   = lambda1 * density;
	u(i, j, PMAX) = p;

  }

}

