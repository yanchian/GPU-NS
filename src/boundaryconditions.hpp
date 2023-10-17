#pragma once

enum Directions {YDIRECTION, XDIRECTION};
enum Streams {UPSTREAM, DOWNSTREAM};

template<BoundaryConditions BCs, bool XDIR, bool downstream>
__global__ void setBoundaryConditionsKernel(Mesh<GPU>::type u);
/*
template<bool XDIR>
__global__ void setSpecialBoundaryConditionsKernel(Mesh<GPU>::type u);
*/
