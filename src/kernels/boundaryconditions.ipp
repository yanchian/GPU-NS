/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#include "boundaryconditions.hpp"

template<BoundaryConditions BCs, bool XDIR, bool downstream>
__global__ void setBoundaryConditionsKernel(Mesh<GPU>::type u) {
  const bool YDIR = !XDIR, upstream = !downstream;

  const int i = blockIdx.x * blockDim.x + threadIdx.x - u.ghostCells();
  
real p[NUMBER_VARIABLES];
real q[NUMBER_VARIABLES];
real p1[NUMBER_VARIABLES];
real p2[NUMBER_VARIABLES];
real p3[NUMBER_VARIABLES];
real p4[NUMBER_VARIABLES];

  if (XDIR && u.exists(i, 0)) {
    for (int k = 0; k < NUMBER_VARIABLES; k++) {
      if (downstream) {
	//no-slip wall  
       //u(i, -2, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, 1, k);
       //u(i, -1, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, 0, k); 
	   //real p[NUMBER_VARIABLES];
   
	//no-slip isothermal wall	
		u(i, -2, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, 1, k);
        u(i, -1, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, 0, k); 		
		conservativeToPrimitive(u(i, 1), p);	 
		conservativeToPrimitive(u(i, 0), q);

		p[DENSITY] = p[PRESSURE]/T_wall;			
		q[DENSITY] = q[PRESSURE]/T_wall;	
		u(i, -2, DENSITY) = p[DENSITY] ;
		u(i, -1, DENSITY) = q[DENSITY] ;	
		/*
		conservativeToPrimitive(u(i, 1), p);	 
		conservativeToPrimitive(u(i, 0), q);

		p[DENSITY] = p[PRESSURE]/T_wall;			
		q[DENSITY] = q[PRESSURE]/T_wall;	
		u(i, -2, DENSITY) = p[DENSITY] ;
		u(i, -1, DENSITY) = q[DENSITY];	
		u(i, 1, DENSITY) = p[DENSITY] ;
		u(i, 0, DENSITY) = q[DENSITY];	
		*/
		//u(i, 0, DENSITY) = p[DENSITY] ;	

    //    u(i, -2, k) = (k == YMOMENTUM ? -1.0 : 1.0) * u(i, 1, k);
    //    u(i, -1, k) = (k == YMOMENTUM ? -1.0 : 1.0) * u(i, 0, k); 
      //  u(i, -2, k) = u(i, u.activeNy() - 2, k);
      //  u(i, -1, k) = u(i, u.activeNy() - 1, k); 
      } 
	}
  }
  if (XDIR && u.exists(i, 0)) {
    for (int k = 0; k < NUMBER_VARIABLES; k++) {
      if (downstream) {
		//u(i, u.activeNy() + 1, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 2, k);
       // u(i, u.activeNy()    , k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 1, k);
		
		u(i, u.activeNy() + 1, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 2, k);
        u(i, u.activeNy()    , k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 1, k);
		conservativeToPrimitive(u(i, u.activeNy() - 2), p);	 
		conservativeToPrimitive(u(i, u.activeNy() - 1), q);

		p[DENSITY] = p[PRESSURE]/T_wall;			
		q[DENSITY] = q[PRESSURE]/T_wall;	
		u(i, u.activeNy() + 1, DENSITY) = p[DENSITY] ;
		u(i, u.activeNy()    , DENSITY) = q[DENSITY] ;	
		
		/*conservativeToPrimitive(u(i, u.activeNy() + 1), p);	 
		//conservativeToPrimitive(u(i, u.activeNy()    ), q);	 
		p[DENSITY] = p[PRESSURE]/T_wall;			
		//q[DENSITY] = q[PRESSURE]/T_wall;	*/
		
		//u(i, u.activeNy() + 1, DENSITY) = u(i, u.activeNy() - 2, PRESSURE)/T_wall ;
		//u(i, u.activeNy()    , DENSITY) = u(i, u.activeNy() - 1, PRESSURE)/T_wall ;	
		
		//u(i, u.activeNy() - 1, DENSITY) = p[DENSITY] ;	
		
	    /*u(i, u.j(YCORNER) + 1, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.j(YCORNER) - 2, k);
        u(i, u.j(YCORNER)    , k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, u.j(YCORNER) - 1, k);
		conservativeToPrimitive(u(i, u.j(YCORNER) + 1), p);	 
		conservativeToPrimitive(u(i, u.j(YCORNER)    ), q);	 
		p[DENSITY] = p[PRESSURE]/T_wall;			
		q[DENSITY] = q[PRESSURE]/T_wall;	
		u(i, u.j(YCORNER) + 1, DENSITY) = p[DENSITY] ;
		u(i, u.j(YCORNER)    , DENSITY) = q[DENSITY] ;			
		*/
   //     u(i, u.activeNy() + 1, k) = (k == YMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 2, k);
   //     u(i, u.activeNy()    , k) = (k == YMOMENTUM ? -1.0 : 1.0) * u(i, u.activeNy() - 1, k);
     //   u(i, u.activeNy() + 1, k) = u(i, 1, k);
     //   u(i, u.activeNy()    , k) = u(i, 0, k);
      }

    }

  }
////////////////////////////////////////////////////////////////

  if (YDIR && u.exists(0, i)) {

    for (int k = 0; k < NUMBER_VARIABLES; k++) {
   
      if (downstream) {
//        u(-2, i, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(1, i, k);
//        u(-1, i, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(0, i, k); 
        u(-2, i, k) = u(1, i, k);
        u(-1, i, k) = u(0, i, k); 
		
      } else {
 //       u(u.activeNx() + 1, i, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(u.activeNx() - 2, i, k);
 //       u(u.activeNx()    , i, k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(u.activeNx() - 1, i, k);
        u(u.activeNx() + 1, i, k) = u(u.activeNx() - 2, i, k);
        u(u.activeNx()    , i, k) = u(u.activeNx() - 1, i, k);
      }

    }

  }

////////////////////////////////////////////////////////////////
for (int n_y = 0; n_y < number_y; n_y++){
	for (int n_x = 0; n_x < number_x; n_x++){
		 for (int k = 0; k < NUMBER_VARIABLES; k++) {
	
if (XDIR && u.exists(i, 0) && i > (u.i(start_x + n_x*space_x - length_x/2)+2) && i < (u.i(start_x + n_x*space_x + length_x/2)-2)) {
//if (XDIR && u.exists(i, 0) && i > u.i(start_x + n_x*space_x - length_x/2) && i < u.i(start_x + n_x*space_x + length_x/2)) {

    const int j = u.j(start_y + n_y*space_y + length_y/2);// bot
    for (int n = 0; n < 2; n++) {
		u(i, j - n - 1,k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, j + n,k);

	conservativeToPrimitive(u(i, j - n - 1), p1);	 
 
		p1[DENSITY] = p1[PRESSURE]/T_wall;			

		u(i, j - n - 1,DENSITY) = p1[DENSITY] ;

    }
	
	const int j_2 = u.j(start_y + n_y*space_y - length_y/2);//top
    for (int n = 0; n < 2; n++) {
		u(i, j_2 + n + 1,k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i, j_2 - n,k);

	  conservativeToPrimitive(u(i, j_2 + n + 1), p2);	 
 
		p2[DENSITY] = p2[PRESSURE]/T_wall;			

		u(i, j_2 + n + 1,DENSITY) = p2[DENSITY] ;
		
    }

  }

  //else if (!XDIR && u.exists(0, i) && i > (u.j(start_y + n_y*space_y - length_y/2) + 2) && i < (u.j(start_y + n_y*space_y + length_y/2) - 2)) { 
  else if (!XDIR && u.exists(0, i) && i > (u.j(start_y + n_y*space_y - length_y/2)) && i < (u.j(start_y + n_y*space_y + length_y/2))) { 
    const int i_1 = u.i(start_x + n_x*space_x - length_x/2); // left
    for (int n = 0; n < 2; n++) {
				u(i_1 + n + 1, i,k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i_1 - n, i,k);
	  	  conservativeToPrimitive(u(i_1 + n + 1, i ), p3);	 
 
		p3[DENSITY] = p3[PRESSURE]/T_wall;			

		u(i_1 + n + 1, i,DENSITY) = p3[DENSITY] ;
		
    }

    const int i_2 = u.i(start_x + n_x*space_x + length_x/2); //right
    for (int n = 0; n < 2; n++) {
		 u(i_2 - n - 1, i,k) = (k == YMOMENTUM || k == XMOMENTUM ? -1.0 : 1.0) * u(i_2 + n, i,k);

	  	  	 conservativeToPrimitive( u(i_2 - n -1, i ), p4);	 
 
		p4[DENSITY] = p4[PRESSURE]/T_wall;			

		u(i_2 - n - 1, i,DENSITY) = p4[DENSITY] ;
		
    }
  }
	}
}
}

}
/*
template<bool XDIR>
__global__ void setSpecialBoundaryConditionsKernel(Mesh<GPU>::type u) {
  const bool YDIR = !XDIR;

  const int k = blockIdx.x * blockDim.x + threadIdx.x - u.ghostCells();

  if (XDIR && u.exists(k, 0) && k < u.i(XCORNER)) {
    const int j = u.j(YCORNER);
    for (int n = 0; n < 2; n++) {
      u(k, j + n + 1) = u(k, j - n);
      u(k, j + n + 1, YMOMENTUM) = -u(k, j + n + 1, YMOMENTUM);

    }
  }
  else if (!XDIR && u.exists(0, k) && k > u.j(YCORNER)) {
    const int i = u.i(XCORNER);
    for (int n = 0; n < 2; n++) {
      u(i - n - 1, k) = u(i + n, k);
      u(i - n - 1, k, YMOMENTUM) = -u(i - n - 1, k, YMOMENTUM);
    }
  }
}
*/
