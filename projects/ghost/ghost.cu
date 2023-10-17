/*
  Copyright © Cambridge Numerical Solutions Ltd 2013
*/
#include "ghost.hpp"
#include <sys/times.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

bool halt = false;

void signalHandler(int signal = 0) {
  halt = true;
}


// Main function on CPU /////////////////////////////////////////////////////////
int main(int argc, char** argv) {

  // capture SIGINT
  signal(SIGINT, signalHandler);

  if (argc < 2) {
    std::cerr << "Invoke with " << argv[0] << " <configuration file>" << std::endl;
    exit(1);
  }

  // select a device
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  std::cout << "#Found " << num_devices << " GPGPUs" << std::endl;
  cudaDeviceProp properties;
  int best_device = 0;
  if (num_devices > 1) {
    // if there's more than one, pick the one with the highest compute capability
    int best_computemode = 0, computemode;
    for (int device = 0; device < num_devices; device++) {
      cudaGetDeviceProperties(&properties, device);
      std::cout << "  #" << device << " " << properties.name << ": " << properties.multiProcessorCount << " processors, compute capability " << properties.major << "." << properties.minor << std::endl;
      computemode = properties.major << 4 + properties.minor;
      if (best_computemode < computemode) {
        best_computemode = computemode;
        best_device = device;
      }
    }
  }
        best_device = atoi(argv[2]);
  cudaGetDeviceProperties(&properties, best_device);
  std::cout << "#  using #" << best_device << " (" << properties.name << ")" << std::endl;
  cudaSetDevice(best_device);

  // start a timer to get the total wall time at the end of the run
  struct tms startTimes, endTimes;
  timespec startClock, endClock;
  times(&startTimes);
  clock_gettime(CLOCK_REALTIME, &startClock);

  Solver solver(argv[1]);

  Solver::Status status = Solver::OUTPUT;

  Mesh<CPU>::type uCPU(*solver.u, Mesh<CPU>::type::Allocate);
  
#ifdef GLOUTPUT
  OpenGLOutputter outputter(argc, argv, *solver.u);
  boost::thread outputterThread(boost::ref(outputter));
#endif

  // open the data file and output a header line
  std::stringstream filename;
  filename << solver.outputDirectory << "data";

  std::ofstream dataFile;
  dataFile.open(filename.str().c_str());

#ifdef GHOST
  dataFile << "#" << solver.u->time() << "\t\t";
  for (int i = 0; i < solver.geometries.size(); i++) {
    if (solver.geometries[i].rotating) {
      dataFile << "torque on level set " << i << "\t\t";
    }
  }
#endif
  for (int i = 0; i < solver.outputRadii.size(); i++) {
    dataFile << "P(r=" << solver.outputRadii[i] << ")\t\t";
    dataFile << "flux(r=" << solver.outputRadii[i] << ")\t\t";
  }
  dataFile << std::endl;
  
   do {
	   
/*#ifdef GLOUTPUT
    if (solver.getStepNumber() % 1 == 0) {
      outputter.dispatchDraw(*solver.u);
      //outputter.paused = true;
      while (outputter.gridToRender != NULL); // SPIN
    }
#endif
*/

  //////////////////////////////////////////////////////////////////////////////////
  // Advance the frame

//Save Conserved Variables and Patch///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//if (uCPU(uCPU.activeNx()-10*uCPU.activeNx()/50, uCPU.activeNy()/2)[PRESSURE] > (1.0+1.0e-4)){
//Find where to start patching	 
uCPU = *solver.u;
	
	int i = 9*uCPU.activeNx()/10;
	int nmax = 0;	
	for (int j = 1*uCPU.activeNy()/5; j < 4*uCPU.activeNy()/5; ++j) {
		real p[NUMBER_VARIABLES];
        conservativeToPrimitive(uCPU(i, j), p);
		if (p[PRESSURE] > 1 + 1.0e-4){
			nmax = 1;
			
		}
	}
	
	
	if (nmax > 0) {
std::stringstream filename;
        filename << solver.outputDirectory << "Density_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile;
        outFile.open(filename.str().c_str());

        outFile.precision(8);
        outFile << "ASCII" << std::endl;
        outFile << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
          //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
            outFile << std::fixed << uCPU(i, j)[DENSITY] << std::endl;
          }
        }

        outFile.close(); 
///////////////////////////////////////////////////////////		
		std::stringstream filename1;
        filename1 << solver.outputDirectory << "Xmom_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile1;
        outFile1.open(filename1.str().c_str());
        outFile1.precision(8);
        outFile1 << "ASCII" << std::endl;
        outFile1 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile1 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile1 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile1 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		 //for (int i = 0; i < uCPU.activeNx(); ++i) {
		 //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile1 << std::fixed << uCPU(i, j)[XMOMENTUM] << std::endl;
          }
        } 

        outFile1.close();  
/////////////////////////////////////////////////////
        std::stringstream filename2;
        filename2 << solver.outputDirectory << "Ymom_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile2;
        outFile2.open(filename2.str().c_str());
        outFile2.precision(8);
        outFile2 << "ASCII" << std::endl;
        outFile2 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile2 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile2 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile2 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile2 << std::fixed << uCPU(i, j)[YMOMENTUM] << std::endl;
          }
        } 

        outFile2.close(); 
//////////////////////////////////////////////////////
        std::stringstream filename3;
        filename3 << solver.outputDirectory << "La1_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile3;
        outFile3.open(filename3.str().c_str());
        outFile3.precision(8);
        outFile3 << "ASCII" << std::endl;
        outFile3 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile3 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile3 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile3 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		 //for (int i = 0; i < uCPU.activeNx(); ++i) {
//		 for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile3 << std::fixed << uCPU(i, j)[LAMBDA0] << std::endl;
          }
        }

        outFile3.close();
//////////////////////////////////////////////////////
        
		std::stringstream filename4;
        filename4 << solver.outputDirectory << "La0_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile4;
        outFile4.open(filename4.str().c_str());
        outFile4.precision(8);
        outFile4 << "ASCII" << std::endl;
        outFile4 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile4 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile4 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile4 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile4 << std::fixed << uCPU(i, j)[LAMBDA0] << std::endl;
          }
        }

        outFile4.close();
		
//////////////////////////////////////////////////////
        std::stringstream filename5;
        filename5 << solver.outputDirectory << "Energy_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile5;
        outFile5.open(filename5.str().c_str());
        outFile5.precision(8);
        outFile5 << "ASCII" << std::endl;
        outFile5 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile5 << "DIMENSIONS " << uCPU.activeNx()/ 3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile5 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile5 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile5 << std::fixed << uCPU(i, j)[ENERGY] << std::endl;
          }
        }

        outFile5.close();  
///////////////////////////////////////////////////////
        std::stringstream filename6;
        filename6 << solver.outputDirectory << "Pmax_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile6;
        outFile6.open(filename6.str().c_str());
        outFile6.precision(8);
        outFile6 << "ASCII" << std::endl;
        outFile6 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile6 << "DIMENSIONS " << uCPU.activeNx() + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile6 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile6 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile6 << std::fixed << uCPU(i, j)[PMAX] << std::endl;
          }
        }
        outFile6.close();		

		
for (int ni = 0; ni < (1*uCPU.activeNx()/2); ++ni) {
  for (int nj = 0; nj < uCPU.activeNy(); ++nj) {  
    uCPU(ni, nj)[DENSITY] = uCPU(ni+1*uCPU.activeNx()/2, nj)[DENSITY];
    uCPU(ni, nj)[XMOMENTUM]  = uCPU(ni+1*uCPU.activeNx()/2, nj)[XMOMENTUM];
    uCPU(ni, nj)[YMOMENTUM] = uCPU(ni+1*uCPU.activeNx()/2, nj)[YMOMENTUM];
    uCPU(ni, nj)[LAMBDA0] = uCPU(ni+1*uCPU.activeNx()/2, nj)[LAMBDA0];
    uCPU(ni, nj)[LAMBDA1] = uCPU(ni+1*uCPU.activeNx()/2, nj)[LAMBDA1];
    uCPU(ni, nj)[ENERGY]  = uCPU(ni+1*uCPU.activeNx()/2, nj)[ENERGY];
	uCPU(ni, nj)[PMAX] = uCPU(ni+1*uCPU.activeNx()/2, nj)[PMAX];
 }
 }
//Initialize new domain///////////////////////////////////////////////////////////////////////////////////////////// 

  //for (int ni = uCPU.activeNx()/3; ni < uCPU.activeNx(); ++ni){
  for (int ni = (1*uCPU.activeNx()/2); ni < uCPU.activeNx(); ++ni){
  for (int nj = 0; nj < uCPU.activeNy(); ++nj){
     uCPU(ni, nj)[DENSITY]    = 1.0;
     uCPU(ni, nj)[XMOMENTUM]  = 0.0;
     uCPU(ni, nj)[YMOMENTUM]  = 0.0;
     uCPU(ni, nj)[LAMBDA0]    = 0.0;
	 uCPU(ni, nj)[LAMBDA1]     = 0.0;
     uCPU(ni, nj)[ENERGY]     = 1.0 / (GAMMA - 1.0);
     uCPU(ni, nj)[PMAX]     = 1.0;	                    
  }
  }
} 
*solver.u = uCPU;
/*
	    if (status == Solver::OUTPUT) {
    
      if (true) {  
	  uCPU = *solver.u;
            std::stringstream filename0;
        filename0 << solver.outputDirectory << "NMAX" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".txt";

        std::ofstream outFile0;
        outFile0.open(filename0.str().c_str());

        outFile0.precision(8);

		int i = 4*uCPU.activeNx()/5;
	int nmax = 0;	
	const real pover = 0;
	for (int j = 1*uCPU.activeNy()/5; j < 4*uCPU.activeNy()/5; ++j) {
		real p[NUMBER_VARIABLES];
        conservativeToPrimitive(uCPU(i, j), p);
		if (p[PRESSURE] > 1 + 1.0e-4){
			nmax = 1;
			
		}
	}
        outFile0 << std::fixed << nmax << std::endl;
        outFile0.close(); 
		
		}
	  }
	  */
	  /*
uCPU = *solver.u;
	
if (Check > 0.0){
  std::cout << "Copying data!";
  Check = -1.0;

// Read in density /////////////////////////////////////////////////
  std::ifstream inFile;
  inFile.open("Density_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile >> uCPU(ni, nj)[DENSITY];     //贴图中3000-4500网格数之间的数据贴进来
      inFile.ignore(256,'\n');
      } else {
      inFile.ignore(256,'\n');
      }
    }
  }
  
  inFile.close();

// Read in x-momentum //////////////////////////////////////////////
  std::ifstream inFile1;
  inFile1.open("Xmom_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile1.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile1.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile1 >> uCPU(ni, nj)[XMOMENTUM];     //贴图中3000-4500网格数之间的数据贴进来
      inFile1.ignore(256,'\n');
      } else {
      inFile1.ignore(256,'\n');
      }
    }
  }
  
  inFile1.close();

// Read in y-momentum //////////////////////////////////////////////
  std::ifstream inFile2;
  inFile2.open("Ymom_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile2.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile2.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile2 >> uCPU(ni, nj)[YMOMENTUM];     //贴图中3000-4500网格数之间的数据贴进来
      inFile2.ignore(256,'\n');
      } else {
      inFile2.ignore(256,'\n');
      }
    }
  }
  
  inFile2.close();

// Read in lambda_0 //////////////////////////////////////////////
  std::ifstream inFile3;
  inFile3.open("La0_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile3.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile3.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile3 >> uCPU(ni, nj)[LAMBDA0];     //贴图中3000-4500网格数之间的数据贴进来
      inFile3.ignore(256,'\n');
      } else {
      inFile3.ignore(256,'\n');
      }
    }
  }
  inFile3.close();

// Read in lambda_1 //////////////////////////////////////////////
  std::ifstream inFile4;
  inFile4.open("La1_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile4.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile4.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile4 >> uCPU(ni, nj)[LAMBDA1];     //贴图中3000-4500网格数之间的数据贴进来
      inFile4.ignore(256,'\n');
      } else {
      inFile4.ignore(256,'\n');
      }
    }
  }
  
  inFile4.close();

// Read in energy //////////////////////////////////////////////
  std::ifstream inFile5;
  inFile5.open("Energy_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile5.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile5.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile5 >> uCPU(ni, nj)[ENERGY];     //贴图中3000-4500网格数之间的数据贴进来
      inFile5.ignore(256,'\n');
      } else {
      inFile5.ignore(256,'\n');
      }
    }
  }
  
  inFile5.close();

// Read in Pmax//////////////////////////////////////////////
  std::ifstream inFile6;
  inFile6.open("Pmax_input000001.vtk");  

  for (int n_skip = 0; n_skip < Skip_lines; ++n_skip) {
    inFile6.ignore(256,'\n');
  }
for (int nj = 0; nj < uCPU.activeNy(); ++nj) {
//    for (int nj = cell*YCORNER; nj > 0; --nj) {
    for (int ni = 0 ; ni < uCPU.activeNx()/2; ++ni) {         //4500是贴图中的总网格数
      if (ni < 0) {                           //小于3000不贴
      inFile6.ignore(256,'\n');
//      } else {
      } else if (ni < uCPU.activeNx()/2) {
      inFile6 >> uCPU(ni, nj)[PMAX];     //贴图中3000-4500网格数之间的数据贴进来
      inFile6.ignore(256,'\n');
      } else {
      inFile6.ignore(256,'\n');
      }
    }
  }
  
  inFile6.close();
}


	int i = 4*uCPU.activeNx()/5;
	int nmax = 0;	
	for (int j = 1*uCPU.activeNy()/5; j < 4*uCPU.activeNy()/5; ++j) {
		real p[NUMBER_VARIABLES];
        conservativeToPrimitive(uCPU(i, j), p);
		if (p[PRESSURE] > 1 + 1.0e-4){
			nmax = 1;
			
		}
	}
	
	
	if (nmax > 0) {
		
std::stringstream filename;
        filename << solver.outputDirectory << "Density_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile;
        outFile.open(filename.str().c_str());

        outFile.precision(8);
        outFile << "ASCII" << std::endl;
        outFile << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
          //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
            outFile << std::fixed << uCPU(i, j)[DENSITY] << std::endl;
          }
        }

        outFile.close(); 
///////////////////////////////////////////////////////////		
		std::stringstream filename1;
        filename1 << solver.outputDirectory << "Xmom_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile1;
        outFile1.open(filename1.str().c_str());
        outFile1.precision(8);
        outFile1 << "ASCII" << std::endl;
        outFile1 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile1 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile1 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile1 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		 //for (int i = 0; i < uCPU.activeNx(); ++i) {
		 //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile1 << std::fixed << uCPU(i, j)[XMOMENTUM] << std::endl;
          }
        } 

        outFile1.close();  
/////////////////////////////////////////////////////
        std::stringstream filename2;
        filename2 << solver.outputDirectory << "Ymom_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile2;
        outFile2.open(filename2.str().c_str());
        outFile2.precision(8);
        outFile2 << "ASCII" << std::endl;
        outFile2 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile2 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile2 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile2 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile2 << std::fixed << uCPU(i, j)[YMOMENTUM] << std::endl;
          }
        } 

        outFile2.close(); 
//////////////////////////////////////////////////////
        std::stringstream filename3;
        filename3 << solver.outputDirectory << "La1_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile3;
        outFile3.open(filename3.str().c_str());
        outFile3.precision(8);
        outFile3 << "ASCII" << std::endl;
        outFile3 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile3 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile3 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile3 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		 //for (int i = 0; i < uCPU.activeNx(); ++i) {
//		 for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile3 << std::fixed << uCPU(i, j)[LAMBDA0] << std::endl;
          }
        }

        outFile3.close();
//////////////////////////////////////////////////////
        
		std::stringstream filename4;
        filename4 << solver.outputDirectory << "La0_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile4;
        outFile4.open(filename4.str().c_str());
        outFile4.precision(8);
        outFile4 << "ASCII" << std::endl;
        outFile4 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile4 << "DIMENSIONS " << uCPU.activeNx()/3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile4 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile4 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile4 << std::fixed << uCPU(i, j)[LAMBDA0] << std::endl;
          }
        }

        outFile4.close();
		
//////////////////////////////////////////////////////
        std::stringstream filename5;
        filename5 << solver.outputDirectory << "Energy_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile5;
        outFile5.open(filename5.str().c_str());
        outFile5.precision(8);
        outFile5 << "ASCII" << std::endl;
        outFile5 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile5 << "DIMENSIONS " << uCPU.activeNx()/ 3 + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile5 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile5 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

       for (int j = 0; j < uCPU.activeNy(); ++j) {
           for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile5 << std::fixed << uCPU(i, j)[ENERGY] << std::endl;
          }
        }

        outFile5.close();  
///////////////////////////////////////////////////////
        std::stringstream filename6;
        filename6 << solver.outputDirectory << "Pmax_input" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".vtk";

        std::ofstream outFile6;
        outFile6.open(filename6.str().c_str());
        outFile6.precision(8);
        outFile6 << "ASCII" << std::endl;
        outFile6 << "DATASET STRUCTURED_POINTS" << std::endl;
        outFile6 << "DIMENSIONS " << uCPU.activeNx() + 1 << " " << uCPU.activeNy() + 1<< " " << 1 << std::endl;
        outFile6 << "SPACING " << uCPU.dx() << " " << uCPU.dy() << " 1" << std::endl;
        outFile6 << "CELL_DATA " << uCPU.activeNx() * uCPU.activeNy() << std::endl;

        for (int j = 0; j < uCPU.activeNy(); ++j) {
          for (int i = (1*uCPU.activeNx()/2); i < uCPU.activeNx(); ++i) {
		  //for (int i = 0; i < uCPU.activeNx(); ++i) {
		  //for (int i = 2*uCPU.activeNx()/3; i < uCPU.activeNx(); ++i) {
 
            outFile6 << std::fixed << uCPU(i, j)[PMAX] << std::endl;
          }
        }
        outFile6.close();		

	
for (int ni = 0; ni < (1*uCPU.activeNx()/2); ++ni) {
  for (int nj = 0; nj < uCPU.activeNy(); ++nj) {  
 
    uCPU(ni, nj)[DENSITY] = uCPU(ni+1*uCPU.activeNx()/2, nj)[DENSITY];
    uCPU(ni, nj)[XMOMENTUM]  = uCPU(ni+1*uCPU.activeNx()/2, nj)[XMOMENTUM];
    uCPU(ni, nj)[YMOMENTUM] = uCPU(ni+1*uCPU.activeNx()/2, nj)[YMOMENTUM];
    uCPU(ni, nj)[LAMBDA0] = uCPU(ni+1*uCPU.activeNx()/2, nj)[LAMBDA0];
    uCPU(ni, nj)[LAMBDA1] = uCPU(ni+1*uCPU.activeNx()/2, nj)[LAMBDA1];
    uCPU(ni, nj)[ENERGY]  = uCPU(ni+1*uCPU.activeNx()/2, nj)[ENERGY];
	uCPU(ni, nj)[PMAX] = uCPU(ni+1*uCPU.activeNx()/2, nj)[PMAX];


 }
 }
//Initialize new domain///////////////////////////////////////////////////////////////////////////////////////////// 

  //for (int ni = uCPU.activeNx()/3; ni < uCPU.activeNx(); ++ni){
  for (int ni = (1*uCPU.activeNx()/2); ni < uCPU.activeNx(); ++ni){
  for (int nj = 0; nj < uCPU.activeNy(); ++nj){

     uCPU(ni, nj)[DENSITY]    = 1.0;
     uCPU(ni, nj)[XMOMENTUM]  = 0.0;
     uCPU(ni, nj)[YMOMENTUM]  = 0.0;
     uCPU(ni, nj)[LAMBDA0]    = 0.0;
     uCPU(ni, nj)[LAMBDA1]    = 0.0;
     uCPU(ni, nj)[ENERGY]     = 1.0 / (GAMMA - 1.0);
	 uCPU(ni, nj)[PMAX]       = 1.0;   
                         
  }
  }
} 
*solver.u = uCPU;
*/
////////////////////////////////////////////////////////////////


    if (status == Solver::OUTPUT) {
    
      if (true) {  
	  uCPU = *solver.u;

        std::stringstream filename1;
        filename1 << solver.outputDirectory << "RESULT" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".txt";

        std::ofstream outFile1;
        outFile1.open(filename1.str().c_str());

        outFile1.precision(8);
       //outFile1 << "VARIABLES = \"P\", \"Z\"" << std::endl;
       // outFile1 << "ZONE I = " << uCPU.activeNx() << " J = " << uCPU.activeNy() << " DATAPACKING = POINT" << std::endl;

        //for (int j = 0; j < uCPU.activeNy(); ++j) { 
		/*
		for (int j = -2; j < 3; ++j) { 
          for (int i = 0; i < uCPU.activeNx(); ++i) {
			  
            real p[NUMBER_VARIABLES];
            conservativeToPrimitive(uCPU(i, j), p);
            //outFile1 << std::fixed << uCPU(i, j)[DENSITY] << " " <<  uCPU(i, j)[YVELOCITY] / uCPU(i, j)[DENSITY] << std::endl;
			outFile1 << std::fixed << i << " " << j << " " << p[PRESSURE] << ' ' << p[DENSITY] << ' ' << p[PRESSURE]/p[DENSITY] << " " <<  p[XVELOCITY] << " " <<  p[YVELOCITY]<< std::endl;
          }
        } 
		*/
		//for (int j = uCPU.activeNy() - 3; j < uCPU.activeNy()+2; ++j) { 
		outFile1 << "i" << " " << "j" << " " << "p" << ' ' << "rho" << ' ' << "t" << " " <<  "u" << " " <<  "v" << " " <<  "LA0" << " " <<  "LA1" << std::endl;
		for (int j = uCPU.activeNy()/5; j < uCPU.activeNy(); j += uCPU.activeNy()/5) {
          for (int i = 0; i < uCPU.activeNx(); ++i) {
			  
            real q[NUMBER_VARIABLES];
            conservativeToPrimitive(uCPU(i, j), q);
            //outFile1 << std::fixed << uCPU(i, j)[DENSITY] << " " <<  uCPU(i, j)[YVELOCITY] / uCPU(i, j)[DENSITY] << std::endl;
			outFile1 << std::fixed << i << " " << j << " " << q[PRESSURE] << ' ' << q[DENSITY] << ' ' << q[PRESSURE]/q[DENSITY] << " " <<  q[XVELOCITY] << " " <<  q[YVELOCITY] << " " <<  q[LAMBDA0]/q[DENSITY] << " " <<  q[LAMBDA1]/q[DENSITY]<< std::endl;
          }
        } 
        outFile1.close(); 

		std::stringstream filename2;
        filename2 << solver.outputDirectory << "VeloGradient" << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".txt";
        std::ofstream outFile2;
        outFile2.open(filename2.str().c_str());

        outFile2.precision(8);
        //outFile2 << "VARIABLES = \"P\", \"Z\"" << std::endl;
        //outFile2 << "ZONE I = " << uCPU.activeNx() << " J = " << uCPU.activeNy() << " DATAPACKING = POINT" << std::endl;

        //for (int j = 0; j < uCPU.activeNy(); ++j) { 
		for (int i = uCPU.activeNx()/5; i < uCPU.activeNx(); i += uCPU.activeNx()/5) {
		for (int j = -2; j < uCPU.activeNy()+2; ++j) { 
			  
            real p[NUMBER_VARIABLES];
            conservativeToPrimitive(uCPU(i, j), p);
            //outFile2 << std::fixed << uCPU(i, j)[DENSITY] << " " <<  uCPU(i, j)[YVELOCITY] / uCPU(i, j)[DENSITY] << std::endl;
			outFile2 << std::fixed << i << " " << j << " " << p[PRESSURE] << ' ' << p[DENSITY] << ' ' << p[PRESSURE]/p[DENSITY] << " " <<  p[XVELOCITY] << " " <<  p[YVELOCITY]<< std::endl;
          }
        } 
        outFile2.close(); 
		
		
    }

      if (true) {
        std::vector<ImageOutputs> outputs;

        outputs.push_back((ImageOutputs){"pressure", PRESSUREFOO, HUE, 10.0, 50.0});
		outputs.push_back((ImageOutputs){"sootfoil", PMAXPLOT, CUBEHELIX, 10.0, 100.0});
		//outputs.push_back((ImageOutputs){"sootfoil", PMAXPLOT, GREYSCALE, 10.0, 80.0});
		outputs.push_back((ImageOutputs){"xvelocity", XVELOCITYPLOT, HUE, 0.0, 6.0});
        // outputs.push_back((ImageOutputs){"pressure2", PRESSUREFOO, HUE, 1.0, 50.0});
        outputs.push_back((ImageOutputs){"density", DISPLAYTYPES+DENSITY, HUE, 1.0, 13.0});
       // outputs.push_back((ImageOutputs){"frac", FRACTIONPLOT, HUE, 0.0, 1.0});
	    outputs.push_back((ImageOutputs){"LAMBDA0", LAMBDA0PLOT, HUE, 0.0, 1.0});
        outputs.push_back((ImageOutputs){"LAMBDA1", LAMBDA1PLOT, HUE, 0.0, 1.0});
        outputs.push_back((ImageOutputs){"schlieren", SCHLIEREN, GREYSCALE, 0.0, 1.0});
        outputs.push_back((ImageOutputs){"temperature", SOUNDSPEED, HUE, 1.0, 10.0});

        for (std::vector<ImageOutputs>::iterator iter = outputs.begin(); iter != outputs.end(); ++iter) {
          std::stringstream filename;
          filename << solver.outputDirectory << (*iter).prefix << std::setw(6) << std::setfill('0') << solver.getOutputNumber() << ".png";
          saveFrame(*solver.u, (*iter).plotVariable, (*iter).colourMode, filename.str().c_str(), (*iter).min, (*iter).max, & (*solver.fluxes)(0, 0, 0));
        }
      }
    }
  } while ((status = solver.step()) != Solver::FINISHED && !halt);

  times(&endTimes);
  clock_gettime(CLOCK_REALTIME, &endClock);
  const double wallTime = (endClock.tv_sec - startClock.tv_sec) + (endClock.tv_nsec - startClock.tv_nsec) * 1e-9;

  std::cout << "CPU time, wall= " << std::setprecision(2) << std::fixed << wallTime << "s, user=" << (endTimes.tms_utime - endTimes.tms_utime) << "s, sys=" << (endTimes.tms_stime - endTimes.tms_stime) << "s.  Time for {fluxes=" << solver.getTimeFluxes() * 1e-3 << "s, sources=" << solver.getTimeSourcing() * 1e-3 << "s, reduction=" << solver.getTimeReducing() * 1e-3 << "s, adding=" << solver.getTimeAdding() * 1e-3 << "s}" << std::endl;
  return 0;
}


