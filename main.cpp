
#include "poisson.h"
#include <Eigen/Dense>
#include <iostream>

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main()
{
  int npoints = 100;

  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> geometry
      = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Random(
          npoints, 3);

  std::cout << geometry << "\n";
}