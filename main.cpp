
#include "poisson.h"
#include <Eigen/Dense>
#include <iostream>

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main()
{
  ufc_form* L = create_form_poisson_L();
  ufc_integral* integral = L->create_cell_integral(-1);
  auto kernel = integral->tabulate_tensor;

  int nelem = 1000;
  int nelem_dofs = 3; // For P1 Poisson on triangle

  // Create a dummy dofmap
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dofmap(
      nelem, nelem_dofs);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
      dofmap(i, j) = i * 2 + j;
  }
  int idx = dofmap.maxCoeff();

  // For a P1 problem, the number of geometry points is the same as the number
  // of dofs
  int npoints = idx + 1;
  std::cout << "npoints = " << npoints << "\n";
  Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> geometry
      = Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>::Random(
          npoints, 3);

  // Global RHS vector
  Eigen::VectorXd global_vector(npoints);
  global_vector.setZero();

  // Assembly loop
  // Coefficient (RHS source term) - constant on each cell
  Eigen::Array<double, 3, 1> w = {1.0, 2.0, 3.0};
  // Element local RHS
  Eigen::Array<double, 3, 1> b;
  // Element local geometry
  Eigen::Array<double, 3, 3, Eigen::RowMajor> cell_geometry;
  for (int i = 0; i < nelem; ++i)
  {
    // Pull out points for this cell
    for (int j = 0; j < 3; ++j)
      cell_geometry.row(j) = geometry.row(dofmap(i, j));

    // Get local values
    kernel(b.data(), w.data(), nullptr, cell_geometry.data(), nullptr, nullptr,
           0);

    // Fill global vector
    for (int j = 0; j < 3; ++j)
      global_vector[dofmap(i, j)] += b[j];
  }

  std::cout << "Vector norm = " << global_vector.norm() << "\n";
}