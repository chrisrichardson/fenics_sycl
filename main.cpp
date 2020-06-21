
#include "poisson.c"
#include <Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <CL/sycl.hpp>

class AssemblyKernel;
class AccumulationKernel;

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main()
{
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

  // Count entries for each dof, and give each entry a column index
  // creating a 2D array to store contributions for each dof on a separate row
  Eigen::ArrayXi dof_counter(npoints);
  dof_counter.setZero();
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dofidx(nelem, nelem_dofs);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
    {
      dofidx(i, j) = dof_counter[dofmap(i, j)];
      ++dof_counter[dofmap(i, j)];    
    }
  }
  int acc_cols = dof_counter.maxCoeff() + 1;
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> accum_buf(npoints, acc_cols);
  accum_buf.setZero();

  // Global RHS vector
  Eigen::VectorXd global_vector(npoints);
  global_vector.setZero();

  {
    // Get a queue
    cl::sycl::default_selector device_selector;
    cl::sycl::queue queue(device_selector);
    
    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";
    
    cl::sycl::buffer<double, 2> geom_buf(geometry.data(), {(std::size_t)npoints, 3});
    cl::sycl::buffer<int, 2> dm_buf(dofmap.data(), {(std::size_t)dofmap.rows(), (std::size_t)dofmap.cols()});
    cl::sycl::buffer<int, 2> col_buf(dofidx.data(), {(std::size_t)dofidx.rows(), (std::size_t)dofidx.cols()});
    cl::sycl::buffer<double, 2> ac_buf(accum_buf.data(), {(std::size_t)accum_buf.rows(), (std::size_t)accum_buf.cols()});
    cl::sycl::range<1> nelem_sycl{(std::size_t)nelem};
    
    queue.submit([&] (cl::sycl::handler& cgh) 
                 {
                   auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_dm = dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_col = col_buf.get_access<cl::sycl::access::mode::read>(cgh);
                   auto access_ac = ac_buf.get_access<cl::sycl::access::mode::write>(cgh);
                   
                   auto kern = [=](cl::sycl::id<1> wiID) 
                     {
                       double cell_geom[9];
                       double b[3] = {0, 0, 0};
                       
                       double w[3] = {1., 2., 3.};
                       
                       // Pull out points for this cell
                       for (int j = 0; j < 3; ++j)
                       {
                         const std::size_t dmi = access_dm[wiID[0]][j];
                         for (int k = 0; k < 3; ++k)
                           cell_geom[j * 3 + k] = access_geom[dmi][k];
                       }
                       
                       // Get local values
                       tabulate_tensor_integral_cell_otherwise_b67e00d4067e0c970c3a0a79f0d0600104ce7791(
                                                                                                        b, w, nullptr, 
                                                                                                        cell_geom, nullptr, nullptr, 0);
                       
                       
                       // Insert result into rows of 2D array corresponding to each dof
                       for (int j = 0; j < nelem_dofs; ++j)
                       {
                         const std::size_t dmi = access_dm[wiID[0]][j];
                         const std::size_t col = access_col[wiID[0]][j];
                         access_ac[dmi][col] = b[j];
                       }
                       
                     };
                   cgh.parallel_for<AssemblyKernel>(nelem_sycl, kern);
                 });
    
  }


  // Second kernel to accumulate row at each dof
  {
    // Get a queue
    cl::sycl::default_selector device_selector;
    cl::sycl::queue queue(device_selector);
    
    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";
    
    cl::sycl::buffer<double, 1> gv_buf(global_vector.data(), global_vector.size());
    cl::sycl::buffer<double, 2> ac_buf(accum_buf.data(), {(std::size_t)accum_buf.rows(), (std::size_t)accum_buf.cols()});
    cl::sycl::range<1> npoints_sycl{(std::size_t)npoints};
    
    queue.submit([&] (cl::sycl::handler& cgh) 
                 {
                   auto access_gv = gv_buf.get_access<cl::sycl::access::mode::write>(cgh);
                   auto access_ac = ac_buf.get_access<cl::sycl::access::mode::read>(cgh);
                   
                   auto kern = [=](cl::sycl::id<1> wiID) 
                     {
                       for (int j = 0; j < acc_cols; ++j)
                         access_gv[wiID[0]] += access_ac[wiID[0]][j];
                     };
                   cgh.parallel_for<AccumulationKernel>(npoints_sycl, kern);
                 });
    
  }

  // Comparison CPU code below
  //--------------------------
  
  Eigen::VectorXd global_vector2(npoints);
  global_vector2.setZero();

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

    b.setZero();
    // Get local values
    tabulate_tensor_integral_cell_otherwise_b67e00d4067e0c970c3a0a79f0d0600104ce7791(
                                                                                     b.data(), w.data(), nullptr, 
                                                                                     cell_geometry.data(), nullptr, nullptr, 0);

    // Fill global vector
    for (int j = 0; j < nelem_dofs; ++j)
      global_vector2[dofmap(i, j)] += b[j];
  }

  std::cout << "Vector norm = " << std::setprecision(20) << global_vector.norm() << "\n";
  std::cout << "Vector norm 2 = " << std::setprecision(20) << global_vector2.norm() << "\n";
}
