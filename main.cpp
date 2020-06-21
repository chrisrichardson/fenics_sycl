
#include "poisson.c"
#include <Eigen/Dense>
#include <iostream>
#include <CL/sycl.hpp>

class MyKernel;

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

  // Global RHS vector
  //  Eigen::VectorXd global_vector(nelem * nelem_dofs);
  Eigen::VectorXd global_vector(npoints);
  global_vector.setZero();

  {
    
  // Get a queue
  cl::sycl::gpu_selector device_selector;
  cl::sycl::queue queue(device_selector);
  
  std::cout << "Running on "
            << queue.get_device().get_info<sycl::info::device::name>()
            << "\n";
    
  cl::sycl::buffer<double, 2> geom_buf(geometry.data(), {(std::size_t)npoints, 3});
  cl::sycl::buffer<int, 2> dm_buf(dofmap.data(), {(std::size_t)dofmap.rows(), (std::size_t)dofmap.cols()});
  cl::sycl::buffer<double, 1> gv_buf(global_vector.data(), global_vector.size());
  cl::sycl::range<1> nelem_sycl{(std::size_t)nelem};

  queue.submit([&] (cl::sycl::handler& cgh) 
               {
                 auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
                 auto access_dm = dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
                 auto access_gv = gv_buf.get_access<cl::sycl::access::mode::write>(cgh);
           
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

                     
//                     printf("i=%ld %f\n", wiID[0], b[0]);

                     for (int j = 0; j < nelem_dofs; ++j)
                     {
                       const std::size_t dmi = access_dm[wiID[0]][j];
                       // access_gv[wiID[0] * nelem_dofs + j] = b[j];
                       access_gv[dmi] += b[j];
                     }
                     
                   };
                 cgh.parallel_for<MyKernel>(nelem_sycl, kern);
               });
  
  }

  // Assembly loop
  //  Eigen::VectorXd global_vector2(nelem * nelem_dofs);
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


    printf("j=%d %f\n", i, b[0]);


    // Fill global vector
    for (int j = 0; j < nelem_dofs; ++j)
      global_vector2[dofmap(i, j)] += b[j];
  }

  std::cout << "Vector norm = " << global_vector.norm() << "\n";
  std::cout << "Vector norm 2 = " << global_vector2.norm() << "\n";
}
