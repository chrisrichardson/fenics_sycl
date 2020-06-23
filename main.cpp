
#include <CL/sycl.hpp>

// Need to include C file in same translation unit as lambda
#include "poisson.c"

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "dolfin_interface.h"

class AssemblyKernel;
class AccumulationKernel;

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main(int argc, char* argv[])
{
  // Get dolfin data
  auto [dofmap, geometry, function_dm] = create_arrays(argc, argv);
  int nelem = dofmap.rows();
  int nelem_dofs = dofmap.cols();

  int idx = dofmap.maxCoeff();
  int npoints = idx + 1;
  std::cout << "npoints = " << npoints << "\n";

  // Count number of entries for each dof and make offset array
  std::vector<int> dof_counter(npoints, 0);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
      ++dof_counter[dofmap(i, j)];
  }
  std::vector<int> dof_offsets(npoints + 1, 0);
  std::partial_sum(dof_counter.begin(), dof_counter.end(),
                   dof_offsets.begin() + 1);
  assert(dof_offsets.back() == nelem * nelem_dofs);

  // Make a "flat index" array listing where each assembly entry should be
  // placed. This is just a permutation, so that the entries that belong to the
  // same dof index are next to each other
  std::vector<int> tmp_offsets(dof_offsets.begin(), dof_offsets.end());
  Eigen::ArrayXi flat_index(nelem * nelem_dofs);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
      flat_index[i * nelem_dofs + j] = tmp_offsets[dofmap(i, j)]++;
  }

  // Device memory to accumulate assembly entries before summing
  cl::sycl::buffer<double, 1> ac_buf(
      cl::sycl::range<1>{(std::size_t)(nelem * nelem_dofs)});
  {
    // Get a queue
    cl::sycl::default_selector device_selector;
    cl::sycl::queue queue(device_selector);

    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    cl::sycl::buffer<double, 2> geom_buf(geometry.data(),
                                         {(std::size_t)npoints, 3});
    cl::sycl::buffer<int, 2> dm_buf(
        dofmap.data(),
        {(std::size_t)dofmap.rows(), (std::size_t)dofmap.cols()});
    cl::sycl::buffer<int, 1> fi_buf(flat_index.data(),
                                    (std::size_t)flat_index.size());
    cl::sycl::range<1> nelem_sycl{(std::size_t)nelem};

    queue.submit([&](cl::sycl::handler& cgh) {
      auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_dm = dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_fi = fi_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_ac = ac_buf.get_access<cl::sycl::access::mode::write>(cgh);

      auto kern = [=](cl::sycl::id<1> wiID) {
        const int i = wiID[0];

        double cell_geom[6];
        double b[3] = {0, 0, 0};

        double w[3] = {1., 2., 3.};

        // Pull out points for this cell
        for (int j = 0; j < 3; ++j)
        {
          const std::size_t dmi = access_dm[i][j];
          for (int k = 0; k < 2; ++k)
            cell_geom[j * 2 + k] = access_geom[dmi][k];
        }

        // Get local values
        tabulate_tensor_integral_cell_otherwise_70ac65295cde4ee9720845a2b33df1637e8afd59(
            b, w, nullptr, cell_geom, nullptr, nullptr, 0);

        // Insert result into array range corresponding to each dof
        for (int j = 0; j < nelem_dofs; ++j)
        {
          const std::size_t idx = access_fi[i * nelem_dofs + j];
          access_ac[idx] = b[j];
        }
      };
      cgh.parallel_for<AssemblyKernel>(nelem_sycl, kern);
    });
  }

  // Global RHS vector (into which accum_buf will be summed)
  Eigen::VectorXd global_vector(npoints);

  // Second kernel to accumulate for each dof
  {
    // Get a queue
    cl::sycl::default_selector device_selector;
    cl::sycl::queue queue(device_selector);

    std::cout << "Running on "
              << queue.get_device().get_info<sycl::info::device::name>()
              << "\n";

    cl::sycl::buffer<double, 1> gv_buf(global_vector.data(),
                                       global_vector.size());
    cl::sycl::buffer<int, 1> off_buf(dof_offsets.data(), dof_offsets.size());
    cl::sycl::range<1> npoints_sycl{(std::size_t)npoints};

    queue.submit([&](cl::sycl::handler& cgh) {
      auto access_gv = gv_buf.get_access<cl::sycl::access::mode::write>(cgh);
      auto access_ac = ac_buf.get_access<cl::sycl::access::mode::read>(cgh);
      auto access_off = off_buf.get_access<cl::sycl::access::mode::read>(cgh);

      auto kern = [=](cl::sycl::id<1> wiID) {
        const int i = wiID[0];
        access_gv[i] = 0.0;
        for (int j = access_off[i]; j < access_off[i + 1]; ++j)
          access_gv[i] += access_ac[j];
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
  Eigen::Array<double, 3, 2, Eigen::RowMajor> cell_geometry;

  for (int i = 0; i < nelem; ++i)
  {
    // Pull out points for this cell
    for (int j = 0; j < 3; ++j)
      cell_geometry.row(j) = geometry.row(dofmap(i, j)).head(2);

    b.setZero();
    // Get local values
    tabulate_tensor_integral_cell_otherwise_70ac65295cde4ee9720845a2b33df1637e8afd59(
        b.data(), w.data(), nullptr, cell_geometry.data(), nullptr, nullptr, 0);

    // Fill global vector
    for (int j = 0; j < nelem_dofs; ++j)
      global_vector2[dofmap(i, j)] += b[j];
  }

  std::cout << "Vector norm = " << std::setprecision(20) << global_vector.norm()
            << "\n";
  std::cout << "Vector norm 2 = " << std::setprecision(20)
            << global_vector2.norm() << "\n";
}
