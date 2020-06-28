
#include <CL/sycl.hpp>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "assemble_rhs.h"
#include "dolfin_interface.h"
#include "poisson.h"

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main(int argc, char* argv[])
{

  // Get dolfin data
  auto [coord_dm, geometry, function_dm] = create_arrays(argc, argv);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coeff(
      function_dm.rows(), 3);
  coeff.col(0) = 1;
  coeff.col(1) = 2;
  coeff.col(2) = 3;

  int nelem = function_dm.rows();
  int nelem_dofs = function_dm.cols();

  const int ndofs = function_dm.maxCoeff() + 1;
  std::cout << "ndofs = " << ndofs << "\n";

  const int npoints = coord_dm.maxCoeff() + 1;
  std::cout << "npoints = " << npoints << "\n";

  // Count number of entries for each dof and make offset array
  std::vector<int> dof_counter(ndofs, 0);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
      ++dof_counter[function_dm(i, j)];
  }
  std::vector<int> dof_offsets(ndofs + 1, 0);
  std::partial_sum(dof_counter.begin(), dof_counter.end(),
                   dof_offsets.begin() + 1);
  assert(dof_offsets.back() == nelem * nelem_dofs);

  // Make a "flat index" array listing where each assembly entry should be
  // placed. This is just a permutation, so that the entries that belong to the
  // same dof index are next to each other.
  std::vector<int> tmp_offsets(dof_offsets.begin(), dof_offsets.end());
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> flat_index(
      nelem, nelem_dofs);
  for (int i = 0; i < nelem; ++i)
  {
    for (int j = 0; j < nelem_dofs; ++j)
      flat_index(i, j) = tmp_offsets[function_dm(i, j)]++;
  }

  // Get a queue
  cl::sycl::default_selector device_selector;
  cl::sycl::queue queue(device_selector);
  std::cout << "Running on "
            << queue.get_device().get_info<sycl::info::device::name>() << "\n";

  auto timer_start = std::chrono::system_clock::now();

  // Device memory to accumulate assembly entries before summing
  cl::sycl::buffer<double, 1> ac_buf(
      cl::sycl::range<1>{(std::size_t)(nelem * nelem_dofs)});
  {
    cl::sycl::buffer<int, 2> coord_dm_buf(
        coord_dm.data(),
        {(std::size_t)coord_dm.rows(), (std::size_t)coord_dm.cols()});

    cl::sycl::buffer<double, 2> geom_buf(geometry.data(),
                                         {(std::size_t)npoints, 3});

    cl::sycl::buffer<double, 2> coeff_buf(
        coeff.data(), {(std::size_t)coeff.rows(), (std::size_t)coeff.cols()});

    cl::sycl::buffer<int, 2> fi_buf(
        flat_index.data(),
        {(std::size_t)flat_index.rows(), (std::size_t)flat_index.cols()});

    assemble_rhs(queue, ac_buf, geom_buf, coord_dm_buf, coeff_buf, fi_buf);
  }

  // Global RHS vector (into which accum_buf will be summed)
  Eigen::VectorXd global_vector(ndofs);

  // Second kernel to accumulate RHS for each dof
  {
    cl::sycl::buffer<double, 1> gv_buf(global_vector.data(),
                                       global_vector.size());
    cl::sycl::buffer<int, 1> off_buf(dof_offsets.data(), dof_offsets.size());
    accumulate_rhs(queue, ac_buf, gv_buf, off_buf);
  }

  auto timer_end = std::chrono::system_clock::now();
  std::chrono::duration<double> dt = (timer_end - timer_start);
  std::cout << "gpu=" << dt.count() << "s.\n";

  timer_start = std::chrono::system_clock::now();

  // Comparison CPU code below
  //--------------------------

  Eigen::VectorXd global_vector2(ndofs);
  global_vector2.setZero();

  auto form = create_form_poisson_L();
  auto integral = form->create_cell_integral(-1);
  auto tabulate_L = integral->tabulate_tensor;

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
      cell_geometry.row(j) = geometry.row(coord_dm(i, j)).head(2);

    b.setZero();
    // Get local values
    tabulate_L(b.data(), w.data(), nullptr, cell_geometry.data(), nullptr,
               nullptr, 0);

    // Fill global vector
    for (int j = 0; j < nelem_dofs; ++j)
      global_vector2[function_dm(i, j)] += b[j];
  }

  timer_end = std::chrono::system_clock::now();
  dt = (timer_end - timer_start);
  std::cout << "cpu=" << dt.count() << "s.\n";

  std::cout << "Vector norm = " << std::setprecision(20) << global_vector.norm()
            << "\n";
  std::cout << "Vector norm 2 = " << std::setprecision(20)
            << global_vector2.norm() << "\n";
}
