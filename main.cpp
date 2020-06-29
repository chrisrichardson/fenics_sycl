
#include <CL/sycl.hpp>

#include <dolfinx.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "assemble_rhs.h"
#include "dolfin_interface.h"
#include "poisson.h"

void exception_handler(sycl::exception_list exceptions)
{
  for (std::exception_ptr const& e : exceptions)
  {
    try
    {
      std::rethrow_exception(e);
    }
    catch (sycl::exception const& e)
    {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
}

using namespace dolfinx;

graph::AdjacencyList<std::int32_t>
create_flat_index(const graph::AdjacencyList<std::int32_t>& dofmap)
{
  const int ndofs = dofmap.array().maxCoeff() + 1;

  // Count the number of occurences of each dof
  std::vector<int> dof_counter(ndofs, 0);
  for (int i = 0; i < dofmap.num_nodes(); ++i)
  {
    auto dofmap_i = dofmap.links(i);
    for (int j = 0; j < dofmap_i.size(); ++j)
      ++dof_counter[dofmap_i[j]];
  }
  std::vector<int> dof_offsets(ndofs + 1, 0);
  std::partial_sum(dof_counter.begin(), dof_counter.end(),
                   dof_offsets.begin() + 1);
  assert(dof_offsets.back() == dofmap.array().size());

  std::vector<int> tmp_offsets(dof_offsets.begin(), dof_offsets.end());
  std::vector<int> flat_index(dof_offsets.back());
  int c = 0;
  for (int i = 0; i < dofmap.num_nodes(); ++i)
  {
    auto dofmap_i = dofmap.links(i);
    for (int j = 0; j < dofmap_i.size(); ++j)
      flat_index[c++] = tmp_offsets[dofmap_i[j]]++;
  }

  return graph::AdjacencyList<std::int32_t>(flat_index, dof_offsets);
}

// Simple code to assemble a dummy RHS vector over some dummy geometry and
// dofmap
int main(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_mpi(argc, argv, 0);
  common::SubSystemsManager::init_petsc(argc, argv);

  auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 0.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{320, 320}}, cmap, mesh::GhostMode::none));

  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // Dofmap will be fixed width for this mesh, so copy to simple 2D array
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coord_dm(
      x_dofmap.num_nodes(), x_dofmap.num_links(0));
  std::copy(x_dofmap.array().data(),
            x_dofmap.array().data()
                + x_dofmap.num_nodes() * x_dofmap.num_links(0),
            coord_dm.data());

  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& geometry
      = mesh->geometry().x();

  auto V = fem::create_functionspace(create_functionspace_form_poisson_a, "u",
                                     mesh);
  std::shared_ptr<fem::Form> L = fem::create_form(create_form_poisson_L, {V});
  
 const graph::AdjacencyList<std::int32_t>& v_dofmap = V->dofmap()->list();
  // Create permuted dofmap for insertion into array
  const graph::AdjacencyList<std::int32_t> flat_dm
      = create_flat_index(v_dofmap);

  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      function_dm(v_dofmap.num_nodes(), v_dofmap.num_links(0));
  std::copy(v_dofmap.array().data(),
            v_dofmap.array().data()
                + v_dofmap.num_nodes() * v_dofmap.num_links(0),
            function_dm.data());

  auto f = std::make_shared<function::Function>(V);
  f->interpolate([](auto& x) {
    auto dx = Eigen::square(x - 0.5);
    return 10.0 * Eigen::exp(-(dx.row(0) + dx.row(1)) / 0.02);
  });
  L->set_coefficients({{"f", f}});

  Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    coeff = fem::pack_coefficients(*L);

  int nelem = function_dm.rows();
  int nelem_dofs = function_dm.cols();

  const int ndofs = function_dm.maxCoeff() + 1;
  std::cout << "ndofs = " << ndofs << "\n";

  const int npoints = coord_dm.maxCoeff() + 1;
  std::cout << "npoints = " << npoints << "\n";

  // Get a queue
  cl::sycl::default_selector device_selector;
  cl::sycl::queue queue(device_selector, exception_handler);
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
        flat_dm.array().data(), {(std::size_t)nelem, (std::size_t)nelem_dofs});

    assemble_rhs(queue, ac_buf, geom_buf, coord_dm_buf, coeff_buf, fi_buf);
  }

  // Global RHS vector (into which accum_buf will be summed)
  Eigen::VectorXd global_vector(ndofs);

  // Second kernel to accumulate RHS for each dof
  {
    cl::sycl::buffer<double, 1> gv_buf(global_vector.data(),
                                       global_vector.size());
    cl::sycl::buffer<int, 1> off_buf(flat_dm.offsets().data(),
                                     flat_dm.offsets().size());
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
    tabulate_L(b.data(), coeff.row(i).data(), nullptr, cell_geometry.data(), nullptr,
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
