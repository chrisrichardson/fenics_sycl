
#include "poisson.h"
#include <dolfinx.h>

using namespace dolfinx;

std::tuple<Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
           Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
           Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
create_arrays(int argc, char* argv[])
{
  common::SubSystemsManager::init_logging(argc, argv);
  common::SubSystemsManager::init_mpi(argc, argv, 0);

  auto cmap = fem::create_coordinate_map(create_coordinate_map_poisson);
  std::array<Eigen::Vector3d, 2> pt{Eigen::Vector3d(0.0, 0.0, 0.0),
                                    Eigen::Vector3d(1.0, 1.0, 0.0)};
  auto mesh = std::make_shared<mesh::Mesh>(generation::RectangleMesh::create(
      MPI_COMM_WORLD, pt, {{32, 32}}, cmap, mesh::GhostMode::none));

  const graph::AdjacencyList<std::int32_t>& x_dofmap
      = mesh->geometry().dofmap();

  // Dofmap will be fixed width for this mesh, so copy to simple 2D array
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> coord_dm(
      x_dofmap.num_nodes(), x_dofmap.num_links(0));
  std::copy(x_dofmap.array().data(),
            x_dofmap.array().data()
                + x_dofmap.num_nodes() * x_dofmap.num_links(0),
            coord_dm.data());

  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& coord_x
      = mesh->geometry().x();

  auto V = fem::create_functionspace(create_functionspace_form_poisson_a, "u",
                                     mesh);

  // Again copy dofmap to a 2D array
  const graph::AdjacencyList<std::int32_t>& v_dofmap = V->dofmap()->list();
  Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      function_dm(v_dofmap.num_nodes(), v_dofmap.num_links(0));
  std::copy(v_dofmap.array().data(),
            v_dofmap.array().data()
                + v_dofmap.num_nodes() * v_dofmap.num_links(0),
            function_dm.data());

  return {coord_dm, coord_x, function_dm};
}
