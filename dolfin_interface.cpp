
#include "poisson.h"
#include <dolfinx.h>

using namespace dolfinx;

// Convert a dofmap (cell_dofs) to a flat index, i.e. for each cell dof, provide
// an index where it should store the value, so that values for the same dof
// are together.
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
  Eigen::ArrayXi flat_index(dof_offsets.back());
  int c = 0;
  for (int i = 0; i < dofmap.num_nodes(); ++i)
  {
    auto dofmap_i = dofmap.links(i);
    for (int j = 0; j < dofmap_i.size(); ++j)
      flat_index[c++] = tmp_offsets[dofmap_i[j]]++;
  }

  return graph::AdjacencyList<std::int32_t>(flat_index, dof_offsets);
}
//-----------------------------------------------------------------------------
// Flat index for LHS
graph::AdjacencyList<std::int32_t>
create_lhs_index(const graph::AdjacencyList<std::int32_t>& dofmap0,
                 const graph::AdjacencyList<std::int32_t>& dofmap1)
{
  assert(dofmap0.num_nodes() == dofmap1.num_nodes());

  const int ndofs0 = dofmap0.array().maxCoeff() + 1;
  std::vector<int> row_counter(ndofs0, 0);

  for (int i = 0; i < dofmap0.num_nodes(); ++i)
  {
    auto dofmap_i0 = dofmap0.links(i);
    for (int j = 0; j < dofmap_i0.size(); ++j)
      row_counter[dofmap_i0[j]] += dofmap1.num_links(i);
  }
  std::vector<int> row_offsets(ndofs0 + 1, 0);
  std::partial_sum(row_counter.begin(), row_counter.end(),
                   row_offsets.begin() + 1);
  std::vector<int> tmp_offsets(row_offsets.begin(), row_offsets.end());

  // Now we have row sizes, fill with column indices
  std::vector<int> cols(row_offsets.back());
  for (int i = 0; i < dofmap0.num_nodes(); ++i)
  {
    auto dofmap_i0 = dofmap0.links(i);
    auto dofmap_i1 = dofmap1.links(i);
    for (int j = 0; j < dofmap_i0.size(); ++j)
    {
      const int row = dofmap_i0[j];
      for (int k = 0; k < dofmap_i1.size(); ++k)
      {
        const int idx = tmp_offsets[row]++;
        cols[idx] = dofmap_i1[k];
      }
    }
  }

  // Get nnz on each row
  std::vector<int> nnz_offset = {0};
  std::vector<int> csr_cols(cols.size());
  nnz_offset.reserve(ndofs0 + 1);
  for (int i = 0; i < ndofs0; ++i)
  {
    std::vector<int> row(cols.data() + row_offsets[i],
                         cols.data() + row_offsets[i + 1]);
    // Get permutation into order
    std::vector<int> p(row.size());
    std::iota(p.begin(), p.end(), 0);
    std::sort(p.begin(), p.end(),
              [&](int a, int b) { return (row[a] < row[b]); });
    // Index unique entries
    std::vector<int> q(row.size());
    int last = -1;
    int nnz = -1;
    for (int j = 0; j < row.size(); ++j)
    {
      if (row[p[j]] != last)
        ++nnz;
      q[p[j]] = nnz;
      last = row[p[j]];
    }
    ++nnz;

    nnz_offset.push_back(nnz_offset.back() + nnz);
  }
}
//-----------------------------------------------------------------------------
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
