
#include "assemble_rhs.h"
#include <iostream>

// Need to include C file in same translation unit as lambda
#include "poisson.c"
#define tabulate_L                                                             \
  tabulate_tensor_integral_cell_otherwise_d5110eeede6a6d5fef6fa5fa9759f7bdb8a989e8

class AssemblyKernel;

void assemble_rhs(cl::sycl::queue& queue,
                  cl::sycl::buffer<double, 1>& accum_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf,
                  cl::sycl::buffer<int, 2>& fi_buf)
{
  queue.submit([&](cl::sycl::handler& cgh) {
    auto access_geom = geom_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_cdm
        = coord_dm_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_fi = fi_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_coeff = coeff_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_ac = accum_buf.get_access<cl::sycl::access::mode::write>(cgh);
    cl::sycl::range<2> coord_dims = coord_dm_buf.get_range();
    cl::sycl::range<2> fi_dims = fi_buf.get_range();
    cl::sycl::range<1> nelem_sycl{fi_dims[0]};
    int nelem_dofs = fi_dims[1];
    int ncoeff = coeff_buf.get_range()[1];
    int gdim = 2;

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];

      double cell_geom[6];
      double w[3] = {1, 2, 3};
      double b[3] = {0};
      double c[1] = {0};
      //      for (int j = 0; j < ncoeff; ++j)
      //        w[j] = access_coeff[i][j];

      // Pull out points for this cell
      for (std::size_t j = 0; j < coord_dims[1]; ++j)
      {
        const std::size_t dmi = access_cdm[i][j];
        for (int k = 0; k < gdim; ++k)
          cell_geom[j * gdim + k] = access_geom[dmi][k];
      }

      // Get local values
      tabulate_L(b, w, c, cell_geom, nullptr, nullptr, 0);

      // Insert result into array range corresponding to each dof
      for (int j = 0; j < nelem_dofs; ++j)
      {
        const std::size_t idx = access_fi[i][j];
        access_ac[idx] = b[j];
      }
    };
    cgh.parallel_for<AssemblyKernel>(nelem_sycl, kern);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}

class AccumulationKernel;

void accumulate_rhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& ac_buf,
                    cl::sycl::buffer<double, 1>& global_vec_buf,
                    cl::sycl::buffer<int, 1>& offset_buf)
{
  // Second kernel to accumulate RHS for each dof

  cl::sycl::range<1> ndofs_sycl = offset_buf.get_range() - 1;

  queue.submit([&](cl::sycl::handler& cgh) {
    auto access_gv
        = global_vec_buf.get_access<cl::sycl::access::mode::write>(cgh);
    auto access_ac = ac_buf.get_access<cl::sycl::access::mode::read>(cgh);
    auto access_off = offset_buf.get_access<cl::sycl::access::mode::read>(cgh);

    auto kern = [=](cl::sycl::id<1> wiID) {
      const int i = wiID[0];
      access_gv[i] = 0.0;
      for (int j = access_off[i]; j < access_off[i + 1]; ++j)
        access_gv[i] += access_ac[j];
    };
    cgh.parallel_for<AccumulationKernel>(ndofs_sycl, kern);
  });

  try
  {
    queue.wait_and_throw();
  }
  catch (sycl::exception const& e)
  {
    std::cout << "Caught synchronous SYCL exception:\n"
              << e.what() << std::endl;
  }
}
