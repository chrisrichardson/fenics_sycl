
#include <CL/sycl.hpp>

#pragma once

// Submit assembly kernels to queue
void assemble_rhs(cl::sycl::queue& queue,
                  cl::sycl::buffer<double, 1>& accum_buf,
                  cl::sycl::buffer<double, 2>& geom_buf,
                  cl::sycl::buffer<int, 2>& coord_dm_buf,
                  cl::sycl::buffer<double, 2>& coeff_buf,
                  cl::sycl::buffer<int, 2>& fi_buf);

// Submit accumulation kernels to queue
void accumulate_rhs(cl::sycl::queue& queue, cl::sycl::buffer<double, 1>& ac_buf,
                    cl::sycl::buffer<double, 1>& global_vec_buf,
                    cl::sycl::buffer<int, 1>& offset_buf);
