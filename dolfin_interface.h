
#include <Eigen/Dense>
#include <tuple>

#pragma once

std::tuple<
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>,
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
> create_arrays(int argc, char* argv[]);
