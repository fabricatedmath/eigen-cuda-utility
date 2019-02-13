#pragma once

#include <Eigen/Dense>

using namespace Eigen;

template<typename T>
using MatrixRX = Matrix<T, Dynamic, Dynamic, RowMajor>;

template<typename T>
using VectorX = Matrix<T, Dynamic, 1>;
