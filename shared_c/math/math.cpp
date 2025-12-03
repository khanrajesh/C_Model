#include<cmath>
#include<vector>
#include<string>
#include <algorithm>
#include <stdexcept>


using namespace std;

class Math{
public:
    vector<vector<double>> transpose(vector<vector<double>>& A){
        if (A.empty()) return {};

        const size_t rows = A.size();
        const size_t cols = A.front().size();
        vector<vector<double>> result(cols, vector<double>(rows, 0.0));

        for (size_t i = 0; i < rows; ++i) {
            if (A[i].size() != cols) {
                throw invalid_argument("Input matrix rows must all be the same length");
            }
            for (size_t j = 0; j < cols; ++j) {
                result[j][i] = A[i][j];
            }
        }

        return result;
    }

    vector<vector<double>> dot(vector<vector<double>>& A, vector<vector<double>>& B){
        if (A.empty() || B.empty()) {
            throw invalid_argument("Input matrices must not be empty");
        }

        const size_t a_rows = A.size();
        const size_t a_cols = A.front().size();
        const size_t b_rows = B.size();
        const size_t b_cols = B.front().size();

        for (const auto& row : A) {
            if (row.size() != a_cols) {
                throw invalid_argument("First matrix rows must all be the same length");
            }
        }
        for (const auto& row : B) {
            if (row.size() != b_cols) {
                throw invalid_argument("Second matrix rows must all be the same length");
            }
        }

        if (a_cols != b_rows) {
            throw invalid_argument("Matrix dimensions are incompatible for dot product");
        }

        vector<vector<double>> result(a_rows, vector<double>(b_cols, 0.0));
        for (size_t i = 0; i < a_rows; ++i) {
            for (size_t k = 0; k < a_cols; ++k) {
                const double a_val = A[i][k];
                for (size_t j = 0; j < b_cols; ++j) {
                    result[i][j] += a_val * B[k][j];
                }
            }
        }

        return result;
    }

    
    vector<vector<double>> inverse(vector<vector<double>> A){
        if (A.empty()) {
            throw invalid_argument("Input matrix must not be empty");
        }

        const size_t n = A.size();
        for (const auto& row : A) {
            if (row.size() != n) {
                throw invalid_argument("Matrix must be square to compute inverse");
            }
        }

        // Augment A with identity matrix: [A | I]
        vector<vector<double>> aug(n, vector<double>(2 * n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                aug[i][j] = A[i][j];
            }
            aug[i][n + i] = 1.0;
        }

        // Gauss-Jordan elimination
        for (size_t i = 0; i < n; ++i) {
            // Pivot: find row with max |aug[row][i]|
            size_t maxRow = i;
            for (size_t k = i + 1; k < n; ++k) {
                if (fabs(aug[k][i]) > fabs(aug[maxRow][i])) {
                    maxRow = k;
                }
            }
            if (fabs(aug[maxRow][i]) < 1e-12) {
                throw runtime_error("Matrix is singular and cannot be inverted");
            }
            if (maxRow != i) {
                swap(aug[i], aug[maxRow]);
            }

            // Normalize pivot row
            double pivot = aug[i][i];
            for (size_t j = 0; j < 2 * n; ++j) {
                aug[i][j] /= pivot;
            }

            // Eliminate other rows
            for (size_t r = 0; r < n; ++r) {
                if (r == i) continue;
                double factor = aug[r][i];
                for (size_t j = 0; j < 2 * n; ++j) {
                    aug[r][j] -= factor * aug[i][j];
                }
            }
        }

        // Extract inverse from augmented matrix
        vector<vector<double>> inv(n, vector<double>(n, 0.0));
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                inv[i][j] = aug[i][n + j];
            }
        }

        return inv;
    }

};
