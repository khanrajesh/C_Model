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

    

};
