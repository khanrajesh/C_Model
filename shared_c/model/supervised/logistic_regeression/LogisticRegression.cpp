
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <cstddef>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <limits>

#include "../../../math/math.cpp"


using namespace std;
Math math;

class LogisticRegression{
public:
    explicit LogisticRegression(): theta_({}){}

    void fit(const vector<vector<double>>& X, vector<double>& Y,int epochs = 1000){

        //check for size 
        if (X.size() == 0 || Y.size()==0) throw invalid_argument("Mismatch training data or labels");


        size_t m = X.size();
        size_t feature = X[0].size(); //feature
        size_t n = feature +1; // intercepter + features

        //init the theta value as 0
        theta_.resize(n,0.0); //1+ no of featurs

        //step 0 -> add intercepter to the input vector
        vector<vector<double>> X_intercept = X;
        for(size_t i=0;i<m;i++){
            X_intercept[i].insert(X_intercept[i].begin(),1.0);
        }


        vector<vector<double>> w(m,vector<double>(m,0.0));
        vector<double> e(m,0.0);
        vector<double> h(m,0.0);
        vector<double> g(n,0.0);



        //into the epoch
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            //step 1 -> calculate z and h
            for (size_t i = 0; i < m; ++i)
            {
                double z_i = 0.0;
                for (size_t j = 0; j < n; ++j)
                {
                    z_i += theta_[j]*X_intercept[i][j];
                }
                h[i] = 1/(1+exp(-1*z_i));
            }
            
            //step 2 -> calculate e & w
            for (size_t i = 0; i < m; i++)
            {
                e[i] = Y[i]-h[i];
                w[i][i] = h[i]*(1-h[i]);
            }

            //step 3 -> calculate gradient
            for (size_t i = 0; i < n; i++)
            {
                double grad = 0.0;
                for (size_t j = 0; j < m; j++)
                {
                    grad += X_intercept[j][i]*e[j];
                }
                g[i] = grad;
            }

            // early stop if gradient is near zero
            double max_grad = 0.0;
            for (size_t i = 0; i < n; ++i) {
                max_grad = max(max_grad, fabs(g[i]));
            }
            if (max_grad < 1e-6) {
                cout << "Stopping early at epoch " << (epoch + 1)
                     << " due to small gradient (max |g|=" << max_grad << ")\n";
                break;
            }
            

            //step 4 -> calculating A
            vector<vector<double>> XT = math.transpose(X_intercept);
            vector<vector<double>> XT_W = math.dot(XT,w);
            vector<vector<double>> A = math.dot(XT_W, X_intercept);

            //add L2 regularization to keep A well-conditioned
            /*double lambda = 1e-3;
            for (size_t i = 0; i < n; ++i) {
                A[i][i] += lambda;
            }*/

            //step 5 -> calculate delta g  
            vector<vector<double>> A_inv = math.inverse(A);

            vector<vector<double>> g_mat(n, vector<double>(1, 0.0));
            for (size_t i = 0; i < n; ++i) {
                g_mat[i][0] = g[i];
            }

            //step 6 -> update theta_
            vector<vector<double>> delta_mat = math.dot(A_inv, g_mat);
            for (size_t i = 0; i < n; ++i) {
                theta_[i] += delta_mat[i][0];
            }

            //step 7 -> calculate mse and report
            double mse_epoch = 0.0;
            for (size_t i = 0; i < m; ++i) {
                mse_epoch += e[i] * e[i];
            }
            mse_epoch /= static_cast<double>(m);

            ostringstream oss;
            oss << "epoch " << (epoch + 1) << " mse=" << mse_epoch << " theta=[";
            for (size_t i = 0; i < theta_.size(); ++i) {
                oss << theta_[i];
                if (i + 1 != theta_.size()) oss << ", ";
            }
            oss << "] h=[";
            size_t show_h = min<size_t>(h.size(), 5);
            for (size_t i = 0; i < show_h; ++i) {
                if (i) oss << ",";
                oss << h[i];
            }
            if (h.size() > show_h) {
                oss << ", ...";
            }
            oss << "] e=[";
            size_t show_e = min<size_t>(e.size(), 5);
            for (size_t i = 0; i < show_e; ++i) {
                if (i) oss << ",";
                oss << e[i];
            }
            if (e.size() > show_e) {
                oss << ", ...";
            }
            oss << "]";
            cout << oss.str() << endl;

        }
        

    }


    double predict(const vector<double>& X)const{
        if (theta_.empty()) {
            throw runtime_error("Model is not trained yet");
        }

        vector<double> features = X;
        //adding intercept
        if (features.size() + 1 == theta_.size()) {
            features.insert(features.begin(), 1.0);
        } else if (features.size() != theta_.size()) {
            throw invalid_argument("Input feature size mismatch");
        }

        //calculate the z value
        double z = 0.0;
        for (size_t i = 0; i < features.size(); ++i) {
            z += features[i] * theta_[i];
        }

        double prob = 1.0 / (1.0 + exp(-z));
        return prob;
    }


private:
    vector<double> theta_;


};

