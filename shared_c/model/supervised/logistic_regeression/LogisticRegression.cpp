
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

using namespace std;

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


        vector<double> w(m,0.0);
        vector<double> e(m,0.0);
        vector<double> h(m,0.0);
        vector<double> g(n,0.0);


        //into the epoch
        for (size_t epoch = 0; epoch < epochs; ++epoch)
        {
            //step 1 -> calculate z and h
            for (size_t i = 0; i < m; ++i)
            {
                double z_i = 0.0;
                for (size_t j = 0; j < n; i++)
                {
                 z_i += theta_[j]*X_intercept[i][j];
                }
                h[i] = 1/(1+exp(-1*z_i));
            }
            
            //step 2 -> calculate e & w
            for (size_t i = 0; i < m; i++)
            {
                e[i] = Y[i]-h[i];
                w[i] = h[i]*(1-h[i]);
            }

            //step 3 -> calculate gradient
            for (size_t i = 0; i < n; i++)
            {
                double grad = 0.0;
                for (size_t j = 0; j < m; j++)
                {
                   grad += X[j][i]*e[i];
                }
                g[i] = grad;
            }
            
            //step 4 -> 


        }
        

    }


    double predict(const vector<double>& X)const{
        //todo
    }


private:
    vector<double> theta_;


};