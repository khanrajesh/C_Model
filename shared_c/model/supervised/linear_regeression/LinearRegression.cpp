
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


class LinearRegression {
public:
    explicit LinearRegression(double learning_rate = 0.000001):learning_rate(learning_rate), theta_({}){}

    void fit(const vector<vector<double>>& X, const vector<double>& Y, int epochs = 1000){
        if (X.size() == 0 || Y.size()==0) throw invalid_argument("Empty training data or labels");

        size_t m = X.size();
        size_t features = X[0].size();

        if (Y.size() != m) throw invalid_argument("Mismatched no of sample and labels");

        //init thera value as 0
        theta_.resize(features+1,0.0);

        //add intercepter
        vector<vector<double>> X_intercept = X;
        for(size_t i = 0; i< m; ++i){
            X_intercept[i].insert(X_intercept[i].begin(),1.0);
        }

        vector<double> error(m,0.0);
        vector<double> gradient(features+1,0.0);
        vector<double> h(m,0.0);


        for(int epoch = 0; epoch <epochs; ++epoch){
            //step 1 : Calculate the hypothesis
            for(size_t i = 0; i <m; ++i){
                double hypo = 0.0;
                for(size_t j = 0; j< features+1;++j){
                    hypo += X_intercept[i][j] * theta_[j];
                }
                h[i] = hypo;
            }

            //step 2 : Calcultate the error
            for (size_t i = 0; i < m; i++){
                error[i] = h[i]-Y[i];
            }

            //step 3 : Calculate the Gradient
            for (size_t i = 0; i < features+1; i++){
                double grad = 0.0;
                for (size_t j = 0; j < m; ++j){
                    grad += error[j] * X_intercept[j][i];
                }
                gradient[i] = (1.0/m) * grad;
            }

            //Step 5: cost function 1/2 SME 
            double mse_epoch = 0.0;
            for(double e : error) mse_epoch += e*e;
            mse_epoch /= (2.0 * m);

            //step 4: Update the theta_
            vector<double> new_theta = theta_;
            for (size_t i = 0; i < features +1; i++){
                double update = theta_[i] - learning_rate * gradient[i];

                new_theta[i] = update;
            }

            theta_ = move(new_theta);

              // Report end-of-epoch progress in one line
            ostringstream oss;
            oss << "epoch " << (epoch + 1) << " half_mse=" << mse_epoch << " theta=[";
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
            oss << "] error=[";
            size_t show_err = min<size_t>(error.size(), 5);
            for (size_t i = 0; i < show_err; ++i) {
                if (i) oss << ",";
                oss << error[i];
            }
            if (error.size() > show_err) {
                oss << ", ...";
            }
            oss << "]";
            cout << oss.str() << endl;
        }

    }

    double predict(const vector<double>& X) const {
        if (theta_.empty()){
            throw runtime_error("Model is not trained yet");
        }

        vector<double> features = X;
        
        if(features.size()+1 == theta_.size()){
            features.insert(features.begin(),1.0); //adding interceptor here
        }else if (features.size() != theta_.size()){
            throw invalid_argument("Input feature size mismatch");
        }

        double result = 0.0;
        for (size_t i = 0; i < features.size(); i++){
            result += features[i]*theta_[i];
        }

        return result;
    }

    vector<double> coefficeients() const {
        return theta_;
    }

private:
    double learning_rate;
    vector<double> theta_;
};

