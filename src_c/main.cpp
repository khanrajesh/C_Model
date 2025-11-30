#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#include "../shared_c/model/supervised/linear_regeression/LinearRegression.cpp"

namespace fs = std::filesystem;
using namespace std;
using namespace fs;

struct Dataset {
    vector<vector<double>>X;
    vector<double>y;
};

Dataset load_dataset(const fs::path& csv_path){
    ifstream in(csv_path);
    if(!in){
        throw runtime_error("Failed to open data set");
    }

    Dataset ds;
    string line;

    //skip header
    getline(in,line);

    while(getline(in, line)){
        if (line.empty()) continue;
        stringstream ss(line);
        string cell;
        vector<double> row;

        while(getline(ss,cell,',')){
            row.push_back(stod(cell));
        };
        
        if (row.size() < 2)continue;
        double target = row.back();
        row.pop_back();
        ds.X.push_back(row);
        ds.y.push_back(target);
    }

    if(ds.X.empty()){
        throw runtime_error("Data set is empty after parsing");
    }

    return ds;
}


int main(){

    try{
        path root = path(__FILE__).parent_path().parent_path();
        path data_path = root / "data" / "Student_Performance.csv";

        cout << "Using dataset: " << data_path.string() << "\n";

        Dataset ds = load_dataset(data_path);

        //import Linear Regeression here
        LinearRegression model = LinearRegression(0.0001);
        cout << "Starting the training" << "";
        model.fit(ds.X,ds.y, 100);

        // auto coeffs = model.coefficeients();
        vector<double> new_sample = {8,79,0,6,2};

        auto pred = model.predict(new_sample);
        cout << "\n \nExpected: 73.0, From 8,79,0,6,2: -> "<< pred << " \n";


    }catch (const exception& ex){
        cout << "Error: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}