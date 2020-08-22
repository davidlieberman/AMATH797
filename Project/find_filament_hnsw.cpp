#define _USE_MATH_DEFINES
#include <cmath>
#include <algorithm>
#include <vector>
#include <queue>
#include <Eigen/dense>
#include "hnswlib/hnswlib.h"
#include <iostream>
#include <iomanip>
#include <fstream>
//#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(cpp17)]]

#define MIN_ELEMS 5 

using namespace Eigen;
using namespace hnswlib;
//using namespace Rcpp;
using namespace std;


float* vectorToPointer(const Vector2f& pt) {
    float* res = new float[2];
    res[0] = pt(0);
    res[1] = pt(1);
    return res;
}

vector<Vector2f> nearest_neighbors(Vector2f x, const vector<Vector2f> points, const float sigma, const HierarchicalNSW<float>& appr_alg) {
    auto* query = vectorToPointer(x);
    vector<Vector2f> nNs;

    float dist = 0.0;
    size_t idx, i = 1;

    while (dist <= 3 * sigma) {
        std::priority_queue<std::pair<float, labeltype>> result = appr_alg.searchKnn(query, i);

        dist = sqrt(result.top().first);
        idx = result.top().second;
    	nNs.push_back(points[idx]);
        ++i;
    }
    delete(query);
    return nNs;
}

//Gaussian Kernal value
float std_normal(const float x) {
    return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

//Computes \hat{p}(x)
float density_estimator(Vector2f x, const vector<Vector2f> points, const float bandwidth, const HierarchicalNSW<float>& appr_alg) {
    float sum = 0.0;

    //Find nearest neighbors
    vector<Vector2f> nNs = nearest_neighbors(x, points, bandwidth, appr_alg);
    for (size_t i = 0; i < nNs.size(); i++) {
        sum += std_normal((x - nNs[i]).norm() / bandwidth);
    }
    return sum / (points.size() * pow(bandwidth, 2));
}

//=====================================================================//
//                         Main Function                               //
//=====================================================================//

//[[Rcpp::export]]
MatrixXf SCMS(const MatrixXf data, const float bandwidth, const float threshold, const int max_iterations, const float epsilon, const bool print_iter) {

    const size_t dim = 2;
    const size_t n_points = data.rows();
    vector<Vector2f> points(n_points);

    L2Space l2space(dim);
    HierarchicalNSW<float> appr_alg(&l2space, n_points);
    
    for (size_t i = 0; i < n_points; ++i) {
        Vector2f pt(data(i, 0), data(i, 1));
        points[i] = pt;

        auto* ptr_pt = vectorToPointer(pt);
        appr_alg.addPoint(ptr_pt, i);
        delete(ptr_pt);
    }

    appr_alg.setEf(100);


    //Step 1: Thresholding: remove x in X if \hat{p}(x) < threshold
    auto thresholded_points = points;
    auto remove_it = remove_if(thresholded_points.begin(), thresholded_points.end(), [&](auto x) {return density_estimator(x, points, bandwidth, appr_alg) < threshold; });
    thresholded_points.erase(remove_it, thresholded_points.end());
	
    std::cout << "No. of thresholded points: " << thresholded_points.size() << endl;

    //Step 2: Peform the following SCMS until convergence
    std::cout << "Performing SCMS..." << endl;

    float max_error = FLT_MAX;
    int iteration = 0;

    vector<pair<Vector2f, float>> thresholded_points_errors;
    for (size_t i = 0; i < thresholded_points.size(); ++i) {
        pair<Vector2f, float> thresholded_point_error(thresholded_points[i], max_error);
        thresholded_points_errors.push_back(thresholded_point_error);
    }

    while ((max_error > epsilon) && (iteration < max_iterations)) {
        if (print_iter == true) std::cout << "    Iteration: " << iteration;
        //For each iteration, we find the maximum error
        max_error = 0.0;
        int points_moved = 0;

        for_each(thresholded_points_errors.begin(), thresholded_points_errors.end(), [&](auto& thresholded_point_error) {
            //Do nothing for converged pts
            if (thresholded_point_error.second >= epsilon) {
                points_moved += 1;
                Vector2f x = thresholded_point_error.first;

                //Compute The Hessian via mu[i] and c[i]
                Matrix2f H = Matrix2f::Zero();
                Vector2f mu(0, 0);
                float c = 0.0;
                Vector2f sum_cx(0, 0); //for msv
                float sum_c = 0.0;      //for msv

                //Find points that are close enough (within 3*bandwidth <=> 3 sigma)
                vector<Vector2f> nNs = nearest_neighbors(x, points, bandwidth, appr_alg);

                for (size_t j = 0; j < nNs.size(); j++) {
                    mu = (x - nNs[j]) / pow(bandwidth, 2);
                    c = std_normal((x - nNs[j]).norm() / bandwidth);
                    H += ((c / nNs.size())
                        * ((mu * mu.transpose()) - (Matrix2f::Identity(2, 2) / pow(bandwidth, 2))));

                    //for msv
                    sum_cx = sum_cx + c * nNs[j];
                    sum_c = sum_c + c;
                }

                //Find smallest d-1 eigenvectors
                SelfAdjointEigenSolver<MatrixXf> es(H);
                Matrix2f eigenvectors = es.eigenvectors();

                //sort in ascending order and drops the first element/col
                Eigen::Matrix<float, 2, 1> ev_red;
                ev_red = (eigenvectors.block<2, 1>(0, 0)).rowwise().reverse();

                //Move the point, msv = mean shift vector
                Vector2f msv = (sum_cx / sum_c) - x;
                Vector2f shift = ev_red * ev_red.transpose() * msv;
                thresholded_point_error.first = thresholded_point_error.first + shift;

                //Update errors
                float difference = shift.norm();
                thresholded_point_error.second = difference;

                max_error = max(max_error, difference);            }
            });
        if (print_iter == true) std::cout << "    Points moved: " << points_moved << endl;
        ++iteration;
    }

    if (iteration == max_iterations) {
        std::cout << "    Max Iterations Reached." << endl;
    }

    std::cout << "Done." << endl;
	
	//Output: the collection of all remaining points
    for (size_t i = 0; i < thresholded_points.size(); ++i) {
        thresholded_points[i] = thresholded_points_errors[i].first;
    }

    auto results = Eigen::Map<Eigen::MatrixXf>(thresholded_points[0].data(), 2, thresholded_points.size());
    //return as<NumericMatrix>(wrap((results.transpose())));
    return results.transpose();
}

MatrixXd readCSV(const string filepath) {
    ifstream input(filepath);
    string line;
    MatrixXd toReturn(3000000, 2);
    int i = 0;
    while (getline(input, line)) {
        double x, y;
        sscanf_s(line.c_str(), "%Lf, %Lf", &x, &y, 2);
        toReturn(i, 0) = x;
        toReturn(i, 1) = y;
        ++i;
    }

    toReturn.conservativeResize(i, 2);
    return toReturn;
}

void writeCSV(const MatrixXf output) {
    ofstream file("C:/Users/darkg/Desktop/output.csv");
    for (int i = 0; i < output.rows(); ++i) {
        file << std::setprecision(15) << output(i, 0) << "," << output(i, 1) << endl;
    }
}

int main() {
    MatrixXd input_d = readCSV("C:/Users/darkg/Desktop/input.csv");
    Eigen::MatrixXf input_f = input_d.cast <float>();
    MatrixXf output = SCMS(input_f, 0.55, 0.025, 10000, 0.001, false);
    writeCSV(output);
    return 0;
}