#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <RcppEigen.h> 
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(cpp17)]]

#define MIN_ELEMS 5 

using namespace Eigen;
using namespace Rcpp;
using namespace std;

//Gaussian Kernal value
float std_normal(float x) {
  return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

//Computes \hat{p}(x)
float density_estimator(float bandwidth, Vector2d x, vector<Vector2d> points) {
    float sum = 0.0;
    for (size_t i = 0; i < points.size(); i++) {
        sum += std_normal((x - points[i]).norm() / bandwidth);
    }
    return sum / (points.size() * pow(bandwidth, 2));
}

//=====================================================================//
//                         Main Function                               //
//=====================================================================//

//[[Rcpp::export]]
DataFrame SCMS(MatrixXd data, float bandwidth, float threshold, int max_iterations, float epsilon, bool print_iter) {

    vector<Vector2d> points;;
    Vector2d pt;

    for (int i = 0; i < data.rows(); ++i) {
        pt(0) = data(i, 0);
        pt(1) = data(i, 1);
        points.push_back(pt);
    }

    //Step 1: Thresholding: remove x in X if \hat{p}(x) < threshold
    auto thresholded_points = points;
    auto remove_it = remove_if(thresholded_points.begin(), thresholded_points.end(), [&](auto x) {return density_estimator(bandwidth, x, points) < threshold; });
    thresholded_points.erase(remove_it, thresholded_points.end());
  
  Rcout << "No. of thresholded points: " << thresholded_points.size() << endl;
  
  //Step 2: Peform the following SCMS until convergence
  Rcout << "Performing SCMS..." << endl;
  
  float max_error = DBL_MAX;
  int iteration = 0;
  
  vector<pair<Vector2d, double>> thresholded_points_errors;
  for(size_t i = 0; i < thresholded_points.size(); ++i){
    pair<Vector2d, double> thresholded_point_error(thresholded_points[i], max_error);
    thresholded_points_errors.push_back(thresholded_point_error);
  }
  
  while ((max_error > epsilon) && (iteration < max_iterations)) {
    if (print_iter == true) Rcout << "    Iteration: " << iteration;
    //For each iteration, we find the maximum error
    max_error = 0.0;
    int points_moved = 0;
    
	for_each(thresholded_points_errors.begin(), thresholded_points_errors.end(), [&](auto& thresholded_point_error) {
		//Do nothing for converged pts
		if (thresholded_point_error.second >= epsilon) {
			points_moved += 1;
			Vector2d x = thresholded_point_error.first;

			//Compute The Hessian via mu[i] and c[i]
			Matrix2d H = Matrix2d::Zero();
			Vector2d mu(0, 0);
			float c = 0.0;
			Vector2d sum_cx(0, 0); //for msv
			float sum_c = 0.0;      //for msv

			for (size_t j = 0; j < points.size(); j++) {
				mu = (x - points[j]) / pow(bandwidth, 2);
				c = std_normal((x - points[j]).norm() / bandwidth);
				H += ((c / points.size())
					* ((mu * mu.transpose()) - (Matrix2d::Identity(2, 2) / pow(bandwidth, 2))));

				//for msv
				sum_cx = sum_cx + c * points[j];
				sum_c = sum_c + c;
			}

			//Find smallest d-1 eigenvectors
			SelfAdjointEigenSolver<MatrixXd> es(H);
			Matrix2d eigenvectors = es.eigenvectors();

			//sort in ascending order and drops the first element/col
			Eigen::Matrix<double, 2, 1> ev_red;
			ev_red = (eigenvectors.block<2, 1>(0, 0)).rowwise().reverse();

			//Move the point, msv = mean shift vector
			Vector2d msv = (sum_cx / sum_c) - x;
			Vector2d shift = ev_red * ev_red.transpose() * msv;
			thresholded_point_error.first = thresholded_point_error.first + shift;

			//Update errors
			float difference = shift.norm();
			thresholded_point_error.second = difference;

			max_error = max(max_error, difference);
		}
		});
    if (print_iter == true) Rcout << "    Points moved: " << points_moved << endl;
    ++iteration;
  }
  
  if (iteration == max_iterations) {
    Rcout << "    Max Iterations Reached." << endl;
  }
  
  Rcout << "Done." << endl;
  
  //Output: the collection of all remaining points
  for(size_t i = 0; i < thresholded_points.size(); ++i){
    thresholded_points[i] = thresholded_points_errors[i].first;
  }
  
  auto results = Eigen::Map<Eigen::MatrixXd>(thresholded_points[0].data(), 2, thresholded_points.size());
  //results = results.transpose();
  
  auto longitude = results.row(0).transpose();
  auto latitude = results.row(1).transpose();
  return DataFrame::create(Named("Longitude") = longitude,
						   Named("Latitude") = latitude);
}