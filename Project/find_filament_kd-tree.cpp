#define _USE_MATH_DEFINES
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <Eigen/dense>
#include <iostream>
#include <iomanip>
#include <fstream>
//#include <RcppEigen.h>
//[[Rcpp::depends(RcppEigen)]]
//[[Rcpp::plugins(cpp17)]]

#define MIN_ELEMS 5 

using namespace Eigen;
//using namespace Rcpp;
using namespace std;

struct point {
    double x, y;
    int subset_no;
    double radius;
};

struct box {
    double min_x, max_x, min_y, max_y;
};

struct sorted_points {
    vector<point> x, y;
};

struct kd_node {
    vector<point> points;
    box bounding_box;
    kd_node* right;
    kd_node* left;
};

typedef struct kd_node* kd_tree;

int largest_dim(sorted_points& P) {
    double dx = P.x.back().x - P.x.front().x;
    double dy = P.y.back().y - P.y.front().y;
    if (dx >= dy) return 0;
    else return 1;
}

//splits the points by the median of the largest dimension
//requires that there are at least 2 points
vector<sorted_points> split(sorted_points& P) {
    vector<sorted_points> split_vectors;
    int dim = largest_dim(P);
    sorted_points left, right;

    vector<point>& xpts = P.x;
    vector<point>& ypts = P.y;

    if (dim == 0) {
        size_t m = xpts.size() / 2;
        double x_cut = xpts[m].x;
        for (size_t i = 0; i < xpts.size(); i++) {
            if (xpts[i].x < x_cut) left.x.push_back(xpts[i]);
            else right.x.push_back(xpts[i]);

            if (ypts[i].x < x_cut) left.y.push_back(ypts[i]);
            else right.y.push_back(ypts[i]);
        }
    }

    else if (dim == 1) {
        size_t m = ypts.size() / 2;
        double y_cut = ypts[m].y;
        for (size_t i = 0; i < ypts.size(); i++) {
            if (ypts[i].y < y_cut) left.y.push_back(ypts[i]);
            else right.y.push_back(ypts[i]);

            if (xpts[i].y < y_cut) left.x.push_back(xpts[i]);
            else right.x.push_back(xpts[i]);
        }
    }

    split_vectors.push_back(left);
    split_vectors.push_back(right);
    return split_vectors;
}

box find_bounding_box(vector<point>& P) {
    box B;
    B.min_x = 99999.9;
    B.max_x = -99999.9;
    B.min_y = 99999.9;
    B.max_y = -99999.9;

    for (size_t i = 0; i < P.size(); i++) {
        point pt = P[i];
        if (pt.x < B.min_x) { B.min_x = pt.x; }
        if (pt.x > B.max_x) { B.max_x = pt.x; }
        if (pt.y < B.min_y) { B.min_y = pt.y; }
        if (pt.y > B.max_y) { B.max_y = pt.y; }
    }
    return B;
}

box merge_box(box B1, box B2) {
    box B0;
    B0.min_x = min(B1.min_x, B2.min_x);
    B0.max_x = max(B1.max_x, B2.max_x);
    B0.min_y = min(B1.min_y, B2.min_y);
    B0.max_y = max(B1.max_y, B2.max_y);
    return B0;
}

bool cmp_pts_x(point pt1, point pt2) {
    return pt1.x < pt2.x;
}
bool cmp_pts_y(point pt1, point pt2) {
    return pt1.y < pt2.y;
}

kd_node* make_kdtree(sorted_points& P) {
    kd_node* T = new kd_node;

    // If there are less then MIN_ELEMS points, this is a leaf
    // The factor of 2 is because we can't split otherwise
    if (P.x.size() < 2 * MIN_ELEMS) {
        T->points = P.x;
        T->bounding_box = find_bounding_box(P.x);
        T->left = NULL;
        T->right = NULL;
    }
    else { //Else it is a node and we recursively build the tree
        vector<sorted_points> split_vectors = split(P);
        T->left = make_kdtree(split_vectors[0]);
        T->right = make_kdtree(split_vectors[1]);
        T->bounding_box = merge_box(T->left->bounding_box,
            T->right->bounding_box);
    }
    return T;
}

//Free Memory of kdtree
void free_kdtree(kd_tree T) {
    if (T != NULL && T->left != NULL) {
        free_kdtree(T->left);
        free_kdtree(T->right);
    }
    delete(T);
}

kd_tree new_kdtree(vector<point> points) {
    sorted_points P;
    vector<point> x_points(points);
    vector<point> y_points(points);
    sort(x_points.begin(), x_points.end(), cmp_pts_x);
    sort(y_points.begin(), y_points.end(), cmp_pts_y);
    P.x = x_points;
    P.y = y_points;
    return make_kdtree(P);
}

//Returns true if T is a leaf node and contains points
bool is_leaf(kd_tree T) {
    if (T != NULL && T->left == NULL && T->right == NULL) {
        return true;
    }
    return false;
}

//Computes minumum distance between two nodes
double min_distance(kd_tree T1, kd_tree T2) {
    double distance = 0.0;
    if (T2->bounding_box.max_x < T1->bounding_box.min_x) {
        double diff = T2->bounding_box.max_x - T1->bounding_box.min_x;
        distance += diff * diff;
    }
    else if (T2->bounding_box.min_x > T1->bounding_box.max_x) {
        double diff = T2->bounding_box.min_x - T1->bounding_box.max_x;
        distance += diff * diff;
    }
    if (T2->bounding_box.max_y < T1->bounding_box.min_y) {
        double diff = T2->bounding_box.max_y - T1->bounding_box.min_y;
        distance += diff * diff;
    }
    else if (T2->bounding_box.min_y > T1->bounding_box.max_y) {
        double diff = T2->bounding_box.min_y - T1->bounding_box.max_y;
        distance += diff * diff;
    }
    return sqrt(distance);
}

//Helper function to compute max of 4 doubles
double maxx(double w, double x, double y, double z) {
    return max(w, max(x, max(y, z)));
}

//Computes maximum distance between two nodes
double max_distance(kd_tree T1, kd_tree T2) {
    double distance = 0.0;
    double diffx = maxx(fabs(T1->bounding_box.min_x - T2->bounding_box.min_x),
        fabs(T1->bounding_box.min_x - T2->bounding_box.max_x),
        fabs(T1->bounding_box.max_x - T2->bounding_box.min_x),
        fabs(T1->bounding_box.max_x - T2->bounding_box.max_x));
    double diffy = maxx(fabs(T1->bounding_box.min_y - T2->bounding_box.min_y),
        fabs(T1->bounding_box.min_y - T2->bounding_box.max_y),
        fabs(T1->bounding_box.max_y - T2->bounding_box.min_y),
        fabs(T1->bounding_box.max_y - T2->bounding_box.max_y));
    distance += diffx * diffx + diffy * diffy;
    return sqrt(distance);
}

//wrapper for min distance
double min_point_distance(Vector2d pt_vector, kd_tree pt_tree) {
    vector<point> pts;
    point pt;
    pt.x = pt_vector(0);
    pt.y = pt_vector(1);

    pts.push_back(pt);
    kd_tree leaf_node = new_kdtree(pts);
    double result = min_distance(leaf_node, pt_tree);
    free_kdtree(leaf_node);

    return result;
}

//returns all points in the kd tree within 3sigma of supplied point
void nearest_neighbors(Vector2d src_point, kd_tree point_tree, float sigma, vector<Vector2d>& accumulator) {
    if (is_leaf(point_tree)) {
        vector<point> pts = point_tree->points;
        for (size_t i = 0; i < pts.size(); i++) {
            point pt = pts[i];
            Vector2d candidate_pt(pt.x, pt.y);
            if (((src_point - candidate_pt).norm()) < 3 * sigma) {
                accumulator.push_back(candidate_pt);
            }
        }
    }

    else if (min_point_distance(src_point, point_tree) < 3 * sigma) {

        //Recursive case
        nearest_neighbors(src_point, point_tree->left, sigma, accumulator);
        nearest_neighbors(src_point, point_tree->right, sigma, accumulator);

    }
}

//Gaussian Kernel value
float std_normal(float x) {
    return exp(-0.5 * x * x) / sqrt(2 * M_PI);
}

//Computes \hat{p}(x)
float density_estimator(float bandwidth, Vector2d x, vector<Vector2d>& points,
    kd_tree points_kdtree) {
    float sum = 0.0;
    //Find nearest neighbors
    vector<Vector2d> nNs;
    nearest_neighbors(x, points_kdtree, bandwidth, nNs);
    for (size_t i = 0; i < nNs.size(); i++) {
        sum += std_normal((x - nNs[i]).norm() / bandwidth);
    }
    return sum / (points.size() * pow(bandwidth, 2));
}

//=====================================================================//
//                         Main Function                               //
//=====================================================================//

//[[Rcpp::export]]
MatrixXd SCMS(MatrixXd data, float bandwidth, float threshold, int max_iterations, float epsilon, bool print_iter) {

    vector<Vector2d> points;
    vector<point> kd_points;
    Vector2d pt;
    point kd_point;

    for (int i = 0; i < data.rows(); ++i) {
        pt(0) = data(i, 0);
        pt(1) = data(i, 1);
        kd_point.x = data(i, 0);
        kd_point.y = data(i, 1);
        points.push_back(pt);
        kd_points.push_back(kd_point);
    }

    kd_tree points_kdtree = new_kdtree(kd_points);

    //Step 1: Thresholding: remove x in X if \hat{p}(x) < threshold
	auto thresholded_points = points;
    auto remove_it = remove_if(thresholded_points.begin(), thresholded_points.end(), [&](auto x) {return density_estimator(bandwidth, x, points, points_kdtree) < threshold; });
    thresholded_points.erase(remove_it, thresholded_points.end());
	
    std::cout << "No. of thresholded points: " << thresholded_points.size() << endl;

    //Step 2: Peform the following SCMS until convergence
    std::cout << "Performing SCMS..." << endl;

    float max_error = DBL_MAX;
    int iteration = 0;

    vector<pair<Vector2d, double>> thresholded_points_errors;
    for (size_t i = 0; i < thresholded_points.size(); ++i) {
        pair<Vector2d, double> thresholded_point_error(thresholded_points[i], max_error);
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
                Vector2d x = thresholded_point_error.first;

                //Compute The Hessian via mu[i] and c[i]
                Matrix2d H = Matrix2d::Zero();
                Vector2d mu(0, 0);
                float c = 0.0;
                Vector2d sum_cx(0, 0); //for msv
                float sum_c = 0.0;      //for msv

                //Find points that are close enough (within 3*bandwidth <=> 3 sigma)
                vector<Vector2d> nNs;
                nearest_neighbors(x, points_kdtree, bandwidth, nNs);

                for (size_t j = 0; j < nNs.size(); j++) {
                    mu = (x - nNs[j]) / pow(bandwidth, 2);
                    c = std_normal((x - nNs[j]).norm() / bandwidth);
                    H += ((c / nNs.size())
                        * ((mu * mu.transpose()) - (Matrix2d::Identity(2, 2) / pow(bandwidth, 2))));

                    //for msv
                    sum_cx = sum_cx + c * nNs[j];
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
                nNs.clear();
            }
            });
        if (print_iter == true) std::cout << "    Points moved: " << points_moved << endl;
        ++iteration;
    }

    if (iteration == max_iterations) {
        std::cout << "    Max Iterations Reached." << endl;
    }

    std::cout << "Done." << endl;

    //Output: the collection of all remaining points
    free_kdtree(points_kdtree);

    for (size_t i = 0; i < thresholded_points.size(); ++i) {
        thresholded_points[i] = thresholded_points_errors[i].first;
    }

    auto results = Eigen::Map<Eigen::MatrixXd>(thresholded_points[0].data(), 2, thresholded_points.size());
    //return as<NumericMatrix>(wrap((results.transpose())));
    return results.transpose();
}

MatrixXd readCSV(const string filepath) {
    std::ifstream input;
    input.open(filepath);
    double x, y;
    char comma;
    MatrixXd toReturn(3000000, 2);
    input >> x >> comma >> y;
    toReturn(0, 0) = x;
    toReturn(0, 1) = y;
    int i = 1;
    while(input >> x >> comma >> y) {
        toReturn(i, 0) = x;
        toReturn(i, 1) = y;
        ++i;
    }

    toReturn.conservativeResize(i, 2);
    return toReturn;
}

void writeCSV(MatrixXd output) {
    ofstream file("C:/Users/darkg/Desktop/output.csv");
    for (int i = 0; i < output.rows(); ++i) {
        file << std::setprecision(15) << output(i, 0) << "," << output(i, 1) << endl;
    }
}

int main() {
    MatrixXd input = readCSV("C:/Users/darkg/Desktop/input.csv");
    MatrixXd output = SCMS(input, 0.55, 0.025, 10000, 0.001, false);
    writeCSV(output);
    return 0;
}