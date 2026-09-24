//
// Created by cronusiius on 2026/9/24.
//

#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Core>
#include "sophus/se3.hpp"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <ceres/ceres.h>
#include <Eigen/Cholesky>
using namespace Eigen;
using namespace std;

struct CURVE_FITTING_COST {
    CURVE_FITTING_COST(double x,double y):_x(x),_y(y){};

    template<typename T>
    bool operator()(const T *const abc,T* residual)const {
        residual[0]=T(_y)-ceres::exp(abc[0]*T(_x)*T(_x)+abc[1]*T(_x)+abc[2]);
        return true;
    }

    const double _x,_y;
};

int main(int argc,char** argv) {
    double ar = 1.0, br = 2.0, cr = 2.0;
    double ae = 2.0, be = 1.0, ce = 5.0;
    int N=100;
    double w_sigma=1.0;
    double inv_sigma=1.0/w_sigma;
    cv::RNG rng;

    vector<double> x_data,y_data;
    for (int i=0;i<N;i++) {
        double x=i/100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr)+rng.gaussian(w_sigma*w_sigma));
    }

    double abc[3]={ae,be,ce};//待优化参数

    ceres::Problem problem;//创建优化问题
    for (int i=0;i<N;i++) {
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(new CURVE_FITTING_COST(x_data[i],y_data[i])),nullptr,abc);
    }
    ceres::Solver::Options options;
    options.linear_solver_type=ceres::DENSE_NORMAL_CHOLESKY;//稠密正规方程加cholesky分解
    options.minimizer_progress_to_stdout=true;
    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1=chrono::steady_clock::now();
    ceres::Solve(options,&problem,&summary);
    chrono::steady_clock::time_point t2=chrono::steady_clock::now();
    chrono::duration<double> time_used=chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout<<"solve time cost"<<time_used.count()<<"seconds"<<endl;
    cout<<summary.BriefReport()<<endl;
    cout<<"estimated a,b,c=";
    for (auto x:abc)
        cout<<x<<" ";
    cout<<endl;
    return 0;
}