//
// Created by cronusiius on 2026/9/24.
//

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <chrono>
#include <memory>

using namespace std;
/*
 *优化变量的维度和保存变量的数据类型，存成了一个vertex
 */
class CurveFittingVertex:public g2o::BaseVertex<3,Eigen::Vector3d> {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    //把顶点恢复到原点函数
    virtual void setToOriginImpl() override {
        _estimate<<0,0,0;
    }

    //更新参数
    virtual void oplusImpl(const double *update) override {
        _estimate+=Eigen::Vector3d(update);
    }

    virtual bool read(istream& in){}
    virtual bool write(ostream& out) const{}
};

/*
 *定义误差怎么算，每条edge只连接一个vertex（unary）：误差维度、measurement类型、连接的vertex类型
 */
class CurveFittingEdge:public g2o::BaseUnaryEdge<1,double,CurveFittingVertex> {
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
public:

    //每个edge对应一个数据点(x_i,y_i)，
    CurveFittingEdge(double x):BaseUnaryEdge(),_x(x) {}

    //计算残差
    virtual void computeError() override {
        const CurveFittingVertex* v=static_cast<const CurveFittingVertex*>(_vertices[0]);
        const Eigen::Vector3d abc=v->estimate();
        _error(0,0)=_measurement-std::exp(abc[0]*_x*_x+abc[1]*_x+abc[2]);
    }

    //计算J
    virtual void linearizeOplus() override {
        const CurveFittingVertex* v=static_cast<const CurveFittingVertex *>(_vertices[0]);
        const Eigen::Vector3d abc=v->estimate();
        double y=exp(abc[0]*_x*_x+abc[1]*_x+abc[2]);
        _jacobianOplusXi[0]=-_x*_x*y;
        _jacobianOplusXi[1]=-_x*y;
        _jacobianOplusXi[2]=-y;
    }

    virtual bool read(std::istream &is) {}
    virtual bool write(std::ostream &os) const {}

private:
     double _x;
};

int main(int argc, char** argv) {
    double ar = 1.0, br = 2.0, cr = 2.0;
    double ae = 20.0, be = 1.0, ce = 5.0;
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

    /*
     *blocksolver是求解器，待优化变量为3,1为保留配置（当前没用到）
     */
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<3,1>> BlockSolverType;
    /*
     *linearsolverdense是稠密矩阵求解器，
     *posematrixtype是通过变量块的大小自动的推导Hessian矩阵对应的类型
     */
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    /*
     *由内向外，创建线性求解器，创建blocksolver,然后高斯牛顿法
     */
    auto solver=new g2o::OptimizationAlgorithmGaussNewton(
        std::make_unique<BlockSolverType>(std::make_unique<LinearSolverType>()));
    //创建整个优化问题的总管理器
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //创建唯一的vertex
    CurveFittingVertex* v=new CurveFittingVertex();
    v->setEstimate(Eigen::Vector3d(ae,be,ce));
    v->setId(0);
    optimizer.addVertex(v);

    //创建100条edge
    for (int i=0;i<N;i++) {
        CurveFittingEdge* edge=new CurveFittingEdge(x_data[i]);
        edge->setId(i);
        edge->setVertex(0,v);//把edge和vertex连接起来，这个edge的第0个vertex就是v
        edge->setMeasurement(y_data[i]);//设置实际观测
        //误差维度为1,所以协方差矩阵就是1x1,信息矩阵就为协方差矩阵的逆
        edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
        optimizer.addEdge(edge);
    }

    cout << "start optimization" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    //开始优化并最多迭代10词
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    Eigen::Vector3d abc_estimate = v->estimate();
    cout << "estimated model: " << abc_estimate.transpose() << endl;
    return 0;
}
