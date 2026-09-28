//
// Created by cronusiius on 2026/9/28.
//

#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <opencv4/opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

/*
 * pts1是变换后矩阵
 * pts2是变换前矩阵
 */
void pose_estimation_3d3d(const std::vector<cv::Point3f>& pts1,
    const std::vector<cv::Point3f>& pts2,
    cv::Mat& R,
    cv::Mat& t) {
    //求两个质心
    cv::Point3f p1(0,0,0),p2(0,0,0);
    int N=pts1.size();
    for (int i=0;i<N;i++) {
        p1+=pts1[i];
        p2+=pts2[i];
    }
    p1/=N;
    p2/=N;
    std::vector<cv::Point3f> q1(N),q2(N);
    for (int i=0;i<N;i++) {
        q1[i]=pts1[i]-p1;
        q2[i]=pts2[i]-p2;
    }
    //计算W
    Eigen::Matrix3d W=Eigen::Matrix3d::Zero();
    for (int i=0;i<N;i++) {
        W+=Eigen::Vector3d(q1[i].x,q1[i].y,q1[i].z)*Eigen::Vector3d(q2[i].x,q2[i].y,q2[i].z).transpose();
    }

    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W,Eigen::ComputeFullU|Eigen::ComputeFullV);
    Eigen::Matrix3d U=svd.matrixU();
    Eigen::Matrix3d V=svd.matrixV();

    Eigen::Matrix3d R_=U*(V.transpose());
    Eigen::Vector3d t_=Eigen::Vector3d(p1.x,p1.y,p1.z)-R_*Eigen::Vector3d(p2.x,p2.y,p2.z);

    cv::eigen2cv(R_,R);
    cv::eigen2cv(t_,t);
}
