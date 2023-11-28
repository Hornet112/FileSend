#pragma once

#ifndef _TYPE_DEFINE_H
#define _TYPE_DEFINE_H

constexpr auto PI = 3.14159265358979323846;
constexpr auto SQ2PI = 2.50662827463100024161;

constexpr auto HERMITE = "Hermite";
constexpr auto LEGENDRE = "Legendre";
constexpr auto LAGUERRE = "Laguerre";
constexpr auto JACOBI = "Jacobi";
constexpr auto FOURIER = "Fourier";
constexpr auto BASIC = "Basic";
constexpr auto EXPONETIAL = "Exponetial";
constexpr auto RADICAL = "Radical";

#include <string>
#include <assert.h>
#include <vector>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;


//模型设置
typedef struct {
	VectorXd num_of_units;
	vector<VectorXd> unit_degree_of_freedom;
	vector<VectorXd> unit_boundaries;
	vector<vector<const char*>> unit_base_type;
	vector<VectorXd> continuity;
}ModelConfig;

//训练设置
typedef struct {
	int max_rank;
	double error_bound;
	int max_iter;
	//special points...
}TrainConfig;

//将模型设置按维度存储
typedef struct {
	//VectorXd coeff;
	VectorXd discontinuous_points;
	VectorXd degree_of_freedom;
	vector<const char*> base_type;
	//vector<vector<MatrixXd>> boundary_value;//第二层vector的[0]是左边界，[1]右边界
	//VectorXd interval_length;
	MatrixXd constrain_matirx;
	MatrixXd Z;
	MatrixXd Yb;
}DimensionInfo;

//将模型设置、建模范围、权重、系数打包的模型信息
typedef struct {
	int dimension;
	vector<DimensionInfo> dimension_info;
	double weight;
	vector<VectorXd> coeff;
	MatrixXd model_range;
}TensorModel;

typedef struct {
	vector<TensorModel> model;
}ECModel;

typedef struct {
	VectorXd betaVector;
	VectorXd Index;
	int complexity;
}pathwiseResult;

typedef struct {
	double alpha;
	double beta;
	double a;
	double b;
} DistributionPara;
typedef struct {
	string disType;
	DistributionPara parameter;
} Distribution;
typedef struct {
	Distribution* disElement;
	double* weight;
	int mixNum;
}MixDistribution;
typedef struct {
	double freqStart;
	double freqEnd;
	double targetTyep;
	double rangeMin;
	double rangeMax;
	double weight;
} targetConfig;
typedef struct {
	double targetTyep;
	double rangeMin;
	double rangeMax;
	double weight;
}targetConfigNoFre;

#endif