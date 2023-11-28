#pragma once

#ifndef _EC_ENGINE_
#define _EC_ENGINE_

#include <iostream>
#include <fstream>
#include <math.h>
#include <regex>
#include <Eigen\Dense>
#include "TypeDefine.h"
#include "GeneralBase.h"
#include <string>
#include <ctime>

using namespace std;



//一些想直接调用的函数
Eigen::VectorXd modelEval(vector<TensorModel> model, Eigen::MatrixXd input);
//eigen&csv
//载入csv格式数据
Eigen::MatrixXd loadDataCsv(const string& filename);
void saveDataCsv(string fileName, Eigen::MatrixXd  matrix);

//回归相关
pathwiseResult pathwiseLearning(Eigen::MatrixXd X, Eigen::VectorXd Y, int max_complexity, double alpha, int flag);

//给pathwise套个壳子
Eigen::VectorXd pathwise_ols(Eigen::MatrixXd X, Eigen::VectorXd Y, int max_complexity, double alpha, int flag);

double S(double x, double y);//坐标下降用

//导出模型
void ecmodel2Json(vector<TensorModel> model, string addr);


class ECEngine {
public:
	ECEngine() {};
	~ECEngine() {};
	ECEngine(vector<ModelConfig> model_config,TrainConfig train_config,Eigen::MatrixXd X,Eigen::VectorXd Y);
	ECEngine(TensorModel patch, Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::MatrixXd epsilon, Eigen::MatrixXd main_model_range, int DF, const char* type);
	ECEngine(TensorModel patch, Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::MatrixXd epsilon, Eigen::MatrixXd main_model_range, Eigen::MatrixXd DF, vector<vector<const char*>> type, Eigen::MatrixXd continuity);
	ECEngine(vector<TensorModel> patch, Eigen::MatrixXd X, Eigen::VectorXd Y, Eigen::MatrixXd epsilon, Eigen::MatrixXd main_model_range, Eigen::MatrixXd DF, vector<vector<const char*>> type, Eigen::MatrixXd continuity);
	ECEngine(vector<TensorModel> model);

	void sortVector(const Eigen::VectorXd& vec, Eigen::VectorXd& sorted_vec, Eigen::VectorXi& ind);
	void yieldDesignMatrix();
	void yieldDesignMatrix(Eigen::MatrixXd x_input);
	Eigen::VectorXd modelEval();
	Eigen::VectorXd modelEval(Eigen::MatrixXd x_input);




	void coeffCorrection(int type, double alpha);
	void modelUpdate();
	void modelTrain();
	void modelTrain(int type, double alpha);

	

	//添加补丁//输入参数有待优化，目前出于实验性质，其他设置还有待添加
	//void appendPatch(TensorModel patch, Eigen::MatrixXd epsilon, int DF, const char* type);
	void appendPatch(TensorModel patch, Eigen::MatrixXd epsilon, Eigen::MatrixXd DF, vector<vector<const char*>> type, Eigen::MatrixXd continuity, int train_type, double alpha);
	void appendPatch(vector<TensorModel> patch, Eigen::MatrixXd epsilon, Eigen::MatrixXd DF, vector<vector<const char*>> type, Eigen::MatrixXd continuity, int train_type, double alpha);
	//提取出模型
	vector<TensorModel> generateTensorModel(Eigen::VectorXd list);
	vector<TensorModel> generateTensorModel();

	//用于归一化输入的辅助函数
	void inputNormalization(Eigen::MatrixXd& x_input);

	//调试用函数
	void displayModelInfo(string tyoe);

	//回归相关
	Eigen::MatrixXd pinv(Eigen::MatrixXd inMatrix);

	//一个基本寄了的求列零空间
	Eigen::MatrixXd orthogonalizeNullSpace(Eigen::MatrixXd matrix);


	//这些是放在private里面不好改才拿出来的变量
	//训练用数据
	
	Eigen::MatrixXd x_normalized;
	Eigen::VectorXd r;
	Eigen::MatrixXd y;

	void resetConfig(vector<ModelConfig> model_config, TrainConfig train_config, int retrain_from_layer);


private:
	vector<vector<DimensionInfo>> model_info;//模型的主要信息
	vector<vector<Eigen::VectorXd>> coeff;
	Eigen::VectorXd weight;
	Eigen::MatrixXd y_per_rank;//存储各张量无权重输出，用于update的时候少算点
	int dimension;
	int rank = 0;
	Eigen::MatrixXd model_range;//建模范围，2*dim，row0是下界，row1是上界
	vector<vector<Eigen::MatrixXd>> design_matrix;

	//trainconfig//为了防止有人(没错正是在下)忘记定义先给个默认值
	int max_iter=10;
	int max_rank=5;
	double error_bound=1e-12;

	//patches;//好像现在没用了
	vector<TensorModel> patches;


	//回归相关
	double error_limit = 1e-3;//from:Eigen::VectorXd ECEngine::coordinateDescentNaive(Eigen::MatrixXd X, Eigen::VectorXd Y, double alpha, double lambda)
};

#endif#pragma once
