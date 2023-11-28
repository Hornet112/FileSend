#pragma once
#pragma once
#ifndef _GENERALBASE_
#define _GENERALBASE_

#include <iostream>
#include <math.h>
#include <string>
#include <Eigen\Dense>
#include "TypeDefine.h"

using namespace std;
using namespace Eigen;

class GeneralBase
{
public:
	GeneralBase(const char* setPolyType);
	GeneralBase(const char* setPolyType, double setAlpha);
	GeneralBase(const char* setPolyType, double setAlpha, double setBeta);
	~GeneralBase() {};

	void setMaxOrder(int order);

	double recurrenceA(int order);
	double recurrenceB(int order);
	double recurrenceC(int order);

	VectorXd calxPower(double x, int order);//calculate xpower of only a sample 
	MatrixXd calXPower(VectorXd x, int order);//calculate xpower of samples 
	double value(VectorXd xPower, int order);//calculate value of the order-th base for only a sample
	VectorXd value(MatrixXd XPower, int order);//calculate value of the order-th base for samples
	VectorXd valueToOrder(VectorXd xPower, int order);//calculate design matrix from 0-th to order-th for only a sample
	MatrixXd valueToOrder(MatrixXd XPower, int order);//calculate design matrix from 0-th to order-th for samples
	//VectorXd valueByXPower(VectorXd x, int order, MatrixXd XPower);

	MatrixXd bondaryMat(int order, int continuty, int flag);//continuty of boundary flag=0 for left,flag=1 for right


private:
	const char* polyType;
	MatrixXd coeffMatrix;
	void updateCoeffMatrix(int order);
	int maxOrder;
	double jcbAlpha;
	double jcbBeta;
	double alpha, beta;



};








#endif // !_GENERALBASE_








