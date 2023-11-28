#include "GeneralBase.h"

using namespace std;
using namespace Eigen;


GeneralBase::GeneralBase(const char* setPolyType)
{
	if (setPolyType == HERMITE) {
		coeffMatrix = MatrixXd::Zero(3, 3);
		coeffMatrix(0, 0) = 1;
		coeffMatrix(1, 0) = 0; coeffMatrix(1, 1) = 1;
		coeffMatrix(2, 0) = -1; coeffMatrix(2, 1) = 0; coeffMatrix(2, 2) = 1;
		setMaxOrder(2);
		polyType = HERMITE;
	}
	else if (setPolyType == LEGENDRE) {
		coeffMatrix = MatrixXd::Zero(3, 3);
		coeffMatrix(0, 0) = 1;
		coeffMatrix(1, 0) = 0; coeffMatrix(1, 1) = 1;
		coeffMatrix(2, 0) = -0.5; coeffMatrix(2, 1) = 0; coeffMatrix(2, 2) = 1.5;
		setMaxOrder(2);
		polyType = LEGENDRE;
	}
	else if (setPolyType == BASIC) {
		coeffMatrix = MatrixXd::Identity(3, 3);
		setMaxOrder(2);
		polyType = BASIC;
	}
	else {
		cout << "Wrong Polynomial Type" << endl;
		coeffMatrix = MatrixXd::Zero(11, 11);
		setMaxOrder(10);
	}
}

GeneralBase::GeneralBase(const char* setPolyType, double setAlpha) {
	if ((setPolyType == LAGUERRE) && (setAlpha > -1)) {
		coeffMatrix = MatrixXd::Zero(3, 3);
		alpha = setAlpha;
		coeffMatrix(0, 0) = 1;
		coeffMatrix(1, 0) = 1 + alpha; coeffMatrix(1, 1) = -1;
		coeffMatrix(2, 0) = (2 + 3 * alpha + alpha * alpha) / 2.0; coeffMatrix(2, 1) = -alpha - 2.0; coeffMatrix(2, 2) = 0.5;;
		setMaxOrder(2);
		polyType = LAGUERRE;
	}
	else {
		cout << "Wrong Polynomial Type" << endl;
		coeffMatrix = MatrixXd::Zero(11, 11);
		setMaxOrder(10);
	}
}

GeneralBase::GeneralBase(const char* setPolyType, double setAlpha, double setBeta) {
	if (setPolyType == JACOBI) {
		coeffMatrix = MatrixXd::Zero(3, 3);
		alpha = setAlpha;
		beta = setBeta;
		coeffMatrix(0, 0) = 1;
		coeffMatrix(1, 0) = (alpha - beta) / 2.0; coeffMatrix(1, 1) = (alpha + beta + 2) / 2.0;
		coeffMatrix(2, 0) = (alpha + beta + 3.0) * (alpha + beta + 4.0) / 8.0 - (alpha + 2.0) * (alpha + beta + 3.0) / 2.0 + (alpha + 1.0) * (alpha + 2.0) / 2.0; coeffMatrix(2, 1) = -(alpha + beta + 3.0) * (alpha + beta + 4.0) / 4.0 + (alpha + 2) * (alpha + beta + 3.0) / 2.0; coeffMatrix(2, 2) = (alpha + beta + 3.0) * (alpha + beta + 4.0) / 8.0;
		setMaxOrder(2);
		polyType = JACOBI;
	}
	else {
		cout << "Wrong Polynomial Type" << endl;
		coeffMatrix = MatrixXd::Zero(11, 11);
		setMaxOrder(10);
	}
}

void GeneralBase::setMaxOrder(int order) {
	maxOrder = order;
}

double GeneralBase::recurrenceA(int order) {
	if (polyType == HERMITE) {
		return 1.0;
	}
	else if (polyType == LEGENDRE) {
		return (2.0 * order + 1.0) / (order + 1.0);
	}
	else if (polyType == LAGUERRE) {
		return -1.0 / (order + 1.0);
	}
	else if (polyType == JACOBI) {
		return ((2.0 * order + alpha + beta + 1.0) * (2.0 * order + alpha + beta + 2.0)) / (2.0 * (order + 1.0) * (order + alpha + beta + 1.0));
	}
	else if (polyType == BASIC) {
		return 1.0;
	}
	else {
		return 0.0;
	}
}

double GeneralBase::recurrenceB(int order) {
	if (polyType == HERMITE) {
		return 0.0;
	}
	else if (polyType == LEGENDRE) {
		return 0.0;
	}
	else if (polyType == LAGUERRE) {
		return (2.0 * order + 1.0 + alpha) / (order + 1.0);
	}
	else if (polyType == JACOBI) {
		return ((pow(alpha, 2) - pow(beta, 2)) * (2.0 * order + alpha + beta + 1.0)) / (2.0 * (order + 1.0) * (order + alpha + beta + 1.0) * (2.0 * order + alpha + beta));
	}
	else {
		return 0.0;
	}
}

double GeneralBase::recurrenceC(int order) {
	if (polyType == HERMITE) {
		return double(order);
	}
	else if (polyType == LEGENDRE) {
		return order / (order + 1.0);
	}
	else if (polyType == LAGUERRE) {
		return (order + alpha) / (order + 1.0);
	}
	else if (polyType == JACOBI) {
		return ((order + alpha) * (order + beta) * (2.0 * order + alpha + beta + 2.0)) / ((2.0 * order + alpha + beta) * (order + 1.0) * (order + alpha + beta + 1.0));/////
	}
	else {
		return 0.0;
	}
}

VectorXd GeneralBase::calxPower(double x, int order) {
	VectorXd xPower = VectorXd::Zero(order + 1);
	xPower(0) = 1.0;
	xPower(1) = x;
	for (int i = 2; i < order + 1; i++) {
		xPower(i) = xPower(i - 1) * x;
	}
	return xPower;
}

MatrixXd GeneralBase::calXPower(VectorXd x, int order) {
	int N = x.size();
	MatrixXd XPower = MatrixXd::Zero(N, order + 1);
	XPower.col(0) = VectorXd::Ones(N);

	//emmmmmm这个函数不兼容order=0的输入，XPower.col(1) = x;这行会爆，加个if规避一下
	//2023.9.16. gkr

	if (!order) {
		return XPower;
	}

	XPower.col(1) = x;
	for (int i = 2; i < order + 1; i++) {
		XPower.col(i) = XPower.col(i - 1).cwiseProduct(x);
	}
	return XPower;
}

double GeneralBase::value(VectorXd xPower, int order) {
	if (order < 0) {
		return 0.0;
	}
	else if (order == 0) {
		return 1.0 * coeffMatrix(0, 0);
	}
	//VectorXd xPower = calxPower(x,order);

	if (order <= maxOrder) {
		return xPower.dot(coeffMatrix.row(order).segment(0, order + 1));
	}
	else {
		updateCoeffMatrix(order);
		return xPower.dot(coeffMatrix.row(order).segment(0, order + 1));
	}
}

VectorXd GeneralBase::value(MatrixXd XPower, int order) {
	int N = XPower.rows();
	if (order < 0) {
		return VectorXd::Zero(N);
	}
	else if (order == 0) {
		return VectorXd::Ones(N) * coeffMatrix(0, 0);
	}
	//MatrixXd xPower = calXPower(x, order);

	if (order <= maxOrder) {
		return XPower * (coeffMatrix.row(order).segment(0, order + 1)).transpose();
	}
	else {
		updateCoeffMatrix(order);
		return XPower * (coeffMatrix.row(order).segment(0, order + 1)).transpose();
	}
}


VectorXd GeneralBase::valueToOrder(VectorXd xPower, int order) {
	//xpower事先计算好，size必须得>=order+1,order>=0

	if (order <= maxOrder) {
		return xPower.segment(0, order + 1).transpose() * coeffMatrix.transpose().block(0, 0, order + 1, order + 1);
	}
	else {
		updateCoeffMatrix(order);
		return xPower.segment(0, order + 1).transpose() * coeffMatrix.transpose();
	}


}


MatrixXd GeneralBase::valueToOrder(MatrixXd XPower, int order) {
	//xpower事先计算好，cols()必须得>=order+1,order>=0
	int N = XPower.rows();

	if (order <= maxOrder) {
		return XPower.block(0, 0, N, order + 1) * coeffMatrix.transpose().block(0, 0, order + 1, order + 1);
	}
	else {
		updateCoeffMatrix(order);
		return XPower.block(0, 0, N, order + 1) * coeffMatrix.transpose();
	}
}

MatrixXd GeneralBase::bondaryMat(int order, int continuty, int flag) {
	if (continuty == -1)
	{
		return MatrixXd::Zero(continuty + 1, order + 1);
	}
	MatrixXd diff_mat = MatrixXd::Zero(continuty + 1, order + 1);
	if (flag == 0)
	{
		diff_mat(0, 0) = 1;
	}
	else
	{
		diff_mat.row(0) = VectorXd::Ones(order + 1);
	}
	for (int i = 1; i <= min(continuty, order); i++)
	{
		for (int j = i; j <= order; j++)
		{
			diff_mat(i, j) = j * diff_mat(i - 1, j - 1);
		}
	}
	if (order <= maxOrder) {
		return diff_mat * coeffMatrix.transpose().block(0, 0, order + 1, order + 1);
	}
	else {
		updateCoeffMatrix(order);
		return diff_mat * coeffMatrix.transpose();
	}
}


void GeneralBase::updateCoeffMatrix(int order) {
	if (order > maxOrder) {
		MatrixXd coeffMatrixNew = MatrixXd::Zero(order + 1, order + 1);
		coeffMatrixNew.block(0, 0, maxOrder + 1, maxOrder + 1) = coeffMatrix;
		for (int i = maxOrder + 1; i < order + 1; i++) {
			VectorXd coeffTemp = VectorXd::Zero(order + 1);
			coeffTemp.segment(1, i) = coeffMatrixNew.row(i - 1).segment(0, i);
			coeffMatrixNew.row(i) = recurrenceA(i - 1) * coeffTemp.transpose() + recurrenceB(i - 1) * coeffMatrixNew.row(i - 1) - recurrenceC(i - 1) * coeffMatrixNew.row(i - 2);
		}
		coeffMatrix = coeffMatrixNew;
		setMaxOrder(order);
	}
}















