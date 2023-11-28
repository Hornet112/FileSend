#include "ECEngine.h"
#include "json/json.h"
#include "assert.h"
#include <iostream>
#include <filesystem>
#include <chrono>
#include <ctime>


using namespace Eigen;

//一些不知道有没有用的路径
string TRAIN = "dataset/train/";
string DATADISPLAY = "data_display/";
string PLOT = "data_display/plot";

//pathwise需要的参数
double error_limit = 1e-3;

VectorXd modelEval(vector<TensorModel> model, MatrixXd input) {
	if (model[0].dimension != input.cols()) {
		return VectorXd::Zero(input.rows());
	}
	ECEngine model_to_response = ECEngine::ECEngine(model);
	model_to_response.inputNormalization(input);
	return model_to_response.modelEval(input);
}

MatrixXd loadDataCsv(const string& filename) {
	ifstream file(filename);
	if (!file.is_open()) {
		cerr << "无法打开文件: " << filename << endl;
		exit(1);
	}

	vector<vector<double>> data;
	string line;
	while (getline(file, line)) {
		vector<double> row;
		size_t start = 0, end = 0;
		while ((end = line.find(",", start)) != string::npos) {
			string cell = line.substr(start, end - start);
			row.push_back(stod(cell));
			start = end + 1;
		}
		string cell = line.substr(start);
		row.push_back(stod(cell));
		data.push_back(row);
	}

	int rows = data.size();
	int cols = data[0].size();
	MatrixXd matrix(rows, cols);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			matrix(i, j) = data[i][j];
		}
	}

	file.close();
	return matrix;
}


pathwiseResult pathwiseLearning(MatrixXd X, VectorXd Y, int max_complexity, double alpha, int flag)
{
	double epsilon = 0.001;
	int K = 100;




	//MatrixXd X = regressionvar.predictor;
	//VectorXd Y = regressionvar.response;
	int N = X.rows();
	int p = X.cols();

	//加一个求列的方均
	VectorXd RMS_2 = VectorXd::Zero(p);
	for (int i = 0; i < p; i++) {
		RMS_2(i) = X.col(i).norm();
		RMS_2(i) = (RMS_2(i) * RMS_2(i)) / N;
	}
	double beta_temp = 0;//beta的临时值
	VectorXd residualVector = Y;//启动时残差向量和响应向量的值是相同的
	VectorXd p_gradient_vector = VectorXd::Zero(p);//协方差更新中的p梯度向量
	MatrixXd dot_Matrix = MatrixXd::Zero(p, p);//协方差更新中的内积矩阵
	MatrixXd result_matrix = MatrixXd::Zero(p, 2);
	VectorXd basic_index = VectorXd::Zero(p);
	double temp = 0;
	double error = 1;//循环终止的误差
	//把lambda初始化为lambda_max;
	double max_dot = 0;
	double lambda = 0, lambda_max = 0;//0
	for (int i = 1; i <= p; i++) {
		double dot = fabs(Y.dot(X.col(i - 1)));
		if (dot > max_dot) { max_dot = dot; }
	}

	if (flag == 2)
	{
		//初始化p梯度向量
		dot_Matrix = X.transpose() * X;
		for (int i = 1; i <= p; i++)
		{
			p_gradient_vector(i - 1) = Y.dot(X.col(i - 1));
			/*
			for (int j = 1; j <= p; j++)
			{
				dot_Matrix(i - 1, j - 1) = X.col(i - 1).dot(X.col(j - 1));
			}
			*/
		}
	}

	lambda_max = max_dot / (N * alpha);
	//cout << "lambda_max=" << lambda << endl;
	VectorXd betaVector = VectorXd::Zero(p);//初始化beta向量
	int complexity;
	for (int k = 0; k <= K; k++) {
		lambda = lambda_max * pow(epsilon, (double)k / K);
		error = 1;
		while (error >= error_limit) {
			error = 0;
			if (flag == 1)
			{
				//compute method=naive
				for (int i = 1; i <= p; i++)
				{
					beta_temp = betaVector(i - 1);
					temp = (residualVector.dot(X.col(i - 1)));
					temp = temp / N + beta_temp * RMS_2(i - 1);
					temp = S(temp, alpha * lambda) / (RMS_2(i - 1) + lambda * (1 - alpha));
					if (temp != beta_temp)
					{
						error = error + (temp - beta_temp) * (temp - beta_temp);
						for (int j = 1; j <= N; j++) {
							residualVector(j - 1) = residualVector(j - 1) - X(j - 1, i - 1) * (temp - beta_temp);//更新残差（这里出过大bug）
						}
						betaVector(i - 1) = temp;
					}
				}
			}
			else
			{
				//compute method=covariance
				for (int i = 1; i <= p; i++)
				{
					beta_temp = betaVector(i - 1);
					temp = p_gradient_vector(i - 1);
					for (int j = 1; j <= p; j++)
					{
						if (fabs(betaVector(j - 1)) > 0)
						{
							temp -= betaVector(j - 1) * dot_Matrix(i - 1, j - 1);
						}
					}
					temp = temp / N + beta_temp * RMS_2(i - 1);
					temp = S(temp, alpha * lambda) / (RMS_2(i - 1) + lambda * (1 - alpha));
					if (temp != beta_temp)
					{
						error = error + (temp - beta_temp) * (temp - beta_temp);
						betaVector(i - 1) = temp;
					}
				}
			}
			error = error / p;
		}
		complexity = 0;
		basic_index = VectorXd::Zero(p);
		for (int i = 0; i < p; i++)
		{
			if (betaVector(i) != 0)
			{
				basic_index(i) = 1.0;
				complexity++;
			}
		}
		if (complexity >= max_complexity)
		{
			break;
		}
	}
	pathwiseResult Result;
	Result.betaVector = betaVector;
	Result.Index = basic_index;
	Result.complexity = complexity;
	result_matrix.col(0) = betaVector;
	result_matrix.col(1) = basic_index;
	//result_matrix.block(0,2,N,1) = residualVector;
	//RegressedBeta regressedBeta;
	//regressedBeta.betaVector = betaVector;
	//regressedBeta.normVar = regressionvar.normVar;
	return Result;
}



VectorXd pathwise_ols(MatrixXd X, VectorXd Y, int max_complexity, double alpha, int flag) {

	int n = X.cols();

	pathwiseResult result = pathwiseLearning(X, Y, max_complexity, alpha, flag);
	VectorXd ans = result.betaVector;

	//cout << result.complexity << endl;

	if (ans.size() != n) {
		cout << endl << "pathwise出大问题辣还整呢" << endl;
	}

	return ans;
};

double S(double x, double y)
{
	if (x > 0 && y < x)
	{
		return x - y;
	}
	else if (x < 0 && y < -x)
	{
		return x + y;
	}
	else
	{
		return 0;
	}
}


void saveDataCsv(string fileName, MatrixXd  matrix)
{
	const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");

	ofstream file(fileName);
	if (file.is_open())
	{
		file << matrix.format(CSVFormat);
		file.close();
	}
}

void ecmodel2Json(vector<TensorModel> model, string addr) {

	//此处使用addr="D:\实习\2307杭州\cpp开发\easyclip\easyclip_LRA\model\"
	// 创建文件夹名称
	std::string mkdir = "mkdir " + addr;

	//编辑json同时mkdir
	int num_of_layers = model.size();
	Json::Value model_json;
	model_json["layers"] = num_of_layers;
	for (int r = 0; r < num_of_layers; r++) {

		system((mkdir + "layer_" + to_string(r + 1)).c_str());

		Json::Value tensor_json;
		tensor_json["dimension"] = model[r].dimension;
		saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "range.csv", model[r].model_range);
		tensor_json["range"] = addr + "layer_" + to_string(r + 1) + "\\" + "range.csv";
		tensor_json["weight"] = model[r].weight;

		for (int d = 0; d < model[r].dimension; d++) {

			system((mkdir + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1)).c_str());

			Json::Value dim_json;
			dim_json["coeff"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "coeff.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "coeff.csv", model[r].coeff[d]);
			dim_json["discontinuous_points"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "discontinuous_points.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "discontinuous_points.csv", model[r].dimension_info[d].discontinuous_points);
			dim_json["degree_of_freedom"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "degree_of_freedom.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "degree_of_freedom.csv", model[r].dimension_info[d].degree_of_freedom);
			dim_json["constrain_matirx"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "constrain_matirx.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "constrain_matirx.csv", model[r].dimension_info[d].constrain_matirx);
			dim_json["Z"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "Z.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "Z.csv", model[r].dimension_info[d].Z);
			dim_json["Yb"] = addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "Yb.csv";
			saveDataCsv(addr + "layer_" + to_string(r + 1) + "\\" + "dimension_" + to_string(d + 1) + "\\" + "Yb.csv", model[r].dimension_info[d].Yb);

			Json::Value base_type_info;
			int units = model[r].dimension_info[d].degree_of_freedom.size();
			for (int u = 0; u < units; u++) {
				base_type_info["Unit_" + to_string(u + 1)] = model[r].dimension_info[d].base_type[u];
			}
			dim_json["base_type"] = Json::Value(base_type_info);

			tensor_json["dimension_" + to_string(d + 1)] = Json::Value(dim_json);
		}

		model_json["layer_" + to_string(r + 1)] = Json::Value(tensor_json);
	}

	Json::StreamWriterBuilder writer;
	std::string jsonString_m = Json::writeString(writer, model_json);

	std::ofstream file_m(addr + "model.json");
	if (file_m.is_open()) {
		file_m << jsonString_m;
		file_m.close();
	}
}


Eigen::MatrixXd doublePointer2EigenMatrix(double** data, int rows, int cols) {
	Eigen::MatrixXd eigenMatrix(rows, cols);

	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			eigenMatrix(i, j) = data[i][j];
		}
	}

	return eigenMatrix;
}

ECEngine::ECEngine(vector<ModelConfig> model_config, TrainConfig train_config, MatrixXd X, VectorXd Y) {//注意默认约束AX=0;

	max_rank = train_config.max_rank;
	max_iter = train_config.max_iter;
	error_bound = train_config.error_bound;
	dimension = X.cols();

	//input normalization
	//请在此前完成特殊点的加入//好像有了patch功能就不需要了
	y = Y;
	r = y;
	int num_of_data = X.rows();
	model_range = MatrixXd::Zero(2, dimension);
	model_range.row(0) = X.colwise().minCoeff();
	model_range.row(1) = X.colwise().maxCoeff();
	x_normalized = MatrixXd::Zero(num_of_data, dimension);
	for (int i = 0; i < dimension; i++) {
		x_normalized.col(i) = (X.col(i) - model_range(0, i) * VectorXd::Ones(num_of_data)) / (model_range(1, i) - model_range(0, i));
	}

	rank = 0;
	weight.resize(max_rank);
	y_per_rank.resize(num_of_data, max_rank);
	model_info.resize(max_rank);
	design_matrix.resize(max_rank);

	MatrixXd left, right;

	for (int k = 0; k < max_rank; k++) {
		model_info[k].resize(dimension);
		for (int d = 0; d < dimension; d++) {
			model_info[k][d].discontinuous_points = model_config[k].unit_boundaries[d];
			model_info[k][d].degree_of_freedom = model_config[k].unit_degree_of_freedom[d];
			model_info[k][d].base_type = model_config[k].unit_base_type[d];

			//coeff[0][d] = VectorXd::Ones(model_config[k].unit_degree_of_freedom[d].sum());

			//约束数model_config[k].continuity[d].sum() + model_config[k].num_of_units(d) + 1在无约束为-1时仍然成立
			model_info[k][d].constrain_matirx = MatrixXd::Zero(model_config[k].continuity[d].sum() + model_config[k].num_of_units(d) + 1, model_config[k].unit_degree_of_freedom[d].sum());
			//model_info[k][d].interval_length = VectorXd::Zero(model_config[k].num_of_units(d));
			//model_info[k][d].boundary_value.resize(model_config[k].num_of_units(d));
			//取消了边界长度和边界值的存储

			int row_mark = 0;
			int col_mark = 0;
			for (int i = 0; i < model_config[k].num_of_units(d); i++) {
				//model_info[k][d].interval_length(i) = model_config[k].unit_boundaries[d](i + 1) - model_config[k].unit_boundaries[d](i);
				double len = model_config[k].unit_boundaries[d](i + 1) - model_config[k].unit_boundaries[d](i);
				//model_info[k][d].boundary_value[i].resize(2);
				GeneralBase base = GeneralBase(model_config[k].unit_base_type[d][i]);
				left = base.bondaryMat(model_config[k].unit_degree_of_freedom[d](i) - 1, model_config[k].continuity[d](i), 0);
				for (int j = 0; j < model_config[k].continuity[d](i) + 1; j++) {//归一化
					left.row(j) = left.row(j) * pow(len, -j);
				}
				right = base.bondaryMat(model_config[k].unit_degree_of_freedom[d](i) - 1, model_config[k].continuity[d](i + 1), 1);
				for (int j = 0; j < model_config[k].continuity[d](i + 1) + 1; j++) {
					right.row(j) = right.row(j) * pow(len, -j);
				}
				model_info[k][d].constrain_matirx.block(row_mark, col_mark, model_config[k].continuity[d](i) + 1, model_config[k].unit_degree_of_freedom[d](i)) = (-1) * left;
				row_mark += model_config[k].continuity[d](i) + 1;
				model_info[k][d].constrain_matirx.block(row_mark, col_mark, model_config[k].continuity[d](i + 1) + 1, model_config[k].unit_degree_of_freedom[d](i)) = right;
				col_mark += model_config[k].unit_degree_of_freedom[d](i);
			}

			FullPivLU<MatrixXd> lu_decomp(model_info[k][d].constrain_matirx);
			model_info[k][d].Z = lu_decomp.kernel();
			//model_info[k][d].Z = loadDataCsv("Z.csv");
			//model_info[k][d].Z = orthogonalizeNullSpace(model_info[k][d].constrain_matirx);
			model_info[k][d].Yb = VectorXd::Zero(model_info[k][d].degree_of_freedom.sum());
		}
	}
	yieldDesignMatrix();

}

ECEngine::ECEngine(TensorModel patch, MatrixXd X, VectorXd Y, MatrixXd epsilon, MatrixXd main_model_range, int DF, const char* type) {//求解缓冲带专用//epsilon为2*dim，0左1右
	dimension = X.cols();
	weight.resize(1);
	rank = 0;
	max_rank = 1;
	model_info.resize(1);
	model_info[0].resize(dimension);
	model_range = main_model_range;
	design_matrix.resize(1);

	int num_of_data = X.rows();

	y_per_rank.resize(num_of_data, max_rank);

	//for (int i = 0; i < dimension; i++) {
	//	X.col(i) = (X.col(i) - model_range(0, i) * VectorXd::Ones(num_of_data)) / (model_range(1, i) - model_range(0, i));
	//}
	x_normalized = X;
	y = Y;



	for (int d = 0; d < dimension; d++) {
		patch.dimension_info[d].discontinuous_points = patch.dimension_info[d].discontinuous_points * (patch.model_range(1, d) - patch.model_range(0, d)) + patch.model_range(0, d) * VectorXd::Ones(patch.dimension_info[d].discontinuous_points.size());
		patch.dimension_info[d].discontinuous_points = (patch.dimension_info[d].discontinuous_points - model_range(0, d) * VectorXd::Ones(patch.dimension_info[d].discontinuous_points.size())) / (model_range(1, d) - model_range(0, d));

		//此处先忽略麻烦的特殊情况，认为补丁加缓冲区全在主模型范围内；
		int leni = patch.dimension_info[d].discontinuous_points.size();
		model_info[0][d].discontinuous_points = VectorXd::Zero(leni + 4);
		model_info[0][d].discontinuous_points(1) = patch.dimension_info[d].discontinuous_points(0) - epsilon(0, d);
		model_info[0][d].discontinuous_points.segment(2, leni) = patch.dimension_info[d].discontinuous_points;
		model_info[0][d].discontinuous_points(leni + 4 - 2) = patch.dimension_info[d].discontinuous_points(leni - 1) + epsilon(1, d);
		model_info[0][d].discontinuous_points(leni + 4 - 1) = 1;

		leni = patch.dimension_info[d].degree_of_freedom.size();
		model_info[0][d].degree_of_freedom = VectorXd::Ones(leni + 4);
		model_info[0][d].degree_of_freedom(1) = DF;
		model_info[0][d].degree_of_freedom.segment(2, leni) = patch.dimension_info[d].degree_of_freedom;
		model_info[0][d].degree_of_freedom(leni + 4 - 2) = DF;

		leni = patch.dimension_info[d].base_type.size();
		vector<const char*> tmp_type;
		tmp_type.resize(leni + 4);
		tmp_type[0] = BASIC;
		tmp_type[1] = type;
		tmp_type[leni + 4 - 1] = BASIC;
		tmp_type[leni + 4 - 2] = type;
		for (int i = 0; i < leni; i++) {
			tmp_type[i + 2] = patch.dimension_info[d].base_type[i];
		}
		model_info[0][d].base_type = tmp_type;

		//最喜欢的一集（恼
		//拼约束方程
		model_info[0][d].constrain_matirx = MatrixXd::Zero(8 + model_info[0][d].degree_of_freedom.sum() - 2 * DF, model_info[0][d].degree_of_freedom.sum());//8代表只能一阶连续

		int row_mark = 0;
		int col_mark = 0;
		MatrixXd left, right;

		leni = model_info[0][d].degree_of_freedom.size();

		for (int i = 0; i < model_info[0][d].degree_of_freedom.size(); i++) {//除了缓冲区外全用单位阵限制，缓冲区用L形的连续性方程约束
			if (i == 1) {
				double len = model_info[0][d].discontinuous_points(i + 1) - model_info[0][d].discontinuous_points(i);
				GeneralBase base = GeneralBase(model_info[0][d].base_type[i]);
				left = base.bondaryMat(DF - 1, 1, 0);//注意此处continuity设置为1，此后有其他要求请调整
				for (int k = 0; k < 1 + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				right = base.bondaryMat(DF - 1, 1, 1);
				for (int k = 0; k < 1 + 1; k++) {//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, 1 + 1, model_info[0][d].degree_of_freedom(i)) = (-1) * left;//
				row_mark += 1 + 1;//
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, 1 + 1, model_info[0][d].degree_of_freedom(i)) = right;//
				col_mark += model_info[0][d].degree_of_freedom(i);

				GeneralBase base_r = GeneralBase(model_info[0][d].base_type[i + 1]);
				len = model_info[0][d].discontinuous_points(i + 2) - model_info[0][d].discontinuous_points(i + 1);
				left = base_r.bondaryMat(model_info[0][d].degree_of_freedom(i + 1) - 1, 1, 0);//注意此处continuity设置为1，此后有其他要求请调整//
				for (int k = 0; k < 1 + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, 1 + 1, model_info[0][d].degree_of_freedom(i + 1)) = (-1) * left;//
				row_mark += 1 + 1;//
				continue;
			}
			if (i == leni - 2) {
				double len = model_info[0][d].discontinuous_points(i) - model_info[0][d].discontinuous_points(i - 1);
				GeneralBase base_l = GeneralBase(model_info[0][d].base_type[i - 1]);
				right = base_l.bondaryMat(model_info[0][d].degree_of_freedom(i - 1) - 1, 1, 1);//注意此处continuity设置为1，此后有其他要求请调整//
				for (int k = 0; k < 1 + 1; k++) {//归一化//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark - model_info[0][d].degree_of_freedom(i - 1), 1 + 1, model_info[0][d].degree_of_freedom(i - 1)) = right;//

				len = model_info[0][d].discontinuous_points(i + 1) - model_info[0][d].discontinuous_points(i);
				GeneralBase base = GeneralBase(model_info[0][d].base_type[i]);
				left = base.bondaryMat(DF - 1, 1, 0);//注意此处continuity设置为1，此后有其他要求请调整
				for (int k = 0; k < 1 + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				right = base.bondaryMat(DF - 1, 1, 1);
				for (int k = 0; k < 1 + 1; k++) {//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, 1 + 1, model_info[0][d].degree_of_freedom(i)) = (-1) * left;//
				row_mark += 1 + 1;//
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, 1 + 1, model_info[0][d].degree_of_freedom(i)) = right;//
				col_mark += model_info[0][d].degree_of_freedom(i);
				row_mark += 1 + 1;//
				continue;
			}
			model_info[0][d].constrain_matirx.block(row_mark, col_mark, model_info[0][d].degree_of_freedom(i), model_info[0][d].degree_of_freedom(i)) = MatrixXd::Identity(model_info[0][d].degree_of_freedom(i), model_info[0][d].degree_of_freedom(i));
			row_mark += model_info[0][d].degree_of_freedom(i);
			col_mark += model_info[0][d].degree_of_freedom(i);
		}

		VectorXd b = VectorXd::Zero(row_mark);
		b.segment(1 + 2 * (1 + 1), patch.dimension_info[d].degree_of_freedom.sum()) = patch.coeff[d];
		MatrixXd Y = pinv(model_info[0][d].constrain_matirx);
		FullPivLU<MatrixXd> lu_decomp(model_info[0][d].constrain_matirx);
		model_info[0][d].Z = lu_decomp.kernel();
		model_info[0][d].Yb = Y * b;
	}
	yieldDesignMatrix();

	//showConstraint();
}


ECEngine::ECEngine(TensorModel patch, MatrixXd X, VectorXd Y, MatrixXd epsilon, MatrixXd main_model_range, MatrixXd DF, vector<vector<const char*>> type, MatrixXd continuity) {
	dimension = X.cols();
	weight.resize(1);
	rank = 0;
	max_rank = 1;
	model_info.resize(1);
	model_info[0].resize(dimension);
	model_range = main_model_range;
	design_matrix.resize(1);

	int num_of_data = X.rows();

	y_per_rank.resize(num_of_data, max_rank);

	x_normalized = X;
	y = Y;

	for (int d = 0; d < dimension; d++) {
		patch.dimension_info[d].discontinuous_points = patch.dimension_info[d].discontinuous_points * (patch.model_range(1, d) - patch.model_range(0, d)) + patch.model_range(0, d) * VectorXd::Ones(patch.dimension_info[d].discontinuous_points.size());
		patch.dimension_info[d].discontinuous_points = (patch.dimension_info[d].discontinuous_points - model_range(0, d) * VectorXd::Ones(patch.dimension_info[d].discontinuous_points.size())) / (model_range(1, d) - model_range(0, d));

		//此处先忽略麻烦的特殊情况，认为补丁加缓冲区全在主模型范围内；

		int leni = patch.dimension_info[d].discontinuous_points.size();
		model_info[0][d].discontinuous_points = VectorXd::Zero(leni + 4);
		model_info[0][d].discontinuous_points(1) = patch.dimension_info[d].discontinuous_points(0) - epsilon(0, d);
		model_info[0][d].discontinuous_points.segment(2, leni) = patch.dimension_info[d].discontinuous_points;
		model_info[0][d].discontinuous_points(leni + 4 - 2) = patch.dimension_info[d].discontinuous_points(leni - 1) + epsilon(1, d);
		model_info[0][d].discontinuous_points(leni + 4 - 1) = 1;

		leni = patch.dimension_info[d].degree_of_freedom.size();
		model_info[0][d].degree_of_freedom = VectorXd::Ones(leni + 4);//先置一
		model_info[0][d].degree_of_freedom(1) = DF(0, d);
		model_info[0][d].degree_of_freedom.segment(2, leni) = patch.dimension_info[d].degree_of_freedom;
		model_info[0][d].degree_of_freedom(leni + 4 - 2) = DF(1, d);

		leni = patch.dimension_info[d].base_type.size();
		vector<const char*> tmp_type;
		tmp_type.resize(leni + 4);
		tmp_type[0] = BASIC;//两边强制0，所以basic用常数0即可
		tmp_type[1] = type[0][d];
		tmp_type[leni + 4 - 1] = BASIC;
		tmp_type[leni + 4 - 2] = type[1][d];
		for (int i = 0; i < leni; i++) {
			tmp_type[i + 2] = patch.dimension_info[d].base_type[i];
		}
		model_info[0][d].base_type = tmp_type;

		//最喜欢的一集（恼
		//拼约束方程
		model_info[0][d].constrain_matirx = MatrixXd::Zero(4 + continuity.col(d).sum() + model_info[0][d].degree_of_freedom.sum() - DF.col(d).sum(), model_info[0][d].degree_of_freedom.sum());//8代表只能一阶连续

		int row_mark = 0;
		int col_mark = 0;
		MatrixXd left, right;

		leni = model_info[0][d].degree_of_freedom.size();

		for (int i = 0; i < model_info[0][d].degree_of_freedom.size(); i++) {//除了缓冲区外全用单位阵限制，缓冲区用L形的连续性方程约束
			if (i == 1) {
				double len = model_info[0][d].discontinuous_points(i + 1) - model_info[0][d].discontinuous_points(i);
				GeneralBase base = GeneralBase(model_info[0][d].base_type[i]);
				left = base.bondaryMat(model_info[0][d].degree_of_freedom(i) - 1, continuity(0, d), 0);//注意此处continuity设置为1，此后有其他要求请调整
				for (int k = 0; k < continuity(0, d) + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				right = base.bondaryMat(model_info[0][d].degree_of_freedom(i) - 1, continuity(1, d), 1);
				for (int k = 0; k < continuity(1, d) + 1; k++) {//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, continuity(0, d) + 1, model_info[0][d].degree_of_freedom(i)) = (-1) * left;//
				row_mark += continuity(0, d) + 1;//
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, continuity(1, d) + 1, model_info[0][d].degree_of_freedom(i)) = right;//
				col_mark += model_info[0][d].degree_of_freedom(i);

				GeneralBase base_r = GeneralBase(model_info[0][d].base_type[i + 1]);
				len = model_info[0][d].discontinuous_points(i + 2) - model_info[0][d].discontinuous_points(i + 1);
				left = base_r.bondaryMat(model_info[0][d].degree_of_freedom(i + 1) - 1, continuity(1, d), 0);//注意此处continuity设置为1，此后有其他要求请调整//
				for (int k = 0; k < continuity(1, d) + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, continuity(1, d) + 1, model_info[0][d].degree_of_freedom(i + 1)) = (-1) * left;//
				row_mark += continuity(1, d) + 1;//
				continue;
			}
			if (i == leni - 2) {
				double len = model_info[0][d].discontinuous_points(i) - model_info[0][d].discontinuous_points(i - 1);
				GeneralBase base_l = GeneralBase(model_info[0][d].base_type[i - 1]);
				right = base_l.bondaryMat(model_info[0][d].degree_of_freedom(i - 1) - 1, continuity(2, d), 1);//注意此处continuity设置为1，此后有其他要求请调整//
				for (int k = 0; k < continuity(2, d) + 1; k++) {//归一化//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark - model_info[0][d].degree_of_freedom(i - 1), continuity(2, d) + 1, model_info[0][d].degree_of_freedom(i - 1)) = right;//
				//问为什么这行row_mark不变的注意上次添加完单位阵到哪里了

				len = model_info[0][d].discontinuous_points(i + 1) - model_info[0][d].discontinuous_points(i);
				GeneralBase base = GeneralBase(model_info[0][d].base_type[i]);
				left = base.bondaryMat(model_info[0][d].degree_of_freedom(i) - 1, continuity(2, d), 0);//注意此处continuity设置为1，此后有其他要求请调整
				for (int k = 0; k < continuity(2, d) + 1; k++) {//归一化//
					left.row(k) = left.row(k) * pow(len, -k);
				}
				right = base.bondaryMat(model_info[0][d].degree_of_freedom(i) - 1, continuity(3, d), 1);
				for (int k = 0; k < continuity(3, d) + 1; k++) {//
					right.row(k) = right.row(k) * pow(len, -k);
				}
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, continuity(2, d) + 1, model_info[0][d].degree_of_freedom(i)) = (-1) * left;//
				row_mark += continuity(2, d) + 1;//
				model_info[0][d].constrain_matirx.block(row_mark, col_mark, continuity(3, d) + 1, model_info[0][d].degree_of_freedom(i)) = right;//
				col_mark += model_info[0][d].degree_of_freedom(i);
				row_mark += continuity(3, d) + 1;//
				continue;
			}
			model_info[0][d].constrain_matirx.block(row_mark, col_mark, model_info[0][d].degree_of_freedom(i), model_info[0][d].degree_of_freedom(i)) = MatrixXd::Identity(model_info[0][d].degree_of_freedom(i), model_info[0][d].degree_of_freedom(i));
			row_mark += model_info[0][d].degree_of_freedom(i);
			col_mark += model_info[0][d].degree_of_freedom(i);
		}

		VectorXd b = VectorXd::Zero(row_mark);
		b.segment(1 + 2 + continuity(0, d) + continuity(1, d), patch.dimension_info[d].degree_of_freedom.sum()) = patch.coeff[d];
		MatrixXd Y = pinv(model_info[0][d].constrain_matirx);
		FullPivLU<MatrixXd> lu_decomp(model_info[0][d].constrain_matirx);
		model_info[0][d].Z = lu_decomp.kernel();
		model_info[0][d].Yb = Y * b;
	}
	yieldDesignMatrix();
}


ECEngine::ECEngine(vector<TensorModel> patch, MatrixXd X, VectorXd Y, MatrixXd epsilon, MatrixXd main_model_range, MatrixXd DF, vector<vector<const char*>> type, MatrixXd continuity) {
	dimension = X.cols();
	max_rank = patch.size();
	weight.resize(max_rank);
	rank = 0;

	model_info.resize(max_rank);
	for (int r = 0; r < max_rank; r++) {
		model_info[r].resize(dimension);
	}

	model_range = main_model_range;
	design_matrix.resize(max_rank);

	int num_of_data = X.rows();

	y_per_rank.resize(num_of_data, max_rank);

	x_normalized = X;
	y = Y;

	for (int r = 0; r < max_rank; r++) {
		for (int d = 0; d < dimension; d++) {
			patch[r].dimension_info[d].discontinuous_points = patch[r].dimension_info[d].discontinuous_points * (patch[r].model_range(1, d) - patch[r].model_range(0, d)) + patch[r].model_range(0, d) * VectorXd::Ones(patch[r].dimension_info[d].discontinuous_points.size());
			patch[r].dimension_info[d].discontinuous_points = (patch[r].dimension_info[d].discontinuous_points - model_range(0, d) * VectorXd::Ones(patch[r].dimension_info[d].discontinuous_points.size())) / (model_range(1, d) - model_range(0, d));

			//此处先忽略麻烦的特殊情况，认为补丁加缓冲区全在主模型范围内；

			int leni = patch[r].dimension_info[d].discontinuous_points.size();
			model_info[r][d].discontinuous_points = VectorXd::Zero(leni + 4);
			model_info[r][d].discontinuous_points(1) = patch[r].dimension_info[d].discontinuous_points(0) - epsilon(0, d);
			model_info[r][d].discontinuous_points.segment(2, leni) = patch[r].dimension_info[d].discontinuous_points;
			model_info[r][d].discontinuous_points(leni + 4 - 2) = patch[r].dimension_info[d].discontinuous_points(leni - 1) + epsilon(1, d);
			model_info[r][d].discontinuous_points(leni + 4 - 1) = 1;

			leni = patch[r].dimension_info[d].degree_of_freedom.size();
			model_info[r][d].degree_of_freedom = VectorXd::Ones(leni + 4);//先置一
			model_info[r][d].degree_of_freedom(1) = DF(0, d);
			model_info[r][d].degree_of_freedom.segment(2, leni) = patch[r].dimension_info[d].degree_of_freedom;
			model_info[r][d].degree_of_freedom(leni + 4 - 2) = DF(1, d);

			leni = patch[r].dimension_info[d].base_type.size();
			vector<const char*> tmp_type;
			tmp_type.resize(leni + 4);
			tmp_type[0] = BASIC;//两边强制0，所以basic用常数0即可
			tmp_type[1] = type[0][d];
			tmp_type[leni + 4 - 1] = BASIC;
			tmp_type[leni + 4 - 2] = type[1][d];
			for (int i = 0; i < leni; i++) {
				tmp_type[i + 2] = patch[r].dimension_info[d].base_type[i];
			}
			model_info[r][d].base_type = tmp_type;

			//最喜欢的一集（恼
			//拼约束方程
			model_info[r][d].constrain_matirx = MatrixXd::Zero(4 + continuity.col(d).sum() + model_info[r][d].degree_of_freedom.sum() - DF.col(d).sum(), model_info[r][d].degree_of_freedom.sum());//8代表只能一阶连续

			int row_mark = 0;
			int col_mark = 0;
			MatrixXd left, right;

			leni = model_info[r][d].degree_of_freedom.size();

			for (int i = 0; i < model_info[r][d].degree_of_freedom.size(); i++) {//除了缓冲区外全用单位阵限制，缓冲区用L形的连续性方程约束
				if (i == 1) {
					double len = model_info[r][d].discontinuous_points(i + 1) - model_info[r][d].discontinuous_points(i);
					GeneralBase base = GeneralBase(model_info[r][d].base_type[i]);
					left = base.bondaryMat(model_info[r][d].degree_of_freedom(i) - 1, continuity(0, d), 0);//注意此处continuity设置为1，此后有其他要求请调整
					for (int k = 0; k < continuity(0, d) + 1; k++) {//归一化//
						left.row(k) = left.row(k) * pow(len, -k);
					}
					right = base.bondaryMat(model_info[r][d].degree_of_freedom(i) - 1, continuity(1, d), 1);
					for (int k = 0; k < continuity(1, d) + 1; k++) {//
						right.row(k) = right.row(k) * pow(len, -k);
					}
					model_info[r][d].constrain_matirx.block(row_mark, col_mark, continuity(0, d) + 1, model_info[r][d].degree_of_freedom(i)) = (-1) * left;//
					row_mark += continuity(0, d) + 1;//
					model_info[r][d].constrain_matirx.block(row_mark, col_mark, continuity(1, d) + 1, model_info[r][d].degree_of_freedom(i)) = right;//
					col_mark += model_info[r][d].degree_of_freedom(i);

					GeneralBase base_r = GeneralBase(model_info[r][d].base_type[i + 1]);
					len = model_info[r][d].discontinuous_points(i + 2) - model_info[r][d].discontinuous_points(i + 1);
					left = base_r.bondaryMat(model_info[r][d].degree_of_freedom(i + 1) - 1, continuity(1, d), 0);//注意此处continuity设置为1，此后有其他要求请调整//
					for (int k = 0; k < continuity(1, d) + 1; k++) {//归一化//
						left.row(k) = left.row(k) * pow(len, -k);
					}
					model_info[r][d].constrain_matirx.block(row_mark, col_mark, continuity(1, d) + 1, model_info[r][d].degree_of_freedom(i + 1)) = (-1) * left;//
					row_mark += continuity(1, d) + 1;//
					continue;
				}
				if (i == leni - 2) {
					double len = model_info[r][d].discontinuous_points(i) - model_info[r][d].discontinuous_points(i - 1);
					GeneralBase base_l = GeneralBase(model_info[r][d].base_type[i - 1]);
					right = base_l.bondaryMat(model_info[r][d].degree_of_freedom(i - 1) - 1, continuity(2, d), 1);//注意此处continuity设置为1，此后有其他要求请调整//
					for (int k = 0; k < continuity(2, d) + 1; k++) {//归一化//
						right.row(k) = right.row(k) * pow(len, -k);
					}
					model_info[r][d].constrain_matirx.block(row_mark, col_mark - model_info[r][d].degree_of_freedom(i - 1), continuity(2, d) + 1, model_info[r][d].degree_of_freedom(i - 1)) = right;//
					//问为什么这行row_mark不变的注意上次添加完单位阵到哪里了

					len = model_info[r][d].discontinuous_points(i + 1) - model_info[r][d].discontinuous_points(i);
					GeneralBase base = GeneralBase(model_info[r][d].base_type[i]);
					left = base.bondaryMat(model_info[r][d].degree_of_freedom(i) - 1, continuity(2, d), 0);//注意此处continuity设置为1，此后有其他要求请调整
					for (int k = 0; k < continuity(2, d) + 1; k++) {//归一化//
						left.row(k) = left.row(k) * pow(len, -k);
					}
					right = base.bondaryMat(model_info[r][d].degree_of_freedom(i) - 1, continuity(3, d), 1);
					for (int k = 0; k < continuity(3, d) + 1; k++) {//
						right.row(k) = right.row(k) * pow(len, -k);
					}
					model_info[r][d].constrain_matirx.block(row_mark, col_mark, continuity(2, d) + 1, model_info[r][d].degree_of_freedom(i)) = (-1) * left;//
					row_mark += continuity(2, d) + 1;//
					model_info[r][d].constrain_matirx.block(row_mark, col_mark, continuity(3, d) + 1, model_info[r][d].degree_of_freedom(i)) = right;//
					col_mark += model_info[r][d].degree_of_freedom(i);
					row_mark += continuity(3, d) + 1;//
					continue;
				}
				model_info[r][d].constrain_matirx.block(row_mark, col_mark, model_info[r][d].degree_of_freedom(i), model_info[r][d].degree_of_freedom(i)) = MatrixXd::Identity(model_info[r][d].degree_of_freedom(i), model_info[r][d].degree_of_freedom(i));
				row_mark += model_info[r][d].degree_of_freedom(i);
				col_mark += model_info[r][d].degree_of_freedom(i);
			}

			VectorXd b = VectorXd::Zero(row_mark);
			b.segment(1 + 2 + continuity(0, d) + continuity(1, d), patch[r].dimension_info[d].degree_of_freedom.sum()) = patch[r].coeff[d];
			MatrixXd Y = pinv(model_info[r][d].constrain_matirx);
			FullPivLU<MatrixXd> lu_decomp(model_info[r][d].constrain_matirx);
			model_info[r][d].Z = lu_decomp.kernel();
			model_info[r][d].Yb = Y * b;
		}
	}
	yieldDesignMatrix();
}


ECEngine::ECEngine(vector<TensorModel> model) {
	//利用提取出的模型生成一个实例
	int length_of_model = model.size();
	rank = length_of_model;
	max_rank = length_of_model;
	model_info.resize(length_of_model);
	coeff.resize(length_of_model);
	weight.resize(length_of_model);
	//由于后续不需要update，y_per_rank没用了


	for (int i = 0; i < length_of_model; i++) {
		dimension = model[0].dimension;//本来dimension是固定的但是放外面要规避model为空不如直接放里面
		coeff[i] = model[i].coeff;
		weight(i) = model[i].weight;
		model_range = model[0].model_range;
		model_info[i] = model[i].dimension_info;
	}
}

void ECEngine::sortVector(const VectorXd& vec, VectorXd& sorted_vec, VectorXi& ind) {//CSDN超的https://blog.csdn.net/juluwangriyue/article/details/122226836

	ind = VectorXi::LinSpaced(vec.size(), 0, vec.size() - 1);//[0 1 2 3 ... N-1]
	auto rule = [vec](int i, int j)->bool {
		return vec(i) < vec(j);//此处递增
	};//正则表达式，作为sort的谓词
	std::sort(ind.data(), ind.data() + ind.size(), rule);
	//data成员函数返回VectorXd的第一个元素的指针，类似于begin()
	sorted_vec.resize(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		sorted_vec(i) = vec(ind(i));
	}
}


void ECEngine::yieldDesignMatrix() {//使用训练数据生成，在训练中复用//这玩意只在训练前调用一次，就不考虑输入数据超出建模的问题了

	vector<MatrixXd> design(dimension);
	int num_of_data = x_normalized.rows();
	//示性函数导致循序随机的data耗时很长（话说只要是多维的数据必有维度乱序），所以还是先排序再计算吧,排序用的是void ECEngine::sortVector(const VectorXd& vec, VectorXd& sorted_vec, VectorXi& ind)

	for (int r = 0; r < max_rank; r++) {
		for (int d = 0; d < dimension; d++) {
			design[d] = MatrixXd::Zero(num_of_data, model_info[r][d].degree_of_freedom.sum());
			MatrixXd tmp = MatrixXd::Zero(num_of_data, model_info[r][d].degree_of_freedom.sum());
			VectorXd x_sorted;
			VectorXi idx;
			sortVector(x_normalized.col(d), x_sorted, idx);
			int num_of_blocks = model_info[r][d].degree_of_freedom.size();
			int block_cnt = 0;
			int head = 0, tail = 0;
			int mark = 0;
			while (tail < num_of_data) {


				//cout << "discontinuous_points:" << model_info[r][d].discontinuous_points << endl << endl;
				//cout << "x:" << x_sorted << endl << endl;

				if (x_sorted(tail) <= model_info[r][d].discontinuous_points(block_cnt + 1))
				{
					tail++;
				}
				else
				{

					//cout << "r" << r << "d" << d << endl << endl << model_info[r][d].base_type[block_cnt] << endl;
					//cout << "head" << head << "tail" << tail << endl << endl;
					GeneralBase base = GeneralBase(model_info[r][d].base_type[block_cnt]);
					//cout << endl << endl << x_sorted.segment(head, tail - head) << endl;
					//cout << endl << endl << model_info[r][d].discontinuous_points(block_cnt) << endl;
					//cout << endl << endl << model_info[r][d].discontinuous_points(block_cnt + 1) - model_info[r][d].discontinuous_points(block_cnt) << endl;
					//cout << endl << endl << block_cnt << endl;

					VectorXd x_seg_nor = (x_sorted.segment(head, tail - head) - model_info[r][d].discontinuous_points(block_cnt) * VectorXd::Ones(tail - head)) / (model_info[r][d].discontinuous_points(block_cnt + 1) - model_info[r][d].discontinuous_points(block_cnt));
					//cout << x_sorted.segment(head, tail - head) << endl << endl << model_info[r][d].degree_of_freedom(block_cnt) - 1 << endl << endl;
					if (x_seg_nor.size()) {
						MatrixXd xpower = base.calXPower(x_seg_nor, model_info[r][d].degree_of_freedom(block_cnt) - 1);
						tmp.block(head, mark, tail - head, model_info[r][d].degree_of_freedom(block_cnt)) = base.valueToOrder(xpower, model_info[r][d].degree_of_freedom(block_cnt) - 1);
					}
					mark += model_info[r][d].degree_of_freedom(block_cnt);
					block_cnt++;
					head = tail;
				}
			}
			//最后一个区间的不会进else，单独处理
			GeneralBase base = GeneralBase(model_info[r][d].base_type[block_cnt]);
			VectorXd x_seg_nor = (x_sorted.segment(head, tail - head) - model_info[r][d].discontinuous_points(block_cnt) * VectorXd::Ones(tail - head)) / (model_info[r][d].discontinuous_points(block_cnt + 1) - model_info[r][d].discontinuous_points(block_cnt));

			if (x_seg_nor.size()) {
				MatrixXd xpower = base.calXPower(x_seg_nor, model_info[r][d].degree_of_freedom(block_cnt) - 1);
				tmp.block(head, mark, tail - head, model_info[r][d].degree_of_freedom(block_cnt)) = base.valueToOrder(xpower, model_info[r][d].degree_of_freedom(block_cnt) - 1);
			}

			for (int n = 0; n < num_of_data; n++) {
				design[d].row(idx(n)) = tmp.row(n);
			}
		}
		design_matrix[r] = design;
	}


}

void ECEngine::yieldDesignMatrix(MatrixXd x_input) {
	//使用指定数据根据已有划分生成，默认数据归一化，对于patches需要考虑超出建模范围的输入数据响应要为0

	int Rank = coeff.size();
	design_matrix.resize(Rank);

	for (int k = 0; k < Rank; k++) {
		vector<MatrixXd> design(dimension);
		int num_of_data = x_input.rows();
		for (int d = 0; d < dimension; d++) {
			design[d] = MatrixXd::Zero(num_of_data, model_info[k][d].degree_of_freedom.sum());
			MatrixXd tmp = MatrixXd::Zero(num_of_data, model_info[k][d].degree_of_freedom.sum());
			VectorXd x_sorted;
			VectorXi idx;
			sortVector(x_input.col(d), x_sorted, idx);
			int num_of_blocks = model_info[k][d].degree_of_freedom.size();
			int block_cnt = 0;
			int head = 0, tail = 0;
			int mark = 0;
			int end = num_of_data - 1;
			int len = model_info[k][d].discontinuous_points.size() - 1;
			while (x_sorted(end) > model_info[k][d].discontinuous_points(len)) {
				end--;
			}
			end += 1;

			while (x_sorted(tail) < model_info[k][d].discontinuous_points(block_cnt)) {
				head++;
				tail++;
			}//先把头尾都移到第一个在建模范围内的数据上；

			while (tail < end) {
				if (x_sorted(tail) <= model_info[k][d].discontinuous_points(block_cnt + 1))
				{
					tail++;
				}
				else
				{
					GeneralBase base = GeneralBase(model_info[k][d].base_type[block_cnt]);
					VectorXd x_seg_nor = (x_sorted.segment(head, tail - head) - model_info[k][d].discontinuous_points(block_cnt) * VectorXd::Ones(tail - head)) / (model_info[k][d].discontinuous_points(block_cnt + 1) - model_info[k][d].discontinuous_points(block_cnt));
					if (x_seg_nor.size()) {
						MatrixXd xpower = base.calXPower(x_seg_nor, model_info[k][d].degree_of_freedom(block_cnt) - 1);
						tmp.block(head, mark, tail - head, model_info[k][d].degree_of_freedom(block_cnt)) = base.valueToOrder(xpower, model_info[k][d].degree_of_freedom(block_cnt) - 1);
					}
					mark += model_info[k][d].degree_of_freedom(block_cnt);
					block_cnt++;
					head = tail;
				}
			}
			GeneralBase base = GeneralBase(model_info[k][d].base_type[block_cnt]);
			VectorXd x_seg_nor = (x_sorted.segment(head, tail - head) - model_info[k][d].discontinuous_points(block_cnt) * VectorXd::Ones(tail - head)) / (model_info[k][d].discontinuous_points(block_cnt + 1) - model_info[k][d].discontinuous_points(block_cnt));

			if (x_seg_nor.size()) {
				MatrixXd xpower = base.calXPower(x_seg_nor, model_info[k][d].degree_of_freedom(block_cnt) - 1);
				tmp.block(head, mark, tail - head, model_info[k][d].degree_of_freedom(block_cnt)) = base.valueToOrder(xpower, model_info[k][d].degree_of_freedom(block_cnt) - 1);
			}

			for (int n = 0; n < num_of_data; n++) {
				design[d].row(idx(n)) = tmp.row(n);
			}
		}
		design_matrix[k] = design;
	}
}

VectorXd ECEngine::modelEval() {
	//仅用在correction中，故不会有patches的响应
	int num_of_data = x_normalized.rows();
	int Rank = rank;
	VectorXd y_hat = VectorXd::Zero(num_of_data);
	for (int i = 0; i < Rank; i++) {
		VectorXd y_tmpr = VectorXd::Ones(num_of_data);
		for (int d = 0; d < dimension; d++) {
			y_tmpr = y_tmpr.cwiseProduct(design_matrix[i][d] * coeff[i][d]);
		}
		y_hat += y_tmpr * weight(i);
	}
	return y_hat;
}

VectorXd ECEngine::modelEval(MatrixXd x_input) {
	//此处默认x_input已经归一化
	int num_of_data = x_input.rows();
	int Rank = rank;
	yieldDesignMatrix(x_input);
	VectorXd y_hat = VectorXd::Zero(num_of_data);
	for (int i = 0; i < Rank; i++) {
		VectorXd y_tmpr = VectorXd::Ones(num_of_data);
		for (int d = 0; d < dimension; d++) {
			y_tmpr = y_tmpr.cwiseProduct(design_matrix[i][d] * coeff[i][d]);
		}
		y_hat += y_tmpr * weight(i);
	}
	return y_hat;
}


void ECEngine::coeffCorrection(int type, double alpha) {

	int num_of_data = x_normalized.rows();

	//初始化coeff和weight
	vector<VectorXd>  coeff_ini(dimension);
	for (int d = 0; d < dimension; d++) {
		coeff_ini[d] = VectorXd::Ones(model_info[rank][d].degree_of_freedom.sum());
	}
	coeff[rank] = coeff_ini;//一定注意coeff的初始化是在correction中进行的
	weight(rank) = 1;

	//初始化误差序列
	int iter = 0;
	VectorXd error = VectorXd::Zero(max_iter + 1);

	VectorXd y_hat = VectorXd::Zero(num_of_data);
	for (int i = 0; i < rank + 1; i++) {
		VectorXd y_tmpr = VectorXd::Ones(num_of_data);
		for (int d = 0; d < dimension; d++) {
			y_tmpr = y_tmpr.cwiseProduct(design_matrix[i][d] * coeff[i][d]);
		}
		y_hat += y_tmpr * weight(i);
	}

	error(0) = (r - y_hat).squaredNorm() / num_of_data;

	double error_delta = 1;
	MatrixXd V = MatrixXd::Ones(num_of_data, dimension);
	if (type) {
		while (iter<max_iter && error_delta>error_bound) {

			for (int d = 0; d < dimension; d++) {
				VectorXd product_of_frozen_dim = VectorXd::Ones(num_of_data);
				for (int p = 0; p < dimension; p++) {
					if (p != d) {
						product_of_frozen_dim = product_of_frozen_dim.cwiseProduct(V.col(p));
					}
				}
				MatrixXd temp = design_matrix[rank][d];
				for (int n = 0; n < num_of_data; n++) {
					temp.row(n) = temp.row(n) * product_of_frozen_dim(n);
				}
				MatrixXd tempz = temp * model_info[rank][d].Z;

				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * ((0*MatrixXd::Identity(tempz.cols(),tempz.cols())+tempz.transpose()*tempz).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tempz.transpose()*(r - temp * model_info[rank][d].Yb)));
				coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * pathwise_ols(tempz, (r - temp * model_info[rank][d].Yb), tempz.cols(), alpha, type);
				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * tempz.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve((r - temp * model_info[rank][d].Yb));//svd so fking slow.
				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * (tempz.transpose() * tempz).ldlt().solve(tempz.transpose() * (r - temp * model_info[rank][d].Yb));
				//cout << endl << coeff[rank][d].transpose() << endl;

				V.col(d) = design_matrix[rank][d] * coeff[rank][d];
			}
			iter += 1;
			error(iter) = (r - V.rowwise().prod()).squaredNorm() / num_of_data;
			error_delta = abs(error(iter) - error(iter - 1));
		}
	}
	else {
		while (iter<max_iter && error_delta>error_bound) {

			for (int d = 0; d < dimension; d++) {
				VectorXd product_of_frozen_dim = VectorXd::Ones(num_of_data);
				for (int p = 0; p < dimension; p++) {
					if (p != d) {
						product_of_frozen_dim = product_of_frozen_dim.cwiseProduct(V.col(p));
					}
				}
				MatrixXd temp = design_matrix[rank][d];
				for (int n = 0; n < num_of_data; n++) {
					temp.row(n) = temp.row(n) * product_of_frozen_dim(n);
				}
				MatrixXd tempz = temp * model_info[rank][d].Z;

				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * ((10000*MatrixXd::Identity(tempz.cols(),tempz.cols())+tempz.transpose()*tempz).bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tempz.transpose()*(r - temp * model_info[rank][d].Yb)));
				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * pathwise_ols(tempz, (r - temp * model_info[rank][d].Yb), tempz.cols(), 0.5, 2);
				coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * tempz.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve((r - temp * model_info[rank][d].Yb));//svd so fking slow.
				//coeff[rank][d] = model_info[rank][d].Yb + model_info[rank][d].Z * (tempz.transpose() * tempz).ldlt().solve(tempz.transpose() * (r - temp * model_info[rank][d].Yb));
				//cout << endl << coeff[rank][d].transpose() << endl;

				V.col(d) = design_matrix[rank][d] * coeff[rank][d];
			}
			iter += 1;
			error(iter) = (r - V.rowwise().prod()).squaredNorm() / num_of_data;
			error_delta = abs(error(iter) - error(iter - 1));
		}
	}
	y_per_rank.col(rank) = V.rowwise().prod();
}

void ECEngine::modelUpdate() {
	int num_of_data = y.size();
	MatrixXd ys = y_per_rank.block(0, 0, num_of_data, rank);
	//weight.segment(0, rank) = (ys.transpose() * ys).ldlt().solve(ys.transpose() * y);
	weight.segment(0, rank) = ys.bdcSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(y);
	r = y - modelEval();
}

void ECEngine::modelTrain() {
	coeff.resize(max_rank);
	for (int r = rank; r < max_rank; r++) {
		coeffCorrection(0, 0);
		rank += 1;
		modelUpdate();
	}
}

void ECEngine::modelTrain(int type, double alpha) {
	coeff.resize(max_rank);
	for (int r = rank; r < max_rank; r++) {
		coeffCorrection(type, alpha);
		rank += 1;
		modelUpdate();
	}
}

//void ECEngine::appendPatch(TensorModel patch,MatrixXd epsilon, int DF, const char* type) {
//	//将patches平滑化后加入主模型
//	ECEngine patch_polished = ECEngine::ECEngine(patch, x_normalized, y, epsilon, model_range, DF, type);
//	patch_polished.r = r;
//	patch_polished.coeffCorrection();//平滑化
//	TensorModel patch_tensor = patch_polished.generateTensorModel(VectorXd::Zero(1))[0];//只提取一层的问题
//
//	//显然append后不能再对weight进行更新；
//	model_info.push_back(patch_tensor.dimension_info);
//	coeff.push_back(patch_tensor.coeff);
//
//	//对weight处理为直接后面加个1 
//	VectorXd weight_tmp = VectorXd::Ones(max_rank+1);
//	weight_tmp.segment(0, max_rank) = weight;
//	weight_tmp(max_rank) = 1;
//	weight = weight_tmp;
//	rank += 1;
//	max_rank += 1;
//	design_matrix.resize(rank);
//	yieldDesignMatrix();
//}

void ECEngine::appendPatch(TensorModel patch, MatrixXd epsilon, MatrixXd DF, vector<vector<const char*>> type, MatrixXd continuity, int train_type, double alpha) {
	//将patches平滑化后加入主模型
	ECEngine patch_polished = ECEngine::ECEngine(patch, x_normalized, y, epsilon, model_range, DF, type, continuity);
	patch_polished.r = r;
	patch_polished.coeffCorrection(train_type, alpha);//平滑化
	TensorModel patch_tensor = patch_polished.generateTensorModel(VectorXd::Zero(1))[0];//只提取一层的问题

	//显然append后不能再对weight进行更新；
	model_info.push_back(patch_tensor.dimension_info);
	coeff.push_back(patch_tensor.coeff);

	//对weight处理为直接后面加个1 
	VectorXd weight_tmp = VectorXd::Ones(max_rank + 1);
	weight_tmp.segment(0, max_rank) = weight;
	weight_tmp(max_rank) = patch.weight;
	weight = weight_tmp;
	rank += 1;
	max_rank += 1;
	design_matrix.resize(rank);
	yieldDesignMatrix();
}


void ECEngine::appendPatch(vector<TensorModel> patch, MatrixXd epsilon, MatrixXd DF, vector<vector<const char*>> type, MatrixXd continuity, int train_type, double alpha) {

	int num_of_newlayers = patch.size();

	ECEngine patch_polished = ECEngine::ECEngine(patch, x_normalized, y, epsilon, model_range, DF, type, continuity);
	patch_polished.r = r;

	
	patch_polished.coeff.resize(num_of_newlayers);
	for (int l = 0; l < num_of_newlayers; l++) {
		patch_polished.coeffCorrection(train_type, alpha);//平滑化
		patch_polished.rank += 1;
		r = r - patch_polished.modelEval();
	}


	vector<TensorModel> patch_tensor = patch_polished.generateTensorModel();

	for (int l = 0; l < num_of_newlayers; l++) {
		model_info.push_back(patch_tensor[l].dimension_info);
		coeff.push_back(patch_tensor[l].coeff);
	}

	//显然append后不能再对weight进行更新；
	//对weight处理为直接后面加个1 
	VectorXd weight_tmp = VectorXd::Ones(max_rank + num_of_newlayers);
	weight_tmp.segment(0, max_rank) = weight;
	for (int l = 0; l < num_of_newlayers; l++) {
		weight_tmp(max_rank + l) = patch[l].weight;
	}
	weight = weight_tmp;
	rank += num_of_newlayers;
	max_rank += num_of_newlayers;
	design_matrix.resize(rank);
	yieldDesignMatrix();
}

vector<TensorModel> ECEngine::generateTensorModel(VectorXd list) {
	//给定一组向量，要求输出这组向量序号的秩的模型
	vector<TensorModel> generated_model;
	//验证list合法性
	if ((list.size() > rank) || (list.maxCoeff() > rank) || (list.minCoeff() < 0)) {
		cout << endl << "要提取的模型非法";
		return generated_model;//先返回一个空的吧
	}
	int length_of_list = list.size();
	generated_model.resize(length_of_list);

	for (int i = 0; i < length_of_list; i++) {
		generated_model[i].dimension = dimension;
		generated_model[i].coeff = coeff[i];
		generated_model[i].weight = weight(i);
		generated_model[i].model_range = model_range;
		generated_model[i].dimension_info = model_info[i];
	}
	return generated_model;
}

vector<TensorModel> ECEngine::generateTensorModel() {
	//直接输出现有的所有层模型
	vector<TensorModel> generated_model;
	generated_model.resize(rank);
	for (int i = 0; i < rank; i++) {
		generated_model[i].dimension = dimension;
		generated_model[i].coeff = coeff[i];
		generated_model[i].weight = weight(i);
		generated_model[i].model_range = model_range;
		generated_model[i].dimension_info = model_info[i];
	}
	return generated_model;
}


void ECEngine::inputNormalization(MatrixXd& x_input) {
	//注意是将input直接改变，不回传
	int num_of_data = x_input.rows();
	for (int i = 0; i < dimension; i++) {
		x_input.col(i) = (x_input.col(i) - model_range(0, i) * VectorXd::Ones(num_of_data)) / (model_range(1, i) - model_range(0, i));
	}
}


void ECEngine::displayModelInfo(string type) {
	for (int i = 0; i < max_rank; i++) {
		for (int j = 0; j < dimension; j++) {
			saveDataCsv(DATADISPLAY + type + "designMatrix/" + to_string(i) + "_" + to_string(j) + ".csv", design_matrix[i][j]);
			saveDataCsv(DATADISPLAY + type + "constraint/" + to_string(i) + "_" + to_string(j) + ".csv", model_info[i][j].constrain_matirx);
			saveDataCsv(DATADISPLAY + type + "Z/" + to_string(i) + "_" + to_string(j) + ".csv", model_info[i][j].Z);
			saveDataCsv(DATADISPLAY + type + "coeff/" + to_string(i) + "_" + to_string(j) + ".csv", coeff[i][j]);
		}
	}
}



MatrixXd ECEngine::pinv(MatrixXd inMatrix)
{
	double pinvtoler = 1e-12; // choose your tolerance wisely!
	Eigen::JacobiSVD<Eigen::MatrixXd> svd(inMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
	Eigen::VectorXd singularValues_inv = svd.singularValues();
	Eigen::VectorXd sv = svd.singularValues();

	int len = singularValues_inv.size();

	for (Eigen::Index i = 0; i < len; ++i)
	{
		if (sv(i) > pinvtoler)
		{
			singularValues_inv(i) = 1.0 / sv(i);
		}
		else
		{
			singularValues_inv(i) = 0;
		}
	}
	Eigen::MatrixXd outMatrix = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose());

	return outMatrix;
}

//Eigen::MatrixXd& pinv(Eigen::MatrixXd& outMatrix, const Eigen::MatrixXd& inMatrix)
//{
//	double pinvtoler = 1.e-6; // choose your tolerance wisely!
//	Eigen::JacobiSVD<Eigen::MatrixXd> svd(inMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
//	Eigen::VectorXd singularValues_inv = svd.singularValues();
//	Eigen::VectorXd sv = svd.singularValues();
//	for (Eigen::Index i = 0; i < svd.cols(); ++i)
//	{
//		if (sv(i) > pinvtoler)
//		{
//			singularValues_inv(i) = 1.0 / sv(i);
//		}
//		else
//		{
//			singularValues_inv(i) = 0;
//		}
//	}
//	outMatrix = (svd.matrixV() * singularValues_inv.asDiagonal() * svd.matrixU().transpose());
//	return outMatrix;
//}//留一个网上的原始版本在这里

MatrixXd ECEngine::orthogonalizeNullSpace(MatrixXd matrix) {
	FullPivLU<MatrixXd> lu(matrix);
	MatrixXd nullSpace = lu.kernel();
	ColPivHouseholderQR<MatrixXd> qr(nullSpace);
	MatrixXd Q = qr.householderQ();

	// 标准化列向量，使其模为1
	for (int i = 0; i < Q.cols(); ++i) {
		Q.col(i) /= Q.col(i).norm();
	}

	return Q;
}


void ECEngine::resetConfig(vector<ModelConfig> model_config, TrainConfig train_config, int retrain_from_layer) {
	max_rank = train_config.max_rank;
	max_iter = train_config.max_iter;
	error_bound = train_config.error_bound;

	//input normalization
	//请在此前完成特殊点的加入//好像有了patch功能就不需要了

	int num_of_data = x_normalized.rows();

	rank = retrain_from_layer;
	weight.conservativeResize(max_rank);
	y_per_rank.conservativeResize(num_of_data, max_rank);
	model_info.resize(max_rank);
	design_matrix.resize(max_rank);

	MatrixXd left, right;

	for (int k = rank; k < max_rank; k++) {
		model_info[k].resize(dimension);
		for (int d = 0; d < dimension; d++) {
			model_info[k][d].discontinuous_points = model_config[k].unit_boundaries[d];
			model_info[k][d].degree_of_freedom = model_config[k].unit_degree_of_freedom[d];
			model_info[k][d].base_type = model_config[k].unit_base_type[d];

			//coeff[0][d] = VectorXd::Ones(model_config[k].unit_degree_of_freedom[d].sum());

			//约束数model_config[k].continuity[d].sum() + model_config[k].num_of_units(d) + 1在无约束为-1时仍然成立
			model_info[k][d].constrain_matirx = MatrixXd::Zero(model_config[k].continuity[d].sum() + model_config[k].num_of_units(d) + 1, model_config[k].unit_degree_of_freedom[d].sum());
			//model_info[k][d].interval_length = VectorXd::Zero(model_config[k].num_of_units(d));
			//model_info[k][d].boundary_value.resize(model_config[k].num_of_units(d));
			//取消了边界长度和边界值的存储

			int row_mark = 0;
			int col_mark = 0;
			for (int i = 0; i < model_config[k].num_of_units(d); i++) {
				//model_info[k][d].interval_length(i) = model_config[k].unit_boundaries[d](i + 1) - model_config[k].unit_boundaries[d](i);
				double len = model_config[k].unit_boundaries[d](i + 1) - model_config[k].unit_boundaries[d](i);
				//model_info[k][d].boundary_value[i].resize(2);
				GeneralBase base = GeneralBase(model_config[k].unit_base_type[d][i]);
				left = base.bondaryMat(model_config[k].unit_degree_of_freedom[d](i) - 1, model_config[k].continuity[d](i), 0);
				for (int j = 0; j < model_config[k].continuity[d](i) + 1; j++) {//归一化
					left.row(j) = left.row(j) * pow(len, -j);
				}
				right = base.bondaryMat(model_config[k].unit_degree_of_freedom[d](i) - 1, model_config[k].continuity[d](i + 1), 1);
				for (int j = 0; j < model_config[k].continuity[d](i + 1) + 1; j++) {
					right.row(j) = right.row(j) * pow(len, -j);
				}
				model_info[k][d].constrain_matirx.block(row_mark, col_mark, model_config[k].continuity[d](i) + 1, model_config[k].unit_degree_of_freedom[d](i)) = (-1) * left;
				row_mark += model_config[k].continuity[d](i) + 1;
				model_info[k][d].constrain_matirx.block(row_mark, col_mark, model_config[k].continuity[d](i + 1) + 1, model_config[k].unit_degree_of_freedom[d](i)) = right;
				col_mark += model_config[k].unit_degree_of_freedom[d](i);
			}

			FullPivLU<MatrixXd> lu_decomp(model_info[k][d].constrain_matirx);
			model_info[k][d].Z = lu_decomp.kernel();
			//model_info[k][d].Z = loadDataCsv("Z.csv");
			//model_info[k][d].Z = orthogonalizeNullSpace(model_info[k][d].constrain_matirx);
			model_info[k][d].Yb = VectorXd::Zero(model_info[k][d].degree_of_freedom.sum());
		}
	}
	yieldDesignMatrix();
}