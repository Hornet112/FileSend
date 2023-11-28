#pragma once
#include<iostream>
#include<fstream>
#include<cmath>
#include<json.h>
#include<vector>
#include<crtdbg.h>
#include"TypeDefine.h"
#include"ECEngine.h"
#include"readjson.h"
using namespace std;

void coutmatrix(vector<VectorXd> in) {
	for (int i = 0; i < in.size(); i++) {
		cout << in[i] << endl;
	}
}
bool fileExists(const std::string& filename) {
	std::ifstream file(filename);
	return file.good();
}
vector<ModelConfig> readModelJson(const string& filepath) {
	Json::Reader reader;
	Json::Value mc_json_val;
	
	vector<ModelConfig> Model_config{};
	int dimension, num_of_layers;
	if (fileExists(filepath)) {
		ifstream in(filepath);
		if (!in.is_open()) { throw runtime_error("error openning" + filepath); }
		if (reader.parse(in, mc_json_val)) {
			dimension = mc_json_val["dimension"].asInt();
			num_of_layers = mc_json_val["num_of_layers"].asInt();

			Model_config.resize(num_of_layers);

			for (int i = 0; i < num_of_layers; i++) {
				Json::Value layer = mc_json_val["layer_" + to_string(i + 1)];
				//Units  一维向量读出即可
				string Unitspath = mc_json_val["layer_" + to_string(i + 1)]["Units"].asString();
				Model_config[i].num_of_units = loadDataCsv(Unitspath);
				// 
				//UnitDF dim维度的XY维向量
				// UnitBound dim维度的 X+1 Y+1维向量
				// Continuity 同上
				vector<VectorXd>UnitDF(dimension);
				vector<VectorXd>UnitBound(dimension);
				vector<VectorXd>Continuity(dimension);

				for (int d = 0; d < dimension; d++) {
					string Continuity_path = layer["dimension_" + to_string(d + 1)]["Continuity"].asString();
					string UnitBound_path = layer["dimension_" + to_string(d + 1)]["UnitBound"].asString();
					string UnitDF_path = layer["dimension_" + to_string(d + 1)]["UnitDF"].asString();

					UnitDF[d] = loadDataCsv(UnitDF_path);
					UnitBound[d] = loadDataCsv(UnitBound_path);
					Continuity[d] = loadDataCsv(Continuity_path);

				}
				Model_config[i].unit_degree_of_freedom = UnitDF;
				Model_config[i].unit_boundaries = UnitBound;
				Model_config[i].continuity = Continuity;
				Model_config[i].unit_base_type.resize(dimension);
				for (int d = 0; d < dimension; d++) {
					Model_config[i].unit_base_type[d].resize(Model_config[i].num_of_units(d));
					for (int j = 0; j < Model_config[i].num_of_units(d); j++) {
						Model_config[i].unit_base_type[d][j] = BASIC;
					}
				}
			}
		}
		in.close();
	}
	else {
		throw runtime_error("File not found:" + filepath);
		//exit(1);
	}
	
	
	return Model_config;
}
TrainConfig readTrainJson(const string& filepath) {
	Json::Reader reader;
	Json::Value tc_json_val;
	TrainConfig train_config;
	if (fileExists(filepath)) {
		ifstream in(filepath);
		if (!in.is_open()) { throw runtime_error("error openning" + filepath); }
		if (reader.parse(in, tc_json_val)) {
			train_config.error_bound = tc_json_val["error_bound"].asDouble();
			train_config.max_iter = tc_json_val["max_iter"].asInt();
			train_config.max_rank = tc_json_val["max_rank"].asInt();
			return train_config;
		}
		else {
			throw runtime_error("Jsonfile read error");
		}
		in.close();
	}
	else {
		throw runtime_error("File not found:" + filepath);
	}

}

vector<string> ReadSetJson(const string& jsonfile) {
	Json::Reader reader;
	Json::Value set_val;
	vector<string> output{};
	if (fileExists(jsonfile)) {
		ifstream in(jsonfile);
		if (!in.is_open()) { throw runtime_error("error opening" + jsonfile); }
		if (reader.parse(in, set_val)) {
			output.push_back(set_val["model_config"].asString());
			output.push_back(set_val["train_config"].asString());
			output.push_back(set_val["X"].asString());
			output.push_back(set_val["Y"].asString());
		}
		in.close();
		return output;
	}
	else {
		throw runtime_error("File not found:" + jsonfile);
	}
}
tuple<vector<string>,int,double> ReadResetJson(const string& jsonfile) {
	Json::Reader reader;
	Json::Value set_val;
	vector<string> output{};
	if (fileExists(jsonfile)) {
		ifstream in(jsonfile);
		int type;
		double alpha;
		if (!in.is_open()) { throw runtime_error("error opening" + jsonfile); }
		if (reader.parse(in, set_val)) {
			output.push_back(set_val["model_config"].asString());
			output.push_back(set_val["train_config"].asString());
			type = set_val["type"].asInt();
			alpha = set_val["alpha"].asDouble();
		}
		in.close();
		return make_tuple(output,type,alpha);
	}
	else {
		throw runtime_error("File not found:" + jsonfile);
	}
}
vector<string> ReadPatchJson(const string& jsonfile) {
	Json::Reader reader;
	Json::Value patch_val;
	vector<string> output{};
	if (fileExists(jsonfile)) {
		ifstream in(jsonfile);
		if (!in.is_open()) { throw runtime_error("error openning" + jsonfile); }
		if (reader.parse(in, patch_val)) {
			output.push_back(patch_val["TensorModel"].asString());
			output.push_back(patch_val["epsilon"].asString());
			output.push_back(patch_val["DF"].asString());
			//output.push_back(patch_val["type"].asString());
			output.push_back(patch_val["patch_continuity"].asString());
			output.push_back(patch_val["train"].asString());
			output.push_back(patch_val["alpha"].asString());
		}
		in.close();
		return output;
	}
	else {
		throw runtime_error("File not found:" + jsonfile);
	}
}

vector<TensorModel> readTensorJson(const string& jsonfile) {
	Json::Reader reader;
	Json::Value model_json;
	vector<TensorModel> Tensor_Model;
	if (fileExists(jsonfile)) {
		ifstream in(jsonfile);
		if (!in.is_open()) { throw runtime_error("error openning" + jsonfile); }
		if (reader.parse(in, model_json)) {

			int num_of_layers = model_json["layers"].asInt();
			Tensor_Model.resize(num_of_layers);

			for (int r = 0; r < num_of_layers; r++) {
				int dimension = model_json["layer_" + to_string(r + 1)]["dimension"].asInt();
				Tensor_Model[r].dimension = dimension;
				Tensor_Model[r].weight = model_json["layer_" + to_string(r + 1)]["weight"].asInt();
				string rangepath = model_json["layer_" + to_string(r + 1)]["range"].asString();
				Tensor_Model[r].model_range = loadDataCsv(rangepath);

				Tensor_Model[r].dimension_info.resize(dimension);
				Tensor_Model[r].coeff.resize(dimension);
				for (int d = 0; d < dimension; d++) {
					Json::Value dimjson = model_json["layer_" + to_string(r + 1)]["dimension_" + to_string(d + 1)];
					string Ybpath = dimjson["Yb"].asString();
					string Zpath = dimjson["Z"].asString();
					string DPpath = dimjson["discontinuous_points"].asString();
					string DFpath = dimjson["degree_of_freedom"].asString();
					string CMpath = dimjson["constrain_matirx"].asString();

					Tensor_Model[r].dimension_info[d].Yb = loadDataCsv(Ybpath);
					Tensor_Model[r].dimension_info[d].Z = loadDataCsv(Zpath);
					Tensor_Model[r].dimension_info[d].discontinuous_points = loadDataCsv(DPpath);
					Tensor_Model[r].dimension_info[d].degree_of_freedom = loadDataCsv(DFpath);
					Tensor_Model[r].dimension_info[d].constrain_matirx = loadDataCsv(CMpath);

					string coeffpath = dimjson["coeff"].asString();
					Tensor_Model[r].coeff[d] = loadDataCsv(coeffpath);

					int units = Tensor_Model[r].dimension_info[d].degree_of_freedom.size();
					Tensor_Model[r].dimension_info[d].base_type.resize(units);
					for (int u = 0; u < units; u++) {
						Tensor_Model[r].dimension_info[d].base_type[u] = "Basic";
					}
				}
			}
		}
		in.close();
		return Tensor_Model;
	}
	else {
		throw runtime_error("File not found:" + jsonfile);
	}
}

//int main() {
//	Json::Reader reader;
//	Json::Value tc;
//	double error_bound;
//	int max_iter, max_rank;
//	ifstream in("model_config/train_config.json");
//	if (reader.parse(in, tc)) {
//			error_bound = tc["error_bound"].asDouble();
//			max_iter = tc["max_iter"].asInt();
//			max_rank = tc["max_rank"].asInt();
//	}
//	in.close();
//	cout << error_bound << endl;
//	return 0;
//}