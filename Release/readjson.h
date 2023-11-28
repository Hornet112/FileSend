#pragma once
#include<iostream>
#include<fstream>
#include<cmath>
#include<json.h>
#include<vector>
#include<crtdbg.h>
#include"TypeDefine.h"
#include"ECEngine.h"

using namespace std;
void coutmatrix(vector<VectorXd> in); //方便输出向量组
bool fileExists(const string& ffilename);
vector<ModelConfig> readModelJson(const string& filepath);
TrainConfig readTrainJson(const string& filepath);
vector<TensorModel> readTensorJson(const string& jsonfile);
vector<string> ReadSetJson(const string& jsonfile);
tuple<vector<string>, int, double> ReadResetJson(const string& jsonfile);
vector<string> ReadPatchJson(const string& jsonfile);