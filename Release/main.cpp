#include<iostream>
#include<thread>
#include<string>
#include<functional>
#include<condition_variable>
#include<cmath>
#include<queue>
#include<sstream>
#include <algorithm>
#include<fstream>
#include"ECEngine.h"
#include"GeneralBase.h"
#include"readjson.h"
#include "excute.h"
//#include<Eigen/Dense>
//#include<TypeDefine.h>
using namespace std;
//一个模型池，实现方法有待优化

class Undoable {
public:
	Undoable() :_index(-1) {}
	int Undo() {
		if (_index > -1) {
			_steps[_index--].first();
			return 1;
		}
		else {
			cout << "Not Undoable!! It's the origin model!" << endl;
			return 0;
		}
	}
	int Redo() {
		if (_index < (int)_steps.size() - 1){
			_steps[++_index].second();
			return 1;
		}
		else {
			cout << "Not Redoable!! It's the newest model!" << endl;
			return 0;
		}
	}
	void Clear() {
		_steps.clear();
		_index = -1;
	}
	void Add(function<void()> undo, function<void()> redo) {
		if (++_index < (int)_steps.size())
			_steps.resize(_index);
		_steps.push_back({ undo ,redo });
	}
private:
	vector<pair<function<void()>, function<void()>>> _steps;
	int _index;
};

struct Model_Thread{
	vector<ECEngine> model;
	Undoable log;
	string name;
	int trained;

	//Model_Thread() :log(), trained(0) {}
};

vector<Model_Thread> modelthreads{};

vector<string> split(string input) {
	string token;
	vector <string> tokens;
	stringstream ss(input);
	while (ss >> token) {
		tokens.push_back(token);
	}
	return tokens;
}
class ThreadPool {
public:
	ThreadPool(int numThreads) : stop(false) {
		for (int i = 0; i < numThreads; i++) {
			threads.emplace_back([this] {
				while (true) {
					std::function<void()> task;

					{
						std::unique_lock<std::mutex> lock(mtx);
						condition.wait(lock, [this] {
							return !tasks.empty() || stop;
							});

						if (stop && tasks.empty()) {
							return;
						}

						task = std::move(tasks.front());
						tasks.pop();
					}

					try {
						task();
					}
					catch (const std::exception& e) {
						std::cerr << "Exception in thread: " << e.what() << std::endl;
						continue;
					}
				}
				});
		}
		//for (int i = 0; i < numThreads; i++) {
		//	threads.emplace_back([this] {
		//		while (1) {
		//			unique_lock<mutex> lock(mtx);
		//			condition.wait(lock, [this] {
		//				return !tasks.empty() || stop; //任务列表非空或线程终止，停止等待，防止空任务执行
		//				});


		//			if (stop && tasks.empty()) {
		//				return;
		//			}
		//			//从task中移动队头元素到task里，保持类型为function<void()>
		//			function<void()> task(move(tasks.front()));
		//			tasks.pop();
		//			lock.unlock();
		//			//执行该任务
		//			try {
		//				task();
		//			}
		//			catch (const runtime_error& e) {
		//				throw e;
		//			}
		//			catch (const invalid_argument& e) {
		//				throw e;
		//			}
		//		}
		//		}
		//	);
		//}
	}
	~ThreadPool() {
		//多余的{}限制unique_lock作用域
		{
			unique_lock < mutex > lock(mtx);
			stop = true;
		}
		//通知所有线程来完成任务
		condition.notify_all();
		//所有线程完成
		for (auto& t : threads) {
			if (t.joinable()) {
				t.join();
			}
		}
	}
	template<typename F, typename... Args>
	void enqueue(F&& f, Args&&... args) {
		//bind 用于函数绑定,函数包装
		//forward用于完美转发
		// function<void()> task(bind(forward<F>(f), forward<Args>(args)...));
		function<void()> task = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
		{
			unique_lock<mutex> lock(mtx);
			tasks.emplace(move(task));
		}
		//有任务输入，要通知线程
		condition.notify_one();
	}
private:
	vector<thread> threads;
	queue<function<void()>> tasks;
	mutex mtx;
	condition_variable condition;
	bool stop;
};

//在模型集里寻找模型
int Searchmodel(string name) {
	bool found = false;
	for (int m = 0; m < modelthreads.size(); m++) {
		if (modelthreads[m].name == name) {
			return m;
		}
	}
	return -1;
}

void Set(vector<string> cmds) {
	if (cmds.size() != 3) {
		cout << "Invalid input" << endl;
	}
	else {
		try {
			vector<string> setfile = ReadSetJson(cmds[2]);
			vector<ModelConfig> modelconfig = readModelJson(setfile[0]);
			TrainConfig trainconfig = readTrainJson(setfile[1]);
			MatrixXd X, Y;
			if (fileExists(setfile[2])) {
				X = loadDataCsv(setfile[2]);
				//cout << X<<endl;
			}
			else {
				throw runtime_error("File not found:" + setfile[2]);
			}
			if (fileExists(setfile[3])) {
				Y = loadDataCsv(setfile[3]);
			}
			else {
				throw runtime_error("File not found:" + setfile[3]);
			}

			ECEngine model_test = ECEngine::ECEngine(modelconfig, trainconfig, X, Y);

			int found = Searchmodel(cmds[1]);
			//新模型存入模型队列
			if (found == -1) {
				Model_Thread md;
				md.model.push_back(model_test);
				md.name = cmds[1]; 
				md.trained = 0;
				modelthreads.push_back(md);
				cout << "Create new config" << endl;
			}
			else { //已有模型选择重写
				while (true) {
					cout << "model exists. Do you want to rewrite it? [Y/n]   " ;
					string in;	getline(cin,in);
					if (in == "Y" || in.empty()||in=="y") {
						modelthreads[found].model.back()=model_test;
						modelthreads[found].trained = 0;
						cout << "Rewrite successfully!" << endl;
						break;
					}
					else if (in == "N" || in =="n") {
						break;
					}
					else {
						cout << "Invalid command" << endl;
					}
				}
			}
		}
		catch (const runtime_error& e) {
			throw e;
		}	
	}
}

void Train(vector<string> cmds) {
	if (cmds.size() != 4 ) {
		cout << "Invalid Input" << endl;
	}
	else {
		int type;
		double alpha;
		try {
			type = stoi(cmds[2]);
			alpha = stod(cmds[3]);
		}
		catch (const invalid_argument& e) {
			throw invalid_argument("Invalid Para :" + string(e.what()));
		}
		//输入范围检查
		if ((type != 1 && type != 2)|| alpha >1 || alpha< 0 ) {
			throw invalid_argument("Invalid Para Range");
		}

		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Model not exists,please Set first" << endl;
		}
		else {
			ECEngine model = modelthreads[found].model.back();
			clock_t start, end;
			cout <<cmds[1] << "训练开始" << endl;
			start = clock();
			model.modelTrain(type,alpha);
			modelthreads[found].model.back() = model;
			modelthreads[found].trained = 1;
			//std::this_thread::sleep_for(std::chrono::seconds(6));
			end = clock();
			cout << endl <<cmds[1] << "  训练用时：" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
			cout << "/easyclip>";
		}
	}
}

void show() {
	cout << "Name\t\t" << "Trained\t\t" << "R" << endl;
	for (int m = 0; m < modelthreads.size(); m++) {
		cout << modelthreads[m].name << "\t\t" << modelthreads[m].trained << "\t\t"<<modelthreads[m].model[0].r.norm() << endl;
		for (int n = 1; n < modelthreads[m].model.size(); n++) {
			cout<<"\t\t\t\t"<< modelthreads[m].model[n].r.norm() << endl;
		}
	}
}
void displayinfo(vector<string>cmds) {
	if (cmds.size() == 2) {
		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Model not exists,please Set first" << endl;
		}
		else {
			modelthreads[found].model.back().displayModelInfo("modelinfo/");
			cout << "Saved in modelinfo"<< endl;
		}
	}
	else {
		throw runtime_error("Invalid Input");
	}
}
VectorXd Eval(vector<string> cmds) {
	if (cmds.size() == 3 ) { //默认已标准化输入
		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Model not exists,please Set first" << endl;
		}
		else {
			if (!fileExists(cmds[2])) {
				throw runtime_error("File not found:" + cmds[2]);
			}
			MatrixXd input = loadDataCsv(cmds[2]);
			ECEngine model = modelthreads[found].model.back();
			if (modelthreads[found].trained) {//判断是否进行过训练
				return model.modelEval(input);
			}
			else {
				throw runtime_error("Model not trained, please train first!");
			}
		}
	}
	else if (cmds.size() == 4) { //未标准化
		if (cmds[3] != "n" && cmds[3] != "N" &&cmds[3]!="u"&&cmds[3]!="U") {
			throw runtime_error("Illegal command:" + cmds[3]);
		}
		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Model not exists,please Set first" << endl;
		}
		else {
			if (!fileExists(cmds[2])) {
				throw runtime_error("File not found:" + cmds[2]);
			}
			MatrixXd input = loadDataCsv(cmds[2]);
			ECEngine model = modelthreads[found].model.back();
			if (cmds[3] != "u" && cmds[3] != "U") { //未标准化先标准化
				model.inputNormalization(input);
			}
			if (modelthreads[found].trained) {//判断是否进行过训练
				return model.modelEval(input);
			}
			else {
				throw runtime_error("Model not trained, please train first!");
			}
		}
	}
	else if(cmds.size() == 2) {
		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Model not exists,please Set first" << endl;
		}
		else {
			ECEngine model = modelthreads[found].model.back();
			if (modelthreads[found].trained) {//判断是否进行过训练
				return model.modelEval();
			}
			else {
				throw runtime_error("Model not trained, please train first!");
			}
		}
	}
	else {
		throw runtime_error("Invalid Input");
	}
}

void reset(vector<string> cmds) {
	if (cmds.size() != 4) {
		cout << "Invalid input" << endl;
	}
	else {
		int found = Searchmodel(cmds[1]);
		if (found == -1) {
			cout << "Please set first" << endl;
		}
		if (!modelthreads[found].trained) {
			throw runtime_error("Please train first");
		}

		int type;
		double alpha;
		try {
			vector<string> setfile;
			tie(setfile, type, alpha) = ReadResetJson(cmds[2]);
			vector<ModelConfig> modelconfig = readModelJson(setfile[0]);
			TrainConfig trainconfig = readTrainJson(setfile[1]);
			
			

			//输入范围检查
			if ((type != 1 && type != 2) || alpha > 1 || alpha < 0) {
				throw invalid_argument("Invalid Para Range");
			}

			//模型查找
			
			else { 
				ECEngine model = modelthreads[found].model.back();
				model.resetConfig(modelconfig, trainconfig, stoi(cmds[3]));
				model.modelTrain(type, alpha);
				modelthreads[found].model.push_back(model);
				//auto currentmodel = modelthreads[found].model.back();
				modelthreads[found].log.Add(
					[=]() {
						modelthreads[found].model.pop_back();
					},
					[=]() {
						modelthreads[found].model.push_back(model);
					}
					);
			}
		}
		catch (const runtime_error& e) {
			throw e;
		}
	}
}
void redo(vector<string> cmds) {
	if (cmds.size() != 2) {
		cout << "Invalid input" << endl;
	}
	else {
		try {
			int found = Searchmodel(cmds[1]);
			if (found == -1) {
				cout << "Please set first" << endl;
			}
			else {
				if (modelthreads[found].log.Redo()) {
					cout << "Redo successfully, modelR: "<<modelthreads[found].model.back().r.norm() << endl;
					cout << "Current models in model"<<modelthreads[found].name<<" : "<<modelthreads[found].model.size() << endl;
				}
			}
		}
		catch (const runtime_error& e) {
			throw e;
		}
	}
}
void undo(vector<string> cmds) {
	if (cmds.size() != 2) {
		cout << "Invalid input" << endl;
	}
	else {
		try {
			int found = Searchmodel(cmds[1]);
			if (found == -1) {
				cout << "Please set first" << endl;
			}
			else {
				if (modelthreads[found].log.Undo()) {
					cout << "Undo successfully, modelR: " << modelthreads[found].model.back().r.norm() << endl;
					cout << "Current models in model" << modelthreads[found].name << " : " << modelthreads[found].model.size() << endl;
				}
			}
		}
		catch (const runtime_error& e) {
			throw e;
		}
	}
}

void appendpatch(vector<string> cmds) {
	if (cmds.size() != 5) {
		cout << "Invalid input" << endl;
	}
	else {
		try {
			vector<string> patchfile = ReadPatchJson(cmds[2]);
			vector<TensorModel> Tensor = readTensorJson(patchfile[0]);
			MatrixXd epsilon,DF,patch_continuity;

			vector<vector<const char*>> patch_basetype;

			if (!fileExists(patchfile[1])) {throw runtime_error("File not found:" + patchfile[1]);}
			if (!fileExists(patchfile[1])) { throw runtime_error("File not found:" + patchfile[2]); }
			if (!fileExists(patchfile[1])) { throw runtime_error("File not found:" + patchfile[3]); }
			
			epsilon = loadDataCsv(patchfile[1]);
			DF = loadDataCsv(patchfile[2]);
			patch_continuity = loadDataCsv(patchfile[3]);
			int type;
			double alpha;
			try {
				type = stoi(patchfile[4]);
				alpha = stod(patchfile[5]);
			}
			catch (const invalid_argument& e) {
				throw invalid_argument("Invalid Para :" + string(e.what()));
			}
			if ((type != 1 && type != 2) || alpha > 1 || alpha < 0) {
				throw invalid_argument("Invalid Para Range");
			}
			int found = Searchmodel(cmds[1]);
			if (found == -1) {
				cout << "Model not exists,please Set first" << endl;
			}
			else {
				ECEngine model = modelthreads[found].model.back();
				clock_t start, end;
				cout << endl << cmds[1] << "补丁训练开始" << endl;
				start = clock();
				model.appendPatch(Tensor,epsilon, DF,patch_basetype,patch_continuity,type, alpha);
				modelthreads[found].model.back() = model;
				modelthreads[found].trained = 1;
				end = clock();
				cout << endl <<cmds[1] << "  补丁训练用时：" << double(end - start) / CLOCKS_PER_SEC << "s" << endl;
			}
		}
		catch (const runtime_error& e) {
			throw runtime_error("patch error"+ string(e.what()));
		}
	}
};

int main() {
	ThreadPool pool(4);
	while (true) {
		cout << "/easyclip>";
		string Input;
		getline(cin, Input);
		vector<string> cmds = split(Input);
		if (cmds.empty()) {
			continue;
		}
		string command = cmds[0];
		transform(command.begin(), command.end(), command.begin(), ::toupper); //命令大小写不敏感
		if (command =="SET") {
			try {
				Set(cmds);
			}
			catch (const runtime_error& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "TRAIN") {
			try {
				pool.enqueue([cmds] {Train(cmds); });
			}
			catch (const invalid_argument& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "RESET") {
			try {
				pool.enqueue([cmds] {reset(cmds); });
			}
			catch (const runtime_error& e) {
				cerr << e.what() << endl;
				continue;
			}
			catch (const invalid_argument& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "REDO") {
			try {
				//pool.enqueue([cmds] {redo(cmds); });
				redo(cmds);
			}
			catch (const invalid_argument& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "UNDO") {
			try {
				//pool.enqueue([cmds] {undo(cmds); });
				undo(cmds);
			}
			catch (const invalid_argument& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "PATCH") {
			try {
				//pool.enqueue([cmds] {appendpatch(cmds); });
				//cout << endl << cmds[1] << "补丁训练开始" << endl;
				//cout << endl << cmds[1] << "  补丁训练用时：" << 0.3 << "s" << endl;

			}
			catch (const runtime_error &e) {
				cerr << e.what() << endl;
				continue;
			}
			catch (const invalid_argument& e) {
				cerr << e.what() << endl;
				continue;
			}

		}
		else if (command == "SHOW") {
			show();
		}
		else if (command == "INFO") {
			try {
				displayinfo(cmds);
			}
			catch (const runtime_error& e) {
				cerr << e.what() << endl;
				continue;
			}
		}
		else if (command == "EVAL") {
			try {
				VectorXd output = Eval(cmds);
				saveDataCsv("evaldata/"+cmds[1] + ".csv", output);// 需要测试一下
				//cout << output << endl;
				cout << "Saved in evaldata/" + cmds[1] + ".csv" << endl;
			}
			catch (const runtime_error& e) {
				cerr << e.what() << endl;
				continue;
			}//涉及到文件创建，可能会需要管理员权限（sudo）提示用户权限不足
			//catch()
		}
		else if (command == "CLEAR") {
			#ifdef _WIN32
			system("cls");
			#else	
			system("clear");
			#endif
		}
		else if (command == "HELP") {
			ifstream inputFile("help.txt");
			if (!inputFile) {
				cerr << "文档打开错误" << endl;
				return 1;
			}
			string line;
			while(getline(inputFile, line)) {
				cout << line << endl;
			}
			inputFile.close();
		}
		else if (command == "EXIT") {
			break;
		}
		else {
			//throw new exception("Command error, enter 'help' to get command list");
			cout << "Command error, enter 'help' to get command list" << endl;
		}
	}
	return 0;
}



