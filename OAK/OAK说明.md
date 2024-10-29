## OAK库说明

针对GP拟合过程中正交核的构造，采用正交加性核使得模型拟合后变量独立解析性提高，利用sobol指数分析多项式各项在拟合中的贡献从而找到多项式中的多个决定项。

## 库启动配置

见setup文件

```
python==3.7.0
gpflow==2.9.2
pytest==7.4.4 
lint
black
mypy
flake8
jupytext
seaborn
jupyter
tqdm
numpy
matplotlib
IPython
scikit-learn
tikzplotlib
seaborn
s3fs
scikit-learn-extra==0.3.0
tensorflow == 2.10.0
tensorflow_probability==0.19.0
```

## 文件说明

### 目录

```
├─data	存储库实验数据
├─mydata	存储自己的实验数据
├─examples
│  ├─uci
│  │  ├─outputs		实验输出，包括sobol指数累加曲线、sobol指数排序列表、高sobol变量曲线
│  │  ├─Myexperiment.py					拟合自己的实验数据
│  │  ├─uci_regression_train.py			回归问题求解库示例
│  │  ├─uci_classification_train.py		分类问题求解库示例
│  │  └─uci_plotting.py					训练结束后绘制高sobol变量的拟合曲线
├─oak	库主体
│  ├─input_measures.py				拟合方法定义
│  ├─model_utils.py					模型构造
│  ├─normalising_flow.py			标准化数据流
│  ├─oak_kernel.py					核构造
│  ├─ortho_binary_kernel.py			二元正交核计算
│  ├─ortho_categorical_kernel.py	分类策略正交核计算
│  ├─ortho_rbf_kernel.py			rbf高斯正交核计算，通常只用该计算方法
│  ├─plotting_utils.py
│  ├─utils.py						主要为sobol指数计算方法及其他模型指数计算函数
│  └─test.py						测试程序
└─setup.py	依赖安装
```

为方便运行路径定义，实验文件统一放在../examples/uci目录下

### 实验运行基本逻辑：

1. 处理输入数据，进行随机排列处理,k折交叉划分训练集与验证集，以下步骤为对每一组数据集的操作
2. 初始模型 oak_model, 示例中输入数据维度参数，inducing_num数据量高于1000时采用稀疏GP使用
3. 训练模型 oak.fit
   1. 数据集进行KL优化处理，bijective映射到高维空间（一对一满射）
   2. 标准化处理后的数据集B(X)、Y
   3. create_oak_model 建立模型，核心即建立正交核
      1. 初始化核 base_kernels=RBF
      2. 初始化OAKKernel  根据拟合方法初始化正交核，实验以RBF正交核为例
         1. OrthogonalRBFKernel 在初始核基础上定义核内运算，并提供高维核计算方法
         2. 定义各维度一阶核的权重指数（variances重要参数），初始数据符合高斯分布 *N(0,1)*
      3. 建立模型model=GPR，输入初始化后的k，data； 提供先验分布gamma(1,0.2)
      4. 优化器opt最优化model超参数

以上为一次模型训练中的基础调用逻辑，具体执行见对应代码注释

### 重要参数

#### 模型参数：

基核函数中有作用尺度l、方差σ，初始分布N(μ，δ^2)，各个加性核对应权重指数var

#### 函数参数：

均指myexperiment中可设置的参数

```
num_inducing 大数据稀疏GP参数
share_var_across_orders	默认为true 表示是否在各个加性核之间共享相同的权重参数，
constrain_orthogonal 是否使用正交核，不用的话就只用加性核
```

### 快速启动

库示例执行uci_regression_train.py

自己的数据执行myexperiment.py
