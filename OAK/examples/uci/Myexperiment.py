import sys
import time
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
from oak.model_utils import oak_model, save_model
from oak.utils import get_model_sufficient_statistics, get_prediction_component
from scipy import io
from sklearn.model_selection import KFold
from pathlib import Path

matplotlib.rcParams.update({"font.size": 20})

data_path_prefix = os.path.abspath(
    # os.path.join(os.path.dirname(__file__), "../../data")
    os.path.join(os.path.dirname(__file__), "../../mydata")
) + '\\'
print(data_path_prefix)

np.set_printoptions(formatter={"float": lambda x: "{0:0.5f}".format(x)})


def main():
    # 数据初始
    filename_x = "x_yield_1.txt"
    filename_y = "y_yield_1.txt"
    k = 5

    output_prefix = os.path.abspath(
        os.path.join(os.path.dirname(__file__), f"./outputs/myexperiment/")
    )
    if not os.path.exists(output_prefix):
        os.mkdir(output_prefix)

    np.random.seed(4)
    tf.random.set_seed(4)

    # 筛选数据，此处为选择1000*22的输入，输出定为1维
    X = np.loadtxt(os.path.join(data_path_prefix, filename_x), delimiter=' ')[:, :22]
    y = np.loadtxt(os.path.join(data_path_prefix, filename_y), delimiter=' ')[:, :1]

    # 随机打乱
    idx = np.random.permutation(range(X.shape[0]))
    X = X[idx, :]
    y = y[idx]

    kf = KFold(n_splits=k)
    fold = 0
    t_start = time.time()
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # 模型训练
        oak = oak_model(
            max_interaction_depth=X.shape[1],
            num_inducing=500,
            share_var_across_orders= False,
            # use_sparsity_prior=False,
        )
        oak.fit(X_train, y_train)

        # 模型评估
        x_max, x_min = X_train.max(0), X_train.min(0)
        y_pred = oak.predict(np.clip(X_test, x_min, x_max))
        rss = ((y_pred - y_test[:, 0]) ** 2).mean()  # mse
        tss = (
                (y_test[:, 0] - y_test[:, 0].mean() * np.ones(y_test[:, 0].shape)) ** 2
        ).mean()
        r2 = 1 - rss / tss
        rmse = np.sqrt(rss)

        # nll = (
        #     -oak.m.predict_log_density(
        #         (
        #             oak._transform_x(np.clip(X_test, x_min, x_max)),
        #             oak.scaler_y.transform(y_test),
        #         )
        #     )
        #     .numpy()
        #     .mean()
        # )

        #sobol指数

        oak.get_sobol(max_dims=2)
        tuple_of_indices, normalised_sobols = (
            oak.tuple_of_indices,  # 所有项的交互搭配
            oak.normalised_sobols,  # 各个项的sobol指数衡量影响
        )

        oak.alpha = get_model_sufficient_statistics(oak.m, get_L=False)  # 张量alpha用于高效预测？
        order = np.argsort(normalised_sobols)[::-1]


        # 参数打印
        sorted_sobols = normalised_sobols[order]
        indexed_sobols = np.column_stack((sorted_sobols, np.array(tuple_of_indices, dtype=object)[order]))

        # 保存为文本文件
        sobolsavedpath = os.path.join(output_prefix + '\\limitdim_sobols\\sorted_normalised_sobols_%d.txt' % fold)
        with open(sobolsavedpath, 'w') as f:
            f.write("Index,Value\n")
            for index, value in indexed_sobols:
                f.write(f"{index},{value}\n")
        print(f"sobol per order is saved at{sobolsavedpath}")

        print(f"fold {fold}, training dataset has size {X_train.shape}")
        print(f"oak test rmse = {rmse}, r2 = {r2}\n")
        # print(f"RBF test nll = {np.round(nll, 4)}\n")

        # save learned model
        save_model(
            oak.m,
            filename=Path(output_prefix + "/model_oak_%d" % fold),
        )

        fold += 1

        # oak.plot(
        #     top_n=10,
        #     semilogy=False,
        #     save_fig=output_prefix + f"/decomposition_{fold}",
        #     max_dims=2,
        # )
        # # 此处plot内的getsobol被注释掉了，因为上边已经get过了
    print(f"Training totally took {time.time() - t_start:.1f} seconds.")


if __name__ == "__main__":
    main()
