import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from torch import Tensor
from torch.quasirandom import SobolEngine
from AlpsInterface import AlpsInterface
from utils import sample
from turbo import turbo
from transform_function import x_cube_y_standard as trans

def FBSS(x1, x2):
    f1 = 3 + 0.1 * (x1 - x2) ** 2 - (x1 + x2) / np.sqrt(2)
    f2 = 3 + 0.1 * (x1 - x2) ** 2 + (x1 + x2) / np.sqrt(2)
    f3 = (x1 - x2) + 6 / np.sqrt(2)
    f4 = -(x1 - x2) + 6 / np.sqrt(2)
    return min(np.array([f1, f2, f3, f4]))


def Bisection_search(f, v, lrange, rrange, tol):
    f_l = f(lrange * v)
    while rrange - lrange > 1e-7:
        mid = (rrange + lrange) / 2
        f_mid = f(mid * v)
        if abs(f_mid) < tol:
            return mid * v
        elif f_mid * f_l > 0:
            lrange = mid
            f_l = f_mid
        else:
            rrange = mid
    return False


def Search1d(f, target, design_param, data, tol):
    '''

    :param f:  function
    :param target:  opt target spec
    :param design_param: x
    :param data: v
    :param tol: tolerance
    :return:  liner search corner
    '''

    # 信任域方法优先
    # def opt(x):
    #     x = x.reshape(1, 1, 44)
    #     return abs(f(design_param, x)[0]-target)**2
    # corner = minimize(opt, data, method="trust-constr", tol=tol, options=dict(maxiter=10))

    x = np.linspace(0, 2, 20)
    mc_value = []
    for xv in x:
        mc_value.append(np.array(f(design_param, xv * data)[0].reshape(-1)-target))
    print(f(design_param, data)[0].reshape(-1)-target)
    plt.figure()
    plt.plot(x,mc_value)
    plt.axhline(0, color='r', linestyle='--')
    plt.show()

    def Bsearch(x):
        return f(design_param, x)[0].reshape(-1) - target

    if np.array(f(design_param, data)[0].reshape(-1)) <= target:
        cor = Bisection_search(Bsearch, data, 0, 1, tol=tol)
        if cor is not False:
            print(f"1dSearch最终统计角{cor}")
            print(f"1dSearch最终输出值{f(design_param, cor)[0].reshape(-1)}")
            return cor
        else:
            print("Select 2nd min v for corner")
            return False
    else:
        right = 1 * 1.5
        try:
            loop = 0
            left = 1
            while np.array(f(design_param, data * right)[0].reshape(-1)) > target:  # 反方向5次以内找到
                left = right
                right = right * 1.5
                loop += 1
                if loop == 5:
                    return False
            cor = Bisection_search(f=Bsearch, v=data, lrange=left, rrange=right, tol=tol)
            if cor is not False:
                print(f"1dSearch最终统计角{cor}")
                print(f"1dSearch最终输出值{f(design_param, cor)[0].reshape(-1)}")
                return cor
            else:
                print("Select 2nd min v for corner")
                return False
        except ValueError as e:
            print(f"Cannot search valid variation in 5 loops:{e}")
            return False



def oss(num, dims, bounds):
    sampler = Sobol(d=dims, scramble=True)
    samples = sampler.random(num) * (bounds[0][0] - bounds[0][1]) + bounds[0][0]
    for dim in range(1, dims):
        sampler = Sobol(d=dims, scramble=True)
        samples = np.append(samples, sampler.random(num) * (bounds[dim][0] - bounds[dim][1]) + bounds[dim][0],
                            axis=0)  # 1
    return samples


def uniform(center, range_percent, num):
    np.random.seed(5)
    length = center * range_percent * 2
    data = np.random.rand(1, num) * length + center - length / 2
    # bound = np.array((0, 1)* length + center - length / 2)
    # print(bound)
    np.random.seed(None)
    return data


def get_random_x(percent):
    data = uniform(5e-7, percent, 6)
    data = np.append(data, uniform(1e-6, percent, 2), axis=1)
    data = np.append(data, uniform(5e-7, percent, 1), axis=1)
    data = np.append(data, uniform(1e-6, percent, 1), axis=1)
    data = np.append(data, uniform(5e-7, percent, 1), axis=1)

    data = np.append(data, uniform(10e-6, percent, 1), axis=1)
    data = np.append(data, uniform(4e-6, percent, 4), axis=1)
    data = np.append(data, uniform(5e-7, percent, 1), axis=1)
    data = np.append(data, uniform(2e-6, percent, 2), axis=1)
    data = np.append(data, uniform(1e-6, percent, 1), axis=1)
    data = np.append(data, uniform(3e-6, percent, 1), axis=1)
    data = np.append(data, uniform(1e-6, percent, 1), axis=1)
    return torch.from_numpy(data)

def get_optDesign():
    data = ([500e-9] * 6 + [1e-6] * 2 + [500e-9, 1e-6, 500e-9, 10e-6] + [4e-6] * 4
            + [500e-9] + [2e-6] * 2 + [1e-6, 3e-6,1e-6])
    return np.array(data)

def get_normal_x(center:Tensor):
    center = center.reshape(-1,)
    std = (center / 12).expand(2 * center.shape[0], center.shape[0])
    design_param = torch.normal(mean=center.expand_as(std), std=std)
    return design_param



# 要有norm点获取的机制，比如第一个是norm点
if __name__ == '__main__':

    target = 105
    target_yield = 0.9986
    design_param_num = 2
    variation_num = 4


    n_dim = 22

    rootPath = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/opamp"  # set your work dir, the results will save here
    bash_path = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/opamp/setup.bash"  # where the setup.bash to be load
    netlist = "opamp_spice_new"
    instance = ["pch_mis", "nch_mis",
                "g45n1svt", "g45p1svt"]
    key = ["dc_gain", "pm", "gbw"]

    alps_simulator = AlpsInterface(netlist, rootPath, bash_path, instance)
    alps_simulator.load_design()


    def sim_alps(x, v):
        simres = alps_simulator.sim_with_given_param(load_design_param=False,
                                                     design_para=x,
                                                     mismatch=v,
                                                     res_keys=key
                                                    )
        return simres[0]

    design_param = torch.DoubleTensor(
        [5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 1e-6, 1e-6, 5e-7, 1e-6, 5e-7,
         10e-6, 4e-6, 4e-6, 4e-6, 4e-6, 5e-7, 2e-6, 2e-6, 1e-6, 3e-6, 1e-6]
    ).reshape(1, -1)
    # 至多五轮优化
    for i in range(10):
        # 这里替换为蒙卡
        # data = oss(128, variation_num,[[-15, 15]])  # num * dim


        num_of_x_mc = 1  # number of x mc actually will always be 1
        num_of_v_mc_sim = 100  # number of v mc in one x
        # torch.manual_seed(114514)
        mismatch_mc = torch.randn(num_of_x_mc, num_of_v_mc_sim, 4 * 11)
        start = time.time()
        Samples = sample(design_param=design_param,
                         mismatch_variation=mismatch_mc,
                         name=f"get {i}th 3 sigma corner",
                         simulator=sim_alps)
        # Samples.verification(f=FBSS, target=target, target_yield=target_yield)
        Specvalue = Samples.Percentile(percentile=target_yield)

        # 选择最近的蒙卡点，可以接入别的方式
        idx = (abs(Samples.Mc_value - Specvalue)).argmin()
        v = mismatch_mc[0, idx, :].reshape(1, 1, 44)  # one time MC
        print(f"初次少量模拟得到的输出值3sigma分点: {Specvalue}")
        print(f"初始统计角: {v}")
        print(f"初始最接近的蒙卡输出值: {Samples.Mc_value[idx]}\n")
        #
        # norminal = torch.zeros(num_of_x_mc, 1, 4 * 11)
        # norminal_mc = Samples.alps_simulator(design_param, norminal)
        # print(norminal_mc)
        # corner = Search1d(f=Samples.alps_simulator, target=Specvalue, design_param=design_param, data=v, tol=1e-3)
        if corner is False:
            mask = np.ones(Samples.Mc_value.shape, dtype=bool)
            mask[idx] = False
            idx_second = (abs(Samples.Mc_value[mask]-Specvalue)).argmin()
            v = mismatch_mc[0, idx_second, :].reshape(1, 1, 44)
            corner = Search1d(f=Samples.alps_simulator, target=Specvalue, design_param=design_param, data=v,
                              tol=1e-6)
            if corner is False: continue
        print(f"Total time for corner:{time.time()-start}")
        # 调整设计参数使输出达到要求
        # 优化算法
        def opt(x):
            x = x.reshape(1, 22)
            result = Samples.alps_simulator(x, corner)[0].reshape(1,-1)
            print(result)
            if result < target:
                return target - result
            else:
                return 0
        def opt_for_turbo(x):
            x = x.reshape(1, 22)
            result = Samples.alps_simulator(x, corner)[0].reshape(-1, 1)
            return result
        #
        # bounds = [(2.5e-7, 15e-6)]*22
        # optDesign = (minimize(opt,
        #                       design_param.numpy(),
        #                       method='Nelder-Mead',
        #                       tol=1e-4,
        #                       # bounds=bounds,
        #                       # options={'maxiter': 10},
        #                       # callback=callback
        #                       )).x
        # optDesign = torch.from_numpy(optDesign.reshape(1,-1))
        # print(f"优化后的设计参数{optDesign}")
        # print(f"优化后的仿真值{Samples.alps_simulator(optDesign, corner)[0]}")

        # TuRBO opt
        # bounds = [(2.5e-7, 15e-6)] * 22
        turbo_x = get_normal_x(design_param)
        optDesign, best_value = turbo(X_turbo=turbo_x, eval_objective=opt_for_turbo)

        # 验证
        ori_num = 1000
        Mc_v = torch.randn(num_of_x_mc, ori_num, 4 * 11)
        verify = sample(design_param=optDesign, mismatch_variation=Mc_v, name=f"verify {i}th opt", verify=True, simulator=sim_alps)
        verify.Percentile(target_yield)   # 查看更新后的分布情况
        verify.verification(target=target, target_yield=target_yield)
        if verify.success:
            print(f"优化完成，design param: {optDesign}")
            break
        design_param = optDesign
        print(f"Not satisfied, start next loop")
