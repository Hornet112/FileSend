import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize
import scipy.stats as stats
from scipy.stats.qmc import Sobol
from torch import Tensor
from torch.quasirandom import SobolEngine
from AlpsInterface import AlpsInterface
from utils import sample
from turbo import turbo
from FFXCorner import corner, CornerState
import scipy.integrate as integrate
import scipy.interpolate as interpolate


def Kde(fun, percentile, draw: bool = True, name: str = "3sigma"):
    tensor_kwargs = {"device": 'cpu', "dtype": torch.double}
    rand_generator = SobolEngine(dimension=44, scramble=True, seed=114514)
    doe_x_cube = rand_generator.draw(400).to(**tensor_kwargs)
    doe_x_cube_clamped = torch.clamp(doe_x_cube, 1e-10, 1 - 1e-10)
    v_mean = torch.zeros(44).to(**tensor_kwargs)
    v_std = (torch.ones(44)).to(**tensor_kwargs)
    doe_x = torch.distributions.Normal(v_mean, v_std).icdf(doe_x_cube_clamped)
    data = fun(doe_x)

    rangel = max(data) - min(data)
    x_min = min(data) - 0.15 * rangel
    x_max = max(data) + 0.15 * rangel
    # x_min = min(data)
    # x_max = max(data)
    kde = stats.gaussian_kde(data, bw_method=0.5)
    x = np.linspace(x_min, x_max, len(data))
    y = kde(x)

    cdf = np.array([integrate.simps(y[:i + 1], x[:i + 1]) for i in range(len(x))])
    interp = interpolate.interp1d(cdf, x, kind="cubic", bounds_error=False, fill_value="extrapolate")
    Spec = interp(1 - percentile)
    plt.figure()
    plt.plot(x, y)
    if draw:
        plt.axvline(Spec, color='r', linestyle='--', label='3-sigma')
    plt.hist(data, bins=30, density=True, alpha=0.5)
    plt.title(name, fontsize=20, verticalalignment='bottom')
    plt.xlabel("value")
    plt.ylabel("f")
    plt.savefig("./pdf/%s.png" % name)

    return Spec


def Kde_for_none_rand(fun, v, anchor_idx=None, idx=None, name: str = "3sigma"):
    data = fun(v)
    x_min = min(data)
    x_max = max(data)
    kde = stats.gaussian_kde(data, bw_method=0.5)
    x = np.linspace(x_min, x_max, len(data))
    y = kde(x)
    plt.figure()
    plt.plot(x, y)

    t = 0
    if idx is not None:
        for x_target in anchor_idx:
            x_val = data[x_target]
            if x_val == 0:
                continue
            y_val = kde(x_val)
            annotation = f"{idx[t]}"
            t += 1
            plt.scatter(x_val, y_val)
            plt.annotate(annotation,
                         xy=(x_val, y_val),
                         xytext=(x_val - 0.02, y_val + (0.04 if t % 2 == 0 else -0.04)),
                         arrowprops=dict(facecolor='black', arrowstyle='->', lw=0.5),
                         fontsize=8
                         )
    plt.hist(data, bins=30, density=True, alpha=0.5)
    plt.title(name, fontsize=20, verticalalignment='bottom')
    plt.xlabel("value")
    plt.ylabel("f")
    plt.savefig("./pdf/%s.png" % name)
# def FourBranch():
#

def get_normal_x(center: Tensor):
    center = center.reshape(-1, )
    std = (center / 6).expand(2 * center.shape[0], center.shape[0])
    design_param = torch.normal(mean=center.expand_as(std), std=std)
    return design_param


def get_icdf_x(center: Tensor):
    tensor_kwargs = {"device": 'cpu', "dtype": torch.double}
    rand_generator = SobolEngine(dimension=22, scramble=True)
    doe_x_cube = rand_generator.draw(44).to(**tensor_kwargs)
    doe_x_cube_clamped = torch.clamp(doe_x_cube, 1e-10, 1 - 1e-10)
    # v_mean = center.to(**tensor_kwargs)
    # v_std = torch.ones_like(center).to(**tensor_kwargs)
    # doe_x = torch.distributions.Normal(v_mean, v_std).icdf(doe_x_cube_clamped)
    # doe_x_clamped = torch.clamp(doe_x, 1e-10, 1 - 1e-10)
    return doe_x_cube_clamped


def get_random_v(rand_generator, doe_num: int, n_dim: int):
    tensor_kwargs = {"device": 'cpu', "dtype": torch.double}
    v_mean = torch.zeros(n_dim).to(**tensor_kwargs)
    v_std = (torch.ones(n_dim) / 2).to(**tensor_kwargs)
    doe_x_cube = rand_generator.draw(doe_num).to(**tensor_kwargs)
    doe_x_cube_clamped = torch.clamp(doe_x_cube, 1e-10, 1 - 1e-10)
    doe_x = torch.distributions.Normal(v_mean, v_std).icdf(doe_x_cube_clamped)
    return doe_x


def margin(
        MC_output: torch.Tensor,  # e.g.  3*100 3 outputs with 100 mc
        bounds: torch.Tensor,  # e.g.  3*2   3 outputs with upper and lower bounds
        scale: torch.Tensor
):
    MC_margin = torch.zeros((0, MC_output.shape[1]))
    # std = torch.std(MC_output, dim=1)  # 3*1
    # minimum, _ = torch.min(MC_output, 1)
    # maximum, _ = torch.max(MC_output, 1)
    # scale = maximum - minimum
    for i in range(MC_output.shape[0]):
        # MC_margin_i = torch.Tensor(
        #     [torch.min((bounds[i][1] - v) / std[i], (v - bounds[i][0]) / std[i]) for v in MC_output[i]])
        MC_margin_i = torch.Tensor(
            [torch.min((bounds[i][1] - v) / scale[i], (v - bounds[i][0]) / scale[i]) for v in MC_output[i]]
        )
        MC_margin = torch.cat((MC_margin, MC_margin_i.reshape(1, -1)), dim=0)

    # negative_sum = (MC_margin < 0).float() * MC_margin
    # margin_result = negative_sum.sum(dim=0)
    # margin result  100 * 1
    margin_result, _ = torch.min(MC_margin, dim=0)

    return margin_result.reshape(-1, 1)


class IntegratedSimulator:
    def __init__(self, design_param, state, netlist, rootPath, bash_path, instance):
        self.torch_para = {"device": 'cpu', "dtype": torch.double}
        self.alps_simulator = AlpsInterface(netlist, rootPath, bash_path, instance)
        self.alps_simulator.load_design()
        self.instance_num = self.alps_simulator.instance_num

        x_lb = [2E-7, 2E-7, 2E-7, 1E-6, 2E-7, 2E-7,
                3E-7, 8E-7, 8E-7, 8E-7, 2E-7,
                1E-6, 1E-6, 1E-6, 7E-7, 1E-6, 1E-6,
                7E-7, 7E-7, 1E-6, 1E-6, 1E-6]
        x_ub = [2E-6, 2E-6, 2E-6, 3E-6, 1E-6, 1E-6,
                2E-6, 3E-6, 3E-6, 3E-6, 2E-6,
                8E-6, 8E-6, 8E-6, 3E-6, 8E-6, 8E-6,
                8E-6, 8E-6, 8E-6, 8E-6, 8E-6]
        self.x_lb = torch.DoubleTensor(x_lb).to(**self.torch_para)
        self.x_ub = torch.DoubleTensor(x_ub).to(**self.torch_para)
        self.sim_bounds = torch.Tensor([[110, 200], [60, 100], [2e+6, 3e+6]])
        self.bounds = self.sim_bounds[:,1]-self.sim_bounds[:,0]

        self.sim_num = 0
        self.design_param = design_param
        self.x_corner = torch.zeros(44).reshape(1, 1, -1)
        self.state = state
        self.anchor = None

    def sim_alps(self, x, v):
        if v.ndim == 2:
            self.sim_num += v.shape[0]
        else:
            self.sim_num += v.shape[1]

        try:
            simres = self.alps_simulator.sim_with_given_param(load_design_param=False,
                                                              design_para=x,
                                                              mismatch=v,
                                                              res_keys=key
                                                              )
        except Exception as e:
            raise("出现错误：", e)

        return simres

    def opt_for_turbo(self, design_x):  # 优化函数，定v
        x = self.unnorm_x(design_x)
        output = self.sim_alps(x[0].reshape(1, -1), self.x_corner)
        result = [[v] for v in output]
        add = False
        for i in range(1, x.shape[0]):
            output = self.sim_alps(x[i].reshape(1, -1), self.x_corner)
            for j in range(len(result)):
                result[j].append(output[j])
        if self.state.sim_turbo is None:  # list(tensor(44*1)
            self.state.sim_turbo = [(torch.cat(result[j], dim=1)).reshape(-1, 1) for j in range(len(result))]
        else:
            add = False
            next_result_tensor = [(torch.cat(result[j], dim=1)).reshape(-1, 1) for j in range(len(result))]
            self.state.sim_turbo = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_turbo, next_result_tensor)]
        mc_result_tensor = torch.stack(self.state.sim_turbo).squeeze()
        result = margin(mc_result_tensor, self.sim_bounds, self.bounds)
        if add:
            return result[-x.shape[0]:].reshape(-1, 1)
        else:
            return result.to(device='cpu', dtype=torch.double)

    def ver_for_sigma(self, v):  # 分布绘制
        mc_result_list = self.sim_alps(self.design_param, v)  # list(tensor(400*1))
        mc_result_tensor = torch.stack(mc_result_list).squeeze()
        result = margin(mc_result_tensor, self.sim_bounds, self.bounds)
        return result.reshape(-1)

    def sim_for_corner(self, v):  # 角提取，定x
        mc_result_list = self.sim_alps(self.design_param, v)  # list(tensor(400*1))
        if self.state.sim_output is None:
            self.state.sim_output = mc_result_list
        else:
            self.state.sim_output = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_output, mc_result_list)]
        mc_result_tensor = torch.stack(self.state.sim_output).squeeze()
        result = margin(mc_result_tensor, self.sim_bounds, self.bounds)

        return result

    def turbo_anchor(self, design_x):  # 优化过程中计算锚点值
        x = self.unnorm_x(design_x)
        anchor_turbo_v_list = []
        for t in range(0, len(self.anchor)):
            output = self.sim_alps(x, self.anchor[t])
            total_result = [torch.cat((l1, l2)) for l1, l2 in zip(self.state.sim_turbo, output)]
            mc_result_tensor = torch.stack(total_result).squeeze()
            anchor_turbo_v_list.append(margin(mc_result_tensor, self.sim_bounds, self.bounds)[-1:])
        anchor_turbo_v_tensor = torch.cat(anchor_turbo_v_list, dim=0)
        dis = [(anchor_turbo_v_tensor[j] - anchor_turbo_v_tensor[j - 1]).item()
               for j in range(1, len(anchor_turbo_v_tensor))]
        return dis

    def norm_x(self, x):
        return (x - self.x_lb) / (self.x_ub - self.x_lb)

    def unnorm_x(self, x):
        return x * (self.x_ub - self.x_lb) + self.x_lb


# 要有norm点获取的机制，比如第一个是norm点
if __name__ == '__main__':

    target = 0
    target_yield = 0.9986
    design_param_num = 2
    variation_num = 4

    rootPath = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/opamp"  # set your work dir, the results will save here
    bash_path = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/opamp/setup.bash"  # where the setup.bash to be load
    netlist = "opamp_spice_new"
    instance = ["pch_mis", "nch_mis",
                "g45n1svt", "g45p1svt"]
    key = ["dc_gain", "pm", "gbw"]
    # design_param = torch.DoubleTensor(
    #     [5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7, 5e-7,
    #      1e-06, 4e-06, 4e-06, 4e-06, 4e-06, 4e-06, 2.e-06, 2.e-06, 1.e-06, 3.e-06, 1.e-06]
    # ).reshape(1, -1)

    design_param = torch.DoubleTensor(
        [4.8045e-07, 5.6722e-07, 2.0019e-07, 2.4222e-06, 2.7402e-07, 7.9335e-07,
         4.5443e-07, 2.8974e-06, 2.4830e-06, 2.3790e-06, 7.0845e-07, 2.6173e-06,
         4.3403e-06, 2.7136e-06, 1.8355e-06, 3.1289e-06, 5.7386e-06, 8.6256e-07,
         5.3565e-06, 2.2399e-06, 6.1370e-06, 5.3333e-06]
    ).reshape(1, -1)

    rand_generator = SobolEngine(dimension=44, scramble=True, seed=114514)
    variation_doe = get_random_v(rand_generator=rand_generator, doe_num=400, n_dim=44)
    variation_corner = get_random_v(rand_generator=rand_generator, doe_num=10000, n_dim=44)  # 取得固定的10000个蒙卡v
    fig_v = torch.cat((variation_doe, variation_corner), dim=0)

    state = CornerState(target=target, yield_per=0.95, v_dim=44,
                        variation_corner=variation_corner, variation_doe=variation_doe)
    alps_sim = IntegratedSimulator(design_param, state, netlist, rootPath, bash_path, instance)
    # 优化目标： dc_gain越大越好，但pm和gbw会相应降低，要求保证后两者下限的同时dc_gain优化上限
      # 2e + 6
    print("当前x对应标准化,如负则范围设置较小",alps_sim.norm_x(design_param))
    with open("CornerLog", 'w') as file_log:
        file_log.writelines(f"边界设置: {alps_sim.sim_bounds.numpy()}\n")
        file_log.writelines(f"初始X: {alps_sim.design_param.numpy()}\n")
        file_log.writelines(f"将x优化至临界")
        opt_design = alps_sim.norm_x(alps_sim.design_param)
        optDesign, best_value = turbo(design_param=opt_design, random_generater=get_icdf_x,
                                      eval_objective=alps_sim.opt_for_turbo, file_log=file_log, ori_opt=True)
        alps_sim.design_param = alps_sim.unnorm_x(optDesign)
        Kde_for_none_rand(fun=alps_sim.ver_for_sigma, v=fig_v, name=f"Original opt")
        file_log.writelines(f"x标准点优化完成")
        print("优化后x:", alps_sim.design_param)
        file_log.writelines(f"优化后x: {alps_sim.design_param}")
        alps_sim.state = CornerState(target=target, yield_per=0.95, v_dim=44,
                        variation_corner=variation_corner, variation_doe=variation_doe)
        for i in range(5):

            file_log.writelines(f"\n第{i}轮优化\n")
            # state = CornerState(target=target, yield_per=0.95, v_dim=44)
            # veri = corner(sim_alps=sim_for_corner, state=state, verify=True)
            anchor_v, anchor, indice = corner(sim_alps=alps_sim.sim_for_corner, state=alps_sim.state)
            alps_sim.anchor = anchor
            position = [(torch.where((fig_v == kk.squeeze()).all(dim=1))[0]) for kk in anchor]
            Kde_for_none_rand(fun=alps_sim.ver_for_sigma, v=fig_v, anchor_idx=position, idx=indice,
                              name=f"{i} loop before opt")

            distance = [(anchor_v[i] - anchor_v[i - 1]).item() for i in range(1, len(anchor_v))]
            file_log.writelines(f"角提取完成： corner：{anchor}\n")
            file_log.writelines(f"当前锚点间距离比： {distance}\n")
            print("当前锚点间距离比：", distance)
            alps_sim.x_corner = anchor[2].reshape(1, 1, -1)

            # turbo_x = get_normal_x(design_param)
            # turbo_x = get_icdf_x(design_param)
            opt_x = alps_sim.norm_x(alps_sim.design_param)
            optDesign, best_value = turbo(design_param=opt_x, random_generater=get_icdf_x,
                                          eval_objective=alps_sim.opt_for_turbo, file_log=file_log,
                                          anchor_objective=alps_sim.turbo_anchor)

            print("优化前：", alps_sim.sim_alps(alps_sim.design_param, alps_sim.x_corner))
            alps_sim.design_param = alps_sim.unnorm_x(optDesign)
            print("优化后x:",alps_sim.design_param)
            file_log.writelines(f"优化后x: {alps_sim.design_param}")
            print("优化后：", alps_sim.sim_alps(alps_sim.design_param, alps_sim.x_corner))
            Kde_for_none_rand(fun=alps_sim.ver_for_sigma, v=fig_v,
                              anchor_idx=position, idx=indice, name=f"{i} loop after opt")
            # 验证  0.95
            alps_sim.state = CornerState(target=target, yield_per=0.95, v_dim=44,
                        variation_corner=variation_corner, variation_doe=variation_doe)

            veri = corner(sim_alps=alps_sim.sim_for_corner, state=alps_sim.state, verify=True)
            if veri:
                print(f"优化完成，design param: {alps_sim.design_param}")
                print(f"x corner: {alps_sim.x_corner}")
                print("sim values:", alps_sim.sim_alps(alps_sim.design_param, alps_sim.x_corner))
                print("total sim num:", alps_sim.sim_num)
                break
        print(f"Not satisfied, start next loop")
