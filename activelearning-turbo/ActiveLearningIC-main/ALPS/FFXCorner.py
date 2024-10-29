from time import time
from deep_kernel_gp import DeepKernelGP
from simulator import SimulatorFunction, SimulatorModel
from corner_generate import MonteCarloSimulation
from learning_function import ScoreLearningFunction
from stopping_criterion import CornerCriterion, MaxSimCriterion
import numpy as np
import torch
import math
from torch.quasirandom import SobolEngine
from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


@dataclass
class CornerState:
    variation_doe: torch.Tensor = None
    variation_corner: torch.Tensor = None
    sim_output: list = None  # 仿真器实际仿真
    sim_turbo: list = None  # 多设计参数的仿真结果
    Ffun: SimulatorModel = None
    x_idx: torch.Tensor = torch.zeros(0)
    lcb_learning: ScoreLearningFunction = None
    max_sim: CornerCriterion = None
    MC: MonteCarloSimulation = None
    failed_sim: int = 0
    idx: list = None  # 所选标签
    # sim_y: torch.Tensor = torch.zeros((0, 1))   # 仿真后的所有margin
    corner_x: torch.Tensor = torch.zeros((0, 1))
    target: float = 100
    yield_per: float = 0.95
    v_dim: int = 44
    Doe: bool = False  # Doe has done


#  涉及到的类型转换：
#  GP需要输入numpy类型
#  sort需要输入tuple（tensor（mean），tensor（std））
def corner(sim_alps, state: CornerState, verify: bool = False, ):
    total_sim_time = 0
    total_model_time = 0
    sim_fun = SimulatorFunction(sim_alps)

    # initial setup
    device = 'cpu'
    dtype = torch.double
    tensor_kwargs = {"device": device, "dtype": dtype}
    # problem setup
    n_dim = state.v_dim
    v_mean = torch.zeros(n_dim).to(**tensor_kwargs)
    v_std = (torch.ones(n_dim) / 2).to(**tensor_kwargs)
    rand_generator = SobolEngine(dimension=n_dim, scramble=True, seed=114514)
    # initial
    doe_num = 400
    sigma = 1 - 0.9986
    scale = 2
    delay = 5
    MC_num = 10000
    corner_len = math.ceil(MC_num * sigma * scale)

    sim_budget = doe_num + corner_len
    if not state.Doe:
        # DOE
        if state.variation_doe is not None:
            doe_x = state.variation_doe
        else:
            doe_x_cube = rand_generator.draw(doe_num).to(**tensor_kwargs)
            doe_x_cube_clamped = torch.clamp(doe_x_cube, 1e-10, 1 - 1e-10)
            doe_x = torch.distributions.Normal(v_mean, v_std).icdf(doe_x_cube_clamped)  # .T).T
        start = time()
        doe_y = sim_fun.sim(doe_x)
        state.Doe = True
        print(f"\nsimulation end,cost: {time() - start}")

        total_sim_time += time() - start
        # indices = torch.randperm(doe_x.size(0))
        # DKGP need numpy input
        # doe_x_perm = doe_x[indices]
        # doe_y_perm = doe_y[indices]

        idx = torch.argsort(doe_y, dim=0)[:corner_len]
        # sim_y = sorted_doe_y[0:corner_len]
        #
        # corner_x = doe_x[idx][0:corner_len].reshape(corner_len,-1)
        # state.sim_y = doe_y_perm
        # state.indices = indices
        corner_x = doe_x
        state.corner_x = doe_x
        state.idx = idx

        # FFX Model
        # FFX = FFXModel(train_x, train_y, test_x, test_y, verbose=False)
        DK_GP = DeepKernelGP(doe_x.numpy(), doe_y.numpy())
        print("start model train")
        start = time()
        # FFX.train()
        DK_GP.train()
        print(f"model train end, cost:{time() - start}")
        total_model_time += time() - start
        # FFX_fun = SimulatorModel(FFX)
        DKGP_fun = SimulatorModel(DK_GP)
        state.Ffun = DKGP_fun
        if verify:
            p = (doe_y < state.target).prod(dim=-1)
            failed_sim = p.sum(dim=-1).item()
            # state.failed_sim = failed_sim
            p_failure = p.to(**tensor_kwargs).mean(dim=-1).item()
            if p_failure > (1 - state.yield_per):
                return False
    else:
        DKGP_fun = state.Ffun
        # failed_sim = state.failed_sim
        idx = state.idx
        corner_x = state.corner_x

    learning_step = 1
    learning_method = "LCB"
    # Yield

    # Learning function
    if not state.max_sim:
        MC = MonteCarloSimulation(DKGP_fun, v_mean, v_std, rand_generator, MC_num, state.variation_corner)
        val_x, val_y_hat, val_y_std = MC.move_forward()
        val_y_hat = torch.from_numpy(val_y_hat)
        val_y_std = torch.from_numpy(val_y_std)
        lcb_learning = ScoreLearningFunction(learning_method, learning_step, (val_y_hat, val_y_std))

        # initial corner len with doe

        x_idx = torch.zeros(0)
        # find 2*sigma points and select the middle point for corner
        max_sim = CornerCriterion(sim_budget, delay, corner_len)
        max_sim.update_sim(doe_num, idx)
    else:
        MC = state.MC
        val_x = MC.sample
        lcb_learning = state.lcb_learning
        x_idx = state.x_idx
        max_sim = state.max_sim

    iter_count = 0
    current_sigma = False
    current_corner = torch.zeros(0)
    anchor = []
    anchor_value = []
    indice = []
    fig, ax = plt.subplots()
    ax.set_yticks([])
    ax.set_title("3-sigma分点")
    i = 0
    point_enum = [[],[],[]]
    while not max_sim.check():
        iter_count += 1
        # Choose new points
        new_idx = lcb_learning.learning_candidate()
        new_x = val_x[new_idx, :]
        start = time()
        new_y = sim_fun.sim(new_x)
        # new_y = torch.cat((new_y[indices], new_y[doe_num:]), dim=0)
        total_sim_time += time() - start

        x_idx = torch.cat((x_idx, new_idx), dim=0)  # 10000个里的坐标
        # print("before:", idx.reshape(-1))
        idx = torch.argsort(new_y, dim=0)[:corner_len]         # 400+个里的
        # print("after:", idx.reshape(-1))
        corner_x = torch.cat((corner_x, new_x), dim=0)

        # Remove simulated samples from val_x
        val_idx = np.setdiff1d(np.linspace(0, val_x.shape[0] - 1, val_x.shape[0], dtype=int), new_idx.detach().numpy())

        # Reset yield analysis
        MC.reset()
        # Update the model
        start = time()
        MC.simulator.model.update(new_x.numpy(), new_y.numpy(), add=False)
        total_model_time += time() - start
        # print(time() - start)
        # Update yield analysis
        MC.load_samples(val_x[val_idx, :])
        val_x, val_y_hat, val_y_std = MC.move_forward()
        max_sim.update_sim(learning_step, idx)
        lcb_learning.reset_model_prediction((torch.from_numpy(val_y_hat), torch.from_numpy(val_y_std)))

        if verify:
            if new_y[-1].item() < state.target:
                failed_sim += 1
                if failed_sim > state.yield_per * (doe_num + MC_num):
                    print("Verification failed,next loop")
                    state.lcb_learning = lcb_learning
                    state.max_sim = max_sim
                    state.x_idx = x_idx
                    state.idx = idx
                    # state.sim_y = sim_y
                    state.corner_x = corner_x
                    state.MC = MC
                    return False

        print('Iteration:', iter_count)
        print('New_y:', new_y[-1],)

        print('current sigma point is:', new_y[idx[math.ceil(MC_num * sigma) - 1]], ' ', idx[math.ceil(MC_num * sigma) - 1])

        point_enum[0].append(new_y[idx[0]].item()-new_y[idx[0]].item())
        point_enum[1].append(new_y[idx[math.ceil(MC_num * sigma) - 1]].item()-new_y[idx[0]].item())
        point_enum[2].append(new_y[idx[-1]].item()-new_y[idx[0]].item())
        i += 1
        # if i==5:
        #     break
    # for t in range(3):
    #     plt.plot(point_enum[t],range(i))
    # plt.show()
    # plt.figure()
    # plt.savefig("./sigma_corner.png")
    for t in range(3):
        plt.plot(point_enum[t], range(i))
    # plt.show()
    plt.savefig("./sigma_corner_norm.png")
    indice = [math.ceil(MC_num * sigma) - 7,
              math.ceil(MC_num * sigma) - 4,
              math.ceil(MC_num * sigma) - 1,
              math.ceil(MC_num * sigma) + 2,
              math.ceil(MC_num * sigma) + 5]
    anchor_v = [new_y[idx[indi]] for indi in indice]
    anchor = [corner_x[idx[indi]].reshape(1, 1, -1) for indi in indice]
    sigma = corner_x[idx[math.ceil(MC_num * sigma) - 1]].reshape(1, 1, -1)
    # anchor = [sigma * 1.5, sigma * 1.25, sigma, sigma * 0.75, sigma * 0.5]
    # anchor_v = sim_fun.sim(torch.cat(anchor, dim=1))[-5:]
    print(anchor_v)
    # print(anchor_value)
    print("total_sim_time:", total_sim_time)
    print("total_model_time:", total_model_time)
    if verify:
        return True
    else:
        return anchor_v, anchor,indice
