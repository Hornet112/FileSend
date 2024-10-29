import torch
from corner_extraction import extract_corner, CornerSetting
from turbo import turbo, TurboSetting
from simulator import IntegratedSimulatorFunction
import numpy as np
from torch import Tensor
from torch.quasirandom import SobolEngine


def f_four_branch(x):
    y = torch.zeros((x.shape[0], 4))
    y[:, 0] = 3 + 0.1 * (x[:, 0] - x[:, 1]) ** 4 - (x[:, 0] + x[:, 1]) / np.sqrt(2)
    y[:, 1] = 3 + 0.1 * (x[:, 0] - x[:, 1]) ** 4 + (x[:, 0] + x[:, 1]) / np.sqrt(2)
    y[:, 2] = (x[:, 0] - x[:, 1]) + 6 / np.sqrt(2)
    y[:, 3] = (x[:, 1] - x[:, 0]) + 6 / np.sqrt(2)
    yy = torch.min(y, 1)[0]
    return yy.reshape(-1, 1)


if __name__ == '__main__':
    random_generator = SobolEngine(dimension=1, scramble=True, seed=114514)
    settings = CornerSetting(
        design_x=Tensor([[0]]),
        val_v=torch.cat((Tensor([[0]]), random_generator.draw(999)), dim=0),
        corner_length=10,
        corner_window=1.5,
        margin_threshold=Tensor([[2.001]]),
        margin_scale=Tensor([[1]]),
        doe_v_size=100,
        model_type='SIMULATION',
        learning_type=None
    )
    function_simulator = IntegratedSimulatorFunction(function=f_four_branch,x_ub=5,x_lb=-5)
    ind_v_corner, ind_v_worst, f_worst = extract_corner(function_simulator, settings)
    function_simulator.v_corner = settings.val_v[ind_v_corner]
    TurboSettings = TurboSetting(
        design_param=settings.design_x,
        NORM_V=Tensor([[0]]),
        corner_norm_distance=f_worst[0]-f_worst[torch.where(ind_v_corner == ind_v_worst)[0]],
        margin_threshold=settings.margin_threshold,
        margin_scale=settings.margin_scale,
    )
    opt_x = turbo(simulator=function_simulator, settings=TurboSettings)
    print(opt_x)


