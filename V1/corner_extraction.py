
from simulator import IntegratedSimulatorFunction

import torch

from dataclasses import dataclass
import warnings
import matplotlib.pyplot as plt
from torch import Tensor
from utils import margin

# warnings.filterwarnings("ignore")

MODEL_TYPE = ['SIMULATION', 'FFX', 'DKGP']
LFTYPE = ['U', 'EFF', 'LCB', 'SORT']


@dataclass
class CornerSetting:
    design_x: Tensor  # design parameter x
    val_v: Tensor  # validation set v
    corner_length: int  # corner position in val_v
    corner_window: float  #
    margin_threshold: Tensor
    margin_scale: Tensor
    doe_v_size: int  # doe size for model
    model_type: str  # string in MODEL_TYPE
    learning_type: str or None


def extract_corner(simulator: IntegratedSimulatorFunction, settings: CornerSetting, verify=False):
    """
    Extract statistical corner form given sample set of variation parameters
    Using MCS based active learning method.
    :return:
    """
    if settings.val_v[0, :].any():
        raise Exception('First sample in validation must be nominal')

    if settings.model_type not in MODEL_TYPE:
        raise Exception('Unsupported corner extraction model')
    else:
        if settings.model_type == 'SIMULATION':
            val_y = simulator.sim_v(settings.val_v, design_x=settings.design_x)
            val_y_margin = margin(val_y, settings.margin_threshold, settings.margin_scale)
            val_y_sort, val_y_sort_ind = torch.sort(val_y_margin[:, 0])
            v_corner = val_y_sort_ind[settings.corner_length]
            v_worst = val_y_sort_ind[0: int(settings.corner_length * settings.corner_window)]
            v_worst = torch.cat((Tensor([0]).to(dtype=v_worst.dtype), v_worst))
            f_worst = val_y[v_worst, :]

            return v_corner, v_worst, f_worst
