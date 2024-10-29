
from torch import Tensor


def margin(
        sim_result: Tensor,  # e.g.  100 * 3 3 outputs with 100 mc
        threshold: Tensor,  # e.g.  1 * 3   3 outputs with upper and lower bounds
        scale: Tensor  # e.g.  1 * 3
):
    margin_result = (sim_result - threshold / scale).min(dim=1)[0]

    return margin_result.reshape(-1, 1)