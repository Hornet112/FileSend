#!/usr/bin/env python3
# coding: utf-8

# ## BO with TuRBO-1 and TS/qEI

import math
import os
import warnings
from dataclasses import dataclass

import gpytorch
import torch
from botorch.acquisition import qExpectedImprovement
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch import Tensor
from torch.quasirandom import SobolEngine
from utils import margin
from simulator import IntegratedSimulatorFunction

warnings.filterwarnings("ignore", category=BadInitialCandidatesWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
SMOKE_TEST = os.environ.get("SMOKE_TEST")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.double


@dataclass
class TurboState:
    dim: int = 22
    batch_size: int = 4
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = float("nan")  # Note: Post-initialized
    success_counter: int = 0
    success_tolerance: int = 10  # Note: The original paper uses 3
    best_value_x: Tensor = torch.zeros(dim).reshape(1, -1)
    best_value: float = -float("inf")
    restart_triggered: bool = False
    opt_num: int = 0

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max([4.0 / self.batch_size, float(self.dim) / self.batch_size])
        )


def get_icdf_x(dim):
    tensor_kwargs = {"device": 'cpu', "dtype": torch.double}
    rand_generator = SobolEngine(dimension=dim, scramble=True)
    doe_x_cube = rand_generator.draw(dim * 22).to(**tensor_kwargs)
    doe_x_cube_clamped = torch.clamp(doe_x_cube, 1e-10, 1 - 1e-10)
    return doe_x_cube_clamped


def update_state(state, y_next, x_next, step):  # give new x\y steps
    state.opt_num += step
    if max(y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
        state.best_value = max(y_next).item()
        state.best_value_x = x_next[y_next.argmax().item()].reshape(1, -1)
    else:
        state.success_counter = 0
        state.failure_counter += 1

    if state.success_counter == state.success_tolerance:  # Expand trust region
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:  # Shrink trust region
        state.length /= 2.0
        state.failure_counter = 0

    if state.length < state.length_min:
        state.restart_triggered = True
        state.length = 0.8
        state.success_counter = 0
        state.failure_counter = 0

    return state


def generate_batch(
        state,
        model,  # GP model
        X,  # Evaluated points on the domain [0, 1]^d
        Y,  # Function values
        batch_size,
        n_candidates=None,  # Number of candidates for Thompson sampling
        num_restarts=10,
        raw_samples=512,
        acqf="ts",  # "ei" or "ts"
):
    assert acqf in ("ts", "ei")
    assert X.min() >= 0.0 and X.max() <= 1.0 and torch.all(torch.isfinite(Y))
    if n_candidates is None:
        n_candidates = min(5000, max(2000, 200 * X.shape[-1]))

    # Scale the TR to be proportional to the lengthscales
    x_center = X[Y.argmax(), :].clone()
    weights = model.covar_module.base_kernel.lengthscale.squeeze().detach().reshape(-1)
    weights = weights / weights.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

    if acqf == "ts":
        dim = X.shape[-1]
        sobol = SobolEngine(dim, scramble=True)
        pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
        pert = tr_lb + (tr_ub - tr_lb) * pert

        # Create a perturbation mask
        prob_perturb = min(20.0 / dim, 1.0)
        mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
        ind = torch.where(mask.sum(dim=1) == 0)[0]
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1

        # Create candidate points from the perturbations and the mask
        X_cand = x_center.expand(n_candidates, dim).clone()
        X_cand[mask] = pert[mask]

        # Sample on the candidate points
        thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
        with torch.no_grad():  # We don't need gradients when using TS
            X_next = thompson_sampling(X_cand, num_samples=batch_size)

    elif acqf == "ei":
        ei = qExpectedImprovement(model, Y.max())
        X_next, acq_value = optimize_acqf(
            ei,
            bounds=torch.stack([tr_lb, tr_ub]),
            q=batch_size,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
        )

    return X_next


@dataclass
class TurboSetting:
    design_param: Tensor
    NORM_V: Tensor
    corner_norm_distance: Tensor
    margin_threshold: Tensor
    margin_scale: Tensor


def turbo(simulator, settings: TurboSetting):  # X:Tensor(num, dim)
    design_param = settings.design_param
    NORM_V = settings.NORM_V
    batch_size = 4
    eval_objective = simulator.sim_x

    dim = design_param.shape[1]
    X_turbo = get_icdf_x(dim)
    Y_turbo = margin(eval_objective(x_norm=X_turbo, v_corner=NORM_V).to(device=device, dtype=dtype),settings.margin_threshold,settings.margin_scale)
    state = TurboState(dim=dim, best_value=max(Y_turbo).item(),
                       best_value_x=X_turbo[Y_turbo.argmax().item()].reshape(1, -1))

    NUM_RESTARTS = 10 if not SMOKE_TEST else 2
    RAW_SAMPLES = 512 if not SMOKE_TEST else 4
    N_CANDIDATES = min(5000, max(2000, 200 * dim)) if not SMOKE_TEST else 4
    max_cholesky_size = float("inf")
    torch.manual_seed(0)

    while state.opt_num < 700 and state.best_value < settings.corner_norm_distance:  # Run until TuRBO converges
        # Fit a GP model
        train_Y = (Y_turbo - Y_turbo.mean()) / Y_turbo.std()
        likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 4.0)
            )
        )
        model = SingleTaskGP(
            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            fit_gpytorch_mll(mll)

            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=N_CANDIDATES,
                num_restarts=NUM_RESTARTS,
                raw_samples=RAW_SAMPLES,
                acqf="ts",
            )

        Y_next = margin(eval_objective(x_norm=X_next, v_corner=NORM_V).to(device=device, dtype=dtype),
                        settings.margin_threshold,settings.margin_scale)

        # Update state
        state = update_state(state=state, y_next=Y_next, x_next=X_next, step=4)
        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.4e}, TR length: {state.length:.2e}"
        )
        line = f"{len(X_turbo)}) Best value: {state.best_value:.4e},  current design param: {state.best_value_x}\n"
    if state.best_value > settings.corner_norm_distance:
        print("Normal condition satisfy")
    return state.best_value_x  # , state.best_value
