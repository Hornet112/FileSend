import gpytorch
from transform_function import *
import numpy as np

from transform_function import *


# from botorch.models.utils.gpytorch_modules import (
#     get_gaussian_likelihood_with_gamma_prior,
#     get_matern_kernel_with_gamma_prior,
# )


class SurrogateBaseModel(ABC):
    def __init__(self, data_x, data_y, seed=None, transformfunction=None):
        """

        :param data_x: raw data x, tensor or ndarray, n * nx
        :param data_y: raw data y, tensor or ndarray, n * ny
        :param seed: random seed
        """

        self.train_x = data_x  # ÕâÀï¿¼ÂÇºóÐø¸üÐÂÊÇ¼ÓÊý¾Ý£¬Òò´ËÀàÖÐÓ¦µ±´æ´¢Ô­Ê¼µÄÊý¾Ý
        self.train_y = data_y
        self.train_n = data_x.shape[0]
        self.nx = data_x.shape[1]
        self.ny = data_y.shape[1]
        if seed is None:
            pass
            self.seed = 1
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
        else:
            self.seed = seed
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        self.model = None

        self.transform = transformfunction

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def update(self, x_new, y_new):
        pass


class DeepKernelGP(SurrogateBaseModel):
    def __init__(self, data_x, data_y, seed=None, use_gpu=False, transformfunction=None):
        super().__init__(data_x, data_y, seed, transformfunction)
        self.likelihood = None
        self.use_gpu = use_gpu
        if use_gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = "cpu"
        dtype = torch.float32
        self.tensor_kwargs = {"device": device, "dtype": dtype}
        self.iteration = 500
        if isinstance(data_x, np.ndarray):
            self.train_x = torch.tensor(self.train_x, **self.tensor_kwargs)
            self.train_y = torch.tensor(self.train_y, **self.tensor_kwargs)

    def train(self):
        if self.transform is None:
            train_x = self.train_x.detach().clone()
            train_y = self.train_y.detach().clone()
        else:
            train_x, train_y = self.transform.data_transform(self.train_x, self.train_y)
        train_y = train_y.squeeze()
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full_like(train_y, 1e-4))
        self.model = GPRegressionModel(train_x, train_y, self.likelihood)

        if torch.cuda.is_available() and self.use_gpu:
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=0.01, weight_decay=1e-4)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for i in range(self.iteration):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = self.model(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            # print(loss)
            loss.backward()
            optimizer.step()

    def predict(self, x, return_cov=False):
        self.model.eval()
        self.likelihood.eval()

        eeva_x = torch.tensor(x, **self.tensor_kwargs)

        if self.transform is not None:
            eeva_x = self.transform.x_transform(eeva_x, recalculate=False)

        with torch.no_grad(), gpytorch.settings.use_toeplitz(False), gpytorch.settings.fast_pred_var():
            post = self.model(eeva_x)

        y_mean = post.mean.reshape((-1, self.ny))

        if self.transform is not None:
            y_mean = self.transform.y_restore(y_mean)
            if self.use_gpu:
                y_mean = y_mean.cpu()
        if return_cov:
            y_std = post.stddev.reshape((-1, self.ny))

            if self.transform is not None:
                y_std = self.transform.y_std_restore(y_std)

            if self.use_gpu:
                y_std = y_std.cpu()

            return y_mean.detach().numpy(), y_std.detach().numpy()
        else:
            return torch.from_numpy(y_mean.detach().numpy())

    def update(self, x_new, y_new, add=True):
        if isinstance(x_new, np.ndarray):
            x_new = torch.tensor(x_new, **self.tensor_kwargs)
            y_new = torch.tensor(y_new, **self.tensor_kwargs)
        self.train_x = torch.cat((self.train_x, x_new), dim=0)
        if add:
            self.train_y = torch.cat((self.train_y, y_new), dim=0)
        else:
            self.train_y = y_new
        # self.train()
        self.update_train()

    def update_train(self):
        if self.transform is None:
            train_x = self.train_x.detach().clone()
            train_y = self.train_y.detach().clone()
        else:
            train_x, train_y = self.transform.data_transform(self.train_x, self.train_y)
        train_y = train_y.squeeze()
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.full_like(train_y, 1e-4))
        model_new = GPRegressionModel(train_x, train_y, self.likelihood, self.model.feature_extractor)

        if torch.cuda.is_available() and self.use_gpu:
            model_new = model_new.cuda()
            self.likelihood = self.likelihood.cuda()

        model_new.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': model_new.covar_module.parameters()},
            {'params': model_new.mean_module.parameters()},
            {'params': model_new.likelihood.parameters()},
        ], lr=0.01, weight_decay=1e-4)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model_new)

        for i in range(self.iteration):
            # Zero backprop gradients
            optimizer.zero_grad()
            # Get output from model
            output = model_new(train_x)
            # Calc loss and backprop derivatives
            loss = -mll(output, train_y)
            # print(loss)
            loss.backward()
            optimizer.step()

        self.model = model_new


class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, out_dim):
        super(LargeFeatureExtractor, self).__init__()
        self.add_module('linear1', torch.nn.Linear(data_dim, 100))
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(100, 100))
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(100, 50))
        self.add_module('relu3', torch.nn.ReLU())
        self.add_module('linear4', torch.nn.Linear(50, out_dim))
# class LargeFeatureExtractor(torch.nn.Sequential):
#     def __init__(self, data_dim, out_dim):
#         super(LargeFeatureExtractor, self).__init__()
#         self.add_module('linear1', torch.nn.Linear(data_dim, 800))
#         self.add_module('relu1', torch.nn.ReLU())
#         self.add_module('linear2', torch.nn.Linear(800, 400))
#         self.add_module('relu2', torch.nn.ReLU())
#         self.add_module('linear3', torch.nn.Linear(400, 100))
#         self.add_module('relu3', torch.nn.ReLU())
#         self.add_module('linear4', torch.nn.Linear(100, out_dim))

class GPRegressionModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, feature=None):
        super(GPRegressionModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=8))
        if feature is None:
            self.feature_extractor = LargeFeatureExtractor(train_x.shape[1], 8)
        else:
            self.feature_extractor = feature

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1., 1.)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"

        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
