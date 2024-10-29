from time import time

import matplotlib
import numpy as np
from typing import List, Optional
import scipy.stats as stats
import matplotlib.pyplot as plt

matplotlib.use("agg")
import scipy.integrate as integrate
import scipy.interpolate as interpolate
from scipy.stats import norm
from torch import Tensor
from AlpsNetlist import create_alps_simulator





def Kde(data, percentile, draw, name):
    rangel = max(data) - min(data)
    x_min = min(data) - 0.15 * rangel
    x_max = max(data) + 0.15 * rangel
    # x_min = min(data)
    # x_max = max(data)
    kde = stats.gaussian_kde(data, bw_method=0.5)
    x = np.linspace(x_min, x_max, len(data)*2)
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
    # plt.legend()  # 显示图例
    # plt.show()

    plt.savefig("./pdf/%s.png" % name)

    return Spec



def wilson(x, n, confidence):
    p = x / n
    z = norm.isf((1 - confidence) / 2)
    lower_bound = (p + z ** 2 / (2 * n) - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (1 + z ** 2 / n)
    upper_bound = (p + z ** 2 / (2 * n) + z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))) / (1 + z ** 2 / n)
    return lower_bound, upper_bound


class sample:
    def __init__(
            self,
            design_param: Tensor,
            mismatch_variation: Tensor,
            value_keys: Optional[List[str]] = None,
            name: str = 'MC PDF',
            verify=False
    ):
        """
        :param design_param: desgin param for sim
        :param mismatch_variation: a vector matrix for var
        :param value_keys: name of keys
        """
        self.success = True
        self.Mc_value = None
        self.design_param = design_param
        self.mismatch_variation = mismatch_variation
        self.value_keys = value_keys
        self.name = name
        self.alps_simulator = create_alps_simulator().sim_with_givenParam

    def Percentile(self, percentile, draw_percentile: bool = True):
        """
        get sim pdf and the percentile point(spec value)
        :param percentile: yield
        :return: spec value
        """
        func = self.alps_simulator
        mismatch_num = self.mismatch_variation.shape[1]
        print(f"Num of v mc  num for one design param:{mismatch_num}\n")
        mc = func(self.design_param, self.mismatch_variation)
        margin = mc[0].reshape(-1)  # 取DC输出，后续高维则应在此处理,  Tensor(10,)
        self.Mc_value = margin
        threesigma = Kde(data=margin, percentile=percentile, draw=draw_percentile, name=self.name)
        return threesigma

    def verification(self,target: float = 0, target_yield: float = 0.9986):
        f = self.alps_simulator
        access = 0
        total = self.mismatch_variation.shape[1]
        mc_ori = f(self.design_param, self.mismatch_variation)
        margin = mc_ori[0].reshape(-1)  # 取DC输出，后续高维则应在此处理,  Tensor(10,1)
        access += sum(1 for v in margin if v > target)
        print(f"初始{total}样本通过数access{access}")

        p = access / total
        sigma = np.sqrt(p * (1 - p) / total)
        wilsonl, wilsonh = wilson(access, total, 0.95)
        mcl, mch = p - sigma ** 1.96, p + sigma ** 1.96
        low = [min(wilsonl, mcl)]
        up = [max(wilsonh, mch)]
        print(f"\n初始蒙特卡洛样本置信区间[{low[-1]},{up[-1]}]")
        n = 5000
        while low[-1] <= target_yield <= up[-1]:
            # np.random.seed(114514)
            v = np.random.randn(1, n, 4 * 11)
            margin = f(self.design_param, v)[0].reshape(-1)
            # print(f"Mc time for {n} samples:{time()-start}")
            for mc in margin:
                if mc > target:
                    access += 1
                total += 1
                p = access / total
                sigma = np.sqrt(p * (1 - p) / total)
                wilsonl, wilsonh = wilson(access, total, 0.95)
                mcl, mch = p - sigma ** 1.96, p + sigma ** 1.96
                low.append(min(wilsonl, mcl))
                up.append(max(wilsonh, mch))
            print(f"\naccess: {access}, total: {total}")
            print(f"样本置信区间[{low[-1]},{up[-1]}]")
            # if total>20000:
            #     print(f"Too many verification with {total}")
            #     break
        print(f"\nFinal bounds[{low[-1]},{up[-1]}]")
        if low[-1] > target_yield:
            print(f"最终仿真次数:{total}")
            print(f"最终通过次数:{access}")
            print(f"最终置信区间:[{low[-1]},{up[-1]}]\n")

            plt.figure()
            x = np.linspace(1000, total, (total - 1000) + 1)
            plt.plot(x, up, label="Upper bound")
            plt.plot(x, low, label="Lower bound")
            plt.axhline(target_yield, color='r', linestyle='--', label='target')
            plt.title("3-sigma Verification", fontsize=20, verticalalignment='bottom')
            plt.xlabel(f"MC numbers")
            plt.ylabel("Yield")
            plt.legend()
            # plt.show()
            plt.savefig("./pdf/verify.png")
            self.success = True
        else:
            self.success = False
