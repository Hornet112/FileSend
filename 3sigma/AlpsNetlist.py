import time

import numpy as np
from collections import namedtuple
from copy import deepcopy
from shutil import copyfile
import os
import re
import subprocess
import torch
from typing import List

from torch import Tensor


# Alps_Mismatch = namedtuple('Alps_Mismatch', ['toxmis', 'vthmis', 'dwmis', 'dlmis'])
# default_AlpsNetlist = "/home/zhangyue/PycharmProjects/YADO/ALPS_SPICE/"
# default_simPath = "/home/zhangyue/PycharmProjects/YADO/test_sram/scbo/simulation"

def pch_mis(l, w, hdifp0=2.4e-7, hdifp1=2.7e-7, multi=1):
    ad = float("{:.2e}".format(2 * hdifp0 * w))
    _as = ad
    pd = float("{:.2e}".format(2 * (2 * hdifp0 + w)))
    ps = pd
    nrd = float("{:.2e}".format(hdifp1 / w))
    nrs = nrd
    return ad, _as, pd, ps, nrd, nrs


def Write_mismath(mismatch: Tensor, index: int):
    output_lines = []
    for i in range(index):
        line = f"{i + 1}"
        for data in mismatch[i]:
            line += " " + f"{data}"
        line += "\n"
        output_lines.append(line)
    return output_lines
class YADO_Alps:
    def __init__(self,
                 netlist: str,
                 rootPath: str,
                 bash_path: str,
                 instance=None,
                 mismatch=None):

        print("Create new simulator")

        if instance is None:
            instance = ["pch3_mis", "nch3_mis", "g45n1svt", "g45p1svt"]
        if not os.path.isdir(rootPath):  #
            raise NotADirectoryError(f"Path '{rootPath}' is not valid.")

        default_AlpsNetlist = os.path.join(rootPath, "ALPS_SPICE/")
        default_simPath = os.path.join(rootPath, "simulation")
        os.makedirs(default_simPath, exist_ok=True)
        os.makedirs(default_AlpsNetlist, exist_ok=True)

        self.netlist_path = default_AlpsNetlist + netlist + "/" + netlist
        self.simNetlist_dir = default_simPath  # + netlist
        self.mc0_path = self.simNetlist_dir + "/" + netlist + ".sim/" + netlist + ".mc0"

        self.simNetlist_path = self.simNetlist_dir + "/" + netlist
        self.simMc0_path = netlist + "_sim.mc0"
        self.simMc0_write_path = self.simNetlist_dir + "/" + self.simMc0_path
        self.measureRes_path = self.simNetlist_dir + "/" + netlist + ".sim/" + netlist + ".measure"

        self.unit_dict = {"n": 1E-9, "u": 1E-6, "m": 1E-3}
        self.instance = instance  # 参数
        self.mismatch = mismatch
        self.script_content = ("""#! /bin/bash\nsource """ + bash_path + """\ncd """ +
                               default_simPath + """\nalps """ + netlist)
        self.Length_dict = None
        self.Width_dict = None
        self.simLength_dict = None
        self.simWidth_dict = None
        self.instance_dict = None
        self.simNetlist = None
        self.mc0 = None
        self.mismatch_dict = None
        self.instance_num = 0

    def load_Design(self):
        start_flag = 0
        output_lines = []
        instance_dict = {}
        Length_dict = {}
        Width_dict = {}
        try:
            # with open(self.simNetlist_path, 'w') as file:
            #     pass
            with open(self.simNetlist_path, 'w') as file_sim:
                with open(self.netlist_path, 'r') as file:
                    for line in file:
                        if ".option" in line:  # select the second mode
                            if ".option m" in line and line.startswith("*"):
                                line = line.lstrip("*")
                            if (not ".option m" in line) and (not line.startswith("*")):
                                line = "*" + line

                        file_sim.writelines(line)

                        if start_flag == 0:
                            output_lines.append(line)
                            if "schematic" in line:
                                start_flag = 1
                                continue

                        if start_flag == 1:
                            updated_line = line
                            if "l=" in line and "w=" in line:
                                for model in self.instance:
                                    if model in line:
                                        self.instance_num += 1
                                        instance_name = line.split(" ")[0].strip()
                                        instance_dict[instance_name] = model

                                        length_match = re.search(r'l=([0-9.eE+-]+)', line)
                                        width_match = re.search(r'w=([0-9.eE+-]+)', line)

                                        if length_match and width_match:
                                            length_value = length_match.group(1)
                                            width_value = width_match.group(1)

                                            length_value = float(length_value)
                                            width_value = float(width_value)

                                            Length_dict[instance_name] = length_value
                                            Width_dict[instance_name] = width_value

                                            # updated_line = re.sub(r"l=\S+", f"l=Lholder", updated_line)
                                            # updated_line = re.sub(r"w=\S+", f"w=Wholder", updated_line)
                            output_lines.append(updated_line)

                self.simNetlist = output_lines
                self.Width_dict = Width_dict
                self.Length_dict = Length_dict
                self.instance_dict = instance_dict
                # print(Length_dict)
                # print(Width_dict)
                # print(instance_dict)
        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        try:
            self.run_sim()
        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        start_flag = 0
        output_lines = []
        misNames = []
        misMatches = []
        try:
            with open(self.mc0_path, 'r') as file:
                for line in file:
                    if start_flag == 0:
                        output_lines.append(line)
                        if "index" in line:
                            start_flag = 1
                            continue

                    if start_flag == 1:
                        output_lines.append(line)
                        if "index" in line:
                            start_flag = 2
                            continue
                        for mismatchType in self.mismatch:
                            if mismatchType in line:
                                # pattern = r'xi0\.([a-zA-Z0-9]+)'
                                pattern = r'xm([0-9]+)'
                                match = re.search(pattern, line).group(1)
                                misName = match + '_' + mismatchType
                                misNames.append(misName)

                    if start_flag == 2:
                        values = list(map(float, line.split()))
                        misMatches.append(values[1:])

                data_transposed = list(zip(*misMatches))
                mismatch_dict = {misNames: col for misNames, col in zip(misNames, data_transposed)}
                # print(data_transposed)
                # print(mismatch_dict)
                self.mc0 = output_lines
                self.mismatch_dict = mismatch_dict
                # print(self.mc0)

        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

    def sim_with_givenParam(
            self,
            design_para: Tensor = torch.zeros(22),
            mismatch: Tensor = torch.zeros(44),
            resKeys: List[str] = ["dc_gain", "pm", "gbw"],
            load_design_param: bool = False,
    ):
        if load_design_param:
            if mismatch.ndim == 2 or mismatch.ndim == 3:
                if mismatch.ndim == 2:
                    mismatch = torch.unsqueeze(mismatch, dim=0)
                num_of_sim = mismatch.shape[1]
                sim_res = self.sim_with_givenParam_one(load_design_param=True,
                                                       mismatch=mismatch[0],
                                                       resKeys=resKeys,
                                                       num_of_sim=num_of_sim
                                                       )
                return sim_res
            else:
                print(f"Wrong mismatch dim! One dim design_param with {mismatch.ndim} dim mismatch!")
                return 0
        else:
            sim_num = design_para.shape[0]
            # if (mismatch.dim() == 2 and sim_num !=1) or (mismatch.dim() == 3 and sim_num != mismatch.shape[0]):
            if (mismatch.ndim == 2 and sim_num != 1) or (mismatch.ndim == 3 and sim_num != mismatch.shape[0]):

                print(f"Wrong mismatch dim! {sim_num} dim design_param with {mismatch.ndim} dim mismatch!")
                return 0
            else:
                if mismatch.ndim == 2:
                    mismatch = torch.unsqueeze(mismatch, dim=0)
                num_of_sim = mismatch.shape[1]
                sim_res = self.sim_with_givenParam_one(param=design_para[0, :],
                                                       mismatch=mismatch[0],
                                                       resKeys=resKeys,
                                                       num_of_sim=num_of_sim
                                                       )
                for i in range(1, sim_num):
                    sim_temp = self.sim_with_givenParam_one(param=design_para[i, :],
                                                            mismatch=mismatch[i],
                                                            resKeys=resKeys,
                                                            num_of_sim=num_of_sim
                                                            )
                    for j in range(len(sim_res)):
                        sim_res[j] = torch.cat((sim_res[j], sim_temp[j]), dim=0)
                return sim_res

    def sim_with_givenParam_one(
            self,
            mismatch,
            param: Tensor = torch.zeros(20),
            resKeys: List[str] = ["dc_gain", "pm", "gbw"],
            num_of_sim: int = 1,
            load_design_param: bool = False,
    ):
        start_flag = 0
        output_lines = []
        try:
            index = 0
            for line in self.simNetlist:
                if "monte=" in line:
                    line = re.sub(r"monte=\S+", f"monte={num_of_sim}", line)

                if ".option" in line:  # select the third mode
                    if ".option m" in line and (not line.startswith("*")):
                        line = "*" + line
                    if ".option g" in line and (not line.startswith("*")):
                        line = "*" + line
                    if ".option s" in line:
                        line = '.option s_mc_param_input_file=\"' + self.simMc0_path + '\"\n'

                if start_flag == 0 and "schematic" in line:
                    start_flag = 1
                    output_lines.append(line)
                    continue

                if start_flag == 1:
                    updated_line = line
                    if not load_design_param:
                        if "l=" in line and "w=" in line:
                            for model in self.instance:
                                if model in line:
                                    inst_name = line.split(" ")[0].strip()
                                    length = float("{:.2e}".format(param[index]))
                                    width = float("{:.2e}".format(param[(index + self.instance_num)]))
                                    index += 1
                                    ad, _as, pd, ps, nrd, nrs = pch_mis(length, width)
                                    if length and width and ad:
                                        updated_line = re.sub(r"l=\S+", f"l={length}", updated_line)
                                        updated_line = re.sub(r"w=\S+", f"w={width}", updated_line)
                                        updated_line = re.sub(r"ad=\S+", f"ad={ad}", updated_line)
                                        updated_line = re.sub(r"as=\S+", f"as={_as}", updated_line)
                                        updated_line = re.sub(r"pd=\S+", f"pd={pd}", updated_line)
                                        updated_line = re.sub(r"ps=\S+", f"ps={ps}", updated_line)
                                        updated_line = re.sub(r"nrd=\S+", f"nrd={nrd}", updated_line)
                                        updated_line = re.sub(r"nrs=\S+", f"nrs={nrs}", updated_line)
                                    break
                    output_lines.append(updated_line)
                else:
                    output_lines.append(line)

            with open(self.simNetlist_path, 'w') as file:
                file.writelines(output_lines)
        except IndexError as e:
            print(f"IndexError: {e}")
        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        output_lines = []
        try:
            for line in self.mc0:
                output_lines.append(line)
            # for index in range(cols):
            mismatch_data = Write_mismath(mismatch, index=num_of_sim)
            output_lines.extend(mismatch_data)

            with open(self.simMc0_write_path, 'w') as file:
                file.writelines(output_lines)

        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        try:
            self.run_sim()
        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        try:
            rows = len(resKeys)
            simRes = [torch.zeros(num_of_sim, 1) for _ in range(rows)]
            col = -1
            with open(self.measureRes_path, 'r') as file:
                for line in file:
                    if "Measure Result" in line:
                        col += 1

                    for key in resKeys:
                        if key in line:
                            value = line.split('=')[1].strip()
                            if value != 'nan':
                                float_value = float(value)
                                simRes[resKeys.index(key)][col, 0] = float_value
            return simRes

        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

    def run_sim(self):
        script_path = './simulation/sim.sh'
        with open(script_path, 'w') as file:
            file.write(self.script_content)
        os.chmod(script_path, 0o755)
        command = './simulation/sim.sh > sim.log'
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.wait()

        if process.returncode != 0:
            print(f'Error executing {command}')
        else:
            print(f'Successfully executed {command}')

# initial set
rootPath = "/home/liyunqi/CornerYO/"  # set your work dir, the results will save here
bash_path = "/home/liyunqi/CornerYO/setup.bash"  # where the setup.bash to be load
netlist = "opamp_spice_new"  # set your spice file name
instance = ["pch_mis", "nch_mis",
            "g45n1svt", "g45p1svt"]
mismatch = ["toxnmis", "toxpmis",
            "vthnmis", "vthpmis",
            "dwnmis", "dwpmis",
            "dlnmis", "dlpmis"]
key = ["dc_gain", "pm", "gbw"]
# if netlist == "sram_spice":
#     key = ["delay", "iread"]
def simulation(design_param, variation):

    # start simulation
    test = YADO_Alps(netlist, rootPath, bash_path, instance, mismatch)
    test.load_Design()
    simres = test.sim_with_givenParam(design_para=design_param,
                                      mismatch=variation
                                      )
    return simres

def create_alps_simulator():
    alps = YADO_Alps(netlist=netlist,
                     rootPath=rootPath,
                     bash_path=bash_path,
                     instance=instance,
                     mismatch=mismatch)
    alps.load_Design()
    return alps

# test the function

if __name__ == '__main__':
    # x (1,22) v(1,100,44)
    num_of_x_mc = 1  # number of x mc actually will always be 1
    num_of_v_mc_sim = 10  # number of v mc in one x
    design_param = torch.rand(num_of_x_mc, 2 * 11) * (1e-6 - 1e-7) + 3e-7  # get x
    mismatch_mc = torch.randn(num_of_x_mc, num_of_v_mc_sim, 4 * 11)
    start = time.time()
    sim_results = simulation(design_param=design_param, variation=mismatch_mc)
    last = time.time() - start
    print(f"10v time:{last}")

    # print(sim_results)

# if __name__ == '__main__':
#     # initial set
#     rootPath = "/home/liyunqi/CornerYO/"  # set your work dir, the results will save here
#     bash_path = "/home/liyunqi/CornerYO/setup.bash"  # where the setup.bash to be load
#     netlist = "opamp_spice_new" # set your spice file name
#     instance = ["pch_mis", "nch_mis",
#                 "g45n1svt", "g45p1svt"]
#     mismatch = ["toxnmis3", "toxpmis3",
#                 "vthnmis3", "vthpmis3",
#                 "dwnmis3", "dwpmis3",
#                 "dlnmis3", "dlpmis3"]
#     num_of_sim = 1             # set the num of simulation for one design param
#     design_param_num = 10        # num of design param sim (num of sim)
#
#     # start simulation
#     test = YADO_Alps(netlist, rootPath, bash_path, instance, mismatch)
#     test.load_Design()
#
#     # prepare the sim data for x & v
#     torch.manual_seed(2)
#     random_param = torch.rand(22, design_param_num) * (1e-6 - 1e-7) + 3e-7
#     # random_param[:, 0] = random_param[:, 1] * torch.rand(1) * 0.2 + 2e-7
#     random_param = random_param.T
#     random_mismatch = torch.randn(design_param_num, num_of_sim, 11*4)
#     # print(random_param)
#     # print(random_mismatch)
#
#     if netlist == "opamp_spice_new":
#         key = ["dc_gain", "pm", "gbw"]
#     if netlist == "sram_spice":
#         key = ["delay", "iread"]
#     simres = test.sim_with_givenParam(design_para=random_param, mismatch=random_mismatch, num_of_sim=num_of_sim, resKeys=key)
#     # simres = test.sim_with_givenParam(
#     #     load_design_param=True,
#     #     mismatch=random_mismatch,
#     #     num_of_sim=num_of_sim,
#     #     resKeys=key
#     # )
#     print(simres)
