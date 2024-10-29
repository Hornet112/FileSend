import os
import re
import subprocess
import sys

import torch
from typing import List
from torch import Tensor


def transistor_para_018(l, w, hdifp0=2.4e-7, hdifp1=2.7e-7, multi=1):
    ad = float("{:.2e}".format(2 * hdifp0 * w))
    _as = ad
    pd = float("{:.2e}".format(2 * (2 * hdifp0 + w)))
    ps = pd
    nrd = float("{:.2e}".format(hdifp1 / w))
    nrs = nrd
    return ad, _as, pd, ps, nrd, nrs


def write_mismatch(mismatch: Tensor, index: int):
    output_lines = []
    for i in range(index):
        line = f"{i + 1}"
        for data in mismatch[i]:
            line += " " + f"{data}"
        line += "\n"
        output_lines.append(line)
    return output_lines


class AlpsInterface:
    def __init__(self, netlist, root_path, bash_path, instance):
        if not os.path.isdir(root_path):  #
            raise NotADirectoryError(f"Path '{root_path}' is not valid.")

        default_alps_template_netlist = os.path.join(root_path, "ALPS_SPICE/")  # default path of template netlist
        default_sim_dir = os.path.join(root_path, "simulation")  # default simulation path
        os.makedirs(default_sim_dir, exist_ok=True)
        os.makedirs(default_alps_template_netlist, exist_ok=True)

        self.netlist_path = default_alps_template_netlist + netlist + "/" + netlist
        self.sim_netlist_dir = default_sim_dir  # +netlist
        self.mc0_path = self.sim_netlist_dir + "/" + netlist + ".sim/" + netlist + ".mc0"

        self.sim_netlist_path = self.sim_netlist_dir + "/" + netlist
        self.sim_mc0_path = netlist + "_sim.mc0"
        self.sim_mc0_write_path = self.sim_netlist_dir + "/" + self.sim_mc0_path
        self.measure_res_path = self.sim_netlist_dir + "/" + netlist + ".sim/" + netlist + ".measure"

        self.unit_dict = {"n": 1E-9, "u": 1E-6, "m": 1E-3}
        self.instance = instance  # 参数
        self.script_content = """#! /bin/bash\nsource """ + bash_path + """\ncd """ + default_sim_dir + """\nalps """ + netlist

        self.sim_netlist = None
        self.mc0 = None
        self.mismatch_dict = None
        self.instance_num = 0
        self.mismatch_para_num = 0
        self.design_para_num = 0

    def load_design(self):
        start_flag = 0
        netlist_template = []
        try:
            with open(self.sim_netlist_path, 'w') as file_sim:
                with open(self.netlist_path, 'r') as file_template:
                    for line in file_template:
                        if ".option" in line:  # select the second mode
                            if ".option mc_file_only" in line and line.startswith("*"):
                                line = line.lstrip("*")
                            if (".option s_mc_param_input_file" in line) and (not line.startswith("*")):
                                line = "*" + line
                            if (".option generate_mc_param_format_file" in line) and (not line.startswith("*")):
                                line = "*" + line

                        file_sim.writelines(line)  # write net list to generate mc0 file

                        updated_line: str = line
                        if "l=" in line and "w=" in line:
                            for model in self.instance:
                                if model in line:
                                    self.instance_num += 1
                                    self.design_para_num += 2
                        netlist_template.append(updated_line)

                self.sim_netlist = netlist_template
        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

        try:
            self.run_sim()
        except FileNotFoundError:
            print("template netlist not found, load design failed")
        except Exception as e:
            print("出现错误：", e)

        start_flag = 0
        mc0_headlines = []
        try:
            with open(self.mc0_path, 'r') as file_template:
                for line in file_template:
                    if start_flag == 0:
                        mc0_headlines.append(line)
                        if "index" in line:
                            start_flag = 1
                            continue

                    if start_flag == 1:
                        mc0_headlines.append(line)
                        self.mismatch_para_num += 1
                        if "index" in line:
                            self.mismatch_para_num -= 1
                            start_flag = 2
                            continue

                self.mc0 = mc0_headlines

        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

    print('successfully loaded')

    def sim_with_given_param(
            self,
            mismatch: Tensor,
            design_para: Tensor or None,
            res_keys: list,
            load_design_param: bool = False,
    ):
        """
        :param mismatch: n * m * self.mismatch_para_num
        :param design_para: design parameters, None or n * self.design_para_num Tensor
        :param res_keys: key name of measurement
        :param load_design_param: whether using template design parameters
        :return:
        """
        if load_design_param:
            if mismatch.ndim == 2 or mismatch.ndim == 3:
                if mismatch.ndim == 2:
                    mismatch = torch.unsqueeze(mismatch, dim=0)

                if mismatch.shape[2] != self.mismatch_para_num:
                    raise Exception('Mismatch parameters do not match the template')

                sim_res = self.sim_with_given_param_one(load_design_param=True, design_para=None, mismatch=mismatch[0],
                                                        res_keys=res_keys)
                return sim_res
            else:
                print(f"Wrong mismatch dim! One dim design_param with {mismatch.ndim} dim mismatch!")
                raise Exception('Wrong mismatch dim')
        else:
            if design_para is None:
                raise Exception('Design parameters must be Tensor when not loading design')

            sim_num = design_para.shape[0]
            if (mismatch.ndim == 2 and sim_num == 1) or (mismatch.ndim == 3 and sim_num == mismatch.shape[0]):
                if mismatch.ndim == 2:
                    mismatch = torch.unsqueeze(mismatch, dim=0)

                if design_para.shape[1] != self.design_para_num or mismatch.shape[2] != self.mismatch_para_num:
                    raise Exception('Design parameters and mismatch parameters do not match the template')

                sim_res = self.sim_with_given_param_one(design_para=design_para[0, :], mismatch=mismatch[0],
                                                        res_keys=res_keys, )
                for i in range(1, sim_num):
                    sim_temp = self.sim_with_given_param_one(design_para=design_para[i, :], mismatch=mismatch[i],
                                                             res_keys=res_keys)
                    for j in range(len(sim_res)):
                        sim_res[j] = torch.cat((sim_res[j], sim_temp[j]), dim=0)
                return sim_res
            else:
                print(f"Wrong mismatch dim! {sim_num} dim design_param with {mismatch.ndim} dim mismatch!")
                raise Exception('Wrong mismatch dim')

    def sim_with_given_param_one(
            self,
            mismatch: Tensor,
            design_para: Tensor or None,
            res_keys: List[str],
            load_design_param: bool = False,
    ):
        """
        :param mismatch: m * self.mismatch_para_num
        :param design_para: design parameters, None or self.design_para_num Tensor
        :param res_keys: key name of measurement
        :param load_design_param: whether using template design parameters
        :return:
        """
        num_of_sim = mismatch.shape[0]
        output_lines = []
        try:
            index = 0
            for line in self.sim_netlist:
                if "monte=" in line:
                    line = re.sub(r"monte=\S+", f"monte={num_of_sim}", line)

                if ".option" in line:  # select the third mode
                    if ".option mc_file_only" in line and (not line.startswith("*")):
                        line = "*" + line
                    if ".option generate_mc_param_format_file" in line and (not line.startswith("*")):
                        line = "*" + line
                    if ".option s_mc_param_input_file" in line:
                        line = '.option s_mc_param_input_file=\"' + self.sim_mc0_path + '\"\n'

                updated_line = line
                if not load_design_param:
                    if "l=" in line and "w=" in line:
                        for model in self.instance:
                            if model in line:
                                length = float("{:.2e}".format(design_para[index]))
                                width = float("{:.2e}".format(design_para[(index + self.instance_num)]))
                                index += 1
                                ad, _as, pd, ps, nrd, nrs = transistor_para_018(length, width)  # need better
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

            with open(self.sim_netlist_path, 'w') as file:
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

            mismatch_data = write_mismatch(mismatch, index=num_of_sim)
            output_lines.extend(mismatch_data)

            with open(self.sim_mc0_write_path, 'w') as file:
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
        # print('run_sim2 done')

        try:
            rows = len(res_keys)
            simRes = [torch.zeros(num_of_sim, 1) for _ in range(rows)]
            col = -1
            with open(self.measure_res_path, 'r') as file:
                for line in file:
                    if "Measure Result" in line:
                        col += 1

                    for key in res_keys:
                        if key in line:
                            value = line.split('=')[1].strip()
                            if value != 'nan':
                                try:
                                    float_value = float(value)
                                except Exception:
                                    if value.endswith('p'):
                                        float_value = float(value[:-1]) * 1e-12
                                    if value.endswith('n'):
                                        float_value = float(value[:-1]) * 1e-9

                                simRes[res_keys.index(key)][col, 0] = float_value
            return simRes

        except FileNotFoundError:
            print("netlist not found")
        except Exception as e:
            print("出现错误：", e)

    def run_sim(self):
        script_path = self.sim_netlist_dir + '/sim.sh'
        with open(script_path, 'w') as file:
            file.write(self.script_content)
        os.chmod(script_path, 0o755)
        command = self.sim_netlist_dir + '/sim.sh > sim.log'
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.wait()

        if process.returncode != 0:
            print(f'Error executing {command}')
            sys.exit()
        else:
            # print(f'Successfully executed {command}')
            pass




# if __name__ == '__main__':
#     # initial set
#     rootPath = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/"  # set your work dir, the results will save here
#     bash_path = "/home/liyunqi/PycharmProjects/ActiveLearningIC-main/ALPS/setup.bash"  # where the setup.bash to be load
#     netlist = "demo.sp"  # set your spice file name
#
#     if netlist == "opamp_spice":
#         instance = ["pch_mis", "nch_mis"]
#         key = ["dc_gain", "pm", "gbw"]
#     if netlist == "sram_spice":
#         instance = ["pch3_mis", "nch3_mis"]
#         key = ["delay", "iread"]
#     if netlist == "demo.sp":
#         instance = ["pckt", "nckt"]
#         key = ["ring_delayb_0", "stage_delayb_0"]
#     if netlist == "demo.sp":
#         instance = ["pckt", "nckt"]
#         key = ["ring_delayb_0", "stage_delayb_0"]
#     if netlist == "8sram_array_spice":
#         instance = ["pch_mis", "nch_mis"]
#         key = ["i_blMax"]
#     if netlist == "16sram_array_spice":
#         instance = ["pch_mis", "nch_mis"]
#         key = ["i_blMax"]
#     if netlist == "32sram_array_spice":
#         instance = ["pch_mis", "nch_mis"]
#         key = ["i_blMax"]
#
#     # start simulation
#     test = AlpsInterface(netlist, rootPath, bash_path, instance)
#     test.load_design()
#     sim_num = 10  # simulation num of design param (num of x)
#     v_sim_num = 10  # simulation num of mismatch param for each design param
#     # prepare the sim data for x & v
#     torch.manual_seed(2)
#     random_param = torch.rand(test.design_para_num, sim_num) * (1e-6 - 1e-7) + 3e-7
#     random_param = random_param.T
#     random_mismatch = torch.randn(sim_num, sim_num, test.mismatch_para_num)
#     # print(random_param)
#     # print(random_mismatch)
#
#     # simres = test.sim_with_given_param(design_para=random_param, mismatch=random_mismatch, res_keys=key)
#     simres = test.sim_with_given_param(
#         load_design_param=True,
#         mismatch=random_mismatch,
#         design_para=None,
#         res_keys=key)
#     print(simres)
