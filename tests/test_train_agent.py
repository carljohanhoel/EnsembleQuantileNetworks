import unittest
import multiprocessing as mp
import time
from shutil import copy
import psutil

import sys
sys.path.append('../src')


def exec_train_agent():
    exec(open("../src/train_agent.py").read())


class Tester(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.test_duration = 30
        copy('../src/parameters.py', 'tmp_parameters_original.py')

    def run_train_agent(self):
        process = mp.Process(target=exec_train_agent)
        process.start()
        time.sleep(self.test_duration)
        if process.is_alive():
            for child_process in psutil.Process(process.pid).children():
                if child_process._name != 'sumo':
                    child_process.send_signal(15)
            process.terminate()
        else:
            raise Exception('Process died')
        process.join()

    def test_dqn(self):
        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("agent_par['env'] = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = False\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = False\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

    def test_ensemble(self):
        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = True\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = False\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = True\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

    def test_iqn(self):
        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = False\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = False\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

    def test_iqn_ensemble(self):
        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = False\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = True\n")
        f.writelines("agent_par['cnn'] = False\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')

        f = open('../src/parameters.py', 'a')
        f.write('\n')
        f.writelines("env = 'intersection'\n")
        f.writelines("agent_par['distributional'] = True\n")
        f.writelines("agent_par['ensemble'] = True\n")
        f.writelines("agent_par['parallel'] = True\n")
        f.writelines("agent_par['cnn'] = True\n")
        f.writelines("agent_par['learning_starts'] = 500\n")
        f.close()
        try:
            self.run_train_agent()
        finally:
            copy('tmp_parameters_original.py', '../src/parameters.py')


if __name__ == '__main__':
    try:
        unittest.main()
    finally:
        copy('tmp_parameters_original.py', '../src/parameters.py')
