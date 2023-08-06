import sys
import matplotlib as mpl
import torch
import numpy as np
mpl.use('Agg')
from config import ConfigObject, build_config_file
import importlib
from typing import List,Dict,Iterable,Set
import os
import json
from collections import defaultdict
import time
from tqdm import tqdm
import torchvision.transforms as T

from helper import AllOf

typenames = {
    "trainer" : "trainer.",
    "network" : "network.",
    "loss" : "loss.",
    "data" : "data.",
    "evaluator" : "evaluator.",
    "logger" : "logger.",
    "tasks" : "task."

}

def mkdir(path):
    if(not os.path.exists(path)):
        os.makedirs(path)


class Runner(ConfigObject):

    def __init__(self, config):
        # unused in runner class see executor for usage
        self.depends : List[str] = []
        self.skip_if_exists : str = None
        self.name : str = "run"
        self.epochs : int = 1000
        self.learning_rate : float = 1e-4
        self.tasks : List["Task"] = []
        self.active_task : "Task" = None
        # self.network = object
        # self.loss = object
        # self.data = object
        name = config.get("name", self.name)
        self.folder = os.path.join("runs", name)
        mkdir(self.folder)
        self.evaluator : "Evaluator" = None
        self.data_path : str = "./"
        self.name = name
        self.perf_test = False

        super().__init__(config)
        if(self.perf_test):
            self.evaluators =  AllOf([])
            self.loggers = AllOf([])
            import logging
            logging.disable(logging.CRITICAL)


    @staticmethod
    def filterdict(config : Dict[str,object]):
        config.pop("runner",None)
        stack : List[Iterable] = [config.values()]
        while len(stack) > 0:
            item = stack.pop()

            for sub_item in item:
                if(isinstance(sub_item,dict)):
                    sub_item.pop("runner",None)
                    stack.append(sub_item.values())
                if isinstance(sub_item,list):
                    stack.append(sub_item)

        return config

    @staticmethod
    def get_config_file(suffix:str, name:str):
        return f".config/{name}_{suffix}.json"

    @staticmethod
    def savehash(suffix: str, config : Dict[str,object]):
        config = Runner.filterdict(config.copy())
        if not os.path.exists(f".config/"):
            os.mkdir(".config")
        filename = Runner.get_config_file(suffix,config["name"])
        file = open(filename,"w")
        file.write(json.dumps(config,skipkeys=True,sort_keys=True))
        file.flush()
        file.close()
        return filename

    @staticmethod
    def comparehash(suffix:str, config : Dict[str,object]) -> bool:
        config = Runner.filterdict(config.copy())
        path = Runner.get_config_file(suffix,config["name"])
        if not os.path.exists(path):
            return False
        file = open(path,"r")
        jsonText = file.read()
        file.close()
        return json.dumps(config,skipkeys=True,sort_keys=True) == jsonText

    @classmethod
    def get_class_for(self, name:str, classname: str):
        basename = typenames[name]
        basename += classname.lower()
        return getattr(importlib.import_module(basename),classname)

    def run(self):

        if(self.perf_test):
                print("Starting Script in testing mode all logging disabled")

        if(self.skip_if_exists is not None and os.path.exists(self.skip_if_exists)):
            return
        if(self.comparehash("runner",self.key_value)):
            self.py_logger.warn(f"Skipping runner {self.name} because config has not changed since last run\n")

        #remove runfile
        runfile = Runner.get_config_file("runner",self.name)

        if(os.path.exists(runfile)):
            os.remove(runfile)

        self.py_logger.info(f"Running runner {self.name}\n")
        current_time = time.time()

        for task in self.tasks:
            self.active_task = task
            task()

        self.py_logger.info(f"Runner {self.name} has finished\n")
        self.savehash("runner",self.key_value)
        if(self.perf_test):
            print(f"Script took {time.time()-current_time}s to compute")


from copy import copy

class argdict(defaultdict):
    #this is a static variable as it is never changed etc
    args : Set[str] = set()
    def __init__(self):
        super().__init__(str)

    def __missing__(self, key):
        self.args.add(key)
        return super().__missing__(key)

    def copy(self):
        d = argdict()
        d.update(self)
        return d


def parse_json_arg(arg:str) -> object:
    if(arg == "null"):
        return None
    if(arg=="true"):
        return True
    if(arg == "false"):
        return False
    if(arg.isnumeric()):
        return int(arg)
    try:
        return float(arg)
    except ValueError:
        return arg

def sphere_tracing(runner, name, N = 512, dist_threshold = 1.0e-3):

    print('Starting Sphere Tracing...')

    time.perf_counter()
    runner.network.cuda()

    pos_cpu = torch.zeros(N * N, 3, requires_grad=True)
    
    path = 'runs/sphere_tracing/'

    with torch.no_grad():
        # rays initialization
        for i in tqdm(range(0, N), 'Rays initialization'):
            for j in range(0, N):
                # Orthogonal projection using Normalized Device Coordinates.
                # if name == 'lucy':
                #     pos_cpu[i * N + j, 0] = i / N - 0.5 
                #     pos_cpu[i * N + j, 1] = 0.6
                #     pos_cpu[i * N + j, 2] = j / N - 0.5
                # else:
                #     pos_cpu[i * N + j, 0] = i / N - 0.5 
                #     pos_cpu[i * N + j, 1] = j / N - 0.5
                    
                #     if name == 'bunny' or name == 'buddha':
                #         pos_cpu[i * N + j, 2] = 0.6
                #     else:
                #         pos_cpu[i * N + j, 2] = -0.6
                pos_cpu[i * N + j, 0] = 2 * i / N - 1.0 
                pos_cpu[i * N + j, 1] = 2 * j / N - 1.0
                pos_cpu[i * N + j, 2] = -0.6

        pos_gpu = pos_cpu.cuda()
        
        # if name == 'lucy':
        #     dir = torch.tensor([0.0, -1.0, 0.0]).cuda()
        # elif name == 'bunny' or name == 'buddha':
        #     dir = torch.tensor([0.0, 0.0, -1.0]).cuda()
        # else:
        #     dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
        #dir = torch.tensor([1.0, 0.0, 0.0]).cuda()

        distances = torch.zeros(N * N, 1).cuda()
        transform = T.ToPILImage()

        start_time = time.perf_counter()

        for j in tqdm(range(0, 100), 'Sphere tracing'):
            # infer distance
            model_input = {'coords': pos_gpu, 'normal_out': None, 'sdf_out': None,
                           'pointcloud': None, 'detach': True, 'istrain': False, 'epoch': 0,
                           'iteration': 0, 'progress' : 0.0, 'force_run': False}
            distances = runner.network(model_input)['sdf']

            # update pos
            pos_gpu = torch.clamp(
                pos_gpu + torch.mul(distances, dir),
                -1.0,
                1.0
            )
        
        img = transform(torch.reshape(distances, (1, N, N)))
        img = img.rotate(90)
        os.makedirs(path, exist_ok=True)
        img.save(path + name + '.png')

    D = torch.cat((distances, distances, distances), dim=-1)

    pbar_str = 'Normals'
    
    # Normals.
    with tqdm(total=200, desc=pbar_str) as pbar:
        result = runner.network(model_input)
        sdf, xyz = result['sdf'], result['detached']
        grad = torch.autograd.grad([sdf], [xyz], torch.ones_like(sdf), retain_graph=False)[0]
        #sdf, xyz, grad = result['base'], result['detached'], result['base_normal']

        pbar.update(70)

        with torch.no_grad():
            # Normalization and distance condition
            grad_norm = torch.linalg.norm(grad, dim=-1)

            unit_grad = grad/grad_norm.unsqueeze(-1)
            unit_grad = torch.abs(unit_grad)

            unit_grad = torch.where(D<=dist_threshold, unit_grad, torch.ones_like(unit_grad))

            pbar.update(30)

    # Rendering
    with torch.no_grad():
        with tqdm(total=100, desc='Rendering') as pbar:
            k_s = 0.5
            k_d = 1.0
            k_a = 1.0
            shininess = 35.0

            ambient = torch.tensor([0.2, 0.2, 0.2]).cuda()
            specular = torch.tensor([1.0, 1.0, 1.0]).cuda()
            diffuse = torch.tensor([0.54, 0.54, 0.54]).cuda()

            # if name == 'lucy':
            #     light_dir = torch.tensor([0.0, 1.0, 0.0]).cuda()
            # else:
            #     light_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
            light_dir = torch.tensor([0.0, 0.0, 1.0]).cuda()
            
            pbar.update(20)

            n = unit_grad
            dot_d = torch.matmul(n, light_dir)
            dot_d = torch.unsqueeze(dot_d, -1)

            reflection = 2.0 * dot_d * n - light_dir

            pbar.update(40)

            dot_d = torch.maximum(dot_d, torch.zeros_like(dot_d))

            dot_s = torch.matmul(reflection, dir)
            dot_s = torch.unsqueeze(dot_s, -1)
            dot_s = torch.maximum(dot_s, torch.zeros_like(dot_s))**shininess

            color = k_a * ambient + k_d * diffuse * dot_d  + k_s * specular * dot_s
            color = torch.clamp(color, max=1.0)

            color = torch.where(D<=dist_threshold, color, torch.ones_like(color))

            pbar.update(40)
    
        frame_time = time.perf_counter() - start_time 

        image_inputs = {'normals' : n, 'colors' : color}

        for key in tqdm(image_inputs, 'Saving results into ./results'):
            transposed = torch.transpose(image_inputs[key], 0, 1)
            
            img = transform(torch.reshape(transposed, (3, N, N)))
            img = img.rotate(90)

            # if multiscale is False and normal_mapping is False:
            #     file_name = '%s_%s_LOD_%d_baseline' % (name, key, lod)
            # else:
            #     file_name = '%s_%s_LOD_%d_multiscale_%s_mapping_%s' % (name, key, lod, multiscale, normal_mapping)

            file_name = '%s_%s' % (name, key)
            img.save(path + file_name + '.png')

        return {'name': file_name, 'time(s)' : frame_time}

if __name__ == "__main__":
    import sys
    if(len(sys.argv) > 1):
        path = sys.argv[1]
    else:
        path = os.path.dirname(__file__) + "/../example.json"

    argDict = argdict()
    force_usage = True

    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    if("--help" not in sys.argv):
        force_usage = False
        currentKey = None
        for arg in sys.argv:
            if(arg.startswith("--")):
                if(currentKey is not None):
                    print(f"stray key {currentKey} ignored!")
                currentKey = arg[2:]

            elif(currentKey is not None):
                argDict[currentKey] = parse_json_arg(arg)
                currentKey = None
            else:
                print(f"stray value {arg} ignored!")

        if(currentKey is not None):
            print(f"stray key {currentKey} ignored!")

    print("\nParsed Arguments:")
    print(argDict.items())
    print("")
    folder = os.path.dirname(path)
    file = path[len(folder):]
    json_obj = build_config_file(file,folder,argDict)


    if(force_usage or len(argdict.args)>0):
        folder = os.path.dirname(path)
        file = path[len(folder):]
        json_obj = build_config_file(file, folder, argDict)
        if( not force_usage):
            print("Not all arguments defined, missing arguments:")
        else:
            print("Usage\n")
            print("Help for script:")
            print(f"{sys.argv[1]}\n")
        for k in argDict.args :
            print(f"--{k} <{k}>")
        print("")
        print("All arguments must be set no defaults! ")
        print("")

        sys.exit(0)

    if(isinstance(json_obj,list)):
        print("")
        print(f"script {path} is an array, try running it with executer.py!")
        print("")
        sys.exit(0)

    ##stop it before it creates any additional files we want to be able to rerun
    for d in json_obj.get("depends",[]):
        runfile = Runner.get_config_file("runner", d)
        if not os.path.exists(runfile):
            print(f"Runner {json_obj['name']} can't run because job {d} did not finish", file=sys.stderr)
            sys.exit(-1)

    runner = Runner(json_obj)

    runner.run()
    sphere_tracing(runner, name='armadillo')
    
    sys.exit(0)
