import json
from pathlib import Path
from itertools import repeat
from collections import OrderedDict
from functools import reduce
from operator import getitem
import wandb
import torch
import os


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    Setup GPU device if available, move model into configured device
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def set_by_path(tree, keys, value):
    '''Set a value in a nested object in tree by sequence of keys.'''
    keys = keys.split(';')
    get_by_path(tree, keys[:-1])[keys[-1]] = value


def get_by_path(tree, keys):
    '''Access a nested object in tree by sequence of keys.'''
    return reduce(getitem, keys, tree)


def msg_box(msg):
    row = len(msg)
    h = ''.join(['+'] + ['-' * row] + ['+'])
    result = h + f"\n|{msg}|\n" + h
    return result

def wandb_save_code(config, base_path='../'):
    """
    Save all files associated with a run to a wandb run
    """
    GLOBAL_FILES = ["models/loss.py",
                    "models/metric.py",
                    "utils/util.py",
                    "base/base_dataloader.py",
                    "base/base_model.py",
                    "base/base_trainer.py",
                    "parse_config.py"]
    
    # Save config file
    wandb.save(str(Path(config.run_args.config)), base_path=base_path)
    
    # Save top-level python scripts
    for file in os.listdir("./"):
        if file.endswith(".py"):
            wandb.save(file, base_path=base_path)
    
    # Save dataset code
    datasets = config.config['datasets']
    for key, value in datasets['train'].items():
        if 'module' in datasets['train'][key]:
            wandb.save(f"data_loaders{datasets['train'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
    for key, value in datasets['valid'].items():
        if 'module' in datasets['valid'][key]:
            wandb.save(f"data_loaders{datasets['valid'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
    for key, value in datasets['test'].items():
        if 'module' in datasets['test'][key]:
            wandb.save(f"data_loaders{datasets['test'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
    
    # Save dataloader code
    data_loaders = config.config['data_loaders']
    for key, value in data_loaders['train'].items():
        if 'module' in data_loaders['train'][key]:
            wandb.save(f"data_loaders{data_loaders['train'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
    for key, value in data_loaders['valid'].items():
        if 'module' in data_loaders['train'][key]:
            wandb.save(f"data_loaders{data_loaders['train'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
    for key, value in data_loaders['test'].items():
        if 'module' in data_loaders['train'][key]:
            wandb.save(f"data_loaders{data_loaders['train'][key]['module']}".replace(".", "/") + ".py", base_path=base_path)
            
    # Save model code
    models = config.config['models']
    for key, value in models.items():
        if 'module' in models[key]:
            wandb.save(f"models{models[key]['module']}".replace(".", "/") + ".py", base_path=base_path)
            
    # Save trainer
    wandb.save(f"trainers{config.config['trainer']['module']}".replace(".", "/") + ".py", base_path=base_path)
    
    # Save global files for all runs
    for file in GLOBAL_FILES:
        wandb.save(file, base_path=base_path)