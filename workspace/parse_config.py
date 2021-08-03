import os
import argparse
import logging
from pathlib import Path
from functools import partial
from datetime import datetime
import importlib

from logger import setup_logging
from utils import ensure_dir, read_json, write_json, set_by_path, get_by_path


class ConfigParser:
    def __init__(self, run_args, modification=None):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param run_args: Dict, running arguments including resume, mode, run_id, log_name.
            - config: String, path to the config file.
            - resume: String, path to the checkpoint being loaded.
            - mode: String, 'train', 'test' or 'inference'.
            - run_id: Unique Identifier for training processes. Used to save checkpoints and training log.
                     Timestamp is being used as default
            - log_name: Change info.log into <log_name>.log.
        :param modification: Dict {keychain: value}, specifying position values to be replaced from config dict.
        """
        # run_args
        self.run_args = run_args
        # load config file and apply modification
        config_json = run_args.config
        config = read_json(Path(config_json))
        self._config = _update_config(config, modification)
        self.resume = Path(run_args.resume) if run_args.resume is not None else None
        self.mode = run_args.mode
        log_name = run_args.log_name

        self.root_dir = self.config['root_dir']
        run_id = run_args.run_id

        save_name = {'train': 'saved/', 'test': 'output/'}
        save_dir = Path(self.root_dir) / save_name[self.mode]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        exp_dir = save_dir / self.config['name'] / run_id

        dirs = {'train': ['log', 'model', 'metrics_best'], 'test': ['log', 'metric', 'fig']}
        self.save_dir = dict()
        for dir_name in dirs[self.mode]:
            dir_path = exp_dir / dir_name
            ensure_dir(dir_path)
            self.save_dir[dir_name] = dir_path

        log_config = {}
        if self.mode == 'train':
            fold_idx = self.config['trainer'].get('fold_idx', 0)
            if fold_idx > 0:
                # multiprocessing is enabled.
                log_config.update({'log_config': 'logger/logger_config_mp.json'})
            if fold_idx <= 1:
                # backup config file to the experiment dirctory
                write_json(self.config, exp_dir / os.path.basename(config_json))

        # configure logging module
        setup_logging(self.save_dir['log'], root_dir=self.root_dir, filename=log_name, **log_config)

    @classmethod
    def from_args(cls, parser, options=''):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        args = parser.parse_args()

        msg_no_cfg = "Configuration file need to be specified. Add '-c config.json', for example."
        assert args.config is not None, msg_no_cfg
        
        if args.device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.device

        modification = None
        for group in parser._action_groups:
            if group.title == 'mod_args':
                # parse custom cli options into dictionary
#                 modification = {opt.target: getattr(args, _get_opt_name(opt.flags)) for opt in options}
                modification = {}
                for opt in options:
                    if isinstance(opt.target, list):
                        for target in opt.target:
                            modification[target] = getattr(args, _get_opt_name(opt.flags))
                    else:
                        modification[opt.target] = getattr(args, _get_opt_name(opt.flags))
            else:
                group_dict = {g.dest: getattr(args, g.dest, None) for g in group._group_actions}
                arg_group = argparse.Namespace(**group_dict)
                if group.title == 'run_args':
                    run_args = arg_group
                elif group.title == 'test_args':
                    cls.test_args = arg_group

        return cls(run_args, modification)

    @staticmethod
    def _update_kwargs(_config, kwargs):
        try:
            _kwargs = dict(_config['kwargs'])
        except KeyError:  # In case no arguments are specified
            _kwargs = dict()
        assert all([k not in _kwargs for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        _kwargs.update(kwargs)
        return _kwargs

    def init_obj(self, keys, module, *args, **kwargs):
        """
        Returns an object or a function, which is specified in config[keys[0]]...[keys[-1]].
        In config[keys[0]]...[keys[-1]],
            'is_ftn': If True, return a function. If False, return an object.
            'module': The module of each instance.
            'type': Class name.
            'kwargs': Keyword arguments for the class initialization.
        keys is the list of config entries.
        module is the package module.
        Additional *args and **kwargs would be forwarded to obj()
        Usage: `objects = config.init_obj(['A', 'B', 'C'], module, a, b=1)`
        """
        obj_config = get_by_path(self, keys)
        try:
            module_name = obj_config['module']
            module_obj = importlib.import_module(module_name, package=module)
        except KeyError:  # In case no 'module' is specified
            module_obj = module
        class_name = obj_config['type']
        obj = getattr(module_obj, class_name)
        kwargs_obj = self._update_kwargs(obj_config, kwargs)

        if obj_config.get('is_ftn', False):
            return partial(obj, *args, **kwargs_obj)
        return obj(*args, **kwargs_obj)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    # read-only attributes
    @property
    def config(self):
        return self._config


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for key, value in modification.items():
        if value is not None:
            set_by_path(config, key, value)
    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith('--'):
            return flg.replace('--', '')
    return flags[0].replace('--', '')
