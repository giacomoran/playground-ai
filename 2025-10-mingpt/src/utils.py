import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

#:


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)


def setup_logging(config):
    """monotonous bookkeeping"""
    work_dir = config.system.work_dir
    # create the work directory if it doesn't already exist
    os.makedirs(work_dir, exist_ok=True)
    # log the args (if any)
    with open(os.path.join(work_dir, "args.txt"), "w") as f:
        f.write(" ".join(sys.argv))
    # log the config itself
    with open(os.path.join(work_dir, "config.json"), "w") as f:
        f.write(json.dumps(dict(config), indent=4))


def merge_from_args(cfg, args):
    """
    Merge command-line arguments into config.
    Expects args in the format: key=value key.nested.value=value
    """
    for arg in args:
        if "=" not in arg:
            continue
        key, value = arg.split("=", 1)

        # Try to convert value to appropriate type
        try:
            # Try literal_eval for lists, tuples, dicts, bools, ints, floats
            value = literal_eval(value)
        except (ValueError, SyntaxError):
            # Keep as string if literal_eval fails
            pass

        # Set nested keys using dot notation
        keys = key.split(".")
        node = cfg
        for k in keys[:-1]:
            node = getattr(node, k)
        setattr(node, keys[-1], value)

    return cfg
