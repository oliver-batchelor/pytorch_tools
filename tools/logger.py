# import numpy as np
import time

import os.path as path
import os

from tools import Histogram, struct, to_dicts, to_structs

import torch
import json
import numpy as np

def create_dir(dir):
    if not path.isdir(dir):
        os.makedirs(dir)


def backup_experiment(log_path, name):
    output_path = os.path.join(log_path, name)
    backup_path = os.path.join(log_path, ".backup")  

    exists = path.exists(output_path) and len(os.listdir(output_path)) > 0

    if exists:
        create_dir(backup_path)

        backup_name = enumerate_name(name, os.listdir(backup_path))
        dest_path = os.path.join(backup_path, backup_name)

        print("backing up: {} -> {}".format(output_path,  dest_path))
        os.rename(output_path, dest_path)        

def make_experiment(log_path, name, dry_run=False, load=False):
    from tensorboard_logger import TensorboardLogger

    output_path = os.path.join(log_path, name)

    if not dry_run:
        if not load:
            backup_experiment(log_path, name)

        create_dir(output_path)
        print("writing experiment to " + output_path)

        logger = CompositeLogger(
            JsonLogger(path.join(output_path, "log.json")),
            TensorboardLogger(output_path))

        return output_path, logger
    else:
        return output_path, NullLogger()


def enumerate_name(base, names):
    i = 1
    name = base
    while name in names:
        name = base + str(i)
        i = i + 1

    return name

class NullLogger:
    def __init__(self):
        pass


    def scalar(self, tag, value):
        pass

    def scalars(self, tag, value):
        pass

    def pr_curve(self, tag, curve):
        pass

    def histogram(self, tag, histogram):
        pass  

    def flush(self):
        pass


class EpochLogger:
    def __init__(self, logger, step):
        self.logger = logger
        self.step = step

    def scalar(self, tag, value):
        self.logger.scalar(tag, value, step=self.step)

    def scalars(self, tag, value):
        self.logger.scalars(tag, value, step=self.step)

    def pr_curve(self, tag, curve):
        self.logger.pr_curve(tag, curve, step=self.step)

    def histogram(self, tag, histogram):
        self.logger.histogram(tag, histogram, step=self.step)    

    def flush(self):
        self.logger.flush()


class CompositeLogger:
    def __init__(self, *loggers):
        self.loggers = loggers

    def scalar(self, tag, value, step):
        for logger in self.loggers:
            logger.scalar(tag, value, step)

    def scalars(self, tag, value, step):
        for logger in self.loggers:
            logger.scalars(tag, value, step)

    def pr_curve(self, tag, curve, step):
        for logger in self.loggers:
            logger.pr_curve(tag, curve, step)

    def histogram(self, tag, histogram, step):
        for logger in self.loggers:
            logger.histogram(tag, histogram, step)

    def flush(self):
       for logger in self.loggers:
            logger.flush()


class JsonLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.entries = {}   

        self.file = open(self.log_file, "a")


    def flush(self):
        self.file.flush()     
        

    def scalar(self, tag, value, step):
        self.append_entry(tag, value, step)


    def scalars(self, tag, value_dict, step):
        self.append_entry(tag, value_dict, step)
                    
    def pr_curve(self, tag, curve, step):
        self.append_entry(tag, curve, step)

    def histogram(self, tag, histogram, step):
        assert isinstance(histogram, Histogram)
        self.append_entry(tag, histogram.to_struct(), step)

    # Internal
    def append_entry(self, tag, value, step):    
        entry = struct(tag=tag, value=value, step=step, time=time.time())

        data = json.dumps(to_dicts(entry))



