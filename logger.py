# import numpy as np
import time
import tensorflow as tf

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
    output_path = os.path.join(log_path, name)

    if not dry_run:
        if not load:
            backup_experiment(log_path, name)

        create_dir(output_path)
        print("writing experiment to " + output_path)

        logger = JsonLogger(path.join(output_path, "log.json"))
        return output_path, logger
    else:
        return output_path, Null()


def enumerate_name(base, names):
    i = 1
    name = base
    while name in names:
        name = base + str(i)
        i = i + 1

    return name


class CompositeLogger:
    def __init__(self, loggers):
        self.loggers = loggers


    def begin_step(self, step):
        for logger in self.loggers:
            logger.begin_step(step)
    
    def scalar(self, tag, value):
        for logger in self.loggers:
            logger.scalar(tag, value)

    def scalars(self, tag, value):
        for logger in self.loggers:
            logger.scalars(tag, value)

    def pr_curve(self, tag, curve):
        for logger in self.loggers:
            logger.pr_curve(tag, curve)

    def histogram(self, tag, histogram):
        for logger in self.loggers:
            logger.histogram(tag, histogram)

    def flush_step():
       for logger in self.loggers:
            logger.flush_step()


class JsonLogger:
    def __init__(self, log_file):
        self.step = None

        self.log_file = log_file
        self.entries = {}   

    def begin_step(self, step):
        if self.step is not None:
            self.flush_step()

        self.step = struct (
            step=step,
            begin=time.time(),
            entries = {}
        )


    def flush_step(self):
        assert self.step is not None

        t = time.time()
        step = self.step._extend(end = t, duration = t - self.step.begin)

        data = json.dumps(to_dicts(self.step))
        with open(self.log_file, "a") as file:
            file.write(data + '\n')        

        self.step = None

    def scalar(self, tag, value):
        self.append_entry(tag, value)


    def scalars(self, tag, value_dict):
        self.append_entry(tag, value_dict)
                    
    def pr_curve(self, tag, curve):
        self.append_entry(tag, curve)

    def histogram(self, tag, histogram):
        assert isinstance(histogram, Histogram)
        self.append_entry(tag, histogram.to_struct())

    # Internal
    def append_entry(self, tag, value):    
        assert tag not in self.step.entries
        self.step.entries[tag] = value


class TensorflowLogger:
    def __init__(self, log_dir):
        self.writers = {None : tf.summary.FileWriter(log_dir)}
        self.log_dir = log_dir
        self.step = 0

    def begin_step(self, step):
        self.step = step


    def flush_step(self):
        for writer in self.writers.values():
            writer.flush()

    def scalar(self, tag, value, run = None):
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,  simple_value=value)])
        self.writer(run=run).add_summary(summary, self.step)

    def scalars(self, tag, value_dict):
        for run, value in value_dict.items():
            summary = tf.Summary(value=[tf.Summary.Value(tag=tag,  simple_value=value)])
            self.writer(run=run).add_summary(summary, self.step)

        append_entry
                    
    def pr_curve(self, tag, curve):
        pass

        # summary_metadata = metadata.create_summary_metadata(display_name=tag, description='', num_thresholds=curve.precision.size(0))
        # summary = tf.Summary()

        # data = torch.stack([curve.true_positives, curve.false_positives, 
        #     curve.true_negatives, curve.false_negatives, 
        #     curve.precision, curve.recall])

        # tensor = tf.make_tensor_proto(np.float32(data.numpy()), dtype=tf.float32)
        # summary.value.add(tag=tag, metadata=summary_metadata, tensor=tensor)         

        # self.add_summary(summary, run=run)


    def histogram(self, tag, histogram):
        assert isinstance(histogram, Histogram)
        # """Logs the histogram of a list/vector of values."""

        # # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = histogram.range[0]
        hist.max = histogram.range[1]
        hist.num = histogram.counts.sum().item()
        
        hist.sum = histogram.sum
        hist.sum_squares = histogram.sum_squares

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin

        #Add bin edges and counts
        bin_edges = histogram.bins()[1:]

        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in histogram.counts:
            hist.bucket.append(c)

        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.add_summary(summary)


    # Internal
    def writer(self, run = None):       
        if not (run in self.writers):
            self.writers[run] = tf.summary.FileWriter(path.join(self.log_dir, str(run)))

        return self.writers[run]

    def add_summary(self, summary, run = None):
        self.writer(run).add_summary(summary, global_step = self.step)
