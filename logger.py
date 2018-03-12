# import numpy as np
import time
import tensorflow as tf

import os.path as path
import os

def make_experiment(log_path, name, dry_run=False, load=False):
    output_path = os.path.join(log_path, name)

    if not dry_run:
        exists = path.exists(output_path) and len(os.listdir(output_path)) > 0

        if exists and not load:
            backup_name = enumerate_name(name, os.listdir(log_path))
            backup_path = os.path.join(log_path, backup_name)


            print("moving old experiment to: " + backup_path)
            os.rename(output_path, backup_path)

        if not path.isdir(output_path):
            os.makedirs(output_path)

        return output_path, Logger(output_path)
    else:
        return output_path, Null()


def enumerate_name(base, names):
    i = 1
    name = base
    while name in names:
        name = base + str(i)
        i = i + 1

    return name


class Null:
    def image(self, name, image, step):
        pass

    def scalar(self, name, value, step=None, wall_time=None):
        pass

    def flush():
        pass

class Logger:
    def __init__(self, log_file):
        self.writer = tf.summary.FileWriter(log_file, flush_secs=1)

    def scalar(self, name, value, step=None, wall_time=None):
        wall_time = wall_time or time.time()

        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name, simple_value=float(value)),
        ])

        event = tf.Event(wall_time=wall_time, step=step, summary=summary)
        self.writer.add_event(event)

    def image(self, name, image, step):
        session = tf.Session()

        assert image.dim() == 3 and image.size(2) <= 4
        image = image.view(1, *image.size())
#        image = image.permute(2, 0, 1)

        p = tf.placeholder("uint8", tuple(image.size()))
        s = tf.summary.image(name, p)

        summary = session.run(s, feed_dict={p: image.numpy()})
        self.writer.add_summary(summary, step)

    def flush(self):
        self.writer.flush()
