import os
import torch

import torch.nn as nn

import argparse
import shlex

from tools import Struct
from tools.parameters import add_arguments, default_parameters

def create(models, params, args):
    assert params.model in models
    model = models[params.model]

    defaults = default_parameters(model.parameters)
    params = defaults.merge(params)

    print(params, args)
    return model.create(params, **args)


def describe_models(models):
    return {name: describe_parameters(model.parameters()) for name, model in models.items()}


def model_stats(model):
    convs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            convs += 1

    parameters = sum([p.nelement() for p in model.parameters()])
    print("Model of {} parameters, {} convolutions".format(parameters, convs))


def parse_params(models, model_desc):

    parser = argparse.ArgumentParser(prog="--model ", description='Model configuration')
    sub_parsers = parser.add_subparsers(help='model and model specific parameters', dest="model")

    for name, model in models.items():

        sub_parser = sub_parsers.add_parser(name)
        add_arguments(sub_parser, model.parameters)

    assert len(model_desc) > 0, "required model parameter missing"

    args = parser.parse_args(shlex.split(model_desc[0]))
    return args




def save(model_file, model, model_params, epoch, score):
    path = os.path.basename(model_file)

    if not os.path.isdir(path):
        os.makedirs(path)

    state = {
        'epoch':    epoch,
        'params':   model_params,
        'state':    model.state_dict(),
        'score':    score
    }

    print('saving model %s' % model_file)
    torch.save(state, model_file)


def load(models, model_file, args):
    print('loading model %s' % model_file)

    state = torch.load(model_file)
    params = state['params']
    model = create(models, params, args)

    model.load_state_dict(state['state'])
    return model, params, state['epoch'], state['score']
