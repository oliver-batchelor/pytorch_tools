import os
import torch

import torch.nn as nn

import argparse
import shlex

def create(models, params, args):
    assert params.model in models
    model = models[params.model]

    print(args)
    return model.create(params, **args)

def model_stats(model):
    convs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            convs += 1

    parameters = sum([p.nelement() for p in model.parameters()])
    print("Model of {} parameters, {} convolutions".format(parameters, convs))

def add_arguments(parser, parameters):
    for name, (default, help) in parameters.items():

        if(type(default) == bool):
            action=('store_false' if default else 'store_true')
            parser.add_argument('--' + name, default=default, type=type(default), help=help)
        else:
            parser.add_argument('--' + name, default=default, type=type(default), help=help)


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
