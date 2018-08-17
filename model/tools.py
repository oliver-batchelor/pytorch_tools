import os
import torch

import torch.nn as nn

import argparse

from tools import Struct
from tools.parameters import add_arguments, default_parameters

def create(models, model_args, dataset_args):

    assert model_args.choice in models, "model not found " + model_args.choice
    model = models[model_args.choice]

    return model.create(model_args.parameters, dataset_args)


def describe_models(models):
    return {name: describe_parameters(model.parameters()) for name, model in models.items()}


def model_stats(model):
    convs = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            convs += 1

    parameters = sum([p.nelement() for p in model.parameters()])
    print("Model of {} parameters, {} convolutions".format(parameters, convs))




def save(model_file, model, model_args, epoch, score):
    path = os.path.dirname(model_file)

    if not os.path.isdir(path):
        os.makedirs(path)

    state = {
        'epoch':    epoch,
        'params':   model_args,
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
