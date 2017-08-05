import os
import torch

import argparse
import shlex

def create(models, params, args):
    assert params.model in models
    model = models[params.model]

    print(args)

    return model.create(params, **args)

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


def save(path, model, model_params, epoch, score):
    model_file = os.path.join(path, 'model.pth')

    if not os.path.isdir(path):
        os.mkdirs(path)

    state = {
        'epoch':    epoch,
        'params':   model_params,
        'state':    model.state_dict(),
        'score':    score
    }

    print('saving model %s' % model_file)
    torch.save(state, model_file)



def load(models, path, args):

    model_file = os.path.join(path, 'model.pth')
    print('loading model %s' % model_file)

    state = torch.load(path)
    params = state['params']
    model = create(models, params, args)

    model.load_state_dict(state['state'])
    return model, params, state['epoch'], state['score']
