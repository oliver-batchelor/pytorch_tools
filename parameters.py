from tools import Struct
import argparse


def describe_parameters(parameters):
    param_descs = {}

    for name, (default, help) in parameters.items():
        descr[name] = {
            'default':default,
            'type':type(default),
            'help':help
        }

    return param_descs


def default_parameters(parameters):
    defaults = {}

    for name, param in parameters.items():
        defaults[name] = param.default

    return Struct(**defaults)


def make_parser(description, parameters):
    assert type(description) is str and type(parameters) is Struct

    parser = argparse.ArgumentParser(description=description)
    return add_arguments(parser, parameters)


def param_type(value):
    return type(value)

def param(default, help='', type=None):
    return Struct (default=default, help=help, type=type or param_type(default))


def add_arguments(parser, parameters):
    for name, parameter in parameters.items():
        default = parameter.default
        param_type = parameter.type

        if(type(default) is bool):
            parser.add_argument('--' + name, default=default,  help=help, action=('store_false' if default else 'store_true'))
        else:
            parser.add_argument('--' + name, default=default, type=param_type, help=parameter.help)
    return parser
