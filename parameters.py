from tools import Struct
import argparse

import shlex


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
        if param.type == 'choice':
            defaults[name] = Struct(choice = param.default, parameters = default_parameters(param.options[param.default]))
        else:
            defaults[name] = param.default

    return Struct(**defaults)


def make_parser(description, parameters):
    assert type(description) is str and type(parameters) is Struct

    parser = argparse.ArgumentParser(description=description)
    return add_arguments(parser, parameters)


def parse_args(parameters, description, cmdArgs=None):
    parser = argparse.ArgumentParser(description=description)
    add_arguments(parser, parameters)

    args = parser.parse_args(cmdArgs)
    return Struct(**args.__dict__)


def parse_choice(name, choice, str):
    tokens = shlex.split(str)
    option, *cmdArgs = tokens

    choices = list(choice.options.keys())
    assert option in choice.options, "option '" + option + "' missing, expected one of " + str(choices)
    args = parse_args(choice.options[option], choice.help, cmdArgs)
    return Struct (choice = option, parameters = args)


def param_type(value):
    return type(value).__name__

def param(default=None, help='', type=None, required=False):
    assert default is not None or type is not None, "type parameter required if default is not provided"
    return Struct (default=default, required=required,  help=help, type=type or param_type(default))

def required(type, help=''):
    return Struct (default=None, required=True, help=help, type=type)

def choice(default, options, help='', required=False):
    return Struct (default=default, options=options, help=help, type='choice', required=required)


def group(title, **parameters):
    return Struct(title=title, type='group', parameters=Struct(**parameters))


def add_arguments(parser, parameters):
    def to_type(name):
        if name == 'bool':
            return bool
        elif name == 'int':
            return int
        elif name == 'str' or name == 'choice':
            return str
        elif name == 'float':
            return float
        else:
            assert false, "unknown type: " + name


    for name, parameter in parameters.items():
        if parameter.type == 'group':
            group = parser.add_argument_group(parameter.title)
            add_arguments(group, parameter.parameters)
        else:
            default = parameter.default
            help = parameter.help

            if parameter.default is not None:
                help = parameter.help + ", default(" + str(default) + ")"

            if(parameter.type == 'bool' and default is not None):
                parser.add_argument('--' + name, required=parameter.required, default=default, help=help, action=('store_false' if default else 'store_true'))
            else:
                parser.add_argument('--' + name, required=parameter.required, default=default, type=to_type(parameter.type), help=help)
    return parser
