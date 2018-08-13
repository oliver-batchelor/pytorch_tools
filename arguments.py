from tools.parameters import param, add_arguments
from tools import Struct

parameters = Struct (
    epochs          = param(1000,   help='number of epochs to train'),
    lr              = param(0.1,    help='learning rate'),
    fine_tuning     = param(0.1,    help='fine tuning as proportion of learning rate'),

    momentum        = param(0.9,    help='SGD momentum'),
    seed            = param(1,      help='random seed'),
    batch_size      = param(16,     help='input batch size for training'),
    epoch_size      = param(1024,   help='epoch size for training'),
    load            = param(False,  help='load progress from previous training'),
    save_interval   = param(2,      help='epochs per save'),
    dry_run         = param(False,  help='run without saving outputs for testing'),
    show            = param(False,  help='view training output'),
    visualize       = param(False,  help='visualize model'),
    num_workers     = param(4,      help='number of workers used to process dataset'),
    name            = param('experiment', help='name for directory to store model and logs'),
    log             = param('log',        help='set output log path')
)

def add(parser):
    add_arguments(parser, parameters)
