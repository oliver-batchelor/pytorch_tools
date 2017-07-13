

def add(parser):
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--epoch_size', type=int, default=512, metavar='E',
                        help='epoch size for training (default: 1024)')

    parser.add_argument('--load', action='store_true', default=False,
                        help='load progress from previous training')

    parser.add_argument('--save_interval', type=int, default=2, metavar='I',
                        help='epochs per save')

    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='run without saving outputs for testing')

    parser.add_argument('--show', action='store_true', default=False,
                        help='view training output')

    parser.add_argument('--visualize', action='store_true', default=False,
                        help='visualize model')

    parser.add_argument('--num-workers', type=int, default=2, metavar='W',
                        help='number of workers used to process dataset')

    parser.add_argument('--name', default='experiment',
                        help='name for directory to store model and logs')

    parser.add_argument('--log', default='log',
                        help='set output log path')


    return parser
