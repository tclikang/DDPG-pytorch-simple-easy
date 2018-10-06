# coding= utf-8

import argparse


def init_hyper_para():
    parser = argparse.ArgumentParser(description='PyTorch on TORCS with Multi-modal')
    parser.add_argument('--num_ep', default=200, type=int, help='episode number')
    parser.add_argument('--capacity', default=100000, type=int, help='memory capacity')
    parser.add_argument('--batch_size', default=128, type=int, help='batch size')
    parser.add_argument('--sigma', default=0.99, type=float, help='discount factor')
    parser.add_argument('--tau', default=0.001, type=float, help='learning para of target network')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                        help='number of episodes with noise (default: 100)')
    args = parser.parse_args()
    return args












