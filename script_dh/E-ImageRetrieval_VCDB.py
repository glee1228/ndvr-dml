from torch.utils.data import DataLoader
import torch

import numpy as np
import argparse
import os

from ndisDB_script import models
from datasets import VCDB, ListDataset
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="...")
    subparsers = parser.add_subparsers()

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--vcdb_root', type=str, required=False)
    common.add_argument('--chunk', type=int, default=100)
    common.add_argument('--margin', type=int, default=1)

    extract = subparsers.add_parser('extract', parents=[common],
                                    help='Extract features and Evaluate VCDB frame ranking')
    extract.add_argument('--model', type=str, choices=models.FRAME_MODELS)
    extract.add_argument('--ckpt', type=str, default=None)
    extract.add_argument('--batch', type=int, default=256)
    extract.add_argument('--worker', type=int, default=4)

    parser_load = subparsers.add_parser('load', parents=[common],
                                        help='Load features and Evaluate VCDB frame ranking')
    parser_load.add_argument('--feature_path', type=str, required=False)
    args = parser.parse_args()
    print(args)

    args.vcdb_root = '/mldisk/nfs_shared_/MLVD/VCDB-core'
    args.feature_path = '/mldisk/nfs_shared_/dh/VCDB/features/resnet50-rmac/frame-features'
    args.chunk = 100
    args.margin = 1
    args.model = 'resnet50-rmac'
    # if not hasattr(args, 'model') and not hasattr(args, 'model'):
    #     parser.print_help()
    #     exit(-1)

    vcdb = VCDB(args.vcdb_root)
    if hasattr(args, 'model'):

        loader = DataLoader(ListDataset([f for v in vcdb.core_videos for f in vcdb.get_frames(v)]),
                            batch_size=args.batch, shuffle=False, num_workers=args.worker)

        model = models.get_frame_model(args.model).cuda()
        # Load checkpoints
        if args.ckpt is not None:
            model.load_state_dict(torch.load(args.ckpt))

        # Check device
        if DEVICE_STATUS and DEVICE_COUNT > 1:
            model = torch.nn.DataParallel(model)

        features = extract_feature(model, loader)
        length = [vcdb.get_framecount(v) for v in vcdb.core_videos]
        start = np.cumsum([0] + length)
        index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    else:
        features, index = load_feature([os.path.join(args.feature_path, v + '.pth')
                                        for v in vcdb.core_videos], progress=False)

    # import pdb;pdb.set_trace()
    avg_rank, avg_dist, avg_rank_per_query, avg_dist_per_query = vcdb_frame_retrieval(vcdb, features,
                                                                                      index, args.chunk, args.margin,
                                                                                      progress=True)
    print(avg_rank, avg_dist, avg_rank_per_query, avg_dist_per_query)