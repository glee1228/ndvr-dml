from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import faiss
import os

from datasets import VCDB
from utils import load_feature, l2_distance, find_video_idx

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate frame triplets for train CNN models with triplet loss.")
    parser.add_argument('--vcdb_root', type=str,
                        default='/mldisk/nfs_shared_/MLVD/VCDB')
    parser.add_argument('--feature_path', type=str,
                        default='/mldisk/nfs_shared_/dh/VCDB/features/resnet50-rmac/frame-features')
    parser.add_argument('--annotation_root', type=str, default='/mldisk/nfs_shared_/MLVD/VCDB/annotation')
    parser.add_argument('--triplet_csv', type=str, default='/mldisk/nfs_shared_/dh/VCDB/vcdb_triplets_1fps.csv')
    parser.add_argument('--negative', type=int, default=10000, help='size of negative video pool.')
    parser.add_argument('--margin', type=float, default=0.3, help='distance margin')
    parser.add_argument('--topk', type=int, default=10, help='maximum negative frame for each positive pair')

    args = parser.parse_args()

    positive_csv = '/mldisk/nfs_shared_/dh/VCDB/vcdb_positive_1fps.csv'
    vcdb_positive = pd.read_csv(positive_csv).to_numpy()

    vcdb = VCDB(args.vcdb_root)
    print(vcdb)

    np.random.seed(0)
    distract_videos = np.random.choice(vcdb.distract_videos, args.negative, replace=False)

    core_feature_path = np.vectorize(lambda x: os.path.join(args.feature_path, x + '.pth'))(vcdb.core_videos)
    distract_feature_path = np.vectorize(lambda x: os.path.join(args.feature_path, x + '.pth'))(distract_videos)

    core_feature, core_idx = load_feature(core_feature_path,
                                          progress=True,
                                          desc='Load core video features')
    distract_feature, distract_idx = load_feature(distract_feature_path,
                                                  progress=True,
                                                  desc='Load distract video features')
    # { feature idx : (video idx, frame idx),  ... }

    distract_idx_table = {i: (n, i - l[0]) for n, l in enumerate(distract_idx) for i in range(l[0], l[1])}

    bg_index = faiss.IndexFlatL2(distract_feature.shape[1])
    bg_index = faiss.index_cpu_to_all_gpus(bg_index)
    bg_index.add(distract_feature)

    # sampled feature index as negative
    used = set()

    # frame triplets
    # fine negative features (dist(a,b) - margin < dist(a, n) < dist(a,b))
    triplets = []
    for i, pair in enumerate(tqdm(vcdb_positive, ncols=150, desc='Triplets Sampling', unit='pair'), start=1):
        a, b = pair
        # find feature a
        a_video_idx = find_video_idx(os.path.dirname(a), vcdb.core_videos)
        a_frame_idx = int(os.path.splitext(os.path.basename(a))[0]) - 1
        a_feature_idx = core_idx[a_video_idx][0] + a_frame_idx
        a_feature = core_feature[a_feature_idx:a_feature_idx + 1, :]
        # find feature b
        b_video_idx = find_video_idx(os.path.dirname(b), vcdb.core_videos)
        b_frame_idx = int(os.path.splitext(os.path.basename(b))[0]) - 1
        b_feature_idx = core_idx[b_video_idx][0] + b_frame_idx
        b_feature = core_feature[b_feature_idx:b_feature_idx + 1, :]

        # dist(a,b)
        pos_distance = l2_distance(a_feature, b_feature)
        if pos_distance != 0:
            neg_distance, neg_rank = bg_index.search(np.concatenate([a_feature, b_feature]), 1024)  # (2, distract)

            # compare a with negative pool
            a_n_distance, a_n_rank = neg_distance[0], neg_rank[0]
            valid_neg_rank = np.where((pos_distance - args.margin < a_n_distance) &
                                      (a_n_distance < pos_distance))[0]
            neg_idx = [a_n_rank[r] for r in valid_neg_rank if a_n_rank[r] not in used][:args.topk]
            triplets += [{'anchor': a,
                          'positive': b,
                          'negative': os.path.join(distract_videos[distract_idx_table[i][0]],
                                                   f'{distract_idx_table[i][1] + 1:06d}.jpg')} for i in neg_idx]

            used.update(set(neg_idx))

            # compare b with negative pool
            b_n_distance, b_n_rank = neg_distance[1], neg_rank[1]
            valid_neg_rank = np.where((pos_distance - args.margin < b_n_distance) &
                                      (b_n_distance < pos_distance))[0]
            neg_idx = [b_n_rank[r] for r in valid_neg_rank if b_n_rank[r] not in used][:args.topk]
            triplets += [{'anchor': b,
                          'positive': a,
                          'negative': os.path.join(distract_videos[distract_idx_table[i][0]],
                                                   f'{distract_idx_table[i][1] + 1:06d}.jpg')} for i in neg_idx]
            used.update(set(neg_idx))

    print(len(triplets))
    pd.DataFrame(triplets).to_csv(args.triplet_csv, index=False)
