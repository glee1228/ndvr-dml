from tqdm import tqdm
import argparse
import warnings
import os
import pandas as pd
from torchvision.transforms import transforms as trn
import albumentations as A
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import torch_optimizer
import cv2
from ndisDB_script import models
from utils import *
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import csv

warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logger = None
writer = None
import numpy as np

def get_color(idx):
    distinct_colors = np.array([
        [255, 165, 150],
        [255, 255, 138],
        [0, 82, 232],
        [196, 141, 0],
        [7, 94, 71],
        [1, 138, 253],
        [152, 255, 143],
        [39, 95, 4],
        [209, 207, 0],
        [52, 255, 213],
        [89, 84, 31],
        [90, 77, 113],
        [255, 102, 141],
        [78, 184, 0],
        [96, 82, 1],
        [238, 42, 0],
        [101, 45, 194],
        [255, 57, 114],
        [1, 248, 160],
        [87, 127, 255],
        [74, 69, 167],
        [153, 255, 86],
        [0, 173, 108],
        [168, 0, 126],
        [255, 230, 75],
        [2, 195, 226],
        [120, 70, 42],
        [138, 166, 0],
        [255, 191, 133],
        [165, 0, 191],
        [0, 162, 104],
        [255, 192, 88],
        [145, 255, 161],
        [1, 247, 184],
        [162, 27, 18],
        [115, 251, 255],
        [255, 157, 168],
        [1, 129, 186],
        [228, 0, 198],
        [2, 164, 126],
        [191, 3, 0],
        [1, 108, 241],
        [255, 149, 208],
        [152, 49, 0],
        [0, 137, 99],
        [255, 244, 181],
        [1, 208, 245],
        [110, 57, 152],
        [0, 104, 91],
        [255, 203, 51],
        [255, 33, 205],
        [0, 115, 135],
        [135, 136, 255],
        [130, 199, 0],
        [115, 214, 0],
        [149, 173, 255],
        [0, 137, 117],
        [220, 255, 158],
        [3, 230, 191],
        [255, 72, 235],
        [122, 66, 72],
        [70, 82, 114],
        [159, 243, 8],
        [255, 105, 35],
        [2, 110, 200],
        [42, 94, 255],
        [254, 123, 255],
        [43, 57, 218],
        [255, 144, 85],
        [111, 75, 55],
        [132, 68, 0],
        [0, 110, 58],
        [138, 56, 67],
        [227, 245, 255],
        [255, 123, 171],
        [206, 145, 255],
        [79, 86, 46],
        [1, 205, 73],
        [0, 87, 162],
        [179, 146, 0],
        [63, 139, 0],
        [1, 140, 22],
        [1, 235, 254],
        [47, 87, 113],
        [133, 255, 237],
        [227, 193, 0],
        [139, 114, 0],
        [137, 15, 169],
        [244, 255, 187],
        [240, 163, 255],
        [166, 0, 78],
        [255, 72, 32],
        [86, 84, 60],
        [87, 255, 124],
        [210, 70, 0],
        [255, 175, 185],
        [0, 205, 203],
        [120, 66, 87],
        [255, 143, 236],
        [255, 248, 243],
        [0, 226, 110],
        [255, 230, 135],
        [255, 59, 163],
        [247, 0, 179],
        [117, 235, 255],
        [219, 96, 0],
        [255, 140, 172],
        [210, 255, 201],
        [78, 84, 78],
        [188, 255, 181],
        [129, 255, 173],
        [255, 154, 21],
        [255, 42, 126],
        [255, 159, 95],
        [200, 0, 175],
        [95, 223, 14],
        [77, 76, 138],
        [157, 255, 186],
        [1, 184, 192],
        [115, 124, 0],
        [167, 255, 121],
        [255, 216, 240],
        [0, 169, 235],
        [145, 50, 46],
        [144, 43, 100],
        [234, 255, 242],
        [117, 176, 255],
        [154, 65, 241],
        [62, 88, 77],
        [125, 46, 150],
        [131, 57, 91],
        [205, 0, 60],
        [255, 169, 45],
        [177, 111, 0],
        [0, 215, 154],
        [61, 108, 0],
        [246, 255, 162],
        [145, 29, 131],
        [138, 85, 255],
        [255, 138, 131],
        [112, 188, 255],
        [244, 154, 0],
        [255, 226, 222],
        [2, 191, 165],
        [166, 205, 0],
        [237, 77, 254],
        [1, 131, 204],
        [251, 0, 34],
        [178, 0, 21],
        [141, 222, 255],
        [1, 168, 204],
        [61, 207, 0],
        [46, 89, 95],
        [100, 170, 0],
        [255, 115, 75],
        [255, 199, 111],
        [255, 184, 252],
        [41, 157, 0],
        [155, 255, 100],
        [1, 199, 181],
        [105, 72, 97],
        [231, 0, 96],
        [213, 0, 142],
        [149, 47, 28],
        [255, 226, 102],
        [255, 52, 144],
        [56, 87, 99],
        [155, 22, 106],
        [179, 82, 0],
        [213, 255, 109],
        [236, 255, 99],
        [227, 0, 40],
        [86, 98, 0],
        [131, 249, 51],
        [1, 118, 89],
        [255, 209, 172],
        [171, 50, 0],
        [245, 81, 0],
        [136, 202, 255],
        [0, 110, 25],
        [227, 128, 255],
        [255, 190, 169],
        [0, 178, 39],
        [230, 255, 63],
        [255, 217, 133],
        [182, 238, 0],
        [182, 255, 209],
        [0, 172, 47],
        [94, 117, 0],
        [210, 0, 123],
        [244, 184, 0],
        [216, 160, 255],
        [134, 255, 207],
        [45, 84, 132],
        [240, 255, 132],
        [0, 147, 143],
        [199, 0, 45],
        [15, 154, 255],
        [49, 255, 249],
        [255, 159, 189],
        [255, 200, 213],
        [105, 114, 255],
        [255, 99, 228],
        [0, 180, 134],
        [212, 178, 255],
        [159, 29, 59],
        [244, 243, 0],
        [198, 164, 255],
        [1, 134, 146],
        [0, 121, 9],
        [137, 131, 0],
        [255, 178, 223],
        [122, 42, 218],
        [0, 143, 83],
        [255, 105, 120],
        [0, 134, 160],
        [164, 116, 0],
        [255, 135, 96],
        [165, 151, 0],
        [159, 96, 0],
        [255, 228, 204],
        [225, 0, 208],
        [255, 165, 135],
        [12, 68, 203],
        [90, 158, 255],
        [1, 102, 174],
        [127, 57, 112],
        [105, 55, 168],
        [203, 95, 255],
        [159, 0, 145],
        [193, 106, 0],
        [255, 131, 207],
        [200, 223, 255],
        [100, 255, 185],
        [1, 102, 116],
        [178, 255, 240],
        [1, 209, 39],
        [236, 205, 255],
        [18, 247, 88],
        [118, 72, 15],
        [1, 153, 55],
        [183, 209, 0],
        [255, 139, 42],
        [171, 125, 255],
        [1, 85, 210],
        [227, 135, 0],
        [183, 0, 96],
        [16, 202, 255],
        [194, 0, 209],
        [255, 232, 168],
        [255, 75, 149],
        [106, 63, 140],
        [101, 79, 41],
        [92, 79, 93],
        [181, 101, 255],
        [0, 235, 234],
        [92, 242, 61],
        [1, 122, 159],
        [224, 0, 173],
        [155, 185, 1],
        [127, 84, 0],
        [0, 101, 148],
        [255, 213, 42],
        [188, 255, 115],
        [255, 53, 184],
        [183, 210, 255],
        [234, 224, 255],
        [76, 88, 17],
        [253, 26, 16],
        [169, 0, 59],
        [0, 163, 81],
        [255, 173, 96],
        [210, 255, 236],
        [1, 182, 242],
        [136, 60, 30],
        [53, 92, 34],
        [1, 103, 210],
        [255, 90, 83],
        [126, 54, 124],
        [193, 249, 255],
        [0, 0, 0]
    ])
    return distinct_colors[idx]
def write_csv(file, data):
    with open(file, 'a', newline='') as outfile:
        writer2 = csv.writer(outfile)
        writer2.writerow(data)

def _retrieve_knn_faiss_gpu_inner_product(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatIP(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices


def _retrieve_knn_faiss_gpu_euclidean(query_embeddings, db_embeddings, k, gpu_id=0):
    """
        Retrieve k nearest neighbor based on inner product

        Args:
            query_embeddings:           numpy array of size [NUM_QUERY_IMAGES x EMBED_SIZE]
            db_embeddings:              numpy array of size [NUM_DB_IMAGES x EMBED_SIZE]
            k:                          number of nn results to retrieve excluding query
            gpu_id:                     gpu device id to use for nearest neighbor (if possible for `metric` chosen)

        Returns:
            dists:                      numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors
                                        for each query
            retrieved_db_indices:       numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors
                                        for each query
    """
    import faiss

    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = gpu_id

    # Evaluate with inner product
    index = faiss.GpuIndexFlatL2(res, db_embeddings.shape[1], flat_config)
    index.add(db_embeddings)
    # retrieved k+1 results in case that query images are also in the db
    dists, retrieved_result_indices = index.search(query_embeddings, k + 1)

    return dists, retrieved_result_indices


def evaluate_recall_at_k(dists, results, query_labels, db_labels, k):
    """
        Evaluate Recall@k based on retrieval results

        Args:
            dists:          numpy array of size [NUM_QUERY_IMAGES x k], distances of k nearest neighbors for each query
            results:        numpy array of size [NUM_QUERY_IMAGES x k], indices of k nearest neighbors for each query
            query_labels:   list of labels for each query
            db_labels:      list of labels for each db
            k:              number of nn results to evaluate

        Returns:
            recall_at_k:    Recall@k in percentage
    """

    self_retrieval = False

    if query_labels is db_labels:
        self_retrieval = True

    expected_result_size = k + 1 if self_retrieval else k

    assert results.shape[1] >= expected_result_size, \
        "Not enough retrieved results to evaluate Recall@{}".format(k)

    recall_at_k = np.zeros((k,))

    for i in range(len(query_labels)):
        pos = 0 # keep track recall at pos
        j = 0 # looping through results
        while pos < k:
            if self_retrieval and i == results[i, j]:
                # Only skip the document when query and index sets are the exact same
                j += 1
                import pdb;
                pdb.set_trace()
                continue
            if query_labels[i] == db_labels[results[i, j]]:
                recall_at_k[pos:] += 1
                # import pdb;pdb.set_trace()
                break
            j += 1
            pos += 1
    # import pdb;
    # pdb.set_trace()
    return recall_at_k/float(len(query_labels))*100.0


def evaluate_float_binary_embedding_faiss(epoch, type, query_embeddings, db_embeddings, query_labels, db_labels,
                                          output, k=1000, gpu_id=0, args=None):
    """
        Wrapper function to evaluate Recall@k for float and binary embeddings
        output recall@k strings for Cars, CUBS, Stanford Online Product, and InShop datasets
    """

    # ======================== float embedding evaluation ==========================================================
    # knn retrieval from embeddings (l2 normalized embedding + inner product = cosine similarity)
    dists, retrieved_result_indices = _retrieve_knn_faiss_gpu_inner_product(query_embeddings,
                                                                            db_embeddings,
                                                                            k,
                                                                            gpu_id=gpu_id)

    # evaluate recall@k
    r_at_k_f = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output
    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[1], r_at_k_f[3], r_at_k_f[7])
    sop_eval_str = "{}(float) R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        type, r_at_k_f[0], r_at_k_f[9], r_at_k_f[99], r_at_k_f[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_f[0], r_at_k_f[9], r_at_k_f[19], r_at_k_f[29], r_at_k_f[39], r_at_k_f[49])

    # print(cars_cub_eval_str)
    print(sop_eval_str)
    # print(in_shop_eval_str)
    # with open(output_file, 'w') as of:
    #     of.write(cars_cub_eval_str + '\n')
    #     of.write(sop_eval_str + '\n')
    #     of.write(in_shop_eval_str + '\n')
    write_csv(output_file, [epoch,args.model, query_embeddings.shape[-1],
                          args.negative_mining_type, args.negative_mining_margin, args.negative_mining_topk, type, r_at_k_f[0], r_at_k_f[9], r_at_k_f[99], r_at_k_f[999]])

    # ======================== binary embedding evaluation =========================================================
    binary_query_embeddings = np.require(query_embeddings > 0, dtype='float32')
    binary_db_embeddings = np.require(db_embeddings > 0, dtype='float32')

    # knn retrieval from embeddings (binary embeddings + euclidean = hamming distance)
    dists, retrieved_result_indices = _retrieve_knn_faiss_gpu_euclidean(binary_query_embeddings,
                                                                        binary_db_embeddings,
                                                                        k,
                                                                        gpu_id=gpu_id)
    # evaluate recall@k
    r_at_k_b = evaluate_recall_at_k(dists, retrieved_result_indices, query_labels, db_labels, k)

    output_file = output + '_binary.eval'

    cars_cub_eval_str = "R@1, R@2, R@4, R@8: {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[1], r_at_k_b[3], r_at_k_b[7])
    sop_eval_str = "{}(binary) R@1, R@10, R@100, R@1000: {:.2f} & {:.2f} & {:.2f} & {:.2f}  \\\\".format(
        type, r_at_k_b[0], r_at_k_b[9], r_at_k_b[99], r_at_k_b[999])
    in_shop_eval_str = "R@1, R@10, R@20, R@30, R@40, R@50: {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.2f} \\\\".format(
        r_at_k_b[0], r_at_k_b[9], r_at_k_b[19], r_at_k_b[29], r_at_k_b[39], r_at_k_b[49])

    # print(cars_cub_eval_str)
    # print(sop_eval_str)
    # print(in_shop_eval_str)
    # with open(output_file, 'w') as of:
    #     of.write(cars_cub_eval_str + '\n')
    #     of.write(sop_eval_str + '\n')
    #     of.write(in_shop_eval_str + '\n')

    # return max(r_at_k_f[0], r_at_k_b[0])
    return r_at_k_f[0], r_at_k_f[9], r_at_k_f[99], r_at_k_f[999]


class ndisDataset(Dataset):
    def __init__(self, ndis_root, transform=None,transform_types=None):
        self.root_dir = ndis_root
        self.db_path = f'{self.root_dir}/db'
        self.query_path = f'{self.root_dir}/query'
        self.origin_path = f'{self.root_dir}/origin'
        self.transform_types = ['cam','caption','contrast','crop','letterbox','logo','noise','rotation','scale']
        if transform_types is not None:
            self.transform_types=transform_types
        self.transformed_img_dict = self.get_transform_img()
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.db_list = self.db_annotation()
        self.origin_list = self.origin_annotation()
        self.query_list = self.query_annotation()

        if transform is not None:
            self.transform = transform

    def get_root_dir(self):
        return self.root_dir

    def get_query_path(self):
        return self.query_path

    def get_origin_path(self):
        return self.origin_path

    def get_db_path(self):
        return self.db_path

    def get_transform_types(self):
        return self.transform_types

    def get_transform_img(self):
        transformed_img_dict = {}
        for transform_type in self.transform_types:
            data_lst = os.listdir(f'{self.query_path}/{transform_type}')
            img_lst = [name for name in data_lst if
            os.path.splitext(os.path.basename(name))[1] in ['.jpg', '.png', '.JPG', '.PNG', 'jpeg', 'JPEG']]
            transformed_img_dict[transform_type]=img_lst
        return transformed_img_dict

    def db_annotation(self):
        db_annotations = []
        for img in [name for name in os.listdir(self.db_path) if os.path.splitext(os.path.basename(name))[1] in ['.jpg','.png','.JPG','.PNG','jpeg','JPEG']]:
            db_annotations.append([img, f'{self.db_path}/{img}'])
        return db_annotations

    def origin_annotation(self):
        origin_annotations = []
        for img in [name for name in os.listdir(self.origin_path) if os.path.splitext(os.path.basename(name))[1] in ['.jpg','.png','.JPG','.PNG','jpeg','JPEG']]:
            origin_annotations.append([img, f'{self.origin_path}/{img}'])
        return origin_annotations

    def query_annotation(self):
        transform_annotations = []
        for type, imgs in self.transformed_img_dict.items():
            for img in imgs:
                transform_annotations.append([img, f'{self.query_path}/{type}/{img}'])
        return transform_annotations


def albumentations_loader(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

class ListDataset(Dataset):
    def __init__(self, query, origin, db, transform=None):
        self.query = query
        self.origin = origin
        self.db = db
        self.l = query+origin+db
        self.transform_types = ['cam', 'caption', 'contrast', 'crop', 'letterbox', 'logo', 'noise', 'rotation', 'scale']

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform
        self.query_bundle = self.get_query_bundle()
        if isinstance(self.transform, A.Compose):
            self.load = lambda x: self.transform(image=albumentations_loader(x))['image']
        elif isinstance(self.transform, trn.Compose):
            self.load = lambda x: self.transform(default_loader(x))
        else:
            raise TypeError('Unsupported image loader')

    def __getitem__(self, idx):
        path = self.l[idx][1]
        frame = self.load(path)

        return path, frame

    def __len__(self):
        return len(self.l)

    def get_csv_name(self):
        return self.query[0][1].split('/')[-4]

    def get_query(self):
        return self.query
    def get_origin(self):
        return self.origin
    def get_db(self):
        return self.db
    def get_query_bundle(self):
        bundle = []
        for type in self.transform_types:
            dummy = []
            for q in self.query:
                q_name,q_path = q
                if type in q_path:
                    dummy.append(q_path)
            bundle.append(dummy)
        return bundle

    def get_query_len(self):
        return len(self.query)
    def get_origin_len(self):
        return len(self.origin)
    def get_db_len(self):
        return len(self.db)
    def __repr__(self):
        fmt_str = f'{self.__class__.__name__}\n'
        fmt_str += f'\tNumber of images : {self.__len__()}\n'
        trn_str = self.transform.__repr__().replace('\n', '\n\t')
        fmt_str += f"\tTransform : \n\t{trn_str}"
        return fmt_str

class TripletDataset(Dataset):
    def __init__(self, triplets_csv, transform=None):
        self.triplets = pd.read_csv(triplets_csv).values

        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if transform is not None:
            self.transform = transform

        if isinstance(self.transform, A.Compose):
            self.load = lambda x: self.transform(image=albumentations_loader(x))['image']
        elif isinstance(self.transform, trn.Compose):
            self.load = lambda x: self.transform(default_loader(x))
        else:
            raise TypeError('Unsupported image loader')

    def load_image(self, p):
        path = os.path.join(p)
        im = self.load(path)

        return path, im

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]

        anc_path, anc = self.load_image(a)
        pos_path, pos = self.load_image(p)
        neg_path, neg = self.load_image(n)

        return (anc_path, pos_path, neg_path), (anc, pos, neg)

    def __len__(self):
        return len(self.triplets)

@torch.no_grad()
def extract_feature(model, loader, progress=True, **kwargs):
    model.eval()
    features = []
    bar = tqdm(loader, desc='Extract features', ncols=150, unit='batch') if progress else loader
    for i, (path, frame) in enumerate(bar):
        out = model(frame, **kwargs).cpu()
        features.append(out)

    features = torch.cat(features).numpy()
    return features


@torch.no_grad()
def eval_rank(net, loader, epoch, args=None):
    features = extract_feature(net, loader, progress=True, single=True)
    # query = loader.dataset.get_query()
    origin = loader.dataset.get_origin()
    o_labels = [int(o.split('.')[0]) for o,o_path in origin]
    db = loader.dataset.get_db()
    db_labels = [int(db.split('.')[0]) for db,db_path in db]
    query_bundle = loader.dataset.get_query_bundle()
    length = [len(bundle) for bundle in query_bundle]
    length.extend([loader.dataset.get_origin_len(),loader.dataset.get_db_len()])
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    ### self.transform_types = ['cam', 'caption', 'contrast', 'crop', 'letterbox', 'logo', 'noise', 'rotation', 'scale']

    # q_cam = features[index[0][0]:index[0][1], ]
    # q_caption = features[index[1][0]:index[1][1], ]
    # q_contrast = features[index[2][0]:index[2][1], ]
    # q_crop = features[index[3][0]:index[3][1], ]
    # q_letterbox = features[index[4][0]:index[4][1], ]
    # q_logo = features[index[5][0]:index[5][1], ]
    # q_noise = features[index[6][0]:index[6][1], ]
    # q_rotation = features[index[7][0]:index[7][1], ]
    # q_scale = features[index[8][0]:index[8][1], ]
    o_features = features[index[-2][0]:index[-2][1], ]
    db_features = features[index[-1][0]:index[-1][1], ]

    o_db_labels = o_labels + db_labels
    db_features = np.concatenate((o_features,db_features),axis=0)
    index2 = index.copy()
    index2[-2,1] = index2[-1,1]
    index2 = index2[:-1]


    # bg_index = faiss.IndexFlatL2(db_features.shape[1])
    # bg_index = faiss.index_cpu_to_all_gpus(bg_index)
    # bg_index.add(db_features)
    csv_file = os.path.join(log_root,f'{loader.dataset.get_csv_name()}_recall.csv')
    print(f'[Epoch {epoch}]{loader.dataset.get_csv_name()} Recall@k')
    # with open(csv_file, 'a') as outfile:
    #     writer = csv.DictWriter(outfile, fieldnames = ['epoch','method', 'Dim', 'Triplet', 'Distortion Type','R@1','R@10','R@100','R@1000'])
    #     writer.writeheader()
    meta_data = []
    label_img = None
    for i, type in enumerate(loader.dataset.transform_types):
        q_features = features[index2[i][0]:index2[i][1],]
        q_names = [os.path.basename(q_name).split('.')[0] for q_name in query_bundle[i]]
        q_labels = [int(q) for q in q_names]
        r1,r10,r100,r1000 = evaluate_float_binary_embedding_faiss(epoch, type, q_features, db_features, q_labels, o_db_labels, csv_file, k=1000, gpu_id=0, args=args)
        writer.add_scalar(f'{loader.dataset.get_csv_name()}/recall_1/{type}', r1, epoch)
        writer.add_scalar(f'{loader.dataset.get_csv_name()}/recall_10/{type}', r10, epoch)
        writer.add_scalar(f'{loader.dataset.get_csv_name()}/recall_100/{type}', r100, epoch)
        writer.add_scalar(f'{loader.dataset.get_csv_name()}/recall_1000/{type}', r1000, epoch)
        q_meta = [f'{type}_{q_label}' for q_label in q_labels]
        meta_data+=q_meta
        for j, q_label in enumerate(q_labels):
            q_label_img = np.full((1, 5, 5, 3), get_color(q_label % 280))
            if i==0 and j==0:
                label_img = q_label_img.copy()
            else:
                label_img = np.concatenate((label_img, q_label_img), axis=0)

    o_meta = [f'origin_{o_label}' for o_label in o_labels]
    meta_data += o_meta
    for o_label in o_labels:
        o_label_img = np.full((1, 5, 5, 3), get_color(o_label % 280))
        label_img = np.concatenate((label_img, o_label_img), axis=0)

    db_meta = [f'db_{db_label}' for db_label in db_labels]
    meta_data += db_meta

    db_label_img = np.full((len(db_labels),5, 5, 3), get_color(280))
    label_img = np.concatenate((label_img, db_label_img), axis=0)
    writer.add_embedding(features, metadata=meta_data, label_img=np.transpose(label_img,(0,3,1,2)))

    del features
    del label_img
    del db_label_img

def train(net, loader, optimizer, criterion, scheduler, epoch):
    losses = AverageMeter()
    net.train()
    bar = tqdm(loader, ncols=150)
    for i, (path, frames) in enumerate(loader, 1):
        optimizer.zero_grad()
        # print()
        # print(path[0][0].split('/')[-3:],path[1][0].split('/')[-3:],path[2][0].split('/')[-3:])
        _output1, _output2, _output3 = net(frames[0].float().cuda(), frames[1].float().cuda(), frames[2].float().cuda())
        loss = criterion(_output1, _output2, _output3)
        losses.update(loss.item())
        loss.backward()
        optimizer.step()
        if scheduler.__class__.__name__ == 'CyclicLR':
            scheduler.step()

        bar.set_description(f'[Epoch {epoch}] '
                            f'lr: {scheduler.get_lr()[0]:4f}, '
                            f'loss: {losses.val:.4f}({losses.avg:.4f}), ')
        bar.update()

    bar.close()

    logger.info(f'[EPOCH {epoch}] '
                f'lr: {scheduler.get_lr()[0]:4f}, '
                f'loss: {losses.avg:4f}, ')
    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/learning_rate', scheduler.get_lr()[0], epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN model.")
    parser.add_argument('--model', type=str, choices=models.FRAME_MODELS, default='Resnet50_RMAC')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--triplet_csv', type=str, required=False)
    parser.add_argument('--train_root', type=str, required=False)
    parser.add_argument('--test_root', type=str, required=False)

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-5)
    parser.add_argument('-wd', '--weight_decay', type=float, default=0)
    parser.add_argument('-m', '--margin', type=float, default=0.3)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch', type=int, default=32)
    parser.add_argument('-tb', '--test_batch', type=int, default=256)
    parser.add_argument('-w', '--worker', type=int, default=4)
    parser.add_argument('-o', '--optim', type=str, default='radam')
    parser.add_argument('-s', '--scheduler', type=str, default='cyclic')

    parser.add_argument('-l', '--log_dir', type=str, default='./log')
    parser.add_argument('-c', '--comment', type=str, default=None)
    args = parser.parse_args()

    args.train_root = '/workspace/data/ndisDB/train'
    args.test_root = '/workspace/data/ndisDB/test'
    args.model = 'Resnet50_avgpool_FC3'
    # args.model = 'Resnet50_ccpool_FC3'
    args.negative_mining_type = 'semi_hard+hard_v2'
    args.negative_mining_margin = 0.2
    args.negative_mining_topk = 1

    args.triplet_path = f'{args.train_root}/ndisDB_triplet_{args.negative_mining_type}_{args.negative_mining_margin}_{args.negative_mining_topk}.csv'
    # args.triplet_path = f'{args.train_root}/ndisDB_triplet.csv'
    transform_types = None
    # args.scheduler = None
    writer, logger, log_dir = initialize_writer_and_log(args.log_dir, args.comment)
    global log_root
    log_root = log_dir
    model_save_dir = os.path.join(log_dir, 'saved_model')
    os.makedirs(model_save_dir)
    logger.info(args)

    # Model
    embed_net = models.get_frame_model(args.model).cuda()

    # Freeze CNN backbone weights
    if args.model == 'Resnet50_avgpool_FC3' or args.model == 'Resnet50_ccpool_FC3':
        embed_net.Resnet50.requires_grad = False

    if args.ckpt is not None:
        embed_net.load_state_dict(torch.load(args.ckpt))

    model = models.TripletNet(embed_net).cuda()


    writer.add_graph(model, [torch.rand((2, 3, 224, 224)).cuda(),
                             torch.rand((2, 3, 224, 224)).cuda(),
                             torch.rand((2, 3, 224, 224)).cuda()])
    logger.info(model)

    if DEVICE_STATUS and DEVICE_COUNT > 1:
        model = torch.nn.DataParallel(model)

    # Data
    transform = {
        # 'train': A.Compose([
        #     A.Resize(256, 256),
        #     # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        #     A.RandomCrop(height=224, width=224),
        #     # A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        #     # A.RandomBrightnessContrast(p=0.5),
        #     # A.OneOf([
        #     #     A.HorizontalFlip(p=1),
        #     #     A.RandomRotate90(p=1),
        #     #     A.VerticalFlip(p=1)
        #     # ], p=.5),
        #     # A.OneOf([
        #     #     A.MotionBlur(p=1),
        #     #     A.OpticalDistortion(p=1),
        #     #     A.GaussNoise(p=1)
        #     # ], p=.5),
        #     # A.ImageCompression(quality_lower=80, quality_upper=100, p=0.1),
        #     # A.ToGray(p=0.05),
        #     # A.IAACropAndPad(p=0.3),
        #     A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     ToTensorV2(p=1.0)
        # ]),
        'train': trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'test': trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }
    trainEvalDataset = ndisDataset(args.train_root, transform_types=transform_types)
    testEvalDataset = ndisDataset(args.test_root,transform_types=transform_types)

    train_loader = DataLoader(TripletDataset(args.triplet_path, transform=transform['train']),
                              shuffle=True, num_workers=args.worker, batch_size=args.batch)
    train_eval_loader = DataLoader(ListDataset(trainEvalDataset.query_annotation(),trainEvalDataset.origin_annotation(),trainEvalDataset.db_annotation(),
                                               transform=transform['test']), shuffle=False, num_workers=args.worker, batch_size=args.test_batch)
    test_eval_loader = DataLoader(ListDataset(testEvalDataset.query_annotation(),testEvalDataset.origin_annotation(),testEvalDataset.db_annotation(),
                                              transform=transform['test']),shuffle=False, num_workers=args.worker, batch_size=args.test_batch)

    # Optimizer
    criterion = nn.TripletMarginLoss(args.margin)
    l2_dist = nn.PairwiseDistance()

    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'radam':
        optimizer = torch_optimizer.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=0.9)

    if args.scheduler == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, args.learning_rate* 0.2, args.learning_rate * 5,
                                                      step_size_up=500, step_size_down=500,
                                                      mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
                                                      cycle_momentum=False, base_momentum=0.8, max_momentum=0.9,
                                                      last_epoch=-1)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15,20,25,30,35,40,45], gamma=0.8)

    eval_rank(model, train_eval_loader, 0, args=args)
    eval_rank(model, test_eval_loader, 0, args=args)
    for e in range(1, args.epoch, 1):
        train(model, train_loader, optimizer, criterion, scheduler, e)
        eval_rank(model, train_eval_loader, e, args=args)
        eval_rank(model, test_eval_loader, e, args=args)

        if args.scheduler != 'cyclic':
            scheduler.step()

        # torch.save({'epoch': e,
        #             'state_dict': model.module.embedding_net.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             }, os.path.join(model_save_dir, f'epoch_{e}_ckpt.pth'))