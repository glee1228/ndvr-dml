import csv
from torch.utils.data import DataLoader
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np
from utils import DEVICE_STATUS, DEVICE_COUNT
from ndisDB_script import models
import torch
import os
from torchvision.transforms import transforms as trn
from torch.utils.data import Dataset
from PIL import Image
import faiss
import pandas as pd

def find_name_idx(name, videos):
    idx = np.where(videos == name)[0]
    name_idx = idx[0] if len(idx) != 0 else -1
    return name_idx

def load(p):
    _, ext = os.path.splitext(p)

    if ext in ['.pt', '.pth']:
        c = torch.load(p)
        f, names = c['features'][0].numpy(), c['paths']

    return f, f.shape[0], names

def l2_distance(a, b):
    idx = faiss.IndexFlatL2(a.shape[1])
    idx.add(a)
    dist, _ = idx.search(b, 1)
    return dist[0][0]

def load_feature(paths, process=4, progress=False, desc='Load features'):
    def update():
        if progress:
            bar.update()

    bar = tqdm(paths, desc=desc, ncols=150) if progress else None

    pool = Pool(process)
    results = [pool.apply_async(load, args=[p], callback=lambda *a: update()) for p in
               paths]
    pool.close()
    pool.join()

    if progress:
        bar.close()

    results = [(f.get()) for f in results]
    features = np.concatenate([r[0] for r in results])
    length = [r[1] for r in results]
    names =  np.concatenate([r[2] for r in results])
    start = np.cumsum([0] + length)
    index = np.concatenate((start[:-1].reshape(-1, 1), start[1:].reshape(-1, 1)), axis=1)

    return features, index, names

# feature extraction
@torch.no_grad()
def extract_frame_features(model, dataset, loader, save_to):
    model.eval()
    features = []
    paths = []
    idx = 0
    progress_bar = tqdm(loader)
    for i_batch, (img, path) in enumerate(progress_bar):
        feat = model(img.cuda()).cpu()
        features.append(feat)
        paths.extend(path)
        features = torch.cat(features)
        features = [features]
        idx += len(path)
    feature_name = dataset.get_feature_name()
    target = os.path.join(save_to, f'{feature_name}_{idx}.pth')
    if not os.path.exists(os.path.dirname(target)):
        os.makedirs(os.path.dirname(target))
    torch.save({'features': features, 'paths': paths}, target)

    del features[:]
    del features
    del paths[:]
    del paths

class ndisDataset(Dataset):
    def __init__(self, ndis_root,data_type='db', transform=None, transform_types=None):
        self.root_dir = ndis_root
        self.db_path = f'{self.root_dir}/db'
        self.query_path = f'{self.root_dir}/query'
        self.origin_path = f'{self.root_dir}/origin'
        self.transform_types = ['cam','caption','contrast','crop','letterbox','logo','noise','rotation','scale']
        if transform_types is not None:
            self.transform_types = transform_types
        self.transformed_img_dict = self.get_transform_img()
        self.transform = trn.Compose([
            trn.Resize((224, 224)),
            trn.ToTensor(),
            trn.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.data_type = data_type
        self.db_list = self.db_annotation()
        self.origin_list = self.origin_annotation()
        self.query_list = self.query_annotation()
        assert self.data_type in ['db','query','origin']

        if transform is not None:
            self.transform = transform

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

    def get_feature_name(self):
        return self.data_type

    def __len__(self):
        if self.data_type=='db':
            return len(self.db_list)
        elif self.data_type=='origin':
            return len(self.origin_list)
        elif self.data_type=='query':
            return len(self.query_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.data_type=='db' :
            img_path = os.path.join(self.db_list[idx][1])
        elif self.data_type == 'origin':
            img_path = os.path.join(self.origin_list[idx][1])
        elif self.data_type == 'query':
            img_path = os.path.join(self.query_list[idx][1])

        image = Image.open(img_path)
        img_name = img_path

        if self.transform:
            image = self.transform(image)
        sample = [image,img_name]
        return sample

nagative_mining_type = 'semi_hard+hard_v2'
margin = 0.2
topk = 1
ndisDB_root = '/workspace/data/ndisDB/train'
positive_path = f'{ndisDB_root}/ndisDB_positive_{nagative_mining_type}_{margin}_{topk}.csv'
feature_path =  f'{ndisDB_root}/features'
triplet_path = f'{ndisDB_root}/ndisDB_triplet_{nagative_mining_type}_{margin}_{topk}.csv'
modelname = 'Resnet50_RMAC'

transform_types = None
db_dataset = ndisDataset(ndisDB_root, data_type='db', transform_types=transform_types)
query_dataset = ndisDataset(ndisDB_root, data_type='query',transform_types=transform_types)
origin_dataset = ndisDataset(ndisDB_root, data_type='origin',transform_types=transform_types)

model = models.get_frame_model(modelname).cuda()

# Check device
if DEVICE_STATUS and DEVICE_COUNT > 1:
    model = torch.nn.DataParallel(model)


db_dataloader = DataLoader(db_dataset, batch_size=64,
                        shuffle=False, num_workers=4)
query_dataloader = DataLoader(query_dataset, batch_size=64,
                        shuffle=False, num_workers=4)
origin_dataloader = DataLoader(origin_dataset, batch_size=64,
                        shuffle=False, num_workers=4)

# if not os.path.exists(feature_path):
#     os.makedirs(feature_path)
#
# if os.path.exists(feature_path) and len(os.listdir(feature_path)) != 0:
#     print(f'Feature Path {feature_path} is not empty.')
#     exit(1)
# extract_frame_features(model, origin_dataset, origin_dataloader, feature_path)
#
# extract_frame_features(model, query_dataset, query_dataloader, feature_path)
#
# extract_frame_features(model, db_dataset, db_dataloader, feature_path)

# positive sample
with open(positive_path, 'w') as f:
    fieldnames = ['a', 'b']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    a_b = []
    for imgname,_ in tqdm(query_dataset.query_annotation()):
        for type in query_dataset.get_transform_types():
            try:
                a_content = f'{query_dataset.get_query_path()}/{type}/{imgname}'
                b_content = f'{query_dataset.get_origin_path()}/{imgname}'
                a_b.append([a_content, b_content])
                writer.writerow({'a': a_content, 'b': b_content})
            except:
                import pdb;
                pdb.set_trace()

### triplet sample
feature_list = os.listdir(feature_path)
db_feature_name, query_feature_name, origin_feature_name = None, None, None
for feature_name in feature_list:
    if 'db' in feature_name:
        db_feature_name = feature_name
    elif 'query' in feature_name:
        query_feature_name = feature_name
    elif 'origin' in feature_name:
        origin_feature_name = feature_name

db_feature, db_idx, db_names = load(f'{feature_path}/{db_feature_name}')
query_feature, query_idx, query_names = load(f'{feature_path}/{query_feature_name}')
origin_feature, origin_idx, origin_names = load(f'{feature_path}/{origin_feature_name}')



distract_idx_table = {n : l for n, l in enumerate(db_names)}

bg_index = faiss.IndexFlatL2(db_feature.shape[1])
bg_index = faiss.index_cpu_to_all_gpus(bg_index)
bg_index.add(db_feature)

# sampled feature index as negative
used = set()

# frame triplets
# fine negative features (dist(a,b) - margin < dist(a, n) < dist(a,b))
triplets = []

with open(positive_path) as csv_file:
    lines = len(csv_file.readlines())

with open(positive_path) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    header = next(csv_reader)
    for i, pair in enumerate(tqdm(csv_reader, desc='Triplets Sampling', unit='pair')):
        a, b = pair

        # find feature a
        a_feature_idx = query_names.index(a)
        a_feature = query_feature[a_feature_idx:a_feature_idx + 1, :]

        # find feature b
        b_feature_idx = origin_names.index(b)
        b_feature = origin_feature[b_feature_idx:b_feature_idx + 1, :]

        # dist(a,b)
        pos_distance = l2_distance(a_feature, b_feature)
        if pos_distance != 0:
            neg_distance, neg_rank = bg_index.search(np.concatenate([a_feature, b_feature]), 2048)  # (2, distract)

            # compare a with negative pool
            a_n_distance, a_n_rank = neg_distance[0], neg_rank[0]
            # valid_neg_rank = np.where((pos_distance - margin < a_n_distance) &
            #                           (a_n_distance < pos_distance))[0]
            if nagative_mining_type =='easy':
                valid_neg_rank = np.where((pos_distance + margin<a_n_distance))[0]
            elif nagative_mining_type in ['semi-hard','semi_hard']:
                valid_neg_rank = np.where((pos_distance<a_n_distance)&(a_n_distance - pos_distance < margin))[0]
            elif nagative_mining_type =='hard':
                valid_neg_rank = np.where((pos_distance>a_n_distance))[0]
            elif nagative_mining_type =='hard_v2':
                valid_neg_rank = np.where((pos_distance-margin>a_n_distance))[0]
            elif nagative_mining_type in ['semi_hard+hard','semi-hard+hard']:
                valid_neg_rank = np.where((a_n_distance - pos_distance < margin))[0]
            elif nagative_mining_type in ['semi_hard+hard_v2','semi-hard+hard_v2']:
                valid_neg_rank = np.where((a_n_distance - pos_distance < (margin/2)) & (a_n_distance >  pos_distance - (margin/2)))[0]

            # valid_neg_rank = np.random.choice(valid_neg_rank, 1)

            # neg_idx = [a_n_rank[r] for r in valid_neg_rank if a_n_rank[r] not in used][:topk]
            neg_idx = [f'{a_feature_idx}+{b_feature_idx}+{a_n_rank[r]}' for r in valid_neg_rank if f'{a_feature_idx}+{b_feature_idx}+{a_n_rank[r]}' not in used][:topk]

            triplets += [{'anchor': a,
                          'positive': b,
                          'negative': distract_idx_table[int(i.split('+')[-1])]} for i in neg_idx]

            used.update(set(neg_idx))

            # # compare b with negative pool
            # b_n_distance, b_n_rank = neg_distance[1], neg_rank[1]
            # valid_neg_rank = np.where((pos_distance - margin < b_n_distance) &
            #                           (b_n_distance < pos_distance))[0]
            # neg_idx = [b_n_rank[r] for r in valid_neg_rank if b_n_rank[r] not in used][:topk]
            # triplets += [{'anchor': b,
            #               'positive': a,
            #               'negative':distract_idx_table[i]} for i in neg_idx]
            # used.update(set(neg_idx))


print(len(triplets))
pd.DataFrame(triplets).to_csv(triplet_path, index=False)