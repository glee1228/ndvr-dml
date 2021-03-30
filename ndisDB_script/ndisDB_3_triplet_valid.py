from datasets.vcdb import VCDB
import csv
from tqdm import tqdm
import shutil
import os
from PIL import Image, ImageFont, ImageDraw
import requests
import io

import numpy as np
import faiss
from utils import load_feature, l2_distance, find_video_idx
import pandas as pd

# vcdb_root = '/mldisk/nfs_shared_/MLVD/VCDB'
# vcdb_frame_root = '/mldisk/nfs_shared_/MLVD/VCDB/frames'
# subset_frame_root = '/workspace/subset/VCDB/frames'
# subset_triplet_vis_root ='/workspace/subset/VCDB/vis_triplet'
# subset_triplet_vis_root2 ='/workspace/subset/VCDB/vis_triplet2'
#
# triplet_path = '/mldisk/nfs_shared_/dh/VCDB/vcdb_triplets_1fps.csv'
#
# sub_triplet_path = '/workspace/subset/VCDB/vcdb_triplets_1fps_subset.csv'
#
# feature_path ='/mldisk/nfs_shared_/dh/VCDB/features/resnet50-rmac/frame-features'

ndisDB_root = '/workspace/data/ndisDB/train'
subset_ndisDB_root = '/workspace/subset/ndisDB/train'
os.makedirs(subset_ndisDB_root,exist_ok=True)
subset_triplet_vis_root =f'{subset_ndisDB_root}/vis_triplet'
subset_triplet_vis_root2 =f'{subset_ndisDB_root}/vis_triplet2'
nagative_mining_type = 'semi_hard+hard_v2'
margin = 0.2
topk = 1
sub_triplet_path = f'{subset_ndisDB_root}/ndisDB_triplet_{nagative_mining_type}_{margin}_{topk}_subset.csv'

triplet_path = f'{ndisDB_root}/ndisDB_triplet_{nagative_mining_type}_{margin}_{topk}.csv'

sub_triplet_list = []
header = None

# read triplet csv and get sub triplet
sub_triplet_len = 500
with open(triplet_path , newline='') as f:
    reader = csv.reader(f, delimiter=' ', quotechar='|')
    header = next(reader)
    if header != None:
        for idx, row in enumerate(reader):
            a, p, n = row[0].split(',')
            # print(a,p,n)
            sub_triplet_list.append([a, p, n])
            if idx==sub_triplet_len-1:
                break


# make triplet subset csv
try:
    with open(sub_triplet_path, 'w') as f:
        fieldnames = ['anchor', 'positive', 'negative']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sub_triplet in sub_triplet_list:
            a_content, p_content, n_content = sub_triplet
            try:
                writer.writerow({'anchor': a_content, 'positive': p_content, 'negative' : n_content})
            except:
                import pdb;
                pdb.set_trace()
except IOError as io_err:
    os.makedirs(os.path.dirname(sub_triplet_path),exist_ok=True)
    with open(sub_triplet_path, 'w') as f:
        fieldnames = ['anchor', 'positive', 'negative']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for sub_triplet in sub_triplet_list:
            a_content, p_content, n_content = sub_triplet
            try:
                writer.writerow({'anchor': a_content, 'positive': p_content, 'negative': n_content})
            except:
                import pdb;

                pdb.set_trace()

# copy subset data from mldisk to my workspace
for sub_triplet in sub_triplet_list:
    a_path , p_path, n_path = sub_triplet
    try:
        shutil.copy(f'{a_path}', f'{os.path.dirname(a_path.replace("data","subset"))}')
        shutil.copy(f'{p_path}', f'{os.path.dirname(p_path.replace("data","subset"))}')
        shutil.copy(f'{n_path}', f'{os.path.dirname(n_path.replace("data","subset"))}')
    except IOError as io_err:
        os.makedirs(os.path.dirname(a_path.replace("data","subset")),exist_ok=True)
        os.makedirs(os.path.dirname(p_path.replace("data","subset")),exist_ok=True)
        os.makedirs(os.path.dirname(n_path.replace("data","subset")),exist_ok=True)
        shutil.copy(f'{a_path}', f'{os.path.dirname(a_path.replace("data","subset"))}')
        shutil.copy(f'{p_path}', f'{os.path.dirname(p_path.replace("data","subset"))}')
        shutil.copy(f'{n_path}', f'{os.path.dirname(n_path.replace("data","subset"))}')


# visualize subset triplet data
image_size = 256
os.makedirs(subset_triplet_vis_root,exist_ok=True)
group_dict = {}
for i, sub_triplet in enumerate(sub_triplet_list):
    a_path , p_path, n_path = sub_triplet
    a_p_path = a_path+','+p_path
    for j, sub_triplet2 in enumerate(sub_triplet_list):
        a_path2, p_path2, n_path2 = sub_triplet2
        a_p_path2 = a_path2+','+p_path2

        if a_p_path == a_p_path2 and a_p_path not in group_dict.keys():
            group_dict[a_p_path]=[n_path2]
        elif a_p_path == a_p_path2 :
            if n_path2 not in group_dict[a_p_path]:
                group_dict[a_p_path].append(n_path2)


for a_p_path, n_paths in group_dict.items():
    a_path, p_path = a_p_path.split(',')
    n_path_len = len(n_paths)
    a_image = Image.open(f'{a_path}')
    p_image = Image.open(f'{p_path}')
    a_image = a_image.resize((image_size, image_size))
    p_image = p_image.resize((image_size, image_size))
    r = requests.get(
        'https://github.com/ProgrammingFonts/ProgrammingFonts/raw/master/Droid-Sans-Mono/droid-sans-mono-1.00/Droid%20Sans%20Mono.ttf',
        allow_redirects=True)
    font = ImageFont.truetype(io.BytesIO(r.content), size=24)
    draw = ImageDraw.Draw(a_image)
    draw.text((10, 10), "Anchor", (255, 255, 255), font=font)
    draw = ImageDraw.Draw(p_image)
    draw.text((10, 10), "Positive", (255, 255, 255), font=font)
    new_image = Image.new('RGB', ((2+n_path_len) * image_size, image_size), (250, 250, 250))
    new_image.paste(a_image, (0, 0))
    new_image.paste(p_image, (image_size, 0))

    for i, n_path in enumerate(n_paths):
        n_image = Image.open(f'{n_path}')
        n_image = n_image.resize((image_size, image_size))
        draw = ImageDraw.Draw(n_image)
        draw.text((10, 10), "Negative", (255, 255, 255), font=font)
        new_image.paste(n_image, ((2*image_size)+(i*image_size), 0))
    new_image.save(f"{subset_triplet_vis_root}/{'-'.join(a_path.split('/')[-3:])}--{'-'.join(p_path.split('/')[-3:])}.jpg", "JPEG")


#
# for a_p_path, n_paths in group_dict.items():
#     a_path, p_path = a_p_path.split(',')
#     n_path_len = len(n_paths)
#     a_image = Image.open(f'{a_path}')
#     p_image = Image.open(f'{p_path}')
#     if '008975' in a_path:
#         if '/workspace/data/ndisDB/train/query/cam/008975.jpg' == a_path:
#             pass
#         else:
#             import pdb;pdb.set_trace()
    # a_image = a_image.resize((image_size, image_size))
    # p_image = p_image.resize((image_size, image_size))
    # r = requests.get(
    #     'https://github.com/ProgrammingFonts/ProgrammingFonts/raw/master/Droid-Sans-Mono/droid-sans-mono-1.00/Droid%20Sans%20Mono.ttf',
    #     allow_redirects=True)
    # font = ImageFont.truetype(io.BytesIO(r.content), size=24)
    # draw = ImageDraw.Draw(a_image)
    # draw.text((10, 10), "Anchor", (255, 255, 255), font=font)
    # draw = ImageDraw.Draw(p_image)
    # draw.text((10, 10), "Positive", (255, 255, 255), font=font)
    # new_image = Image.new('RGB', ((2+n_path_len) * image_size, image_size), (250, 250, 250))
    # new_image.paste(a_image, (0, 0))
    # new_image.paste(p_image, (image_size, 0))
    #
    # for i, n_path in enumerate(n_paths):
    #     n_image = Image.open(f'{n_path}')
    #     n_image = n_image.resize((image_size, image_size))
    #     draw = ImageDraw.Draw(n_image)
    #     draw.text((10, 10), "Negative", (255, 255, 255), font=font)
    #     new_image.paste(n_image, ((2*image_size)+(i*image_size), 0))
    # new_image.save(f"{subset_triplet_vis_root}/{'-'.join(a_path.split('/')[-3:])}--{'-'.join(p_path.split('/')[-3:])}.jpg", "JPEG")
