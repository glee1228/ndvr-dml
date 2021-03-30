from datasets.vcdb import VCDB
import csv
from tqdm import tqdm

vcdb_root = '/mldisk/nfs_shared_/MLVD/VCDB'
positive_path = '/mldisk/nfs_shared_/dh/VCDB/vcdb_positive_1fps_Deduplication.csv'
dataset = VCDB(root=vcdb_root)

with open(positive_path, 'w') as f:
    fieldnames = ['a', 'b']
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    a_b = []
    for key, values in tqdm(dataset.frame_annotation.items()):
        for value in values:
            for time in value[1]:
                try:
                    if len(a_b)>10000:
                        a_b = []
                    a_content = f'{key}/{time[0] + 1:06}.jpg'
                    b_content = f'{value[0]}/{time[1] + 1:06}.jpg'
                    if not [b_content,a_content] in a_b:
                        a_b.append([a_content, b_content])
                        writer.writerow({'a': a_content, 'b': b_content})
                except:
                    import pdb;
                    pdb.set_trace()


