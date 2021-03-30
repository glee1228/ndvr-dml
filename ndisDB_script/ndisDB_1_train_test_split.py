import os
from sklearn.model_selection import train_test_split
import shutil

ndisDB_root = '/workspace/data/ndisDB'
origin_path = f'{ndisDB_root}/ndisDB_origin'
train_path = f'{ndisDB_root}/train'
test_path = f'{ndisDB_root}/test'
original_path = f'{origin_path}/original'
query_path = f'{origin_path}/query'

train_query_path = f'{train_path}/query'
train_origin_path = f'{train_path}/origin'
train_db_path = f'{train_path}/db'
test_query_path = f'{test_path}/query'
test_origin_path = f'{test_path}/origin'
test_db_path = f'{test_path}/db'

original_data_lst = os.listdir(original_path)
origin_img_lst = sorted([name for name in original_data_lst if os.path.splitext(os.path.basename(name))[1] in ['.jpg','.png','.JPG','.PNG','jpeg','JPEG']])
transform_types = ['cam','caption','contrast','crop','letterbox','logo','noise','rotation','scale']
transformed_img_dict = {}

for transform_type in transform_types:
    data_lst = os.listdir(f'{query_path}/{transform_type}')
    img_lst = [name for name in data_lst if
     os.path.splitext(os.path.basename(name))[1] in ['.jpg', '.png', '.JPG', '.PNG', 'jpeg', 'JPEG']]
    transformed_img_dict[transform_type]=img_lst

query_set = []
for key,value in transformed_img_dict.items():
    query_set.extend(sorted(value))

query_set = sorted(list(set(query_set)))
db_set = [item for item in origin_img_lst if item not in query_set]
train_query_set, test_query_set, _, _  = train_test_split(query_set, list(range(0,len(query_set))), test_size=0.2, shuffle=True, random_state=777)
train_db_set , test_db_set, _, _ = train_test_split(db_set, list(range(0,len(db_set))),test_size=0.3, shuffle=True,random_state=777)

os.makedirs(train_path)
os.makedirs(test_path)
os.makedirs(train_db_path)
os.makedirs(train_origin_path)
os.makedirs(test_db_path)
os.makedirs(test_origin_path)
for type in transform_types:
    os.makedirs(f'{train_query_path}/{type}')
    os.makedirs(f'{test_query_path}/{type}')

for imgname in train_query_set:
    query = None
    for type in transform_types:
        query = f'{query_path}/{type}/{imgname}'
        train_query = f'{train_query_path}/{type}'
        shutil.copy(query, train_query)
    origin = f'{train_origin_path}'
    shutil.copy(query,origin)

for imgname in train_db_set:
    db = f'{original_path}/{imgname}'
    train_db = f'{train_db_path}'
    shutil.copy(db, train_db)

for imgname in test_query_set:
    query = None
    for type in transform_types:
        query = f'{query_path}/{type}/{imgname}'
        test_query = f'{test_query_path}/{type}'
        shutil.copy(query, test_query)
    origin = f'{test_origin_path}'
    shutil.copy(query,origin)

for imgname in test_db_set:
    db = f'{original_path}/{imgname}'
    test_db = f'{test_db_path}'
    shutil.copy(db, test_db)
