from utils import *


# dxy 数据集有 423 个训练集 和 104 个测试集
dxy_path = 'data/dxy/raw/dxy_dialog_data_dialog_v2.pickle'

# mz4 数据集有多个版本
# 第一个版本为 ACL 2018 年文章的版本，共 568 个训练集 和 142 个测试集合。
# 第二个版本为 HRL 2020 年文章的版本，共 1214 个训练集，174 个验证集 和 345 个测试集。
mz4_path = 'data/mz4/raw/acl2018-mds.p'
mz4_1k_path = 'data/mz4-1k/raw/'

# mz10 数据集有 1214 个训练集，174 个验证集 和 345 个测试集。
mz10_path = 'data/mz10/raw/'


# dxy
data = pickle.load(open(dxy_path, 'rb'))

train_samples, test_samples = convert_dxy(data['train'], prefix='train'), convert_dxy(data['test'], prefix='test')

json_dump(train_samples, test_samples, 'dxy')

# mz4
data = pickle.load(open(os.path.join(mz4_path), 'rb'))

train_samples = convert_mz4(data['train'], prefix='train')
test_samples = convert_mz4(data['test'], prefix='test')

json_dump(train_samples, test_samples, 'mz4')

# mz4-1k
train_data = pickle.load(open(os.path.join(mz4_1k_path, 'goal_set.p'), 'rb'))
test_data = pickle.load(open(os.path.join(mz4_1k_path, 'goal_test_set.p'), 'rb'))

train_samples = convert_mz4(train_data['train'] + train_data['dev'], prefix='train')
test_samples = convert_mz4(test_data['test'], prefix='test')

json_dump(train_samples, test_samples, 'mz4-1k')

# mz10
train_data = load_json(os.path.join(mz10_path, 'train.json'))
dev_data = load_json(os.path.join(mz10_path, 'dev.json'))
test_data = load_json(os.path.join(mz10_path, 'test.json'))

train_samples = convert_mz10(train_data) + convert_mz10(dev_data)
test_samples = convert_mz10(test_data)

json_dump(train_samples, test_samples, 'mz10')
