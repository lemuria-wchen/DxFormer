# this is all the parameters to train our decoder-encoder based agent
import argparse

# parameters (training)
parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--train_dataset', default='mz4', help='choose the training dataset')
parser.add_argument('-d2', '--test_dataset', default='mz4', help='choose the testing dataset')
args = parser.parse_args()

# dataset for training
train_dataset = args.train_dataset

# dataset for testing
test_dataset = train_dataset if train_dataset != 'all' else args.test_dataset

# check validity
ds_range = ['all', 'dxy', 'mz4', 'mz10']
# train_ds_range = ds_range + ['all']

assert train_dataset in ds_range
assert test_dataset in ds_range

# gpu device number
# device_num = {'dxy': 0, 'mz4': 1, 'mz10': 2}.get(train_dataset)
device_num = 0

# train/test data path
train_path = []
if train_dataset != 'all':
    train_path.append('data/{}/train_set.json'.format(train_dataset))
else:
    for ds in ds_range[1:]:
        train_path.append('data/{}/train_set.json'.format(ds))

test_path = []
if test_dataset != 'all':
    test_path.append('data/{}/test_set.json'.format(test_dataset))
else:
    for ds in ds_range[1:]:
        test_path.append('data/{}/test_set.json'.format(ds))
# test_path = ['data/{}/test_set.json'.format(test_dataset)]

best_pt_path = 'saved/{}/best_pt_model.pt'.format(train_dataset)
last_pt_path = 'saved/{}/last_pt_model.pt'.format(train_dataset)


# global settings
# suffix = {'0': '-Negative', '1': '-Positive', '2': '-Not-Sure'}
suffix = {'0': '-Negative', '1': '-Positive', '2': '-Negative'}
min_sx_freq = None
max_voc_size = None
keep_unk = True
digits = 4

# model hyperparameter setting
# group 1: position embeddings
pos_dropout = 0.2
pos_max_len = 80

# group 2: transformer decoder
sx_one_hot = False
attr_one_hot = False
num_attrs = 5

dec_emb_dim = 128 if train_dataset == 'dxy' else 512
# dec_emb_dim = 128 if train_dataset == 'dxy' else 256
dec_dim_feedforward = 2 * dec_emb_dim

dec_num_heads = 4
dec_num_layers = 4
dec_dropout = 0.2

dec_add_pos = True
exclude_exp = True if train_dataset != 'mz10' else False

# group 3: transformer encoder
enc_emb_dim = dec_emb_dim
enc_dim_feedforward = 2 * enc_emb_dim

enc_num_heads = 4
enc_num_layers = 1
enc_dropout = 0.2 if train_dataset == 'dxy' else 0.2

# group 3: training
num_workers = 0

pt_learning_rate = 3e-4 if train_dataset == 'dxy' else 1e-4
# pt_train_epochs = 100 if train_dataset == 'dxy' else 50
pt_train_epochs = 200

learning_rate = 1e-4
train_epochs = {'all': 200, 'dxy': 40, 'mz4': 30, 'mz10': 20}.get(train_dataset)
warm_epoch = train_epochs // 2

train_bsz = 128
test_bsz = 128

alpha = 0.2
exp_name = 'dense_all'
num_turns = 10
num_repeats = 1
verbose = True
