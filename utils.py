import json
import os
import pickle


"""
# symptom attribute mapping
'0' -> negative
'1' -> positive
'2' -> not sure
"""


# pre-process
def convert_dxy(samples, prefix):
    return [{
        'pid': prefix + '-' + str(pid + 1),
        'exp_sxs': {sx: '1' if attr else '0' for sx, attr in sample['goal']['explicit_inform_slots'].items()},
        'imp_sxs': {sx: '1' if attr else '0' for sx, attr in sample['goal']['implicit_inform_slots'].items()},
        'label': sample['disease_tag']
    } for pid, sample in enumerate(samples)]


def convert_mz4(samples, prefix):
    return convert_dxy(samples, prefix)


def convert_mz10(samples, keep_ns: bool = True):
    # keep_ns: map 'not sure (NS)' as 'negative (NEG)' or as a new category
    if keep_ns:
        attr_map = {'0': '0', '1': '1', '2': '2'}
    else:
        attr_map = {'0': '0', '1': '1', '2': '0'}
    return [{
        'pid': pid,
        'exp_sxs': {sx: '1' for sx in sample['explicit_info']['Symptom']},
        'imp_sxs': {sx: attr_map.get(attr) for sx, attr in sample['implicit_info']['Symptom'].items()},
        'label': sample['diagnosis']
    } for pid, sample in samples.items()]


def filter_duplicate(samples):
    _samples = []
    for sample in samples:
        exp_sxs, imp_sxs = sample['exp_sxs'], sample['imp_sxs']
        for sx in exp_sxs:
            if sx in imp_sxs:
                imp_sxs.pop(sx)
        _samples.append({'exp_sxs': exp_sxs, 'imp_sxs': imp_sxs, 'label': sample['label']})
    return _samples


def load_data(paths) -> list:
    if not isinstance(paths, (tuple, list)):
        assert isinstance(paths, str)
        paths = [paths]
    data = []
    for path in paths:
        with open(path, encoding='utf-8') as f:
            data.extend(json.load(f))
            print('loading json object from {}.'.format(path))
    return data


def load_json(path):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_data(obj, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        print('dumping json object to {}.'.format(path))


def json_dump(train_samples, test_samples, ds):
    print('-' * 50)
    print('# num of total/train/test examples: {}/{}/{}'.format(
        len(train_samples) + len(test_samples), len(train_samples), len(test_samples)))
    write_data(train_samples, 'data/{}/train_set.json'.format(ds))
    write_data(test_samples, 'data/{}/test_set.json'.format(ds))


def tokenizer(x):
    return x.split()


def load_pickle(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print('loading pickle formatted model from {}.'.format(path))
    return obj


def save_pickle(obj, path, verbose: bool = False):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    if verbose:
        print('dumping pickle formatted model to {}.'.format(path))


def make_dirs(paths):
    if not isinstance(paths, (tuple, list)):
        assert isinstance(paths, str)
        paths = [paths]
    for path in paths:
        dir_path = os.path.dirname(path)
        if not os.path.exists(dir_path):
            print('making dir {} ...'.format(dir_path))
        os.makedirs(dir_path, exist_ok=True)


def set_path(name, dataset, exp_name, max_turn, num, num_repeats):
    if num_repeats == 1:
        path = 'saved/{}/{}/{}_{}.pt'.format(dataset, exp_name, name, max_turn)
    else:
        path = 'saved/{}/{}/{}_{}_{}.pt'.format(dataset, exp_name, name, max_turn, num + 1)
    return path
