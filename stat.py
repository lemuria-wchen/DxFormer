from utils import load_data
import numpy as np
from conf import *


class STAT:
    def __init__(self):
        train_samples, test_samples = load_data(train_path), load_data(test_path)
        self.samples = train_samples + test_samples

    def count_num_samples(self):
        print('number of diseases: {}'.format(len(self.samples)))

    def compute_num_diseases(self):
        # num of diseases
        dis = set()
        for sample in self.samples:
            dis.add(sample['label'])
        print('number of diseases: {}'.format(len(dis)))
        return len(dis)

    def compute_num_symptoms(self):
        # num of unique symptoms
        sxs = set()
        for sample in self.samples:
            for sx in sample['exp_sxs']:
                sxs.add(sx)
            for sx in sample['imp_sxs']:
                sxs.add(sx)
        print('number of symptoms: {}'.format(len(sxs)))

    def count_symptom_lens(self):
        # statistics of num of symptoms
        exp_sxs_lens = np.array([len(sample['exp_sxs']) for sample in self.samples])
        print('number of avg./max. exp: {}/{}'.format(
            np.round(np.mean(exp_sxs_lens), digits), np.max(exp_sxs_lens)))
        imp_sxs_lens = np.array([len(sample['imp_sxs']) for sample in self.samples])
        print('number of avg./max. imp: {}/{}'.format(
            np.round(np.mean(imp_sxs_lens), digits), np.max(imp_sxs_lens)))

    def count_zero_symptom(self):
        # statistics of zero symptoms
        exp_0, imp_0, both = 0, 0, 0
        for sample in self.samples:
            exp_0 += len(sample['exp_sxs']) == 0
            imp_0 += len(sample['imp_sxs']) == 0
            both += len(sample['exp_sxs']) + len(sample['imp_sxs']) == 0
        print('number of zero exp/imp/both symptoms: {}/{}/{}'.format(exp_0, imp_0, both))

    def count_duplicates(self):
        num_duplicates = 0
        total = 0
        for sample in self.samples:
            for imp_sx, _ in sample['imp_sxs'].items():
                if imp_sx in sample['exp_sxs']:
                    num_duplicates += 1
                total += 1
        print('duplicates of implicit symptoms from explicit symptoms: {}/{}'.format(
            num_duplicates, round(num_duplicates / total, digits)))


stat = STAT()
stat.count_num_samples()
stat.compute_num_diseases()
stat.compute_num_symptoms()
stat.count_zero_symptom()
stat.count_symptom_lens()
stat.count_duplicates()


# results
#           #samples    #diseases   #symptoms   #avg/max exps   #avg/max imps
# dxy       527         5           41          3.07/7          1.67/6
# mz4       1733        4           230         2.091/10        5.463/21
# mz10      4116        10          331         1.73/12         6.6/25
