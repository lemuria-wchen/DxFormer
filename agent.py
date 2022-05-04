# this script is to reproduce the random agent and rule-based agent
import numpy as np
from conf import *
from utils import load_data
from data_utils import SymptomVocab, DiseaseVocab, compute_sscom, random_agent, rule_agent

# load dataset
train_samples, test_samples = load_data(train_path), load_data(test_path)


exclude_exp = True

# # ----------------------------------------------------------------------------------------------------------------------
# # random agent
# sv = SymptomVocab(samples=train_samples, add_special_sxs=False)
# recs = random_agent(test_samples, sv, max_turn, exclude_exp)
#
# print('rec of random agent: mean/std: {}/{}'.format(
#     np.round(np.mean(recs), digits), np.round(np.std(recs), digits)))

# ----------------------------------------------------------------------------------------------------------------------
# rule-based agent
sv = SymptomVocab(samples=train_samples, add_special_sxs=False)
dv = DiseaseVocab(samples=train_samples)
cm = compute_sscom(train_samples, sv)
for max_turn in range(41):
    # max_turn = 20
    rec, pos_rec, neg_rec = rule_agent(test_samples, cm, sv, max_turn, exclude_exp)
    # print('rec of rule agent: {}/{}/{}'.format(
    #     np.round(rec, digits), np.round(pos_rec, digits), np.round(neg_rec, digits)))
    print(rec)
