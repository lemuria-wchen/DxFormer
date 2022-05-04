from torch.utils.data import DataLoader
from utils import load_data
from layers import Agent
from tqdm import tqdm
import numpy as np

from data_utils import *
from conf import *


# load dataset
train_samples, test_samples = load_data(train_path), load_data(test_path)
test_size = len(test_samples)

# construct symptom & disease vocabulary
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, shuffle=False, collate_fn=pg_collater)

# compute disease-symptom co-occurrence matrix
dscom = compute_dscom(train_samples, sv, dv)

# init reward distributor
rd = RewardDistributor(sv, dv, dscom)

# init patient simulator
ps = PatientSimulator(sv)

# init agent
model = Agent(num_sxs, num_dis).to(device)

max_turn = 12
metric_model_path = 'saved/{}/dense_once/metric_model_{}.pt'.format('mz10', max_turn)
model.load(metric_model_path)


# 当症状的不确定性很大，确定性很小
# 疾病的确定性很大，确定性很大

eps = [0.950, 0.955, 0.960, 0.965, 0.970, 0.975, 0.980, 0.985, 0.990, 0.995, 1.0]

turns = []
accs = []

for e in tqdm(eps):
    test_num_hits = 0
    actual_turns = []
    max_probs = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_ds_loader):
            actual_turn, is_success, max_prob = model.execute(batch, ps, sv, max_turn, e)
            test_num_hits += is_success
            actual_turns.append(actual_turn)
            max_probs.append(max_prob)
    test_acc = test_num_hits / test_size
    avg_turn = np.mean(actual_turns)
    turns.append(avg_turn)
    accs.append(test_acc)
    # print('eps: {}, avg turn: {}, acc: {}.'.format(
    #     eps, np.round(np.mean(actual_turns), digits), np.round(test_acc, digits)))

# eps: 1.0, avg turn: 10.0, acc: 0.7536.
# eps: 0.99, avg turn: 8.0696, acc: 0.742.
# eps: 1.0, avg turn: 20.0, acc: 0.6942.
# eps: 0.99, avg turn: 14.966, acc: 0.6869.
# eps: 0.95, avg turn: 10.3131, acc: 0.6408.


# dec/enc eps: 1.0/0.0, avg turn: 20.0, acc: 0.6796, avg dec/enc entropy: 0.7219/0.173
# dec/enc eps: 1.0/0.005, avg turn: 16.6748, acc: 0.6772, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.01, avg turn: 14.1092, acc: 0.6699, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.02, avg turn: 10.6942, acc: 0.6456, avg dec/enc entropy: 0.6711/0.173
# dec/enc eps: 1.0/0.03, avg turn: 8.0825, acc: 0.6165, avg dec/enc entropy: 0.6294/0.173
# dec/enc eps: 1.0/0.04, avg turn: 6.9442, acc: 0.6044, avg dec/enc entropy: 0.6294/0.173
# dec/enc eps: 1.0/0.05, avg turn: 5.8374, acc: 0.585, avg dec/enc entropy: 0.6294/0.1667
# dec/enc eps: 1.0/0.08, avg turn: 2.7718, acc: 0.5534, avg dec/enc entropy: 0.5923/0.1667
# dec/enc eps: 1.0/0.1, avg turn: 1.4417, acc: 0.5291, avg dec/enc entropy: 0.5923/0.1667


entropys = []
model.eval()
with torch.no_grad():
    for batch in tqdm(test_ds_loader):
        entropys.append(model.execute(batch, ps, sv, max_turn))

ess = []
for i in range(max_turn):
    es = sum([entropy[i] for entropy in entropys])
    ess.append(es)

