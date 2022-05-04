# this script is the definition for neural layers
import os
import torch
import torch.nn as nn
from torch.nn import functional
from torch.distributions import Categorical
from tqdm import tqdm

import math

from data_utils import SymptomVocab, PatientSimulator, device
from conf import *


# sinusoid position embedding
class PositionalEncoding(nn.Module):

    def __init__(self, emb_dim: int):
        super().__init__()
        self.dropout = nn.Dropout(p=pos_dropout)

        position = torch.arange(pos_max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_dim, 2) * (-math.log(10000.0) / emb_dim))
        pe = torch.zeros(pos_max_len, 1, emb_dim)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# transformer-based decoder
class SymptomDecoderXFMR(nn.Module):

    def __init__(self, sx_embedding, attr_embedding, pos_embedding, num_sxs: int, emb_dim: int):
        super().__init__()

        self.num_sxs = num_sxs

        self.sx_embedding = sx_embedding
        self.attr_embedding = attr_embedding
        self.pos_embedding = pos_embedding

        self.decoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=emb_dim,
                nhead=dec_num_heads,
                dim_feedforward=dec_dim_feedforward,
                dropout=dec_dropout,
                activation='relu'),
            num_layers=dec_num_layers)

        self.sx_fc = nn.Linear(emb_dim, num_sxs)

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        if not sx_one_hot and not attr_one_hot:
            inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids)
        else:
            inputs = torch.cat([self.sx_embedding(sx_ids), self.attr_embedding(attr_ids)], dim=-1)
        if dec_add_pos:
            inputs = self.pos_embedding(inputs)
        outputs = self.decoder(inputs, mask, src_key_padding_mask)
        return outputs

    def get_features(self, outputs):
        features = self.sx_fc(outputs)
        return features

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / math.log(self.num_sxs)

    def init_repeat_score(self, bsz: int, sv: SymptomVocab, batch: dict = None):
        prob = torch.zeros(bsz, self.num_sxs, device=device)
        prob[:, :sv.num_special] = float('-inf')
        prob[:, sv.end_idx] = 0
        if exclude_exp:
            assert batch is not None
            for idx in range(bsz):
                for sx in batch['exp_sx_ids'][:, idx]:
                    if sx != sv.pad_idx:
                        prob[idx, sx] = float('-inf')
        return prob

    @staticmethod
    def update_repeat_score(action, score):
        for act, sc in zip(action, score):
            sc[act.item()] = float('-inf')

    def simulate(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, max_turn: int, inference: bool = False):
        # 初始化输入
        _, bsz = batch['exp_sx_ids'].shape
        sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
        attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
        # 初始化重复分数，手动将选择特殊symptom的action的概率设置为无穷小
        repeat_score = self.init_repeat_score(bsz, sv, batch)
        actions, log_probs = [], []
        # 采样 trajectory
        if max_turn > 0:
            for step in range(max_turn):
                # 前向传播计算选择每个action的概率
                src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()
                outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
                features = self.get_features(outputs[-1])
                if inference:
                    # greedy decoding
                    action = (features + repeat_score).argmax(dim=-1)
                else:
                    # 根据policy网络当前的参数抽样
                    policy = Categorical(functional.softmax(features + repeat_score, dim=-1))
                    action = policy.sample()
                    log_probs.append(policy.log_prob(action))
                # 让已经选择的action再次被解码出的概率为无穷小
                self.update_repeat_score(action, repeat_score)
                # 与病人模拟器进行交互，病人模拟器告知agent病人是否具有该症状
                _, q_attr_ids = ps.answer(action, batch)
                # 更新 transformer 的输入
                sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
                attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])
                # 记录选择的动作和对数概率（便于之后计算回报和优化）
                actions.append(action)
        else:
            actions.append(torch.tensor([sv.end_idx] * bsz, device=device))
            log_probs.append(torch.tensor([0] * bsz, device=device))
        # 返回整个batch的 trajectory 和对数概率
        si_actions = torch.stack(actions, dim=1)
        si_log_probs = None if inference else torch.stack(log_probs, dim=1)
        si_sx_ids = sx_ids.permute((1, 0))
        si_attr_ids = attr_ids.permute((1, 0))
        return si_actions, si_log_probs, si_sx_ids, si_attr_ids

    def inference(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, max_turn: int):
        return self.simulate(batch, ps, sv, max_turn, inference=True)

    def generate(self, ds_loader, ps: PatientSimulator, sv: SymptomVocab, max_turn: int):
        from data_utils import to_list
        ds_sx_ids, ds_attr_ids, ds_labels = [], [], []
        self.eval()
        with torch.no_grad():
            for batch in tqdm(ds_loader):
                _, _, sx_ids, attr_ids = self.inference(batch, ps, sv, max_turn)
                ds_sx_ids.extend(to_list(sx_ids))
                ds_attr_ids.extend(to_list(attr_ids))
                ds_labels.extend(to_list(batch['labels']))
        return ds_sx_ids, ds_attr_ids, ds_labels


class SymptomEncoderXFMR(nn.Module):

    def __init__(self, sx_embedding, attr_embedding, num_dis):
        super().__init__()

        self.num_dis = num_dis
        self.sx_embedding = sx_embedding
        self.attr_embedding = attr_embedding

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=enc_emb_dim,
                nhead=enc_num_heads,
                dim_feedforward=enc_num_layers,
                dropout=enc_dropout,
                activation='relu'),
            num_layers=enc_num_layers)

        self.dis_fc = nn.Linear(enc_emb_dim, num_dis, bias=True)

    def forward(self, sx_ids, attr_ids, mask=None, src_key_padding_mask=None):
        if not sx_one_hot and not attr_one_hot:
            inputs = self.sx_embedding(sx_ids) + self.attr_embedding(attr_ids)
        else:
            inputs = torch.cat([self.sx_embedding(sx_ids), self.attr_embedding(attr_ids)], dim=-1)
        outputs = self.encoder(inputs, mask, src_key_padding_mask)
        return outputs

    # mean pooling feature
    def get_mp_features(self, sx_ids, attr_ids, pad_idx):
        src_key_padding_mask = sx_ids.eq(pad_idx).transpose(1, 0).contiguous()
        outputs = self.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
        seq_len, batch_size, emb_dim = outputs.shape
        mp_mask = (1 - sx_ids.eq(pad_idx).int())
        mp_mask_ = mp_mask.unsqueeze(-1).expand(seq_len, batch_size, emb_dim)
        avg_outputs = torch.sum(outputs * mp_mask_, dim=0) / torch.sum(mp_mask, dim=0).unsqueeze(-1)
        features = self.dis_fc(avg_outputs)
        return features

    def predict(self, sx_ids, attr_ids, pad_idx):
        outputs = self.get_mp_features(sx_ids, attr_ids, pad_idx)
        labels = outputs.argmax(dim=-1)
        return labels

    def inference(self, sx_ids, attr_ids, pad_idx):
        return self.simulate(sx_ids, attr_ids, pad_idx, inference=True)

    def compute_entropy(self, features):
        return torch.distributions.Categorical(functional.softmax(features, dim=-1)).entropy().item() / self.num_dis

    @staticmethod
    def compute_max_prob(features):
        return torch.max(functional.softmax(features, dim=-1))


class Agent(nn.Module):

    def __init__(self, num_sxs: int, num_dis: int):

        super().__init__()

        if sx_one_hot:
            sx_embedding = nn.Embedding(num_sxs, num_sxs)
            sx_embedding.weight.data = torch.eye(num_sxs)
            sx_embedding.weight.requires_grad = False
            self.sx_embedding = sx_embedding
        else:
            self.sx_embedding = nn.Embedding(num_sxs, dec_emb_dim, padding_idx=0)

        if attr_one_hot:
            attr_embedding = nn.Embedding(num_attrs, num_attrs)
            attr_embedding.weight.data = torch.eye(num_attrs)
            attr_embedding.weight.requires_grad = False
            self.attr_embedding = attr_embedding
        else:
            self.attr_embedding = nn.Embedding(num_attrs, dec_emb_dim, padding_idx=0)

        if self.sx_embedding.weight.data.shape[-1] != self.attr_embedding.weight.data.shape[-1]:
            emb_dim = self.sx_embedding.weight.data.shape[-1] + self.attr_embedding.weight.data.shape[-1]
        else:
            emb_dim = dec_emb_dim

        self.pos_embedding = PositionalEncoding(emb_dim)

        self.symptom_decoder = SymptomDecoderXFMR(
            self.sx_embedding, self.attr_embedding, self.pos_embedding, num_sxs, emb_dim)

        self.symptom_encoder = SymptomEncoderXFMR(
           self.sx_embedding, self.attr_embedding, num_dis
        )

    def forward(self):
        pass

    def load(self, path):
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            if verbose:
                print('loading pre-trained parameters from {} ...'.format(path))

    def save(self, path):
        torch.save(self.state_dict(), path)
        if verbose:
            print('saving best model to {}'.format(path))

    def execute(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, max_turn: int, eps: float):
        from data_utils import make_features_xfmr
        _, bsz = batch['exp_sx_ids'].shape
        sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
        attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
        repeat_score = self.symptom_decoder.init_repeat_score(bsz, sv, batch)
        for step in range(max_turn + 1):
            # 每一个step，先观察 encoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
            si_sx_ids = sx_ids.clone().permute((1, 0))
            si_attr_ids = attr_ids.clone().permute((1, 0))
            si_sx_feats, si_attr_feats = make_features_xfmr(
                sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
            dc_outputs = self.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
            prob = self.symptom_encoder.compute_max_prob(dc_outputs).item()
            if prob > eps or step == max_turn:
                is_success = batch['labels'].eq(dc_outputs.argmax(dim=-1)).item()
                max_prob = prob
                return step, is_success, max_prob
            # 再观察 decoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
            src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()
            outputs = self.symptom_decoder.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
            features = self.symptom_decoder.get_features(outputs[-1])
            action = (features + repeat_score).argmax(dim=-1)
            self.symptom_decoder.update_repeat_score(action, repeat_score)
            _, q_attr_ids = ps.answer(action, batch)
            sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
            attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])

    # def execute(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, max_turn: int, dec_eps: float, enc_eps: float):
    #     from data_utils import make_features_xfmr
    #     _, bsz = batch['exp_sx_ids'].shape
    #     sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
    #     attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
    #     repeat_score = self.symptom_decoder.init_repeat_score(bsz, sv, batch)
    #     actual_turn, is_success, dec_ent, enc_ent = None, None, None, None
    #     for step in range(max_turn + 1):
    #         # 每一个step，先观察 encoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
    #         si_sx_ids = sx_ids.clone().permute((1, 0))
    #         si_attr_ids = attr_ids.clone().permute((1, 0))
    #         si_sx_feats, si_attr_feats = make_features_xfmr(
    #             sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
    #         dc_outputs = self.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
    #         enc_entropy = self.symptom_encoder.compute_entropy(dc_outputs)
    #         # 再观察 decoder 的 entropy 是否已经有足够的 confidence 给出诊断结果
    #         src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()
    #         outputs = self.symptom_decoder.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
    #         features = self.symptom_decoder.get_features(outputs[-1])
    #         dec_entropy = self.symptom_decoder.compute_entropy(features + repeat_score)
    #         # 当症状的不确定性很大，确定性很小
    #         # 疾病的确定性很大，确定性很大
    #         if enc_entropy < enc_eps or (step > 5 and dec_entropy > dec_eps) or step == max_turn:
    #             actual_turn = step
    #             is_success = batch['labels'].eq(dc_outputs.argmax(dim=-1)).item()
    #             dec_ent, enc_ent = dec_entropy, enc_entropy
    #             break
    #         action = (features + repeat_score).argmax(dim=-1)
    #         self.symptom_decoder.update_repeat_score(action, repeat_score)
    #         _, q_attr_ids = ps.answer(action, batch)
    #         sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
    #         attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])
    #     return actual_turn, is_success, dec_ent, enc_ent

    # def execute(self, batch: dict, ps: PatientSimulator, sv: SymptomVocab, max_turn: int):
    #     from data_utils import make_features_xfmr
    #     _, bsz = batch['exp_sx_ids'].shape
    #     sx_ids = torch.cat([batch['exp_sx_ids'], ps.init_sx_ids(bsz)])
    #     attr_ids = torch.cat([batch['exp_attr_ids'], ps.init_attr_ids(bsz)])
    #     repeat_score = self.symptom_decoder.init_repeat_score(bsz, sv, batch)
    #     enc_entropys = []
    #     for step in range(max_turn):
    #         src_key_padding_mask = sx_ids.eq(sv.pad_idx).transpose(1, 0).contiguous()
    #         outputs = self.symptom_decoder.forward(sx_ids, attr_ids, src_key_padding_mask=src_key_padding_mask)
    #         features = self.symptom_decoder.get_features(outputs[-1])
    #         action = (features + repeat_score).argmax(dim=-1)
    #         self.symptom_decoder.update_repeat_score(action, repeat_score)
    #         _, q_attr_ids = ps.answer(action, batch)
    #         sx_ids = torch.cat([sx_ids, action.unsqueeze(dim=0)])
    #         attr_ids = torch.cat([attr_ids, q_attr_ids.unsqueeze(dim=0)])
    #         si_sx_ids = sx_ids.clone().permute((1, 0))
    #         si_attr_ids = attr_ids.clone().permute((1, 0))
    #         si_sx_feats, si_attr_feats = make_features_xfmr(
    #             sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
    #         dc_outputs = self.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
    #         enc_entropy = self.symptom_encoder.compute_entropy(dc_outputs)
    #         enc_entropys.append(enc_entropy)
    #     return enc_entropys
