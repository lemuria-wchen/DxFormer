from torch.utils.data import DataLoader

from utils import load_data, set_path
from layers import Agent

from utils import save_pickle, make_dirs
from data_utils import *
from conf import *


# load dataset
train_samples, test_samples = load_data(train_path), load_data(test_path)
train_size, test_size = len(train_samples), len(test_samples)

# construct symptom & disease vocabulary
sv = SymptomVocab(samples=train_samples, add_special_sxs=True, min_sx_freq=min_sx_freq, max_voc_size=max_voc_size)
dv = DiseaseVocab(samples=train_samples)
num_sxs, num_dis = sv.num_sxs, dv.num_dis

# init dataloader
train_ds = SymptomDataset(train_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
train_ds_loader = DataLoader(train_ds, batch_size=train_bsz, num_workers=num_workers, shuffle=True, collate_fn=pg_collater)

test_ds = SymptomDataset(test_samples, sv, dv, keep_unk, add_tgt_start=True, add_tgt_end=True)
test_ds_loader = DataLoader(test_ds, batch_size=test_bsz, num_workers=num_workers, shuffle=False, collate_fn=pg_collater)

# compute disease-symptom co-occurrence matrix
dscom = compute_dscom(train_samples, sv, dv)

# init reward distributor
rd = RewardDistributor(sv, dv, dscom)

# init patient simulator
ps = PatientSimulator(sv)

print('training...')

mtra_path = 'saved/{}/tra_{}.pt'.format(train_dataset, exp_name)
results = []

for max_turn in range(num_turns, -1, -1):
    # for max_turn in range(1, num_turns + 1):
    for num in range(num_repeats):
        # init epoch recorder
        train_sir = SIRecorder(num_samples=len(train_ds), num_imp_sxs=compute_num_sxs(train_samples), digits=digits)
        test_sir = SIRecorder(num_samples=len(test_ds), num_imp_sxs=compute_num_sxs(test_samples), digits=digits)
        # init agent
        model = Agent(num_sxs, num_dis).to(device)
        # load parameters from pre-trained models if exits
        model.load(best_pt_path)
        # init optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        # compute loss of disease classification
        criterion = torch.nn.CrossEntropyLoss().to(device)
        # model path
        rec_model_path = set_path('rec_model', train_dataset, exp_name, max_turn, num, num_repeats)
        acc_model_path = set_path('acc_model', train_dataset, exp_name, max_turn, num, num_repeats)
        metric_model_path = set_path('metric_model', train_dataset, exp_name, max_turn, num, num_repeats)
        log_path = set_path('sir', train_dataset, exp_name, max_turn, num, num_repeats)
        make_dirs([rec_model_path, acc_model_path, metric_model_path, log_path])
        # start
        epochs = train_epochs if max_turn > 0 else warm_epoch
        for epoch in range(epochs):
            num_hits_train, num_hits_test = 0, 0
            # training
            for batch in train_ds_loader:
                np_batch = to_numpy(batch)
                # symptom inquiry
                # simulate
                model.train()
                si_actions, si_log_probs, _, _ = model.symptom_decoder.simulate(batch, ps, sv, max_turn)
                # compute reward of each step of symptom recovery
                si_rewards = rd.compute_sr_reward(si_actions, np_batch, epoch, train_sir)
                # compute loss
                si_loss = - torch.sum(to_tensor_(si_rewards) * si_log_probs)
                # disease classification
                model.eval()
                with torch.no_grad():
                    _, _, si_sx_ids, si_attr_ids = model.symptom_decoder.inference(batch, ps, sv, max_turn)
                model.train()
                if epoch < warm_epoch and max_turn > 0:
                    # 热身阶段（即前一半epochs）只训练 decoder
                    loss = si_loss
                else:
                    # 热身阶段完毕之后，decoder和encoder一起训练（原因是decoder更难训练）
                    si_sx_feats, si_attr_feats = make_features_xfmr(
                        sv, batch, si_sx_ids, si_attr_ids, merge_act=True, merge_si=True)
                    dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                    # 在训练疾病分类器时，将预测的数据和完整的数据混合作为输入
                    double_labels = torch.cat([batch['labels'], batch['labels']])
                    dc_loss = criterion(dc_outputs, double_labels)
                    num_hits_train += torch.sum(double_labels.eq(dc_outputs.argmax(dim=-1))).item()
                    loss = si_loss + dc_loss
                # compute the gradient and optimize the parameters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            train_acc = num_hits_train / (2 * train_size)
            train_sir.update_acc(epoch, train_acc)
            if verbose:
                train_sir.epoch_summary(epoch)
            # evaluation
            model.eval()
            with torch.no_grad():
                for batch in test_ds_loader:
                    np_batch = to_numpy(batch)
                    # symptom inquiry
                    si_actions, _, si_sx_ids, si_attr_ids = model.symptom_decoder.inference(batch, ps, sv, max_turn)
                    # compute reward of each step of symptom recovery
                    _ = rd.compute_sr_reward(si_actions, np_batch, epoch, test_sir)
                    # make features
                    si_sx_feats, si_attr_feats = make_features_xfmr(
                        sv, batch, si_sx_ids, si_attr_ids, merge_act=False, merge_si=True)
                    # make diagnosis
                    dc_outputs = model.symptom_encoder.get_mp_features(si_sx_feats, si_attr_feats, sv.pad_idx)
                    num_hits_test += torch.sum(batch['labels'].eq(dc_outputs.argmax(dim=-1))).item()
            test_acc = num_hits_test / test_size
            test_sir.update_acc(epoch, test_acc)
            if verbose:
                test_sir.epoch_summary(epoch)
            cur_rec, best_rec, _, _, cur_acc, best_acc, _, _, cur_met, best_met, _, _, _ = test_sir.report(epoch, digits, alpha, verbose)
            if cur_rec >= best_rec:
                model.save(rec_model_path)  # save the model with best recall
            if cur_acc >= best_acc:
                model.save(acc_model_path)  # save the model with best accuracy
            if cur_met >= best_met:
                model.save(metric_model_path)   # save the model with best metric
            if verbose:
                print('-' * 100)
        # end training
        _, best_rec, best_rec_epoch, best_rec_acc, _, best_acc, best_acc_epoch, best_acc_rec, _, _, best_met_epoch, best_met_rec, best_met_acc = test_sir.report(epochs - 1, digits, alpha, verbose)
        result = {
            'max_turn': max_turn,
            'num': num,
            'best_rec_epoch': best_rec_epoch,
            'best_rec': round(best_rec, digits),
            'best_rec_acc': round(best_rec_acc, digits),
            'best_acc_epoch': best_acc_epoch,
            'best_acc_rec': round(best_acc_rec, digits),
            'best_acc': round(best_acc, digits),
            'best_met_epoch': best_met_epoch,
            'best_met_rec': round(best_met_rec, digits),
            'best_met_acc': round(best_met_acc, digits),
        }
        print(result)
        results.append(result)
        save_pickle((train_sir, test_sir), log_path)
        save_pickle(results, mtra_path)
