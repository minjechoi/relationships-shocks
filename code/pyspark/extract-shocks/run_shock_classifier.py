import sys
import torch
import argparse

sys.path.insert(0, "/home/minje/libraries/transformers-bertweet/src")

import gzip, bz2, os
from os.path import join
import matplotlib.pyplot as plt
import ujson as json
import seaborn as sns
import random
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, TextClassificationPipeline
from transformers import RobertaModel, RobertaConfig, RobertaForSequenceClassification
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, precision_recall_curve


def main():
    print('pid: ', os.getpid())
    parser = argparse.ArgumentParser()

    # directory hyperparameters
    parser.add_argument(
        "--train_file_or_dir",
        default='/shared/0/projects/relationships/working-dir/life-events/train-data/',
        type=str,
        help="The input training data file or directory containing these files (if latter, train on all files with 'train' in name)"
    )
    parser.add_argument(
        "--val_file_or_dir",
        default='/shared/0/projects/relationships/working-dir/life-events/train-data/reg.valid.df.tsv',
        type=str, help="The validation data file or directory"
    )
    parser.add_argument(
        "--test_file_or_dir",
        default='/shared/0/projects/relationships/working-dir/life-events/train-data/reg.test.df.tsv',
        type=str, help="The test data file or directory"
    )
    parser.add_argument(
        "--infer_file_or_dir",
        default='/shared/0/projects/relationships/working-dir/life-events/filtered-tweets/',
        type=str, help="The infer data file or directory"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
        default='/shared/0/projects/relationships/working-dir/life-events/saved-weights/'
    )
    parser.add_argument(
        "--model_name", type=str, help="Prefix for saving model name (reg, cls1, cls2,...)"
    )
    parser.add_argument(
        "--event_type", type=str, default=None,
        help="If set to none, runs for all categories. If set to event type, trains classifier only dealing that data"
    )
    parser.add_argument(
        "--model_file_or_dir",
        help="Path to the actual pickle for for the trained classifier model or path (for evaluation), if path runs all checkpoints within"
    )
    parser.add_argument(
        "--thresh", type=float, default=0.5,
        help="Threshold for model when inferring positive tweets using model"
    )

    # script-related hyperparameters
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_val", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true", help="Whether to run eval on the test set.")
    parser.add_argument("--do_infer", action="store_true",
                        help="Whether to run infer probabilities from an unlabeled set.")
    parser.add_argument("--do_extract", action="store_true",
                        help="Whether to run infer probabilities from an unlabeled set.")

    # model & data hyperparameters
    parser.add_argument("--tokenize", action="store_true", help="whether to tokenize the data, if given in text form")
    parser.add_argument("--lr", type=float, default=3e-6)

    # other hyperparameters
    parser.add_argument("--start_from", default=None, type=int, help="Checkpoint to start from")
    parser.add_argument("--cuda", action="store_true", help="whether to use GPUs")
    parser.add_argument("--gpu_id", default=None, type=int, help="GPU id to use (optional)")
    parser.add_argument("--seed", default=42, type=int, help="Seed to use for reproducibility")

    args = parser.parse_args()

    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    if args.do_train:
        # script for training
        train_bert(args)

    if args.do_test:
        print("Running evaluation script for ", args.model_file_or_dir)
        eval_bert(args)

    if args.do_infer:
        infer_bert(args)

    if args.do_extract:
        extract_high_score_tweets(args)
    return


class CustomDataset(Dataset):
    def __init__(self, typ, file_path=None, dataframe=None):
        if file_path:
            if (typ == 'train') & (os.path.isdir(file_path)):
                df = pd.DataFrame([], columns=['text', 'is_shock'])
                for filename in os.listdir(file_path):
                    if 'train' in filename:
                        df_tmp = pd.read_csv(join(file_path, filename), sep='\t')
                        df = df.append(df_tmp[['text', 'is_shock']])
            else:
                df = pd.read_csv(join(file_path), sep='\t')[['text', 'is_shock']]
        elif dataframe is not None:
            df = dataframe[['text', 'is_shock']]
        else:
            print("Error! Either the path to a dataframe file or an adtual dataframe should be provided")
        df = df.dropna()
        self.data = df.values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_bert(args):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    # load training data
    def collate_fn(arr):
        X, y = [x[0] for x in arr], [int(x[1]) for x in arr]
        enc = tokenizer(text=X, padding='longest', truncation=True, max_length=128)
        return torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']), torch.LongTensor(y)

    tr_dataset = CustomDataset(typ='train', file_path=args.train_file_or_dir)
    tr_loader = DataLoader(tr_dataset, batch_size=16, shuffle=True, num_workers=1, collate_fn=collate_fn)
    val_dataset = CustomDataset(typ='val', file_path=args.val_file_or_dir)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # load model
    bertweet = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # reinitialize

    device = torch.device('cuda')
    bertweet.to(device)
    optimizer = AdamW(bertweet.parameters(), lr=args.lr,
                      eps=1e-8)  # To reproduce BertAdam specific behavior set correct_bias=False
    n_epochs = 50
    num_warmup_steps = 100
    num_training_steps = n_epochs * len(tr_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)  # PyTorch scheduler
    max_grad_norm = 1.0

    answers = [x[1] for x in tr_dataset.data]
    n_pos = sum(answers)
    weight = (len(answers) - n_pos) / n_pos
    print(len(answers), n_pos, weight)

    # train the model
    total_loss = []
    best_score = np.inf
    best_score = 0
    for epoch in range(n_epochs):
        bertweet.train()
        pbar = tqdm(tr_loader)
        preds = []
        scores = []
        tr_loss = 0.0
        for X, masks, y in pbar:
            optimizer.zero_grad()

            n_pos = y.sum().item()
            n_neg = len(y) - n_pos
            if n_pos > 0:
                X_pos = torch.stack([X[idx] for idx, n in enumerate(y.tolist()) if n == 1])
                masks_pos = torch.stack([masks[idx] for idx, n in enumerate(y.tolist()) if n == 1])
                y_pos = torch.tensor([1] * n_pos)
                X_pos, masks_pos, y_pos = X_pos.to(device), masks_pos.to(device), y_pos.to(device)
                pos_loss, _ = bertweet(input_ids=X_pos, labels=y_pos, attention_mask=masks_pos)
                pos_loss *= weight
            if n_neg > 0:
                X_neg = torch.stack([X[idx] for idx, n in enumerate(y.tolist()) if n == 0])
                masks_neg = torch.stack([masks[idx] for idx, n in enumerate(y.tolist()) if n == 0])
                y_neg = torch.tensor([0] * n_neg)
                X_neg, masks_neg, y_neg = X_neg.to(device), masks_neg.to(device), y_neg.to(device)
                neg_loss, _ = bertweet(input_ids=X_neg, labels=y_neg, attention_mask=masks_neg)
            if (n_pos > 0) and (n_neg > 0):
                loss = pos_loss + neg_loss
            elif n_pos == 0:
                loss = neg_loss
            elif n_neg == 0:
                loss = pos_loss

            # X, masks, y = X.to(device), masks.to(device), y.to(device)
            # loss, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
            tr_loss += loss.item()
            total_loss.append(loss.item())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bertweet.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            pbar.set_description("Epoch %d/%d, loss: %1.3f" % (epoch + 1, n_epochs, loss.item()))
            # logits = logits.softmax(1)
            # scores.extend(logits[:, -1].reshape(-1).tolist())
            # preds.extend(np.round(logits[:, -1].reshape(-1).tolist()))

        # validate after each epoch
        val_loss = 0.0
        answers = [x[1] for x in val_dataset.data]
        scores = []
        # vanilla
        with torch.no_grad():
            bertweet.eval()
            for X, masks, y in val_loader:
                X, masks, y = X.to(device), masks.to(device), y.to(device)
                loss, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
                val_loss += loss.item()
                logits = logits.softmax(1)
                scores.extend(logits[:, -1].reshape(-1).tolist())

        # weighted validation loss
        n_pos = sum(answers)
        # weight = (len(answers)-n_pos)/n_pos
        # with torch.no_grad():
        #     bertweet.eval()
        #     for X, masks, y in val_loader:
        #         n_pos = y.sum().item()
        #         n_neg = len(y)-n_pos
        #         if n_pos>0:
        #             X_pos = torch.stack([X[idx] for idx,n in enumerate(y.tolist()) if n==1])
        #             masks_pos = torch.stack([masks[idx] for idx,n in enumerate(y.tolist()) if n==1])
        #             y_pos = torch.tensor([1]*n_pos)
        #             X_pos, masks_pos, y_pos = X_pos.to(device), masks_pos.to(device), y_pos.to(device)
        #             pos_loss,_ = bertweet(input_ids=X_pos, labels=y_pos, attention_mask=masks_pos)
        #             val_loss+=pos_loss*weight
        #         if n_neg>0:
        #             X_neg = torch.stack([X[idx] for idx,n in enumerate(y.tolist()) if n==0])
        #             masks_neg = torch.stack([masks[idx] for idx,n in enumerate(y.tolist()) if n==0])
        #             y_neg = torch.tensor([0]*n_neg)
        #             X_neg, masks_neg, y_neg = X_neg.to(device), masks_neg.to(device), y_neg.to(device)
        #             neg_loss,_ = bertweet(input_ids=X_neg, labels=y_neg, attention_mask=masks_neg)
        #             val_loss+=neg_loss

        # val_loss += loss.item()
        # logits = logits.softmax(1)
        # scores.extend(logits[:, -1].reshape(-1).tolist())
        # precision, recall, thresholds = precision_recall_curve(answers,scores)
        # f1_thresh_list = []
        # for t in thresholds:
        #     preds = [int(x>=t) for x in scores]
        #     f1 = f1_score(answers, preds)
        #     f1_thresh_list.append((f1,t))
        # f1,t = sorted(f1_thresh_list)[-1] # threshold that induces the largest f1 score

        # use the best f-1 from the different thresholds
        preds = [int(x >= 0.5) for x in scores]
        acc = accuracy_score(answers, preds)
        pre = precision_score(answers, preds)
        rec = recall_score(answers, preds)
        f1 = f1_score(answers, preds)
        print('Validation of Epoch %d - loss: %1.3f, acc: %1.3f, f-1: %1.3f, pre: %1.3f, rec: %1.3f' % (
            epoch + 1, val_loss, acc, f1, pre, rec))
        # print('Validation of Epoch %d - loss: %1.3f,' % (
        #     epoch + 1, val_loss))

        # update best weights and best threshold if our F-1 beats the model's best F-1
        if best_score < f1:
            best_score = f1
            # if best_score > val_loss:
            #     best_score = val_loss
            torch.save(bertweet.state_dict(), join(args.output_dir, '%s.best_weights.pth' % args.model_name))
            # with open(join(args.output_dir,'%s.best_thresh.txt'%args.model_name),'w') as outf1:
            #     outf1.write(str(t))

    return


def eval_bert(args):
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

    # load test data
    def collate_fn(arr):
        X, y = [x[0] for x in arr], [int(x[1]) for x in arr]
        enc = tokenizer(text=X, padding='longest', truncation=True, max_length=128)
        return torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']), torch.LongTensor(y)

    t_dataset = CustomDataset(typ='val', file_path=args.test_file_or_dir)
    t_loader = DataLoader(t_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)

    # load model from saved weights & best threshold
    bertweet = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # reinitialize
    bertweet.load_state_dict(torch.load(args.model_file_or_dir))  # load models
    if os.path.isfile(args.model_file_or_dir.replace('best_weights.pth', 'best_thresh.txt')):
        with open(args.model_file_or_dir.replace('best_weights.pth', 'best_thresh.txt')) as f:
            thresh = float(f.read().strip())
        print("Exported threshold ", thresh)
    else:
        thresh = 0.5
        print("Default threshold ", thresh)
    device = torch.device('cuda')
    bertweet.to(device)

    scores = []
    with torch.no_grad():
        bertweet.eval()
        for X, masks, y in t_loader:
            X, masks, y = X.to(device), masks.to(device), y.to(device)
            loss, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
            logits = logits.softmax(1)
            scores.extend(logits[:, -1].reshape(-1).tolist())
    preds = [int(x >= thresh) for x in scores]
    # preds.extend(np.round(logits[:, -1].reshape(-1).tolist()))
    # acc = accuracy_score(answers, preds)
    # f1 = f1_score(answers, preds)
    # pre = precision_score(answers, preds)
    # rec = recall_score(answers, preds)
    # print('Test file - acc: %1.3f, f-1: %1.3f, pre: %1.3f, rec: %1.3f' % (acc, f1, pre, rec))

    # save this somewhere
    model_name = args.model_file_or_dir.split('/')[-1].split('.best_weight')[0]
    save_file = '/shared/0/projects/relationships/working-dir/life-events/predicted-scores/reg.test_%s.df.tsv' % model_name
    df = pd.read_csv(args.test_file_or_dir, sep='\t')
    df['scores'] = scores
    df['predictions'] = preds
    # additional stuff to remove names including 'extra'
    if 'file_name' not in df.columns:
        df['file_name'] = 'tmp'
    df['file_name'] = [x.replace('_extra', '') for x in df['file_name']]
    df = df.sort_values(by='tweet_id', ascending=True)
    df.to_csv(save_file, sep='\t', index=False)
    print("Saved predictions to ", save_file)

    out = []
    for cat in [model_name]:
        for file_name in sorted(df['file_name'].unique()):
            df2 = df
            # df2 = df[(df['category']==cat)&(df['file_name']==file_name)]
            if len(df2) == 0:
                continue
            y_score, y_true, y_pred = df2['scores'], df2['is_shock'], df2['predictions']
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            out.append((model_name, cat, file_name, acc, f1, pre, rec))
            print('[%s] Test (%s-%s) - acc: %1.3f, f-1: %1.3f, pre: %1.3f, rec: %1.3f' % (
            model_name, cat, file_name, acc, f1, pre, rec))

            precision, recall, thresholds = precision_recall_curve(y_true, y_score)
            f1_thresh_list = []
            for t in thresholds:
                preds = [int(x >= t) for x in y_score]
                f1 = f1_score(y_true, preds)
                f1_thresh_list.append((f1, t))
            f1, t = sorted(f1_thresh_list)[-1]  # threshold that induces the largest f1 score
            print("Custom test set performance: F-1 %1.3f, thresh: %1.3f" % (f1, t))
            preds = [int(x >= 0.5) for x in y_score]
            f1 = f1_score(y_true, preds)
            print("Performance at 0.5 threshold: F-1: %1.3f" % f1)

    df2 = pd.DataFrame(out, columns=['model_name', 'category', 'file_name', 'accuracy', 'f1', 'precision', 'recall'])
    save_file = '/shared/0/projects/relationships/working-dir/life-events/predicted-scores/reg.test_scores_%s.df.tsv' % model_name
    df2.to_csv(save_file, sep='\t', index=False)
    return


def infer_bert(args):
    # load model
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    bertweet = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # reinitialize
    bertweet.load_state_dict(torch.load(args.model_file_or_dir))
    device = torch.device('cuda')
    bertweet.to(device)

    from nltk.tokenize import TweetTokenizer
    tokenize = TweetTokenizer().tokenize

    # get model name
    model_name = args.model_file_or_dir.split('/')[-1].split('.best_weight')[0]

    tweet_dir = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets-cavium/'
    label_dir = '/shared/0/projects/relationships/working-dir/life-events/labeled-tweets'
    score_dir = '/shared/0/projects/relationships/working-dir/life-events/predicted-scores'
    train_dir = args.train_file_or_dir

    cat = args.event_type

    def collate_fn(arr):
        X, y = [x[0] for x in arr], [int(x[1]) for x in arr]
        enc = tokenizer(text=X, padding='longest', truncation=True, max_length=128)
        return torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']), torch.LongTensor(y)

    # get a list of all tweets that were included in our previous training data to remove duplicates
    S_tid = []

    # from train data
    for file in os.listdir(train_dir):
        S_tid.extend([x[1:] for x in pd.read_csv(join(train_dir, file), sep='\t', engine='python')['tweet_id']])
    S_tid = set(S_tid)
    print(len(S_tid), ' samples already labeled!')

    # round 1: try to see if we can suffice from what has previously been labeled
    input_data = []
    df = pd.read_csv(join(tweet_dir, 'pre-labeled.%s.df.tsv' % cat), sep='\t', engine='python')
    df['category'] = cat
    for line in df.values:
        tid = line[0][1:]
        if tid in S_tid:
            continue
        else:
            S_tid.add(tid)  # add this tid to what we've labeled
            text = line[2]
            line[2] = text.lower()
            input_data.append(line)
    df = pd.DataFrame(input_data, columns=df.columns)
    df_in = df[['text', 'is_shock']]
    print("Round 1: test with %d labeled samples" % len(df_in))
    pre_dataset = CustomDataset(typ='infer', dataframe=df_in)
    pre_loader = DataLoader(pre_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)
    bertweet.load_state_dict(torch.load(args.model_file_or_dir))

    # predict scores
    preds = []
    scores = []
    with torch.no_grad():
        bertweet.eval()
        pbar = tqdm(pre_loader)
        for i, (X, masks, y) in enumerate(pbar):
            X, masks, y = X.to(device), masks.to(device), y.to(device)
            _, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
            logits = logits.softmax(1)
            scores.extend(logits[:, -1].reshape(-1).tolist())
            preds.extend(np.round(logits[:, -1].reshape(-1).tolist()))

    df['pred_score'] = scores
    df['is_labeled'] = True
    df_labeled = df[['tweet_id', 'user_id', 'category', 'text', 'pred_score', 'is_shock', 'is_labeled']]
    # print(len(df_in),' tweets matching our criterion!')
    # if len(df_in)==400:
    #     df_in.to_csv(join(label_dir, '%s.boundary.df.tsv' % (model_name)), sep='\t', index=False)
    #     print('%d samples collected from pre-labeled tweets'%len(df_in))
    #     return

    # round 2: then we have to manually look for the previous tweets
    input_data = []
    with open(join(tweet_dir, 'reg3.%s.json' % cat.split('.')[0])) as f:
        for line in f:
            obj = json.loads(line)
            obj['text'] = obj['text'].lower()
            if obj['tid'] in S_tid:
                continue
            else:
                input_data.append(obj)
                S_tid.add(obj['tid'])
    print(len(input_data), ' unlabeled tweets in total!')

    # load text into Dataset
    df_un = pd.DataFrame(list(zip([x['text'].lower() for x in input_data], [0] * len(input_data))),
                         columns=['text', 'is_shock'])
    un_dataset = CustomDataset(typ='infer', dataframe=df_un)
    un_loader = DataLoader(un_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)

    bertweet.load_state_dict(torch.load(args.model_file_or_dir))

    # predict scores
    preds = []
    scores = []
    with torch.no_grad():
        bertweet.eval()
        pbar = tqdm(un_loader)
        for i, (X, masks, y) in enumerate(pbar):
            X, masks, y = X.to(device), masks.to(device), y.to(device)
            _, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
            logits = logits.softmax(1)
            scores.extend(logits[:, -1].reshape(-1).tolist())
            preds.extend(np.round(logits[:, -1].reshape(-1).tolist()))

    # additionally, save random tweets for (1) tweets near the boundary, and (2) high-predicted tweets
    thresh = 0.5
    scores_abs = [np.abs(x - thresh) for x in scores]
    indices = np.array(scores_abs).argsort()[:400]

    # save samples in dataframe
    samples = []
    for idx in indices:
        obj = input_data[idx]
        tid = 't' + obj['tid']
        uid = 'u' + obj['uid']
        # timestamp = obj['timestamp']
        text = ' '.join(obj['text'].replace('\n', ' ').split()).strip()
        # text = ' '.join(tokenize(obj['text']))
        pred_score = scores[idx]
        samples.append((tid, uid, cat, text, pred_score, 0))
    df_unlabeled = pd.DataFrame(samples, columns=['tweet_id', 'user_id', 'category', 'text', 'pred_score', 'is_shock'])
    df_unlabeled['is_labeled'] = False
    df_all = df_labeled.append(df_unlabeled)

    # get score threshold here again
    scores = df_all['pred_score']
    thresh = 0.5
    scores_abs = [np.abs(x - thresh) for x in scores]
    indices = np.array(scores_abs).argsort()[:400]
    indices = set(indices)
    out = []
    for i, line in enumerate(df_all.values):
        if i in indices:
            out.append(line)
    df_out = pd.DataFrame(out, columns=df_all.columns)
    ln1, ln2 = len(df_out[df_out['is_labeled'] == True]), len(df_out[df_out['is_labeled'] == False])
    df_out.to_csv(join(label_dir, '%s.boundary.df.tsv' % (model_name)), sep='\t', index=False)
    print('%d samples collected from %d pre-labeled and %d unlabeled tweets' % (len(df_out), ln1, ln2))
    return


def extract_high_score_tweets(args):
    """
    Similar to infer_bert, but saves tweets with high predicted (0.5) scores
    :param args:
    :return:
    """
    # load model
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    bertweet = RobertaForSequenceClassification.from_pretrained("vinai/bertweet-base", num_labels=2)  # reinitialize
    bertweet.load_state_dict(torch.load(args.model_file_or_dir))
    device = torch.device('cuda')
    bertweet.to(device)

    from nltk.tokenize import TweetTokenizer
    tokenize = TweetTokenizer().tokenize

    # get model name
    model_name = args.model_file_or_dir.split('/')[-1].split('.best_weight')[0]

    tweet_dir = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets-cavium/'
    label_dir = '/shared/0/projects/relationships/working-dir/life-events/labeled-tweets'
    score_dir = '/shared/0/projects/relationships/working-dir/life-events/predicted-scores'
    score_dir = '/shared/0/projects/relationships/working-dir/life-events/predicted-scores/'
    load_dir = '/shared/0/projects/relationships/working-dir/life-events/filtered-tweets/'
    save_dir = '/shared/0/projects/relationships/working-dir/life-events/predicted-tweets/'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.event_type:
        categories = args.event_type.split(',')
    else:
        categories = ['crime', 'separation', 'job-loss', 'death']

    def collate_fn(arr):
        X, y = [x[0] for x in arr], [int(x[1]) for x in arr]
        enc = tokenizer(text=X, padding='longest', truncation=True, max_length=128)
        return torch.tensor(enc['input_ids']), torch.tensor(enc['attention_mask']), torch.LongTensor(y)

    for cat in categories:
        out = []

        # collect the tweet objects from all other tweet ids that are not included in our data
        print(cat, len(out))
        # get a list of all tweets that were included in our previous training data to remove duplicates

        train_dir = args.train_file_or_dir
        texts, uids, tids = [], [], []
        S_tid = []
        for file in os.listdir(train_dir):
            df = pd.read_csv(join(train_dir, file), sep='\t', engine='python')
            S_tid.extend(df['tweet_id'])
            df2 = df[df.is_shock == 1]
            texts.extend(df2.text.tolist())
            uids.extend(df2.user_id.tolist())
            tids.extend(df2.tweet_id.tolist())
        S_tid = set(S_tid)
        print('%d/%d positive tweets obtained from labels' % (len(tids), len(S_tid)))

        with open(join(tweet_dir, 'reg3.%s.json' % cat.split('.')[0])) as f:
            for ln, line in enumerate(f):
                obj = json.loads(line)
                obj['text'] = obj['text'].lower()
                if 't' + obj['tid'] in S_tid:
                    continue
                else:
                    out.append(obj)
                    S_tid.add('t' + obj['tid'])
        print('%d/%d potential sentences to be inferred in total!' % (len(out), ln))
        # for year in [2019,2020]:
        #     with open(join(tweet_dir, '%s.%d.json' % (cat.split('.')[0],year))) as f:
        #         for line in f:
        #             obj = json.loads(line)
        #             obj['text'] = obj['text'].lower()
        #             tid = 't'+obj['tid']
        #             if tid in S_tid:
        #                 continue
        #             else:
        #                 out.append(obj)
        #                 S_tid.add(tid)
        # print(len(out),' potential unlabeled sentences!')

        # load text into Dataset
        df_un = pd.DataFrame(list(zip([x['text'].lower() for x in out], [0] * len(out))), columns=['text', 'is_shock'])
        un_dataset = CustomDataset(typ='infer', dataframe=df_un)
        un_loader = DataLoader(un_dataset, batch_size=16, shuffle=False, num_workers=1, collate_fn=collate_fn)

        bertweet.load_state_dict(torch.load(args.model_file_or_dir))

        # predict scores
        preds = []
        scores = []
        with torch.no_grad():
            bertweet.eval()
            pbar = tqdm(un_loader)
            for i, (X, masks, y) in enumerate(pbar):
                X, masks, y = X.to(device), masks.to(device), y.to(device)
                _, logits = bertweet(input_ids=X, labels=y, attention_mask=masks)
                logits = logits.softmax(1)
                scores.extend(logits[:, -1].reshape(-1).tolist())
                preds.extend(np.round(logits[:, -1].reshape(-1).tolist()))

        # get additional tweets
        cnt = 0
        for i, s in enumerate(scores):
            if s >= args.thresh:
                obj = out[i]
                texts.append(obj['text'])
                uids.append('u' + obj['uid'])
                tids.append('t' + obj['tid'])
                cnt += 1

        print("%d additional samples from inference!" % cnt)
        texts = [' '.join(tokenize(x)).replace('\n', ' ').replace('\t', ' ').strip() for x in texts]

        # add dates to these tweets
        dates = []
        names = []
        tid_info = {k: None for k in tids}
        cnt = 0
        with open(join(tweet_dir, 'reg3.%s.json' % cat.split('.')[0])) as f:
            for line in f:
                obj = json.loads(line)
                tid = 't' + obj['tid']
                if tid in tid_info:
                    cnt += 1
                    tid_info[tid] = (obj['name'], obj['timestamp'], obj['timestamp2'])
        print(cnt, len(tid_info))

        out = []
        for tid, uid, text in zip(tids, uids, texts):
            if tid_info[tid] != None:
                name, timestamp, timestamp2 = tid_info[tid]
                out.append((tid, timestamp, timestamp2, uid, name, text))
        # for tid in tids:
        #     if tid_info[tid]:
        #         name,timestamp = tid_info[tid]
        #         names.append(name)
        #         dates.append(timestamp)
        #
        #
        # out = list(zip(tids,dates,uids,names,texts))
        df = pd.DataFrame(out, columns=['tweet_id', 'timestamp', 'timestamp2', 'user_id', 'user_name', 'text'])
        df = df.drop_duplicates()
        df.to_csv(join(save_dir, '%s.df.tsv' % cat), sep='\t', index=False)
        print(len(df), ' tweets extracted for %s' % cat)

    return


if __name__ == '__main__':
    main()

"""

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_extract --model_name=cls.total.crime \
  --event_type=crime  --thresh=0.5 \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.total.crime.best_weights.pth
CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_extract --model_name=cls.total.death \
  --event_type=death  --thresh=0.5 \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.total.death.best_weights.pth
CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_extract --model_name=cls.total.breakup \
  --event_type=breakup  --thresh=0.5 \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.total.breakup.best_weights.pth
CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_extract --model_name=cls.total.job-loss \
  --event_type=job-loss  --thresh=0.5 \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.total.job-loss.best_weights.pth

Train model
CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_train --model_name=cls.6.death \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/ \
  --val_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/reg.3.val.df.tsv | tee log.death.cls.6.txgt

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_test --model_name=cls.6.death \
  --test_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/reg.3.test.df.tsv \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.6.death.best_weights.pth

CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_infer --model_name=cls.5.death --event_type=death \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.5.death.best_weights.pth


CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_train --model_name=cls.6.breakup --lr=1e-5 \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/ \
  --val_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/reg.3.val.df.tsv

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_test --model_name=cls.6.breakup \
  --test_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/reg.3.test.df.tsv \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/final/cls.6.breakup.best_weights.pth

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_infer --model_name=cls.6.breakup --event_type=breakup \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.6.breakup.best_weights.pth


CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_train --model_name=cls.6.crime \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/ \
  --val_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/reg.3.val.df.tsv --lr=1e-5 | tee log.crime.cls.5.txt

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_test --model_name=cls.6.crime \
  --test_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/reg.3.test.df.tsv \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.6.crime.best_weights.pth

CUDA_VISIBLE_DEVICES=4 python run_shock_classifier.py --do_infer --model_name=cls.5.crime --event_type=crime \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.5.crime.best_weights.pth

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_test --model_name=cls.9.crime \
  --test_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/labeled-tweets/reg.3.crime.df.tsv \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.9.crime.best_weights.pth





CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_extract --model_name=cls.6.breakup --event_type=breakup \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/breakup/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/final/cls.6.breakup.best_weights.pth

CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_extract --model_name=cls.5.job-loss \
  --event_type=job-loss \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/ \
  --thresh=0.5 \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/final/cls.5.job-loss.best_weights.pth  

CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_extract --model_name=cls.5.death \
  --event_type=death \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/death/ \
  --thresh=0.5 \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/final/cls.5.death.best_weights.pth 

CUDA_VISIBLE_DEVICES=7 python run_shock_classifier.py --do_extract --model_name=cls.5.crime \
  --event_type=crime \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/crime/ \
  --thresh=0.5 \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/final/cls.5.crime.best_weights.pth 


CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_train --model_name=cls.6.job-loss \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/ \
  --val_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/reg.3.val.df.tsv \
  --lr=3e-5 | tee log.job-loss.cls.6.txt

CUDA_VISIBLE_DEVICES=2 python run_shock_classifier.py --do_test --model_name=cls.6.job-loss \
  --test_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/reg.3.test.df.tsv \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.6.job-loss.best_weights.pth

CUDA_VISIBLE_DEVICES=5 python run_shock_classifier.py --do_infer --model_name=cls.5.job-loss --event_type=job-loss \
  --train_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/train-data/job-loss/ \
  --model_file_or_dir=/shared/0/projects/relationships/working-dir/life-events/saved-weights/cls.5.job-loss.best_weights.pth


  """