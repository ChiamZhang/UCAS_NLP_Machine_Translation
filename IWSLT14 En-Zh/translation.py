import random
import numpy as np
import pkuseg
import nltk

from nltk.translate.bleu_score import corpus_bleu
import argparse
import os


from tqdm import trange,tqdm
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import AdamW,get_linear_schedule_with_warmup

from model import Encoder,Attention,Decoder,seq2seq,LanguageModelCriterion

def setseed():
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)

def load_file(path,tgt_add_bos=True):
    en = []
    cn = []
    seg = pkuseg.pkuseg()
    count=0
    with open(path,'rb') as f:
        for line in f.readlines():
            line = line.decode().strip().split('\t')

            if line[0]=="":
                continue
            if len(line)==1:
                continue

            count=count+1

            print(count)

            en.append(["BOS"] + nltk.word_tokenize(line[0].lower()) + ["EOS"])
            # test时tgt不加开头结束，用于BLEU计算
            if tgt_add_bos:
                cn.append(["BOS"] + seg.cut(line[1]) + ["EOS"])
            else:
                cn.append(seg.cut(line[1]))

    return en,cn

def build_tokenizer(sentences,args):
    word_count = Counter()
    for sen in sentences:
        for word in sen:
            word_count[word] += 1
    ls = word_count.most_common(args.max_vocab_size)
    word2idx = {word:idx+2 for idx,(word,_) in enumerate(ls)}
    word2idx['UNK'] = args.UNK_IDX
    word2idx['PAD'] = args.PAD_IDX

    id2word = {v:k for k,v in word2idx.items()}
    total_vocab = len(ls) + 2

    return word2idx,id2word,total_vocab

def tokenize2num(en_sentences,cn_sentences,en_word2idx,cn_word2idx, sort_reverse = True):
    length = len(en_sentences)

    out_en_sents = [[en_word2idx.get(word,1) for word in sen] for sen in en_sentences]
    out_cn_sents = [[cn_word2idx.get(word, 1) for word in sen] for sen in cn_sentences]

    def sort_sents(sents):
        return sorted(range(len(sents)),key = lambda x : len(sents[x]),reverse=True)
    if sort_reverse:
        sorted_index =  sort_sents(out_en_sents)
        out_en_sents = [out_en_sents[idx] for idx in sorted_index]
        out_cn_sents = [out_cn_sents[idx] for idx in sorted_index]

    return out_en_sents,out_cn_sents

class Tokenizer(object):
    def __init__(self,word2idx,id2word,vocab_size):
        self.word2idx = word2idx
        self.id2word = id2word
        self.vocab_size = vocab_size


class DataProcessor(object):
    def __init__(self,args):
        cached_en_tokenizer = os.path.join(args.data_dir,"cached_{}".format("en_tokenizer"))
        cached_cn_tokenizer = os.path.join(args.data_dir, "cached_{}".format("cn_tokenizer"))

        if not os.path.exists(cached_en_tokenizer) or not os.path.exists(cached_cn_tokenizer):
            en_sents, cn_sents = load_file(args.data_dir + "train.txt")
            en_word2idx, en_id2word, en_vocab_size = build_tokenizer(en_sents,args)
            cn_word2idx, cn_id2word, cn_vocab_size = build_tokenizer(cn_sents, args)

            torch.save([en_word2idx, en_id2word, en_vocab_size],cached_en_tokenizer)
            torch.save([cn_word2idx, cn_id2word, cn_vocab_size],cached_cn_tokenizer)
        else:
            en_word2idx, en_id2word, en_vocab_size = torch.load(cached_en_tokenizer)
            cn_word2idx, cn_id2word, cn_vocab_size = torch.load(cached_cn_tokenizer)

        self.en_tokenizer = Tokenizer(en_word2idx, en_id2word, en_vocab_size)
        self.cn_tokenizer = Tokenizer(cn_word2idx, cn_id2word, cn_vocab_size)

    def get_train_examples(self,args):
        return self._create_examples(os.path.join(args.data_dir,"train.txt"),"train",args)


    def get_dev_examples(self,args):
        return self._create_examples(os.path.join(args.data_dir,"dev.txt"),"dev",args)

    def _create_examples(self,path,set_type,args):
        en_sents,cn_sents = load_file(path)
        out_en_sents,out_cn_sents = tokenize2num(en_sents,cn_sents,
                                                 self.en_tokenizer.word2idx,self.cn_tokenizer.word2idx)
        minibatches = getminibatches(len(out_en_sents),args.batch_size)

        all_examples = []
        for minibatch in minibatches:
            mb_en_sentences = [out_en_sents[i] for i in minibatch]
            mb_cn_sentences = [out_cn_sents[i] for i in minibatch]

            mb_x,mb_x_len = prepare_data(mb_en_sentences)
            mb_y,mb_y_len = prepare_data(mb_cn_sentences)

            all_examples.append((mb_x,mb_x_len,mb_y,mb_y_len))

        return all_examples

def prepare_data(seqs):
    # 处理每个batch句子（一个batch中句子长度可能不一致，需要pad）
    batch_size = len(seqs)
    lengthes = [len(seq) for seq in seqs]  # 每个句子的长度列表

    max_length = max(lengthes)  # 句子最大长度
    # 初始化句子矩阵都为0
    x = np.zeros((batch_size, max_length)).astype("int32")
    for idx in range(batch_size):
        # 按行将每行句子赋值进去
        x[idx, :lengthes[idx]] = seqs[idx]

    x_lengths = np.array(lengthes).astype("int32")
    return x, x_lengths

def getminibatches(n,batch_size,shuffle=True):
    minibatches = np.arange(0,n,batch_size)
    if shuffle:
        np.random.shuffle(minibatches)

    result = []
    for idx in minibatches:
        result.append(np.arange(idx,min(n,idx+batch_size)))
    return result


def train(args,model, data,loss_fn,eval_data):
    LOG_FILE = "translation_model.log"
    tb_writer = SummaryWriter('./runs')

    t_total = args.num_epoch * len(data)
    optimizer = AdamW(model.parameters(), lr=args.learnning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    global_step = 0
    total_num_words = total_loss = 0.
    logg_loss = 0.
    logg_num_words = 0.
    val_losses = []
    train_iterator = trange(args.num_epoch,desc='epoch')
    for epoch in train_iterator:
        model.train()
        epoch_iteration = tqdm(data, desc='iteration')
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(epoch_iteration):
            # （英文batch，英文长度，中文batch，中文长度）
            mb_x = torch.from_numpy(mb_x).to(args.device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(args.device).long()
            # 前n-1个单词作为输入，后n-1个单词作为输出，因为输入的前一个单词要预测后一个单词
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(args.device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(args.device).long()
            mb_y_len = torch.from_numpy(mb_y_len - 1).to(args.device).long()
            # 输入输出的长度都减一。
            mb_y_len[mb_y_len <= 0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)
            mb_out_mask = torch.arange(mb_y_len.max().item(), device=args.device)[None, :] < mb_y_len[:, None]
            # batch,seq_len . 其中每行长度超过自身句子长度的为false
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)
            # 损失函数

            # 更新模型
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.GRAD_CLIP)
            # 为了防止梯度过大，设置梯度的阈值
            optimizer.step()
            scheduler.step()

            global_step += 1
            num_words = torch.sum(mb_y_len).item()
            # 一个batch里多少个单词
            total_loss += loss.item() * num_words
            # 总损失，loss计算的是均值损失，每个单词都是都有损失，所以乘以单词数
            total_num_words += num_words
            # 总单词数

            if (it+1) % 100 == 0:
                loss_scalar = (total_loss - logg_loss) / (total_num_words-logg_num_words)
                logg_num_words = total_num_words
                logg_loss = total_loss

                with open(LOG_FILE, "a") as fout:
                    fout.write("epoch: {}, iter: {}, loss: {},learn_rate: {}\n".format(epoch, it, loss_scalar,
                                                                                       scheduler.get_lr()[0]))
                print("epoch: {}, iter: {}, loss: {}, learning_rate: {}".format(epoch, it, loss_scalar,
                                                                                scheduler.get_lr()[0]))
                tb_writer.add_scalar("learning_rate", scheduler.get_lr()[0], global_step)
                tb_writer.add_scalar("loss", loss_scalar, global_step)

        print("Epoch", epoch, "Training loss", total_loss / total_num_words)
        eval_loss = evaluate(args, model, eval_data, loss_fn)  # 评估模型
        with open(LOG_FILE, "a") as fout:
            fout.write("===========" * 20)
            fout.write("EVALUATE: epoch: {}, loss: {}\n".format(epoch, eval_loss))
        if len(val_losses) == 0 or eval_loss < min(val_losses):
            # 如果比之前的loss要小，就保存模型
            print("best model, val loss: ", eval_loss)
            torch.save(model.state_dict(), "translate-best.th")
        val_losses.append(eval_loss)


def evaluate(args,model, data,loss_fn):
    model.eval()
    total_num_words = total_loss = 0.
    eval_iteration = tqdm(data, desc='eval iteration')
    with torch.no_grad():#不需要更新模型，不需要梯度
        for it, (mb_x, mb_x_len, mb_y, mb_y_len) in enumerate(eval_iteration):
            mb_x = torch.from_numpy(mb_x).to(args.device).long()
            mb_x_len = torch.from_numpy(mb_x_len).to(args.device).long()
            mb_input = torch.from_numpy(mb_y[:, :-1]).to(args.device).long()
            mb_output = torch.from_numpy(mb_y[:, 1:]).to(args.device).long()
            mb_y_len = torch.from_numpy(mb_y_len-1).to(args.device).long()
            mb_y_len[mb_y_len<=0] = 1

            mb_pred, attn = model(mb_x, mb_x_len, mb_input, mb_y_len)

            mb_out_mask = torch.arange(mb_y_len.max().item(), device=args.device)[None, :] < mb_y_len[:, None]
            mb_out_mask = mb_out_mask.float()

            loss = loss_fn(mb_pred, mb_output, mb_out_mask)

            num_words = torch.sum(mb_y_len).item()
            total_loss += loss.item() * num_words
            total_num_words += num_words
    print("Evaluation loss", total_loss/total_num_words)
    return total_loss/total_num_words

def test(args,model,processor):
    model.eval()
    en_sents, cn_sents = load_file(args.data_dir+'test.txt',tgt_add_bos=False)
    en_sents, _ = tokenize2num(en_sents, cn_sents,
                               processor.en_tokenizer.word2idx,
                               processor.cn_tokenizer.word2idx,
                               sort_reverse=False)

    top_hypotheses = []
    test_iteration = tqdm(en_sents, desc='test bleu')
    with torch.no_grad():
        for idx, en_sent in enumerate(test_iteration):
            mb_x = torch.from_numpy(np.array(en_sent).reshape(1, -1)).long().to(args.device)
            mb_x_len = torch.from_numpy(np.array([len(en_sent)])).long().to(args.device)
            bos = torch.Tensor([[processor.cn_tokenizer.word2idx['BOS']]]).long().to(args.device)
            completed_hypotheses = model.beam_search(mb_x, mb_x_len,
                                                     bos, processor.cn_tokenizer.word2idx['EOS'],
                                                     topk=args.beam_size,
                                                     max_length=args.max_beam_search_length)
            top_hypotheses.append([processor.cn_tokenizer.id2word[id] for id in completed_hypotheses[0].value])

    bleu_score = corpus_bleu([[ref] for ref in cn_sents],
                             top_hypotheses)

    print('Corpus BLEU: {}'.format(bleu_score * 100))

    return bleu_score


def main():
    parse = argparse.ArgumentParser()

    parse.add_argument("--data_dir",default='./data/',type=str,required=False,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",)
    parse.add_argument("--batch_size", default=16, type=int)
    parse.add_argument("--do_train",default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--do_test",default=True, action="store_true", help="Whether to run test.")
    parse.add_argument("--do_translate",default=True, action="store_true", help="Whether to run training.")
    #parse.add_argument("--do_gaoyi", default=True, action="store_true", help="Whether to run training.")
    parse.add_argument("--learnning_rate", default=5e-4, type=float)
    parse.add_argument("--dropout", default=0.2, type=float)
    parse.add_argument("--num_epoch", default=10, type=int) #
    parse.add_argument("--max_vocab_size",default=50000,type=int)
    parse.add_argument("--embed_size",default=300,type=int)
    parse.add_argument("--enc_hidden_size", default=512, type=int)
    parse.add_argument("--dec_hidden_size", default=512, type=int)
    parse.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parse.add_argument("--GRAD_CLIP", default=1, type=float)
    parse.add_argument("--UNK_IDX",default=1,type=int)
    parse.add_argument("--PAD_IDX", default=0, type=int)
    parse.add_argument("--beam_size", default=5, type=int)
    parse.add_argument("--max_beam_search_length", default=100, type=int)

    args = parse.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    setseed()

    processor = DataProcessor(args)

    encoder = Encoder(processor.en_tokenizer.vocab_size,args.embed_size,
                      args.enc_hidden_size,args.dec_hidden_size,args.dropout)
    decoder = Decoder(processor.cn_tokenizer.vocab_size,args.embed_size,
                      args.enc_hidden_size,args.dec_hidden_size,args.dropout)
    model = seq2seq(encoder,decoder)
    if os.path.exists("translate-best.th"):
        model.load_state_dict(torch.load("translate-best.th"))
    model.to(device)
    loss_fn = LanguageModelCriterion().to(device)

    train_data = processor.get_train_examples(args)
    eval_data = processor.get_dev_examples(args)

    #if args.do_train:
    train(args,model,train_data,loss_fn,eval_data)

   # if args.do_test:
    test(args,model,processor)

   # if args.do_translate:
    model.load_state_dict(torch.load("translate-best.th"))
    model.to(device)
    while True:
        title = input("请输入要翻译的英文句子:\n")
        if len(title.strip()) == 0:
            continue
        title = ['BOS'] + nltk.word_tokenize(title.lower()) + ['EOS']
        title_num = [processor.en_tokenizer.word2idx.get(word,1) for word in title]
        mb_x = torch.from_numpy(np.array(title_num).reshape(1,-1)).long().to(device)
        mb_x_len = torch.from_numpy(np.array([len(title_num)])).long().to(device)

        bos = torch.Tensor([[processor.cn_tokenizer.word2idx['BOS']]]).long().to(device)

        completed_hypotheses = model.beam_search(mb_x, mb_x_len,
                                                     bos,processor.cn_tokenizer.word2idx['EOS'],
                                                     topk=args.beam_size,
                                                     max_length=args.max_beam_search_length)

        for hypothes in completed_hypotheses:
                result = "".join([processor.cn_tokenizer.id2word[id] for id in hypothes.value])
                score = hypothes.score
                print("翻译后的中文结果为:{},score:{}".format(result,score))




if __name__ == '__main__':
    main()



