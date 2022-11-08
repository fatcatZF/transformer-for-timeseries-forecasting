from __future__ import division
from __future__ import print_function

import random

import time
import argparse
import pickle
import os
import datetime

import torch.optim as optim
from torch.optim import lr_scheduler

from utils import *
from models import *

from scipy.stats import pearsonr

parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--batch-size', type=int, default=64,
                    help='Number of samples per batch.')
parser.add_argument('--lr', type=float, default=0.0007,
                    help='Initial learning rate.')
parser.add_argument("--lr-decay", type=int, default=200, 
                    help="lr decay steps.")
parser.add_argument("--gamma", type=float, default=0.5, 
                    help="LR decay factor.")
parser.add_argument("--teach-max", type=float, default=1., 
                    help="Initial teacher forcing rate.")
parser.add_argument("--teach-min", type=float, default=0.,
                    help="Final teacher forcing rate.")
parser.add_argument("--teach-steps", type=int, default=200,
                    help="Teacher Forcing steps.")
parser.add_argument("--dim", type=int, default=1, 
                    help="dimension of time series.")
parser.add_argument("--d-model", type=int, default=64, 
                    help="dimension of transformer attention.")
parser.add_argument("--dim-feedforward", type=int, default=128,
                    help="dimension of transformer feedforward net.")
parser.add_argument("--nhead", type=int, default=4, 
                    help="number of heads of attention.")
parser.add_argument("--num-enlayers", type=int, default=4,
                    help="number of transformer encoder layers.")
parser.add_argument("--num-delayers", type=int, default=4,
                    help="number of transformer decoder layers.")
parser.add_argument("--dropout", type=float, default=0.2,
                    help="dropout rate.")
parser.add_argument("--max-len", type=int, default=15, 
                    help="maximal length of time series.")
parser.add_argument("--test-part", type=int, default=2, 
                    help="test part.")
parser.add_argument("--training-steps", type=int, default=10,
                    help="time steps used for training (observed steps).")
parser.add_argument("--save-folder", type=str, default="logs",
                    help="Where to save the trained model.")
parser.add_argument("--load-folder", type=str, default='',
                    help="where to load the trained model.")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

print(args)


# Save model and meta-data. Always saves in a new sub-folder.
if args.save_folder:
    exp_counter = 0
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    save_folder = '{}/exp{}/'.format(args.save_folder, timestamp)
    os.mkdir(save_folder)
    meta_file = os.path.join(save_folder, 'metadata.pkl')
    encoder_file = os.path.join(save_folder, 'encoder.pt')
    decoder_file = os.path.join(save_folder, 'decoder.pt')

    log_file = os.path.join(save_folder, 'log.txt')
    log = open(log_file, 'w')

    pickle.dump({'args': args}, open(meta_file, "wb"))
else:
    print("WARNING: No save_folder provided!" +
          "Testing (within this script) will throw an error.")


train_loader, valid_loader, test_loader, train_max, train_min = load_data_ili(training_steps=args.training_steps,
                                                                 test_part=args.test_part,
                                                                 batch_size=args.batch_size)




encoder = TimeSeriesEncoder(n_in=args.dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                            nhead=args.nhead, num_enlayers=args.num_enlayers, dropout=args.dropout,
                            max_len=args.max_len)

decoder = TimeSeriesDecoder(n_in=args.dim, d_model=args.d_model, dim_feedforward=args.dim_feedforward,
                            nhead=args.nhead, num_delayers=args.num_delayers, dropout=args.dropout,
                            max_len=args.max_len)

if args.load_folder:
    encoder_file = os.path.join(args.load_folder, 'encoder.pt')
    encoder.load_state_dict(torch.load(encoder_file))
    decoder_file = os.path.join(args.load_folder, 'decoder.pt')
    decoder.load_state_dict(torch.load(decoder_file))

    args.save_folder = False


optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_decay, 
                        gamma=args.gamma)

if args.cuda:
    encoder.cuda()
    decoder.cuda()



def train(epoch, best_val_loss, teach_rate):
    t = time.time()
    mse_train = []
    mse_val = []
    encoder.train()
    decoder.train()
     
    for batch_idx, (x, y) in enumerate(train_loader):
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = x.permute(1,0,2), y.permute(1,0,2)
        #shape: [seq_len, n_batch, dim]
        optimizer.zero_grad() 
        memory = encoder(x)
        teacher_forcing = (random.random() < teach_rate)
        if teacher_forcing:
            # training with teacher forcing (given previous ground truth input)
            x_last = x[-1:,:,:]
            x_de = torch.cat([x_last, y[:-1,:,:]], dim=0)
            tgt_mask = torch.zeros(x_de.size(0), x_de.size(0))-torch.inf
            tgt_mask = torch.triu(tgt_mask, diagonal=1)
            if args.cuda:
                tgt_mask = tgt_mask.cuda()
            y_predict = decoder(x_de, memory, tgt_mask)
        else:
            seq_target = y.size(0)
            x_de = x[-1:,:,:]
            predicts = []
            for i in range(seq_target):
                seq_de = x_de.size(0)
                if seq_de > 1:
                     tgt_mask = torch.zeros(seq_de, seq_de)-torch.inf
                     tgt_mask = torch.triu(tgt_mask, diagonal=1)
                     if args.cuda: tgt_mask = tgt_mask.cuda()
                else: tgt_mask = None
                x_de_next = decoder(x_de, memory, tgt_mask)
                predicts.append(x_de_next[-1:,:,:])
                x_de = torch.cat([x_de, x_de_next[-1:,:,:]], dim=0)
            y_predict = torch.cat(predicts, dim=0)
        
        loss = F.mse_loss(y_predict, y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        mse_train.append(loss.item())

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(valid_loader):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = x.permute(1,0,2), y.permute(1,0,2)
            #shape: [seq_len, n_batch, dim]
            memory = encoder(x)
            seq_target = y.size(0)
            x_de = x[-1:,:,:]
            predicts = []
            for i in range(seq_target):
                seq_de = x_de.size(0)
                if seq_de > 1:
                     tgt_mask = torch.zeros(seq_de, seq_de)-torch.inf
                     tgt_mask = torch.triu(tgt_mask, diagonal=1)
                     if args.cuda: tgt_mask = tgt_mask.cuda()
                else: tgt_mask = None
                x_de_next = decoder(x_de, memory, tgt_mask)
                predicts.append(x_de_next[-1:,:,:])
                x_de = torch.cat([x_de, x_de_next[-1:,:,:]], dim=0)
            y_predict = torch.cat(predicts, dim=0)

            loss = F.mse_loss(y_predict, y)
            mse_val.append(loss.item())

    
    print("Epoch: {:04d}".format(epoch+1),
          "mse_train: {:.10f}".format(np.mean(mse_train)),
          "mse_val: {:.10f}".format(np.mean(mse_val)),
           "teach_rate: {:.10f}".format(teach_rate))
    if args.save_folder and np.mean(mse_val) < best_val_loss:
        torch.save(encoder, encoder_file)
        torch.save(decoder, decoder_file)
        print("Best model so far, saving...")
        print("Epoch: {:04d}".format(epoch+1),
          "mse_train: {:.10f}".format(np.mean(mse_train)),
          "mse_val: {:.10f}".format(np.mean(mse_val)),
           "teach_rate: {:.10f}".format(teach_rate), file=log)
        log.flush()
    
    return np.mean(mse_val)



def test():
    mse_test = []
    mse_test_real = []
    pearson_real = []
    encoder = torch.load(encoder_file)
    decoder = torch.load(decoder_file)
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            if args.cuda:
                x, y = x.cuda(), y.cuda()
            x, y = x.permute(1,0,2), y.permute(1,0,2)
            #shape: [seq_len, n_batch, dim]
            x_real, y_real = (train_max-train_min)*x+train_min, (train_max-train_min)*y+train_min
            memory = encoder(x)
            seq_target = y.size(0)
            x_de = x[-1:,:,:]
            predicts = []
            for i in range(seq_target):
                seq_de = x_de.size(0)
                if seq_de > 1:
                     tgt_mask = torch.zeros(seq_de, seq_de)-torch.inf
                     tgt_mask = torch.triu(tgt_mask, diagonal=1)
                     if args.cuda: tgt_mask = tgt_mask.cuda()
                else: tgt_mask = None
                x_de_next = decoder(x_de, memory, tgt_mask)
                predicts.append(x_de_next[-1:,:,:])
                x_de = torch.cat([x_de, x_de_next[-1:,:,:]], dim=0)
            y_predict = torch.cat(predicts, dim=0)
            y_predict_real = (train_max-train_min)*y_predict+train_min

            loss = F.mse_loss(y_predict, y)
            loss_real = F.mse_loss(y_predict_real, y_real)
            mse_test.append(loss.item())
            mse_test_real.append(loss_real.item())

            y_predict_numpy = y_predict_real.cpu().squeeze().numpy()
            y_numpy = y_real.cpu().squeeze().numpy()
            pc, _ = pearsonr(y_predict_numpy, y_numpy)
            pearson_real.append(pc)
    
    print('--------------------------------')
    print('--------Testing-----------------')
    print('--------------------------------')
    print("mse_test: {:.10f}".format(np.mean(mse_test)),
        "real mse_test: {:.10f}".format(np.mean(mse_test_real)),
        "real pearson correlation: {:.10f}".format(np.mean(pearson_real)))


#train model
t_total = time.time()
best_val_loss = np.inf
best_epoch = 0


teach_rate = args.teach_max
teach_delta = (args.teach_max-args.teach_min)/args.teach_steps


for epoch in range(args.epochs):
    val_loss = train(epoch, best_val_loss, teach_rate)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch=epoch
    teach_rate= max(teach_rate-teach_delta, args.teach_min)
    

print("Optimization Finished!")
print("Best Epoch: {:04d}".format(best_epoch+1))
if args.save_folder:
    print("Best Epoch: {:04d}".format(best_epoch), file=log)
    log.flush()

test()

log.close()






            





        