from __future__ import division
from __future__ import print_function

import datetime
import json
import logging
import os
import pickle
import time

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import NCModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics
import torch.nn.functional as F



import warnings
warnings.filterwarnings('ignore')


def run(args, block, only_pseudo):

    #logPath = './logs/'
    args.message_block = str(block) + '/'
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    logging.info(" ".join([f'========================Training Block {block}=================================']))

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))

    # Load data
    data = load_data(args)

    args.n_nodes, args.feat_dim = data['features'].shape
    Model = NCModel
    args.n_classes = int(data['labels'].max() + 1)
    logging.info(f'Num classes: {args.n_classes}')

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(f'Type of manifold: {args.manifold}')
    logging.info(str(model))

    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )


    tot_params = sum([np.prod(p.size()) for p in model.parameters()])
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_model = None

    total_loss = []
    if only_pseudo:
        args.epochs = 1

    for epoch in range(args.epochs):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'])
        if only_pseudo:
            embeddings_n = []
            for i in data['indices']:
                embeddings_n.append(embeddings[i].detach().numpy())
            embeddings_n = np.array(embeddings_n)
            embeddings = torch.Tensor(embeddings_n)
        train_metrics = model.compute_metrics(embeddings, data, args)

        l = train_metrics['loss']
        total_loss.append(float(l))

        if not only_pseudo:
            train_metrics['loss'].backward()
            optimizer.step()
            lr_scheduler.step()


        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))

        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = train_metrics

            if model.has_improved(best_val_metrics, val_metrics):
                best_model = model
                best_val_metrics = val_metrics
                best_test_metrics = val_metrics
                counter = 0
            else:
                counter += 1
                if counter >= args.patience and epoch >= args.min_epochs:
                    logging.info("Early stopping")
                    break

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if not best_model:
        model.eval()
        best_model = model

    logging.info(" ".join(["Best results:", format_metrics(best_test_metrics, 'test')]))
    ari = best_test_metrics['ari']
    ami = best_test_metrics['ami']
    acc = best_test_metrics['acc']
    nmi = best_test_metrics['nmi']
    args.message += f'Block: {block}: Best results: ACC: {acc}, NMI: {nmi}, ARI: {ari}, AMI: {ami} \n'
    if block != 0:
        args.total_ari += ari
        args.total_ami += ami
    return


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dataset == 'Event2012':
        blocks = [i for i in range(22)]
    elif args.dataset == 'Event2018':
        blocks = [i for i in range(17)]
    else:
        blocks = []


    for block in blocks:
        only_pseudo = True # directly clustering pseudo lables
        teacher_model = run(args, block, only_pseudo)

    print(f'Avg ARI: {args.total_ari/(len(blocks) - 1)}, Avg AMI: {args.total_ami/(len(blocks) - 1)}')


    args.message += f'LLM model:{args.llm_model}, Sampling Style:{args.sampling_style}, ' \
                    f'Vectorization Method:{args.method}, Embedding Space:{args.manifold}, ' \
                    f'Clustering method:{args.clustering_method} \n' \
                    f'Margin: {args.margin}, H_Dim:{args.dim}, H_L:{args.num_layers - 1}, ' \
                    f'ETA:{args.eta}, Random_k:{args.k}, Metapath:Label_similarity: {args.metapath}:{args.label_similarity}, ' \
                    f'Only pseudo: {only_pseudo} \n'


    print(args.message)


