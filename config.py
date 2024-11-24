import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'total_ari': (0, 'total_ari'),
        'total_ami': (0, 'total_ami'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (100, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (20, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.1, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'min-epochs': (20, 'do not early stop before min-epochs'),
        'message': ('', 'storage training information'),
        'clustering_method': ('kmean', 'kmean, dbscan, hdbscan')
    },
    'model_config': {
        'task': ('nc', 'task type node classification]'),
        'sampling_style': ('enumerate_random', "enumerate_hard,enumerate_soft, enumerate_random, random_hard, "
                                             "random_soft, random_random"),
        'model': ('HMLP', 'which encoder to use'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'dim': (512, 'embedding dimension'),
        'k': (5, 'random positive times'),
        'margin': (1, 'delta for triplet loss'),
        'eta': (0.7, 'label similarity'),
        'metapath': (True, 'use metapaht selection triplet'),
        'label_similarity': (True, 'use llm label similarity'),
        'llm_model': ('llama3.1', 'using llama3 to summary the original text'),
        'method': ('sbert', 'embedding method sbert or w2v'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not')
    },
    'data_config': {
        'datapath': ('./data/', 'the path of dataset'),
        'dataset': ('Event2018', 'which dataset to use: twitter, mini-twitter,kawarith,crisislext, cora, citeseer'),
        'dataset-model': ('/incremental', 'which dataset model to use (incremental/offline)'),
        'message-block': ('0/', 'which message block to use(0~21). offline only have 0'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
