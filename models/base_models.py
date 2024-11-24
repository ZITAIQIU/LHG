"""Base model class."""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.cluster import KMeans, OPTICS
from scipy.stats import mode
from tqdm import tqdm
#from info_nce import InfoNCE, info_nce

from layers.layers import FermiDiracDecoder

import manifolds
import models.encoders as encoders
from models.decoders import model2decoder
from utils.eval_utils import get_metrics




class BaseModel(nn.Module):
    """
    Base model for graph embedding tasks.
    """

    def __init__(self, args):
        super(BaseModel, self).__init__()
        self.manifold_name = args.manifold
        if args.c is not None:
            self.c = torch.tensor([args.c])
            if not args.cuda == -1:
                self.c = self.c.to(args.device)
        else:
            self.c = nn.Parameter(torch.Tensor([1.]))
            print('self,c cpu')
        self.manifold = getattr(manifolds, self.manifold_name)()
        if self.manifold.name == 'Hyperboloid':
            args.feat_dim = args.feat_dim + 1
        self.nnodes = args.n_nodes
        self.encoder = getattr(encoders, args.model)(self.c, args)

    def encode(self, x):
        if self.manifold.name == 'Hyperboloid':
            o = torch.zeros_like(x)
            x = torch.cat([o[:, 0:1], x], dim=1)
        h = self.encoder.encode(x)
        return h

    def compute_metrics(self, embeddings, data, args):
        raise NotImplementedError

    def init_metric_dict(self):
        raise NotImplementedError

    def has_improved(self, m1, m2):
        raise NotImplementedError

class NCModel(BaseModel):
    """
    Base model for node classification task.
    """

    def __init__(self, args):
        super(NCModel, self).__init__(args)

        self.decoder = model2decoder[args.model](self.c, args)
        self.args = args


        if args.n_classes > 2:
            self.f1_average = 'micro'
        else:
            self.f1_average = 'binary'

        self.weights = torch.Tensor([1.] * args.n_classes)
        if not args.cuda == -1:
            self.weights = self.weights.to(args.device)


    def decode(self, h):
        output = self.decoder.decode(h)

        return output

    def alignment_predict(self, teacher_preds, current_dim):
        teacher_dim = teacher_preds.shape[1]
        net = torch.nn.Sequential(torch.nn.Linear(teacher_dim, current_dim))
        return net(teacher_preds)

    def run_kmean(self, extract_features, extract_labels):

        # Extract labels
        extract_labels = extract_labels.cpu().numpy()
        labels_true = extract_labels

        # Extract features
        # X = extract_features[indices, :]
        X = extract_features.cpu().detach().numpy()
        assert labels_true.shape[0] == X.shape[0]
        n_test_tweets = X.shape[0]  # 100

        n_classes = len(set(labels_true.tolist()))

        # k-means clustering
        kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
        labels = kmeans.labels_

        y_pred = np.zeros_like(labels)
        for i in range(n_classes):
            mask = (labels == i)
            y_pred[mask] = mode(labels_true[mask])[0]

        nmi = metrics.normalized_mutual_info_score(y_pred, labels_true)
        ami = metrics.adjusted_mutual_info_score(y_pred, labels_true)
        ari = metrics.adjusted_rand_score(y_pred, labels_true)
        acc = metrics.accuracy_score(y_pred, labels_true)
        f1 = metrics.f1_score(y_pred, labels_true, average=self.f1_average)

        return acc, f1, nmi, ari, ami

    def run_dbscan(self,extract_features, extract_labels):

        # Extract labels
        extract_labels = extract_labels.cpu().numpy()
        labels_true = extract_labels

        # Extract features
        # X = extract_features[indices, :]
        X = extract_features.cpu().detach().numpy()
        assert labels_true.shape[0] == X.shape[0]

        n_classes = len(set(labels_true.tolist()))

        db = OPTICS(min_cluster_size=8, xi=0.01)
        db.fit(X)
        labels = db.labels_

        y_pred = np.zeros_like(labels)
        for i in range(n_classes):
            mask = (labels == i)
            y_pred[mask] = mode(labels_true[mask])[0]

        nmi = metrics.normalized_mutual_info_score(labels_true, labels)
        ami = metrics.adjusted_mutual_info_score(labels_true, labels)
        ari = metrics.adjusted_rand_score(labels_true, labels)
        acc = metrics.accuracy_score(y_pred, labels_true)
        f1 = metrics.f1_score(y_pred, labels_true, average=self.f1_average)

        return acc, f1, nmi, ari, ami

    def run_hdbscan(self, extract_features, extract_labels):

        # Extract labels
        extract_labels = extract_labels.cpu().numpy()
        labels_true = extract_labels

        # Extract features
        # X = extract_features[indices, :]
        X = extract_features.cpu().detach().numpy()
        assert labels_true.shape[0] == X.shape[0]

        n_classes = len(set(labels_true.tolist()))

        hdb = HDBSCAN(min_cluster_size=8)
        hdb.fit(X)
        labels = hdb.labels_

        y_pred = np.zeros_like(labels)
        for i in range(n_classes):
            mask = (labels == i)
            y_pred[mask] = mode(labels_true[mask])[0]

        nmi = metrics.normalized_mutual_info_score(labels_true, labels)
        ami = metrics.adjusted_mutual_info_score(labels_true, labels)
        ari = metrics.adjusted_rand_score(labels_true, labels)
        acc = metrics.accuracy_score(y_pred, labels_true)
        f1 = metrics.f1_score(y_pred, labels_true, average=self.f1_average)

        return acc, f1, nmi, ari, ami




    def compute_metrics(self, embeddings, data, args):

        idx = data['indices']
        #output = self.decode(embeddings, data['adj_train_norm'])

        """if args.inference:
            output = self.alignment_predict(output, args.n_classes)
            self.weights = torch.Tensor([1.] * args.n_classes)"""

        idx, anchor, positive, negative = self.get_triplet(idx, data['triplet_set'], embeddings)


        Loss = nn.TripletMarginLoss(margin=args.margin)
        #Loss = InfoNCE()

        loss = Loss(anchor, positive, negative)


        if args.clustering_method == 'hdbscan':
            acc, f1, nmi, ari, ami = self.run_hdbscan(embeddings, data['labels'])
        elif args.clustering_method == 'dbscan':
            acc, f1, nmi, ari, ami = self.run_dbscan(embeddings, data['labels'])
        else:
            acc, f1, nmi, ari, ami = self.run_kmean(embeddings, data['labels'])

        metrics = {'loss': loss, 'acc': acc, 'f1': f1, 'nmi': nmi, 'ari': ari, 'ami': ami}


        return metrics

    def init_metric_dict(self):
        return {'acc': -1, 'nmi': -1, 'ari': -1, 'ami': -1}

    def has_improved(self, m1, m2):
        return m1['ami'] < m2['ami']


    def get_triplet(self, idx, triplet_set, embeddings):
        anchor =[]
        positive = []
        negative = []
        for i in idx:
            if i in triplet_set.keys():
                anchor.append(i)
                positive.append(triplet_set[i][0])
                negative.append(triplet_set[i][1])

        return anchor, embeddings[anchor], embeddings[positive], embeddings[negative]


