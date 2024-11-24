"""Data utils functions for pre-processing and data loading."""
import os
import pickle
import pickle as pkl
import sys
from scipy import sparse
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from random import choice
from tqdm import tqdm



# add
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(args):
    if args.task == 'nc':
        data = load_data_nc(args)

    data['adj_train_norm'], data['features'], data['label_features'] = process(
        data['adj_train'], data['features'], data['label_features'], args.normalize_adj, args.normalize_feats
    )

    return data

# ############### FEATURES PROCESSING ####################################



def process(adj, features, label_features, normalize_adj, normalize_feats):
    if sp.isspmatrix(features):
        features = np.array(features.todense())
        label_features = np.array(label_features.todense())
        
    if normalize_feats:
        features = normalize(features)
        label_features = normalize(label_features)
        
    features = torch.Tensor(features)
    label_features = torch.Tensor(label_features)
    
    if normalize_adj:
        adj = normalize(adj + sp.eye(adj.shape[0]))
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    return adj, features, label_features


def normalize(mx):
    """Row-normalize sparse matrix."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo()
    indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.Tensor(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


# ############### NODE CLASSIFICATION DATA LOADERS ####################################


def load_data_nc(args):

    if args.dataset in ['Event2012', 'Event2018']:
        adj, features, labels, indices, triplet_set, label_features = load_dataset_data(args)
    else:
        raise Exception('Dataset error!')

    labels = torch.LongTensor(labels)

    data = {'adj_train': adj, 'features': features, 'labels': labels, 'indices': indices,
            'triplet_set': triplet_set, 'label_features': label_features}
    return data


# ############### DATASETS ####################################


def load_dataset_data(args):


    if args.dataset_model == '/offline':
        loadpath = args.datapath + args.dataset + f'{args.dataset_model}/' + args.message_block
    else:
        loadpath = args.datapath + args.dataset + f'{args.dataset_model}_{args.llm_model}_{args.method}' \
               + '/' + args.message_block


    sampling_style = args.sampling_style
    alignment_triplet_set_save_path = loadpath + f'alignment_triplet_{sampling_style}_{args.eta}_{args.k}.pkl'
    if not args.metapath and not args.label_similarity:
        args.sampling_style = "None"
        alignment_triplet_set_save_path = loadpath + f'alignment_triplet_full_random.pkl'
    if not args.metapath and args.label_similarity:
        args.sampling_style = "None"
        alignment_triplet_set_save_path = loadpath + f'alignment_triplet_no_metapath.pkl'
    if not args.label_similarity and args.metapath:
        args.sampling_style = "None"
        alignment_triplet_set_save_path = loadpath + f'alignment_triplet_no_similarity.pkl'





    if args.llm_model != 'original':
        t = 'llm'
    else:
        t = 'original'
    features_og = np.load(os.path.join(loadpath, f'features_{t}.npy'))
    labels_og = np.load(os.path.join(loadpath, 'labels.npy'))
    adj_og = sparse.load_npz(loadpath + '/s_bool_A_tid_tid.npz')
    indices_og = np.load(os.path.join(loadpath, 'indices.npy'))


    adj = sp.csr_matrix(adj_og, dtype=np.int64)
    features = sp.coo_matrix(features_og).tolil()

    labels = encode_onehot(labels_og)
    labels = np.argmax(labels, 1)

    # generatge current triplet
    label_features_path = args.datapath + args.dataset + '/llm_label_features_time.pkl'
    metapath_pair_path = args.datapath + args.dataset + '/' + 'metapath_pair.pkl'


    indices = indices_og.tolist()

    print('Generate Metapath based triplet set')

    with open(metapath_pair_path, 'rb') as f:
        metapath_pair = pickle.load(f)

    with open(label_features_path, 'rb') as f:
        features_label_og = pickle.load(f)

    if not os.path.exists(alignment_triplet_set_save_path):
        alignment_triplet = get_triplet_set(metapath_pair, features_label_og, indices, args, sampling_style)
        with open(alignment_triplet_set_save_path, 'wb') as f:
            pickle.dump(alignment_triplet, f)
    else:
        with open(alignment_triplet_set_save_path, 'rb') as f:
            alignment_triplet = pickle.load(f)

    if args.dataset_model == '/incremental':
        indices = list(range(len(indices)))

    label_features = sp.coo_matrix(features_label_og).tolil()

    return adj, features, labels, indices, alignment_triplet, label_features

def get_triplet_set(meta, label_features, indices, args, sampling_style):
    triplet_set = {}
    negative_list = []
    # meta + similarity
    if args.metapath and args.label_similarity:
        print('Adopt Meta-path and Label Similarity')
        p_s, n_s = sampling_style.split('_')
        for i in tqdm(indices):
            if i in meta.keys() and len(meta[i]) != 0:
                positive_list = []
                for p in meta[i]:
                    if p in indices:
                        positive_list.append(p)
                if len(positive_list) != 0:
                    if p_s == 'enumerate':
                        anchor, positive = get_positive_pair_enumerate(i, positive_list, label_features)
                    else:
                        anchor, positive = get_positive_pair_random(i, positive_list, label_features, args.eta, args.k)
                    if i not in negative_list and len(negative_list) != 0 and n_s == 'hard':
                        negative = get_negative_pair(i, negative_list, label_features, args.eta, args.k)
                    elif n_s == 'soft':
                        n_l = list(set(indices).difference((set(positive_list))))
                        negative = get_negative_pair(i, n_l, label_features, args.eta, args.k)
                    else:
                        negative = get_negative_pair(i, indices, label_features, args.eta, args.k)
                else:
                    anchor, positive = get_positive_pair_random(i, indices, label_features, args.eta, args.k)
                    negative_list.append(i)
                    if i not in negative_list and len(negative_list) != 0 and n_s == 'hard':
                        negative = get_negative_pair(i, negative_list, label_features, args.eta, args.k)
                    else:
                        negative = get_negative_pair(i, indices, label_features, args.eta, args.k)
                triplet_pair = [indices.index(positive), indices.index(negative)]
            else:
                anchor, positive = get_positive_pair_random(i, indices, label_features, args.eta, args.k)
                negative_list.append(i)
                negative = get_negative_pair(i, indices, label_features, args.eta, args.k)
                triplet_pair = [indices.index(positive), indices.index(negative)]

            triplet_set[indices.index(anchor)] = triplet_pair
    # similarity only
    elif not args.metapath and args.label_similarity:
        print("Adopt Label Similarity Only")
        for i in tqdm(indices):
            anchor, positive = get_positive_pair_random(i, indices, label_features, args.eta, args.k)
            negative = get_negative_pair(i, indices, label_features, args.eta, args.k)
            triplet_pair = [indices.index(positive), indices.index(negative)]
            triplet_set[indices.index(anchor)] = triplet_pair
    # meta path only
    elif not args.label_similarity and args.metapath:
        print("Adopt Meta-paht Only")
        for i in tqdm(indices):
            if i in meta.keys() and len(meta[i]) != 0:
                positive_list = []
                for p in meta[i]:
                    if p in indices:
                        positive_list.append(p)
                if len(positive_list) != 0:
                    anchor, positive = get_positive_pair_random_no_similarity(i, positive_list)
                else:
                    anchor, positive = get_positive_pair_random_no_similarity(i, indices)
                negative = get_negative_pair(i, indices, label_features, args.eta, args.k)
                triplet_pair = [indices.index(positive), indices.index(negative)]
            else:
                anchor, positive = get_positive_pair_random_no_similarity(i, indices)
                negative_list.append(i)
                negative = get_negative_pair_no_similarity(i, indices)
                triplet_pair = [indices.index(positive), indices.index(negative)]

            triplet_set[indices.index(anchor)] = triplet_pair
    else:
        print('No Meta-paht and No Label Similarity')
        for i in tqdm(indices):
            anchor, positive = get_positive_pair_random_no_similarity(i, indices)
            negative = get_negative_pair_no_similarity(i, indices)
            triplet_pair = [indices.index(positive), indices.index(negative)]
            triplet_set[indices.index(anchor)] = triplet_pair

    return triplet_set



def get_cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    magnitude_A = np.linalg.norm(A)
    magnitude_B = np.linalg.norm(B)

    cosine_similarity = dot_product / (magnitude_A * magnitude_B)
    return cosine_similarity

def get_positive_pair_random(anchor_index, positive_index_list, label_features, eta, k):
    anchor_index = anchor_index
    count = 1
    random_positive_list = []
    cs_list = []
    random_positive_index = int(choice(positive_index_list))
    cosine_similarity = get_cosine_similarity(label_features[anchor_index], label_features[random_positive_index])
    random_positive_list.append(random_positive_index)
    cs_list.append(cosine_similarity)
    while cosine_similarity < eta and count < 20:
        random_positive_index = choice(positive_index_list)
        cosine_similarity = get_cosine_similarity(label_features[anchor_index], label_features[random_positive_index])
        random_positive_list.append(random_positive_index)
        cs_list.append(cosine_similarity)
        count += 1
    if count == k:
        random_positive_index = random_positive_list[cs_list.index(max(cs_list))]
    return anchor_index, random_positive_index

def get_positive_pair_random_no_similarity(anchor_index, positive_index_list):
    anchor_index = anchor_index
    random_positive_index = int(choice(positive_index_list))
    while random_positive_index == anchor_index:
        random_positive_index = int(choice(positive_index_list))
    return anchor_index, random_positive_index


def get_positive_pair_enumerate(anchor_index, positive_index_list, label_features):
    anchor_index = anchor_index
    enumerate_positive_list = []
    cs_list = []
    for i in positive_index_list:
        if i != anchor_index:
            enumerate_positive_list.append(i)
            cosine_similarity = get_cosine_similarity(label_features[anchor_index], label_features[i])
            cs_list.append(cosine_similarity)

    enumerate_positive_index = enumerate_positive_list[cs_list.index(max(cs_list))]
    return anchor_index, enumerate_positive_index

def get_negative_pair(anchor_index, index_list, label_features, eta, k):
    cs_list = []
    negative_list = []
    count = 0
    random_negative_index = choice(index_list)
    while random_negative_index == anchor_index:
        random_negative_index = choice(index_list)
    cosine_similarity = get_cosine_similarity(label_features[anchor_index], label_features[random_negative_index])
    while cosine_similarity > eta and count <= k:
        random_negative_index = choice(index_list)
        cosine_similarity = get_cosine_similarity(label_features[anchor_index], label_features[random_negative_index])
        cs_list.append(cosine_similarity)
        negative_list.append(random_negative_index)
        count += 1
    if len(negative_list) != 0:
        random_negative_index = negative_list[cs_list.index(min(cs_list))]
    return random_negative_index

def get_negative_pair_no_similarity(anchor_index, index_list):
    random_negative_index = choice(index_list)
    while random_negative_index == anchor_index:
        random_negative_index = choice(index_list)
    return random_negative_index










