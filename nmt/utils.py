import os, os.path
import logging
from datetime import timedelta
import subprocess

import numpy
import torch
import torch.nn as nn

import nmt.all_constants as ac
import networkx as nx
from scipy.sparse.linalg import eigs


def get_logger(logfile=None):
    _logfile = logfile if logfile else './DEBUG.log'
    """Global logger for every logging"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s:%(filename)s:%(lineno)s - %(funcName)20s(): %(message)s')

    if not logger.handlers:
        debug_handler = logging.FileHandler(_logfile)
        debug_handler.setFormatter(formatter)
        debug_handler.setLevel(logging.DEBUG)
        logger.addHandler(debug_handler)

    return logger


def shuffle_file(input_file):
    shuffled_file = input_file + '.shuf'
    scriptdir = os.path.dirname(os.path.abspath(__file__))
    commands = 'bash {}/../scripts/shuffle_file.sh {} {}'.format(scriptdir, input_file, shuffled_file)
    subprocess.check_call(commands, shell=True)
    subprocess.check_call('mv {} {}'.format(shuffled_file, input_file), shell=True)


def get_validation_frequency(train_length_file, val_frequency, batch_size):
    with open(train_length_file) as f:
        line = f.readline().strip()
        num_train_toks = int(line)

    return int(num_train_toks * val_frequency / batch_size)


def format_seconds(seconds):
    return str(timedelta(seconds=seconds))


def get_vocab_masks(config, src_vocab_size, trg_vocab_size):
    masks = []
    for vocab_size, lang in [(src_vocab_size, config['src_lang']), (trg_vocab_size, config['trg_lang'])]:
        if config['tie_mode'] == ac.ALL_TIED:
            mask = numpy.load(os.path.join(config['data_dir'], 'joint_vocab_mask.{}.npy'.format(lang)))
        else:
            mask = numpy.ones([vocab_size], numpy.float32)

        mask[ac.PAD_ID] = 0.
        mask[ac.BOS_ID] = 0.
        masks.append(torch.from_numpy(mask).type(torch.uint8))

    return masks


def get_vocab_sizes(config):
    def _get_vocab_size(vocab_file):
        vocab_size = 0
        with open(vocab_file) as f:
            for line in f:
                if line.strip():
                    vocab_size += 1
        return vocab_size

    src_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['src_vocab_size'], config['src_lang']))
    trg_vocab_file = os.path.join(config['data_dir'], 'vocab-{}.{}'.format(config['trg_vocab_size'], config['trg_lang']))

    return _get_vocab_size(src_vocab_file), _get_vocab_size(trg_vocab_file)

def get_sine_encoding(dim, sentence_length):
    div_term = numpy.power(10000.0, - (numpy.arange(dim) // 2).astype(numpy.float32) * 2.0 / dim)
    div_term = div_term.reshape(1, -1)
    pos = numpy.arange(sentence_length, dtype=numpy.float32).reshape(-1, 1)
    encoded_vec = numpy.matmul(pos, div_term)
    encoded_vec[:, 0::2] = numpy.sin(encoded_vec[:, 0::2])
    encoded_vec[:, 1::2] = numpy.cos(encoded_vec[:, 1::2])

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return torch.from_numpy(encoded_vec.reshape([sentence_length, dim])).type(dtype)

def get_cycle_graph_lapes(dim, sentence_length):
    """Compute laplacian eigenvectors for large cycle graph (unnormalized Laplacian)"""
    big_length = int(1e6)
    cos_encs = []
    sine_encs = []
    for i in range((big_length//2)+1):
        cos_enc = numpy.cos((2*numpy.pi*i/big_length)*numpy.arange(dim)).reshape(-1,1)
        if i != 0 and i != big_length//2:
            sin_enc = numpy.sin((2*numpy.pi*i/big_length)*numpy.arange(dim)).reshape(-1,1)
            sine_encs.append(sin_enc)
        cos_encs.append(cos_enc)
    cos_encs = numpy.concatenate(cos_encs, axis=1)
    sine_encs = numpy.concatenate(sine_encs, axis=1)
    cos_prefix = cos_encs[:, :(big_length//2)-1]
    evec_prefix = numpy.vstack((cos_prefix, sine_encs)).reshape(dim, big_length-2, order='F')
    evecs = numpy.concatenate((evec_prefix, cos_encs[:, (big_length)//2-1:big_length]), axis=1)
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    return torch.from_numpy(evecs[:dim, :sentence_length].T).type(dtype)

def get_lape_encoding(dim, sentence_length, graph_size=None):
    p_n = nx.cycle_graph(graph_size if graph_size else sentence_length)
    laplacian = nx.normalized_laplacian_matrix(p_n)
    evals, evecs = numpy.linalg.eig(laplacian.A)
    idx = evals.argsort()
    evals, evecs = evals[idx], numpy.real(evecs[:, idx])
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    if graph_size is not None:
        evecs = evecs[:sentence_length,1:dim+1]
        numpy.random.shuffle(evecs)
        pos_enc = torch.from_numpy(evecs).type(dtype)
    else:
        pos_enc = torch.from_numpy(evecs[:, 1:dim+1]).type(dtype)
    return pos_enc

def normalize(x, scale=True):
    mean = x.mean(-1, keepdim=True)
    std = x.std(-1, keepdim=True) + 1e-6
    if scale:
        std = std * x.size()[-1] ** 0.5
    return (x - mean) / std


def gnmt_length_model(alpha):
    def f(time_step, prob):
        return prob / ((5.0 + time_step + 1.0) ** alpha / 6.0 ** alpha)
    return f

class AutomatonPELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_states = config['num_states']
        self.directed = config['directed']
        embed_dim = config['embed_dim']

        self.embedding_pos_enc = nn.Linear(self.num_states, embed_dim)
        self.pos_initial = nn.Parameter(torch.Tensor(self.num_states, 1), requires_grad=True)
        self.pos_transition = nn.Parameter(torch.Tensor(self.num_states, self.num_states), requires_grad=True)

        nn.init.normal_(self.pos_initial)
        nn.init.orthogonal_(self.pos_transition)

    def kron(self, mat1, mat2):
        n1 = mat1.size(0)
        m1 = mat2.size(0)
        n2 = mat1.size(1)
        m2 = mat2.size(1)
        return torch.einsum("ab,cd->acbd", mat1, mat2).view(n1*m1,  n2*m2)

    def forward(self, sentence_len):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        ones = torch.ones(sentence_len-1)
        if self.directed:
            adj = torch.diag(ones, 1)
        else:
            adj = torch.diag(ones, -1) + torch.diag(ones, 1)

        # z = torch.zeros(self.num_states, g.num_nodes()-1, requires_grad=False, device=device)

        # vec_init = torch.cat((self.pos_initial, z), dim=1)
        vec_init = torch.cat([self.pos_initial for _ in range(adj.shape[0])], dim=1)
        vec_init = vec_init.transpose(1, 0).flatten()

        adj = adj.reshape(adj.shape[1], adj.shape[0]).to(device)
        kron_prod = self.kron(adj, self.pos_transition)
        B = torch.eye(kron_prod.shape[1], device=device) - kron_prod

        vec_init = vec_init.reshape(-1, 1)

        encs, _ = torch.solve(vec_init, B)
        stacked_encs = torch.stack(encs.split(self.num_states), dim=1)
        stacked_encs = stacked_encs.transpose(1, 0).squeeze(2)

        pe = self.embedding_pos_enc(stacked_encs)

        return pe
