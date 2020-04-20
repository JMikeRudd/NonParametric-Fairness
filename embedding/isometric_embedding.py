import os
import numpy as np
import logging
from itertools import combinations
from tqdm import tqdm
from matplotlib import pyplot as plt

from torch.nn import Module, MSELoss, Parameter
import torch

from .embedding_space import EmbeddingSpace
from .metrics import Metric


class IsometricEmbedding(Module):

    def __init__(self, emb_space, metric, proportional=False):

        super().__init__()

        assert issubclass(type(emb_space), EmbeddingSpace)
        self.emb_space = emb_space

        assert issubclass(type(metric), Metric)
        self.metric = metric

        self.loss_fn = MSELoss()

        # Only train the scale parameter if proportional is True
        assert isinstance(proportional, bool)
        self.K = Parameter(torch.ones(1), requires_grad=proportional)

        # Don't want to train any metric parameters
        # self.model_params = chain(self.K, self.emb_space.parameters())

    def forward(self, input):
        return None

    def real_distance(self, x, y):
        return self.metric(x, y)

    def latent_distance(self, x, y):
        return self.emb_space.compute_dist(x, y)

    def _train_batch(self, batch, optim):
        optim.zero_grad()

        loss = 0
        if not isinstance(batch, dict):
            bs = batch.size(0)
        else:
            bs = [v for (k, v) in batch.items()][0].size(0)

        # Get unique pairs
        [I, J] = [list(idx) for idx in zip(*list(combinations(range(bs), 2)))]

        if not isinstance(batch, dict):
            x, y = batch[I], batch[J]
        else:
            x = {k: v[I] for (k, v) in batch.items()}
            y = {k: v[J] for (k, v) in batch.items()}

        # Get estimated distances in latent space and true distances
        est = self.latent_distance(x, y)
        real = self.real_distance(x, y)

        loss = self.loss_fn(self.K * est, real)
        loss.backward()
        optim.step()

        return loss.item()

    def embed(self, x):
        return self.emb_space.embed(x)


def train_isometric_embedding_epoch(isom_embedding, data_loader, optim):
    epoch_losses = 0
    N = len(data_loader)

    for (_, batch) in enumerate(data_loader):
        if isinstance(batch, list) and len(batch) == 1:
            batch = batch[0]

        if not isinstance(batch, dict):
            assert issubclass(type(batch), torch.Tensor)
        else:
            for (_, v) in batch.items():
                assert issubclass(type(v), torch.Tensor)

        batch_loss = isom_embedding._train_batch(batch, optim)

        epoch_losses += batch_loss / N

    return epoch_losses


def train_isometric_embedding(isom_embedding, epochs, data_loader, optim,
                              print_every=1, save_every=1, save_dir=None, **kwargs):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for e in tqdm(range(epochs)):
        epoch_loss = train_isometric_embedding_epoch(
            isom_embedding, data_loader, optim)

        if e % print_every == 0:
            _log_losses(e, epoch_loss, logger)

        if e % save_every == 0:
            # save the model
            torch.save(isom_embedding,
                       os.path.join(save_dir, 'isometric_embedding'))
            plt_svd_embs(isom_embedding, os.path.join(save_dir, 'embs'), idx=e, **kwargs)


def _log_losses(epoch, epoch_loss, logger):

    log_msg = 'Epoch {}: {}'.format(epoch, epoch_loss)
    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)

def plt_svd_embs(isom_embedding, save_dir, idx, plt_cols=None):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)    

    embs = isom_embedding.emb_space.mapping.model.weight.data.cpu().numpy()
    embs -= np.mean(embs, 0)

    U, S, V = np.linalg.svd(embs, full_matrices=False, compute_uv=True)
    embs_svd = np.dot(U, np.diag(S))

    eigvals = S * S / (len(embs) - 1)
    plt.bar([i for i in range(embs.shape[1])], eigvals)
    plt.savefig(os.path.join(save_dir, 'eigenvalue_plot_{}'.format(idx)))
    plt.close()

    if plt_cols is not None:
        c=list(plt_cols)
    else:
        c=[0] * len(embs)

    for i in range(3):
        for j in range(i):
            plt.scatter(embs_svd[:,i], embs_svd[:,j], s=1, c=c, cmap='winter')
            plt.title('Embeddings d{} vs d{}'.format(i,j))
            plt.savefig(os.path.join(save_dir, 'emb_plt_{}_{}_{}'.format(i,j,idx)))
            plt.close()
