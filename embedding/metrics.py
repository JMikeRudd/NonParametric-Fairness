import torch
from copy import copy
from itertools import product

from .utils import pi
from .embedding_models import MixedEmbMapping

METRICS = ['euclidean', 'angular', 'tabular']


class Metric(torch.nn.Module):
    ''' Class to hold all metrics, includes utilities for computing distances
        validating inputs, and updating the metric (if it is estimated)
        Arguments:
            None
        Methods:
            forward:
                Validate two inputs and compute distance between them. Inputs
                assumed to have shape (b, d) where b is batch size and d is the
                dimension of the inputs. d can change batch-by-batch as long as
                inputs have same dim and dim is not declared at initialization.
            _validate_inputs:
                Checks that inputs are torch tensors with the right dimension
            _compute_dist:
                Implemented by subclasses. Does the distance computation.
                Inputs same as forward. Returns tensor of scalar distances for
                each row in the batch.
            update_metric:
                Some metrics are estimated from data and will change over time.
                This method handles training on one batch and returns a dict of
                any info (e.g. loss) that the caller of the method might want.
    '''
    def __init__(self, dim=None):
        super().__init__()

        if dim is not None:
            assert isinstance(dim, int) and dim > 0
        self.dim = dim

    def forward(self, x, y):
        x, y = self._validate_inputs(x, y)
        return self._compute_dist(x, y)

    def _validate_inputs(self, x, y):
        assert (issubclass(type(x), torch.Tensor) and
                issubclass(type(y), torch.Tensor))
        if x.dim() == 1 and y.dim() == 1:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
        assert x.dim() == 2 and y.dim() == 2
        assert x.size(1) == y.size(1)
        if self. dim is not None:
            assert x.size(1) == self.dim
        return x, y

    def _compute_dist(self, x, y):
        raise NotImplementedError('Implemented by subclasses')

    def update_metric(self, batch):
        raise NotImplementedError('Implemented by subclasses')


class EuclideanMetric(Metric):
    ''' Metric is the Euclidean distance.
    '''
    def __init__(self, dim=None):
        super().__init__(dim)

    def _compute_dist(self, x, y):
        return ((x - y) ** 2).sum(dim=-1)

    def update_metric(self, batch):
        return {}


class AngularDistanceMetric(Metric):
    ''' Metric is the angular distance.
    '''
    def __init__(self, epsilon=0.001, dim=None):
        super().__init__(dim)

        assert isinstance(epsilon, float) and epsilon > 0
        self.epsilon = epsilon

    def _compute_dist(self, x, y):
        cos_sim = (x * y).sum(dim=-1) / (1 + self.epsilon)
        # assert cos_sim.max() < 1 and cos_sim.min() > -1

        return torch.acos(cos_sim) / pi

    def update_metric(self, batch):
        return {}


class TabularMetric(Metric):
    ''' Metric is a lookup function of a supplied distance matrix.
    '''
    def __init__(self, dist_matrix, dim=None):
        super().__init__(dim)

        assert issubclass(type(dist_matrix), torch.Tensor)
        assert dist_matrix.dim() == 2 and dist_matrix.size(0) == dist_matrix.size(1)
        self.dist_matrix = dist_matrix
        self.max_int = dist_matrix.size(0)

    def _compute_dist(self, x, y):
        x, y = x[0], y[0]
        #import pdb; pdb.set_trace()
        #assert isinstance(x, torch.LongTensor) and isinstance(y, torch.LongTensor)
        assert (x >= 0).all() and (x < self.max_int).all() and (y >= 0).all() and (y < self.max_int).all()
        ds = torch.zeros_like(x).float()
        for i in range(len(x)):
            ds[i] += self.dist_matrix[x[i], y[i]]
        #cat_idx = [tuple(c) for c in torch.cat([x.unsqueeze(1), y.unsqueeze(1)], axis=1)]
        return ds

    def update_metric(self, batch):
        return {}


class LinearCombinationMetric(Metric):
    ''' Class for when the metric is a linear combination of other metrics.
        Restriction that each coefficient must be positive.
        Arguments:
            metrics (required):
                list of Metric objects
            weights (optional):
                list of floats. If None then defaults to 1/|metrics|
        Methods:
            forward:
                return linear combination of submetrics evaluated on inputs
            update_metric:
                return list of results of update_metric for submetrics
    '''
    def __init__(self, metrics, weights=None):
        super().__init()

        # Check that metrics passed as list
        assert isinstance(metrics, list)

        # If no weights given then assign each equal weight
        weights = [float(1 / len(metrics)) for _ in range(len(metrics))] if\
            weights is None else weights

        # Check weights are list equal in length to metrics
        assert isinstance(weights, list) and (len(metrics) == len(weights))

        # Check each metric is Metric subclass and w are all positive floats
        for (m, w) in zip(metrics, weights):
            assert issubclass(type(m), Metric)
            assert isinstance(w, float) and w > 0

        self.metrics = metrics
        self.weights = weights

        # Check that any metrics with dim not None have same dim
        dims = [m.dim for m in self.metrics if m.dim is not None]
        dim = dims[0] if len(dims) > 0 else None
        assert all([d == dim for d in dims])
        self.dim = dim

    def forward(self, x, y):

        dist = 0
        for (m, w) in zip(self.metrics, self.weights):
            dist += w * m(copy(x), copy(y))

        return dist

    def update_metric(self, batch):
        return [m.update_metric(batch) for m in self.metrics]


def get_metric(metric, model=None, optim=None, **metric_kwargs):
    assert metric in METRICS

    if metric == 'euclidean':
        return EuclideanMetric()
    elif metric == 'angular':
        return AngularDistanceMetric()
    elif metric == 'tabular':
        return TabularMetric(**metric_kwargs)