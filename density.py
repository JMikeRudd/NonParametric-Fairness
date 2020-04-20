import numpy as np
from tqdm import tqdm
from scipy.stats import norm

class Estimator(object):

    def __init__(self, summary, h=None):
        assert isinstance(summary, list)
        self.summary = summary

        self.cat_cols = [i for i in range(len(summary)) if summary[i]['type'] == 'cat']
        cat_dims = [s['dim'] for s in summary if s['type'] == 'cat']
        self.cat_table = np.zeros(cat_dims)

        if h is None:
            self.h = np.ones(len([s for s in summary if s['type'] == 'cont']))
        else:
            self.h = h

        self.cont_cols = [i for i in range(len(summary)) if summary[i]['type'] == 'cont']

    def learn_estimates(self, X, learn_h=True):
        self.cats = X[:, self.cat_cols]
        if np.sum(self.cat_table) == 0.:
            self.cat_table = Estimator._learn_cat_table(self.cat_table, X[:, self.cat_cols])

        assert isinstance(learn_h, bool)
        self.means = X[:, self.cont_cols]
        if learn_h:
            self._learn_h(X)
        self.log_norm_const = -0.5 * (len(self.cont_cols) * np.log(2 * np.pi) + np.sum(np.log(self.h)))
        
        # est cdfs
        self.cdfs = {}
        # categoricals
        for c in self.cat_cols:
            mcdf = np.zeros(self.summary[c]['dim'])
            for i in range(len(mcdf)):
                mcdf[i] += (X[:,0]<=i).sum()/len(X)

            self.cdfs[c] = mcdf

        # continuous
        for c in self.cont_cols:
            cont_idx = self.cont_cols.index(c)
            Uxs = np.unique(X[:,c])
            mcdf = {}
            for ux in Uxs:
                mcdf[ux] = norm.cdf((ux - self.means[:, cont_idx]) / self.h[cont_idx]).mean()
            '''
            for i in range(len(X[:,c])):
                zs = norm.cdf((X[i, c] - self.means[:, cont_idx]) / self.h[cont_idx]).mean()
                ps = (zs)
                mcdf[i] += ps.mean()
            '''
            self.cdfs[c] = mcdf



    @staticmethod
    def _learn_cat_table(cat_table, X):
        if cat_table.ndim > 1:
            unique, counts = np.unique(X[:,0], return_counts=True)
            c_dict = dict(zip(unique, counts))
            for i in range(cat_table.shape[0]):
                if c_dict.get(i, 0) > 0:
                    # Get conditional prob that first var = i
                    p_x0_i = float(c_dict[i] / X.shape[0])
                    # Find rows where this is true
                    row_idxs = np.where(X[:,0]==i)[0]
                    # Get conditional probs of all future assignments given this & prev assignments
                    cond_table = Estimator._learn_cat_table(cat_table[i], X[row_idxs,1:])
                    # Joint prob table is marginal * cond
                    cat_table[i, :] = p_x0_i * cond_table
                else:
                    # If it's zero skip recursion and just return that
                    cat_table[i, :] = np.zeros_like(cat_table[i])
            return cat_table
        else:
            # if it's the last axis can compute marginals usual way
            unique, counts = np.unique(X, return_counts=True)
            c_dict = dict(zip(unique, counts))
            return np.array([c_dict.get(i, 0) / len(X) for i in range(len(cat_table))])

    def _learn_h(self, X):
        return

    def log_prob(self, x):
        assert isinstance(x, np.ndarray) and x.ndim == 2 and x.shape[1] == len(self.cat_cols + self.cont_cols)
        import pdb; pdb.set_trace()
        lps = np.zeros((x.shape[0],1))
        for i in range(x.shape[0]):
            lps[i] += np.log(self.cat_table[tuple(x[i, self.cat_cols])])
            cat_means = self.means[np.where(np.sum(np.abs(self.cats - x[i, self.cat_cols]), axis=1) == 0)[0]]
            z = (x[i, self.cont_cols] - cat_means) / self.h
            lps[i] += -0.5 * np.sum(z ** 2)

        lps += self.log_norm_const

        return lps

    def prob_dist(self, x1, x2):

        # enumerate every categorical assignment between x1 and x2
        '''
        def rec_find(x1_cat, x2_cat):
            assert len(x1_cat) == len(x2_cat) >= 1
            min_idx, max_idx = min(x1_cat[0], x2_cat[0]), max(x1_cat[0], x2_cat[0])
            if len(x1_cat) == 1:
                return [[i] for i in range(min_idx, max_idx + 1)]
            else:
                ret_list = []
                # list of every assignment of downstream variables
                next_list = rec_find(x1_cat[1:], x2_cat[1:])
                for i in range(min_idx, max_idx + 1):
                    ret_list += [[i] + nl for nl in next_list]
                return ret_list

        all_cats = rec_find(x1[self.cat_cols], x2[self.cat_cols])

        # get cont dist for each category and sum
        p_dist = 0
        x1_cont, x2_cont = x1[self.cont_cols], x2[self.cont_cols]
        for c in all_cats:
            p_cat = self.cat_table[tuple(c)]
            if p_cat > 0:
                p_dist += p_cat * self._cont_dist(np.array(c), x1_cont, x2_cont)
        '''
        '''
        assert len(x1) == len(x2)
        cdfs1, cdfs2 = np.zeros_like(x1, dtype=np.float32), np.zeros_like(x2, dtype=np.float32)
        for c in range(len(x1)):
            if c in self.cont_cols and x1[c] not in self.cdfs[c].keys():
                cont_idx = self.cont_cols.index(c)
                cdfs1[c] = norm.cdf((x1[c] - self.means[:, cont_idx]) / self.h[cont_idx]).mean()
            else:
                cdfs1[c] = self.cdfs[c][x1[c]]

            if c in self.cont_cols and x2[c] not in self.cdfs[c].keys():
                cont_idx = self.cont_cols.index(c)
                cdfs2[c] = norm.cdf((x2[c] - self.means[:, cont_idx]) / self.h[cont_idx]).mean()
            else:
                cdfs2[c] = self.cdfs[c][x2[c]]

        p_dist = np.abs(cdfs1 - cdfs2).sum()
        '''
        
        '''
        cat_dist = (x1[self.cat_cols] != x2[self.cat_cols]).sum()
        cont_dist = np.sqrt(((x1[self.cont_cols] - x2[self.cont_cols]) ** 2).mean())
        p_dist = (cat_dist + cont_dist) / len(x1)
        '''

        p_dist = np.sqrt(((x1 - x2) ** 2).mean())

        return p_dist

    def _cont_dist(self, cat, x1_cont, x2_cont):
        cat_means = self.means[np.where(np.sum(np.abs(self.cats - cat), axis=1) == 0)[0]]
        z1, z2 = (x1_cont - cat_means) / self.h, (x2_cont - cat_means) / self.h

        # We assume a mixture of isotropic Gaussians so we can compute the distances independently
        p_diff = np.abs(norm.cdf(z1) - norm.cdf(z2)).prod(axis=1).mean()
        return p_diff

def est_densities(X, summary, h=None):
    estimator = Estimator(summary, h)
    estimator.learn_estimates(X)
    return estimator


def est_distances(estimator, X):
    assert isinstance(X, np.ndarray) and X.ndim == 2
    dists = np.zeros((len(X),len(X)))

    for i in tqdm(range(len(X))):
        for j in range(i):
            dists[i, j] += estimator.prob_dist(X[i], X[j])
            dists[j, i] += dists[i, j]

    return dists
