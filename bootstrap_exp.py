import os
import logging

from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import *
from density import est_densities, est_distances
from embedding.metrics import *
from embedding.embedding_space import *
from embedding.embedding_models import *
from embedding.isometric_embedding import *
from embedding.utils import linear_stack


dataset = 'german_credit'

n_bootstraps = 300
bootstrap_size = 1000

emb_dim = 30
lr = 0.01
epochs=250
lip_K = 15
device = 'cuda'
save_dir = 'trained_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_dir = os.path.join(save_dir, dataset)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(filename=os.path.join(model_dir, 'train.log'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load Data
X, y, summary, protected = load_data(dataset)
dl = DataLoader(TensorDataset(torch.arange(len(X)).long().to(device)), batch_size=32, shuffle=True)

reduced_X = X[np.random.choice(len(X), 100, replace=False)]
dists = np.zeros((n_bootstraps, len(reduced_X), len(reduced_X)))

for b in range(n_bootstraps):
    # Get bootstrap sample
    samp_inds =  np.random.choice(len(X), bootstrap_size, replace=True)
    X_b, y_b = X[samp_inds], y[samp_inds]
    # Estimate Densities
    estimator = est_densities(X_b, summary)
    # Compute Distances
    dists_b = est_distances(estimator, reduced_X)
    dists[b,:,:] += dists_b
    #torch.save(torch.tensor(dists), os.path.join(model_dir, 'dist_matrix'))
    #dists = torch.load(os.path.join(model_dir, 'dist_matrix')).to(device)

torch.save(dists, os.path.join(model_dir, 'bootstrap_dists'))

n_comps = int(len(reduced_X) * (len(reduced_X) - 1) / 2)
means = []#np.zeros(n_comps)
devs = []#np.zeros(n_comps)

for i in range(dists.shape[1]):
    for j in range(i):
        means.append(dists[:,i,j].mean())
        devs.append(dists[:,i,j].std())

means, devs = np.array(means), np.array(devs)
print(means.mean())
print(devs.mean())
print((devs/np.abs(means)).mean())


from matplotlib import pyplot as plt
plt.scatter(means, devs, s=1)
plt.title('Bootstrap Sample Deviation vs Mean')
plt.xlabel('Bootstrap Distance Mean')
plt.ylabel('Bootstrap Distance Deviation')
plt.savefig(os.path.join(model_dir, 'bootstrap_deviation_plot'))

'''
# Declare Embedding Components
metric = get_metric('tabular', dist_matrix=dists)
mapping = DiscreteEmbMapping(len(X), emb_dim).to(device)
emb_space = EuclideanEmbeddingSpace(mapping)
isom_emb = IsometricEmbedding(emb_space, metric).to(device)

train_isometric_embedding(isom_emb, epochs=epochs, data_loader=dl, optim=Adam(mapping.parameters(), lr=lr),
                          print_every=1, save_every=5, save_dir=model_dir, plt_cols=y)

isom_emb = torch.load(os.path.join(model_dir, 'isometric_embedding'))

# Train Model
embs = isom_emb.emb_space.mapping.model.weight.data.to(device)
embs -= embs.mean(0)
embs *= lip_K
y = (torch.tensor(y) - 1).long().to(device)

# Get Model
model = torch.nn.Sequential(linear_stack([emb_dim, 2*emb_dim, 2], spec_norm=True), torch.nn.Softmax()).to(device)
train_model(model, X=embs, Y=y, epochs=100, opt=Adam(model.parameters(), lr=0.001), save_dir=model_dir)
#model = torch.load(os.path.join(model_dir, 'model'))

model.eval()
pred_y = model(embs).max(dim=1)[1]
acc = (pred_y==y).float().mean()
tpr = ((y==1)*(pred_y==1)).float().sum()/(y==1).float().sum()
tnr = ((y==0)*(pred_y==0)).float().sum()/(y==0).float().sum()
logger.info('Acc: {}\nTPR: {}\nTNR: {}'.format(acc, tpr, tnr))
'''