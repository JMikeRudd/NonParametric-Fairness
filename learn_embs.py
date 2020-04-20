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


dataset = 'adult_income'
emb_dim = 30
lr = 0.01
epochs=200
device = 'cuda'
save_dir = 'trained_models'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

model_dir = os.path.join(save_dir, 'adult_income_euclidean')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

logging.basicConfig(filename=os.path.join(model_dir, 'train.log'))
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load Data
X, y, summary, protected = load_data(dataset)
if dataset == 'adult_income':
    X, y, protected = X[:3000], y[:3000], protected[:3000]

dl = DataLoader(TensorDataset(torch.arange(len(X)).long().to(device)), batch_size=32, shuffle=True)

# Estimate Densities
estimator = est_densities(X, summary)

# Compute Distances
dists = est_distances(estimator, X)
torch.save(torch.tensor(dists), os.path.join(model_dir, 'dist_matrix'))
dists = torch.load(os.path.join(model_dir, 'dist_matrix')).to(device)


# Declare Embedding Components
metric = get_metric('tabular', dist_matrix=dists)
mapping = DiscreteEmbMapping(len(X), emb_dim).to(device)
emb_space = EuclideanEmbeddingSpace(mapping)
isom_emb = IsometricEmbedding(emb_space, metric).to(device)

train_isometric_embedding(isom_emb, epochs=epochs, data_loader=dl, optim=Adam(mapping.parameters(), lr=lr),
                          print_every=1, save_every=5, save_dir=model_dir, plt_cols=y)

print('Loading Embs')
isom_emb = torch.load(os.path.join(model_dir, 'isometric_embedding'))
#
train_inds = torch.tensor(np.random.choice(len(X), int(0.8 * len(X)), replace=False))
test_inds = torch.tensor(list(set([i for i in range(len(X))]).difference(set([i.item() for i in train_inds]))))
embs = isom_emb.emb_space.mapping.model.weight.data.clone().to(device)
embs -= embs.mean(0)
embs /= embs.std(0)
if dataset == 'german_credit':
    y = (torch.tensor(y) - 1).long().to(device)
elif dataset == 'adult_income':
    y = torch.tensor(y).long().to(device)

X_train, X_test, y_train, y_test = embs[train_inds], embs[test_inds], y[train_inds], y[test_inds]

all_test_preds = {}
for lip_K in range(1,16):
    print('Training K={}'.format(lip_K))
    # Train Model
    X_train, X_test = embs[train_inds] * torch.tensor(lip_K), embs[test_inds] * torch.tensor(lip_K)    
    # Get Model
    model = torch.nn.Sequential(linear_stack([emb_dim, 4*emb_dim, 2], spec_norm=True), torch.nn.Softmax()).to(device)
    train_model(model, X=X_train, Y=y_train, epochs=50, opt=Adam(model.parameters(), lr=0.001), save_dir=model_dir)
    #model = torch.load(os.path.join(model_dir, 'model'))
    model.eval()
    pred_y = model(X_test).max(dim=1)[1]
    acc = (pred_y==y_test).float().mean()
    tpr = ((y_test==1)*(pred_y==1)).float().sum()/(y_test==1).float().sum()
    tnr = ((y_test==0)*(pred_y==0)).float().sum()/(y_test==0).float().sum()
    logger.info('Acc: {}\nTPR: {}\nTNR: {}'.format(acc, tpr, tnr))
    all_test_preds[lip_K] = {'pred': pred_y,
                             'protected': torch.tensor(protected)[test_inds],
                             'y_test': y_test,
                             'acc': acc, 'tpr': tpr, 'tnr': tnr}
    model.train()
    del model
    del X_train
    del X_test


torch.save(all_test_preds, os.path.join(model_dir, 'all_test_preds'))
