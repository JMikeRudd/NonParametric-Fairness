import torch
import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

model_dir = 'trained_models/adult_income_euclidean'
all_test_preds = torch.load('trained_models/adult_income_euclidean/all_test_preds')

ks = [int(k) for k in all_test_preds.keys()]
accs = [m['acc'] for (k,m) in all_test_preds.items()]
tprs = [m['tpr'] for (k,m) in all_test_preds.items()]
tnrs = [m['tnr'] for (k,m) in all_test_preds.items()]

e_opps = []
e_odds = []
all_tprs, all_tnrs = {}, {}
for k in ks:
    y_test, pred_y, protected = all_test_preds[k]['y_test'].cpu(), all_test_preds[k]['pred'].cpu(), all_test_preds[k]['protected'].cpu()
    group_tprs, group_tnrs = {}, {}
    for j in range(2):
        group_tprs[j] = ((y_test==1)*(pred_y==1)*(protected==j)).float().sum()/((y_test==1)*(protected==j)).float().sum()
        group_tnrs[j] = ((y_test==0)*(pred_y==0)*(protected==j)).float().sum()/((y_test==0)*(protected==j)).float().sum()
    all_tprs[k] = group_tprs
    all_tnrs[k] = group_tnrs


plt.plot(ks, accs)
plt.title('Model Accuracy vs. Lipschitz Bound')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Test Accuracy')
plt.savefig(os.path.join(model_dir, 'lipschitz_accuracy'))
plt.close()

plt.plot(ks, tprs)
plt.title('Model TPR vs. Lipschitz Bound')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Test True Positive Rate')
plt.savefig(os.path.join(model_dir, 'lipschitz_tpr'))
plt.close()

plt.plot(ks, tnrs)
plt.title('Model TNR vs. Lipschitz Bound')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Test True Negative Rate')
plt.savefig(os.path.join(model_dir, 'lipschitz_tnr'))
plt.close()

plt.plot(ks, [s[0] for (k, s) in all_tprs.items()], 'r-', label='Group 0 TPR') # grp 0 tprs
plt.plot(ks, [s[0] for (k, s) in all_tnrs.items()], 'r--', label='Group 0 TNR') # grp 0 tnrs
#plt.plot(ks, [base_group_tprs[0]] * len(ks), 'k-', label='Group 0 Base TPR') # grp 0 base tprs
#plt.plot(ks, [base_group_tnrs[0]] * len(ks), 'k--', label='Group 0 Base TPR') # grp 0 base tprs
plt.plot(ks, [s[1] for (k, s) in all_tprs.items()], 'b-', label='Group 1 TPR') # grp 1 tprs
plt.plot(ks, [s[1] for (k, s) in all_tnrs.items()], 'b--', label='Group 1 TNR') # grp 1 tnrs
#plt.plot(ks, [base_group_tprs[1]] * len(ks), 'g-', label='Group 1 Base TPR') # grp 1 base tprs
#plt.plot(ks, [base_group_tnrs[1]] * len(ks), 'g--', label='Group 1 Base TPR') # grp 1 base tprs

plt.title('Group TPRs/TNRs vs. Lipschitz Bound')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Group Score')
fontP = FontProperties()
fontP.set_size('small')
plt.legend(prop=fontP)
#plt.savefig(os.path.join(model_dir, 'lipschitz_group_scores_w_base'))
plt.savefig(os.path.join(model_dir, 'lipschitz_group_scores'))
plt.close()


import numpy as np
euo_fairness = (np.array([s[1] for (k, s) in all_tprs.items()]) - np.array([s[0] for (k, s) in all_tprs.items()])) ** 2 + (np.array([s[1] for (k, s) in all_tnrs.items()]) - np.array([s[0] for (k, s) in all_tnrs.items()])) ** 2
euc_fairness = (np.array([s[1] for (k, s) in all_tprs.items()]) - np.array([s[0] for (k, s) in all_tprs.items()])) ** 2 + (np.array([s[1] for (k, s) in all_tnrs.items()]) - np.array([s[0] for (k, s) in all_tnrs.items()])) ** 2
met_fairness = (np.array([s[1] for (k, s) in all_tprs.items()]) - np.array([s[0] for (k, s) in all_tprs.items()])) ** 2 + (np.array([s[1] for (k, s) in all_tnrs.items()]) - np.array([s[0] for (k, s) in all_tnrs.items()])) ** 2

plt.plot(ks, met_fairness, 'b-', label='Non-Parametric')
plt.plot(ks, euc_fairness, 'r-', label='Euclidean (Ordered)')
plt.plot(ks, euo_fairness, 'g-', label='Euclidean')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Unfairness Score')
plt.title('Metric Fairness Comparison')
plt.legend()
plt.savefig('metric_fairness')
plt.close()

import os
import numpy as np
from embedding.embedding_models import *
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from utils import *

dataset = 'adult_income'
emb_dim = 30
lr = 0.01
epochs=1000
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
if dataset == 'german_credit':
    y = (torch.tensor(y) - 1).long().to(device)    
else:
    X, y, protected = X[:5000], y[:5000], protected[:5000]

X, y = torch.tensor(X).to(device), torch.tensor(y).long().to(device)
protected= torch.tensor(protected).to(device)

emb_dict = {}
cat_dim = 0
for j in range(len(summary)):
    print(cat_dim)
    if summary[j]['type'] == 'cat':
        emb_dict[str(j)] = DiscreteEmbMapping(summary[j]['dim'], summary[j]['dim'])
        cat_dim += summary[j]['dim']
    #else:
    #    X[:,j] -= X[:,j].float().mean(0).long()
    #    #X[:,j] = X[:,j] / X[:,j].float().std(0).long()
    #    emb_dict[str(j)] = IDEmbMapping(1)
    #    cat_dim += 1


train_inds = torch.tensor(np.random.choice(len(X), int(0.8 * len(X)), replace=False))
test_inds = torch.tensor(list(set([i for i in range(len(X))]).difference(set([i.item() for i in train_inds]))))
#test_inds = torch.tensor([i for i in range(len(X)) if i not in list(train_inds)])
X_train, X_test, y_train, y_test = X[train_inds], X[test_inds], y[train_inds], y[test_inds]

model = torch.nn.Sequential(
    MixedEmbMapping(
        emb_model_dict=emb_dict,
            emb_dim=2,
            comb_model=MLPEmbMapping([cat_dim, 10*cat_dim, 2], batch_norm=False)),
    torch.nn.Softmax()
    ).to(device)

#model = torch.nn.Sequential(linear_stack([X.size(1), 4*emb_dim, 4*emb_dim, 2], spec_norm=False), torch.nn.Softmax()).to(device)
train_model(model, X=X_train, Y=y_train, epochs=100, opt=Adam(model.parameters(), lr=0.001), save_dir=model_dir)

model.eval()
inp_dict = {str(j): X_test[:,j] for j in range(X_test.size(1))}
pred_y = model(inp_dict).max(dim=1)[1]
acc = (pred_y==y_test).float().mean()
tpr = ((y_test==1)*(pred_y==1)).float().sum()/(y_test==1).float().sum()
tnr = ((y_test==0)*(pred_y==0)).float().sum()/(y_test==0).float().sum()

group_tprs, group_tnrs = {}, {}
test_protected = protected[test_inds]
for j in range(2):
    group_tprs[j] = ((y_test==1)*(pred_y==1)*(test_protected==j)).float().sum()/((y_test==1)*(test_protected==j)).float().sum()
    group_tnrs[j] = ((y_test==0)*(pred_y==0)*(test_protected==j)).float().sum()/((y_test==0)*(test_protected==j)).float().sum()

base_group_tprs = group_tprs
base_group_tnrs = group_tnrs
print(base_group_tprs, base_group_tnrs)







import torch
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

mods = ['', '_euclidean', '_euclidean_unord']
dataset = 'adult_income'

fair_scores = {}
for m in mods:
    model_dir = 'trained_models/' + dataset + m
    all_test_preds = torch.load(os.path.join(model_dir, 'all_test_preds'))
    ks = [int(k) for k in all_test_preds.keys()]
    accs = [m['acc'] for (k,m) in all_test_preds.items()]
    tprs = [m['tpr'] for (k,m) in all_test_preds.items()]
    tnrs = [m['tnr'] for (k,m) in all_test_preds.items()]
    e_opps = []
    e_odds = []
    all_tprs, all_tnrs = {}, {}
    for k in ks:
        y_test, pred_y, protected = all_test_preds[k]['y_test'].cpu(), all_test_preds[k]['pred'].cpu(), all_test_preds[k]['protected'].cpu()
        group_tprs, group_tnrs = {}, {}
        for j in range(2):
            group_tprs[j] = ((y_test==1)*(pred_y==1)*(protected==j)).float().sum()/((y_test==1)*(protected==j)).float().sum()
            group_tnrs[j] = ((y_test==0)*(pred_y==0)*(protected==j)).float().sum()/((y_test==0)*(protected==j)).float().sum()
        all_tprs[k] = group_tprs
        all_tnrs[k] = group_tnrs
    plt.plot(ks, accs)
    plt.title('Model Accuracy vs. Lipschitz Bound')
    plt.xlabel('Lipschitz Bound (K)')
    plt.ylabel('Test Accuracy')
    plt.savefig(os.path.join(model_dir, 'lipschitz_accuracy'))
    plt.close()
    plt.plot(ks, tprs)
    plt.title('Model TPR vs. Lipschitz Bound')
    plt.xlabel('Lipschitz Bound (K)')
    plt.ylabel('Test True Positive Rate')
    plt.savefig(os.path.join(model_dir, 'lipschitz_tpr'))
    plt.close()
    plt.plot(ks, tnrs)
    plt.title('Model TNR vs. Lipschitz Bound')
    plt.xlabel('Lipschitz Bound (K)')
    plt.ylabel('Test True Negative Rate')
    plt.savefig(os.path.join(model_dir, 'lipschitz_tnr'))
    plt.close()
    plt.plot(ks, [s[0] for (k, s) in all_tprs.items()], 'r-', label='Group 0 TPR') # grp 0 tprs
    plt.plot(ks, [s[0] for (k, s) in all_tnrs.items()], 'r--', label='Group 0 TNR') # grp 0 tnrs
    #plt.plot(ks, [base_group_tprs[0]] * len(ks), 'k-', label='Group 0 Base TPR') # grp 0 base tprs
    #plt.plot(ks, [base_group_tnrs[0]] * len(ks), 'k--', label='Group 0 Base TPR') # grp 0 base tprs
    plt.plot(ks, [s[1] for (k, s) in all_tprs.items()], 'b-', label='Group 1 TPR') # grp 1 tprs
    plt.plot(ks, [s[1] for (k, s) in all_tnrs.items()], 'b--', label='Group 1 TNR') # grp 1 tnrs
    #plt.plot(ks, [base_group_tprs[1]] * len(ks), 'g-', label='Group 1 Base TPR') # grp 1 base tprs
    #plt.plot(ks, [base_group_tnrs[1]] * len(ks), 'g--', label='Group 1 Base TPR') # grp 1 base tprs
    plt.title('Group TPRs/TNRs vs. Lipschitz Bound')
    plt.xlabel('Lipschitz Bound (K)')
    plt.ylabel('Group Score')
    fontP = FontProperties()
    fontP.set_size('small')
    plt.legend(prop=fontP)
    #plt.savefig(os.path.join(model_dir, 'lipschitz_group_scores_w_base'))
    plt.savefig(os.path.join(model_dir, 'lipschitz_group_scores'))
    plt.close()
    fairness = (np.array([s[1] for (k, s) in all_tprs.items()]) - np.array([s[0] for (k, s) in all_tprs.items()])) ** 2 + (np.array([s[1] for (k, s) in all_tnrs.items()]) - np.array([s[0] for (k, s) in all_tnrs.items()])) ** 2
    fair_scores[m] = fairness

euo_fairness = fair_scores['_euclidean_unord']
euc_fairness = fair_scores['_euclidean']
met_fairness = fair_scores['']
plt.plot(ks[:-1], met_fairness, 'b-', label='Non-Parametric')
plt.plot(ks, euc_fairness, 'r-', label='Euclidean (Ordered)')
plt.plot(ks, euo_fairness, 'g-', label='Euclidean')
plt.xlabel('Lipschitz Bound (K)')
plt.ylabel('Unfairness Score')
plt.title('Metric Fairness Comparison')
plt.legend()
plt.savefig('metric_fairness')
plt.close()
