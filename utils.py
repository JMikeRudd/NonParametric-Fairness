import pandas as pd
import os
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

data_dir = '.data'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

gc_mapping = {
    0: {'A14': 0, 'A11': 1, 'A12': 2, 'A13': 3},
    1: 'numerical',
    2: {'A30': 0, 'A31': 1, 'A32': 2, 'A33': 3, 'A34': 4},
    3: 'remove',
    4: 'numerical',
    5: {'A65': 0, 'A61': 1, 'A62': 2, 'A63': 3, 'A64': 4},
    6: {'A71': 0, 'A72': 1, 'A73': 2, 'A74': 3, 'A75': 4},
    7: {1: 0, 2: 1, 3: 2, 4: 3},
    8: 'remove',
    9: {'A101': 0, 'A102': 1, 'A103': 2},
    10: {1: 0, 2: 1, 3: 2, 4: 3},
    11: {'A124': 0, 'A123': 1, 'A122': 2, 'A121': 3},
    12: 'numerical',
    13: {'A141': 0, 'A142': 1, 'A143': 2},
    14: {'A151': 0, 'A152': 1, 'A153': 2},
    15: {1: 0, 2: 1, 3: 2, 4: 3},
    16: {'A171': 0, 'A172': 1, 'A173': 2, 'A174': 3},
    17: {1: 0, 2: 1},
    18: {'A191': 0, 'A192': 1},
    19: {'A201': 0, 'A202': 1}
}

ai_mapping = {
    0: 'numerical',
    1: [' ?', ' Never-worked', ' Without-pay', ' Self-emp-not-inc', ' Self-emp-inc', ' Local-gov', ' State-gov', ' Federal-gov', ' Private'],
    2: 'remove',
    3: [' Preschool', ' 1st-4th', ' 5th-6th', ' 7th-8th', ' 9th', ' 10th', ' 11th', ' 12th', ' HS-grad', ' Assoc-voc', ' Assoc-acdm', ' Some-college', ' Bachelors', ' Masters', ' Doctorate', ' Prof-school'],
    4: 'numerical',
    5: {' Never-married': 0, ' Married-spouse-absent': 1, ' Married-AF-spouse': 1, ' Married-civ-spouse': 1, ' Divorced': 2, ' Separated': 2, ' Widowed': 3},
    6: [' ?', ' Other-service', ' Handlers-cleaners', ' Farming-fishing', ' Transport-moving', ' Priv-house-serv', ' Protective-serv', ' Armed-Forces', ' Machine-op-inspct', ' Adm-clerical', ' Craft-repair', ' Tech-support', ' Sales', ' Exec-managerial', ' Prof-specialty'],
    7: {' Not-in-family': 0, ' Unmarried': 1, ' Wife': 2, ' Husband': 2, ' Own-child': 3, ' Other-relative': 3},
    8: 'remove',
    9: 'remove',
    10: 'numerical',
    11: 'numerical',
    12: 'numerical',
    13: 'remove',
    14: {' <=50K': 0, ' >50K': 1}
}


def load_data(dataset):
    if dataset == 'german_credit':
        data_path = os.path.join(data_dir, 'german_credit.csv')
        df = pd.read_csv(data_path, header=None)

        mappings = gc_mapping
        X = df.loc[:, :19]
        y = df.loc[:, 20]
        protected = df.loc[:, 8].map({'A91': 0, 'A93': 0, 'A94': 0, 'A92': 1, 'A95': 1})

    elif dataset == 'adult_income':
        data_path = os.path.join(data_dir, 'adult.csv')
        df = pd.read_csv(data_path, header=None)

        mappings = ai_mapping
        X = df.loc[:, :13]
        y = df.loc[:, 14].map(mappings[14])
        protected = df.loc[:, 8].map({' White': 0, ' Asian-Pac-Islander': 1, ' Amer-Indian-Eskimo': 1, ' Other': 1, ' Black': 1})

    else:
        raise ValueError

    new_cols = []
    new_maps = []
    for i in range(X.shape[1]):
        if isinstance(mappings[i], dict):
            new_cols.append(X.loc[:, i].map(mappings[i]))
            new_maps.append({'type': 'cat', 'dim': len(mappings[i])})
        elif isinstance(mappings[i], list):
            map_list = mappings[i]
            map_dict = {m: map_list.index(m) for m in map_list}
            new_cols.append(X.loc[:, i].map(map_dict))
            new_maps.append({'type': 'cat', 'dim': len(mappings[i])})
        elif mappings[i] == 'numerical':
            new_cols.append(X.loc[:, i])
            new_maps.append({'type': 'cont', 'dim': 1})

    new_X = pd.concat(new_cols, axis=1)

    return new_X.to_numpy(), y.to_numpy(), new_maps, protected.to_numpy()


def train_batch(x, y, model, opt):
    opt.zero_grad()
    pred_y = model(x)
    loss = torch.nn.functional.cross_entropy(pred_y, y)
    loss.backward()
    opt.step()
    return loss.item()

def train_epoch(model, X, Y, opt, batch_size=32):
    dl = DataLoader(TensorDataset(torch.arange(len(X)).to(device)), batch_size=batch_size, shuffle=True)
    epoch_losses = 0
    N = len(dl)
    for (_, batch) in enumerate(dl):
        xb, yb = X[batch], Y[batch]
        #xb = {str(i): xb[:,i] for i in range(xb.size(1))}
        b_loss = train_batch(xb, yb, model, opt)
        epoch_losses += b_loss / N

    return epoch_losses

def _log_losses(epoch, epoch_loss, logger):

    log_msg = 'Epoch {}: {}'.format(epoch, epoch_loss)
    if logger is not None:
        logger.info(log_msg)
    else:
        print(log_msg)

def train_model(model, X, Y, epochs, opt,
                print_every=1, save_every=1, save_dir=None):

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    for e in tqdm(range(epochs)):
        epoch_loss = train_epoch(model, X, Y, opt)

        if e % print_every == 0:
            _log_losses(e, epoch_loss, logger)

        if e % save_every == 0:
            # save the model
            torch.save(model, os.path.join(save_dir, 'model'))
