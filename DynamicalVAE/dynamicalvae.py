# -*- coding: utf-8 -*-


import torch
import os
import pickle as pkl
os.chdir('/content/drive/MyDrive/JSALT/DynamicalVAE/DVAE')
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import numpy as np
from copy import deepcopy
from typing import List, Union
import random
from collections import Counter
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')


from dvae.model import build_VAE, build_DKF, build_STORN, build_VRNN, build_SRNN, build_RVAE, build_DSAE
import graphviz
from dvae.model import vrnn,srnn
from dvae import learning_algo,learning_algo_new
from sklearn.cluster import KMeans



"""# Test"""

!python train_model.py --cfg ./config/speech/cfg_rvae_Causal.ini

train_dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
        transform=transforms.ToTensor()),
    batch_size=16, shuffle=True)

from dvae.model import vrnn,srnn

x_dim = 28
z_dim = 16
device = 'cpu'
dvae_model = vrnn.VRNN(x_dim=x_dim, z_dim=z_dim).to(device)
model_info = dvae_model.get_info()
#print(model_info)
dvae_model

from dvae.utils import myconf, get_logger, loss_ISD, loss_KLD, loss_MPJPE

for batch, (X,y) in enumerate(train_dataloader):
  print(X.shape)

  plt.axis("off")
  plt.imshow(X[0].squeeze(), cmap="gray")
  plt.show()
  X = X.squeeze(1)
  X = X.permute(2, 0, 1)



  out = dvae_model(X)
  print(out.shape)
  plt.axis("off")
  plt.imshow(out[0].detach().numpy(), cmap="gray")
  plt.show()


  break

"""# Create data"""

from datasets import load_dataset
from sentence_transformers import SentenceTransformer

multiwoz_dset = load_dataset("multi_woz_v22")

utterances = []
ids = []

for n, i in enumerate(multiwoz_dset['test']):
  print(n)
  utterances.append(i['turns']['utterance'])
  ids.append(i['dialogue_id'])

model = SentenceTransformer("sentence-transformers/LaBSE")

model.cuda()

df = pd.DataFrame(list(zip(ids, utterances)), columns =['id', 'utterances'])

df['embeds'] = df.utterances.apply(lambda x : model.encode(x))

df.to_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_test.pkl')

df = pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_val.pkl')

"""# Experiment"""

# train_dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=True, download=True,
#         transform=transforms.ToTensor()),
#     batch_size=2, shuffle=True)
#
# val_dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST('data', train=False, download=True,
#         transform=transforms.ToTensor()),
#     batch_size=2, shuffle=False)

class LoadDataset (Dataset):
  def __init__(self,data, train = False):
    self.data = data
    if train :
      self.data = data.sample(frac = 1)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self,i):
    return self.data.iloc[i]

def collate_data(data):
  ids = []
  tensors = []

  max = 0
  pad = np.zeros(768)

  for i in data:
    ids.append(i[0])

    if i[2].shape[0] > max:
      max = i[2].shape[0]


  for i in data:
    shape = i[2].shape[0]
    arr = i[2]

    for j in range( shape , max):
      arr=np.vstack([arr, pad])

    tensors.append(arr)

  tensors = np.array(tensors)
  tensors = torch.tensor(tensors).float()

  return ids, tensors


train_data = LoadDataset(pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl'), True)
train_loader = DataLoader(train_data, 32, True, collate_fn=collate_data )

val_data = LoadDataset(pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_val.pkl'))
val_loader = DataLoader(val_data, 32, True, collate_fn=collate_data )

params = {}
params['cfg'] = './config/speech/cfg_vrnn.ini'
params['pretrain_dict'] = None
params['model_dir'] = './saved_model/WSJ0_2023-08-01-14h06_VRNN_z_dim=16'
params['reload'] = True

learning = learning_algo_new.LearningAlgorithm(params, train_loader, val_loader)

learning.train()

"""# Generate Clustering"""

df = pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl')
df.head()

class LoadDataset (Dataset):
  def __init__(self,data, train = False):
    self.data = data
    if train :
      self.data = data.sample(frac = 1)

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self,i):
    return self.data.iloc[i]

def collate_data(data):
  ids = []
  tensors = []

  max = 0
  pad = np.zeros(768)

  for i in data:
    ids.append(i[0])

    if i[2].shape[0] > max:
      max = i[2].shape[0]


  for i in data:
    shape = i[2].shape[0]
    arr = i[2]

    for j in range( shape , max):
      arr=np.vstack([arr, pad])

    tensors.append(arr)

  tensors = np.array(tensors)
  tensors = torch.tensor(tensors).float()

  return ids, tensors


train_data = LoadDataset(pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl'), True)
train_loader = DataLoader(train_data, 32, True, collate_fn=collate_data )

val_data = LoadDataset(pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_val.pkl'))
val_loader = DataLoader(val_data, 32, True, collate_fn=collate_data )

params = {}
params['cfg'] = './config/speech/cfg_vrnn.ini'
params['pretrain_dict'] = None
params['model_dir'] = './saved_model/WSJ0_2023-08-01-14h06_VRNN_z_dim=16'
params['reload'] = True

learning = learning_algo_new.LearningAlgorithm(params, train_loader, val_loader)

cp_file = os.path.join(params['model_dir'] , '{}_checkpoint.pt'.format(learning.model_name))
checkpoint = torch.load(cp_file, map_location=torch.device('cpu'))
learning.build_model()

learning.model.load_state_dict(checkpoint['model_state_dict'])

train_data = LoadDataset(pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl'))
train_loader = DataLoader(train_data, 1, True, collate_fn=collate_data )

z_dict = {}
y_dict = {}
h_dict = {}

with torch.no_grad():
  for n, i in enumerate(train_loader):
    print(n)
    batch_data = (i[1].permute(1, 0, 2))
    y_dict[i[0][0]], z_dict[i[0][0]], h_dict[i[0][0]] = learning.model(batch_data, True)

len(df.id.tolist())

df['z'] = df.id
df['h'] = df.id
df['y'] = df.id

df.z = df.z.apply(lambda x : z_dict[x] )
df.h = df.h.apply(lambda x : h_dict[x] )
df.y = df.y.apply(lambda x : y_dict[x] )

df.head()

df.iloc[:, 2]

df.head()

t = df.iloc[:, 5].tolist()
arr = t[0]

for i in range(1, len(t)):
  #print(i)
  arr = np.vstack((arr, t[i]))

kmeans = KMeans(n_clusters=20, random_state=0, n_init="auto").fit(arr)

kmeans.labels_

l = df.embeds.tolist()

l = [i.shape[0] for i in l ]

to_df = []
start = 0
for i in l:
  to_df.append(kmeans.labels_[start : start+ i])
  start += i

df['y_clusters'] = to_df



"""# Cluster"""

counter = Counter(merge_list)
len(Counter({k: c for k, c in counter.items() if c >= 2}))

#df.to_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl')

df = pd.read_pickle('/content/drive/MyDrive/JSALT/DynamicalVAE/data/multiwoz_sentence_embeds_train.pkl')
df.head()

utts = df.utterances.tolist()
x_clusters = df.x_clusters.tolist()
y_clusters = df.y_clusters.tolist()

utts = [j for sub in utts for j in sub]
x_clusters = [j for sub in x_clusters for j in sub]
y_clusters = [j for sub in y_clusters for j in sub]

x = {}
y = {}

for i in range(20):
  x[str(i)] = []
  y[str(i)] = []

for n, i in enumerate(x_clusters):
  x[str(i)].append(utts[n])

for n, i in enumerate(y_clusters):
  y[str(i)].append(utts[n])

x_to_y_link = {}

for i in x:
  max = 0
  x_to_y_link[i] = None
  for j in y:
    merge_list = []
    merge_list.extend(list(set(x[i])))
    merge_list.extend(list(set(y[j])))

    counter = Counter(merge_list)
    curr = len(Counter({k: c for k, c in counter.items() if c >= 2}))

    if curr > max :
      max = curr
      x_to_y_link[i] = j

x_to_y_link

def compute_transitions(domain_convs: List[list], cluster_ixs) -> np.ndarray:
    """Compute the transitions from one cluster to the other, based on
    the actual conversation/dialogue flow. Also compute the assigment of sentence embeddings to
    the cluster indices - mainly start of dialogue and end of dialogue sentences"""

    print('Computing transition matrix ..')
    n_clusters = np.unique(cluster_ixs).size
    print(' n_clusters', n_clusters)

    soc_clusters = []  # start of conversation clusters
    eoc_clusters = []  # end of conversation clusters
    # transitions: row(from) - col(to)
    transitions = np.zeros(shape=(n_clusters, n_clusters))

    i = 0
    k = 0
    while i < len(domain_convs): # iterate over conversations/dialogs, where each conv is a seq of utts
        j = 0
        prev_cix = -1
        while j < len(domain_convs[i]):  # iterate over utts in the current dialogue

            cix = cluster_ixs[k]

            if j == 0:  # start of conversation
                soc_clusters.append(cix)
            else:
                # from, to
                transitions[prev_cix, cix] += 1

            prev_cix = deepcopy(cix)

            k += 1
            j += 1

        # end of conversation
        eoc_clusters.append(prev_cix)

        i += 1

    return transitions, np.asarray(soc_clusters), np.asarray(eoc_clusters)

domain_convs = [list(i) for i in df.y_clusters.tolist()] #change
idsx = [j for sub in domain_convs for j in sub]

trans, soc, eoc = compute_transitions (domain_convs,idsx)

def get_cluster_assignments(occurrences):
    """Get percentage of embeddings assigned to each cluster"""

    ixs, count = np.unique(occurrences, return_counts=True)
    max_ix = np.argmax(count)
    print(
        "Cluster ix:",
        np.array2string(ixs, precision=0, formatter={'int_kind': lambda x: "%4d" % x}, separator=' |')
    )
    print(
        "Percentage:",
        np.array2string(count * 100/count.sum(), precision=1,
                        formatter={'float_kind': lambda x: "%4.1f" % x},
                        separator=' |')
    )
    print("{:.1f} % from cluster index {:d} is the most dense.".format(count[max_ix]*100./count.sum(), ixs[max_ix]))
    return count/count.sum(), ixs

soc_prob, soc_ixs = get_cluster_assignments(soc)


print("\nEnd of conversation..")
# get the clusters where the conversation usually END
eoc_prob, eoc_ixs = get_cluster_assignments(eoc)

for i in range(trans.shape[0]):
    trans[i, :] /= trans[i, :].sum()

def visualize_graph(trans, soc_prob, soc_ixs, eoc_prob, eoc_ixs, s_thresh=0.1, thresh=0.1):
    # low threshold will show many arcs, higher threshold will only show dominant arcs
    # s_thresh - threshold for drawing begin, end transitions
    # thresh -  threshold for drawing an intermediate arc

    # see https://graphviz.readthedocs.io/en/stable/  for more documentation on graphviz

    dot = graphviz.Digraph(f"MutliWoz_", format='png', graph_attr={'rankdir':'LR'})  # initialize a graph

    n_clusters = trans.shape[0]

    # add BEGIN and END nodes
    dot.node('BEGIN', shape='doublecircle')
    dot.node('END', shape='doublecircle')

    # Add a node representing each cluster
    for i in range(n_clusters):
        dot.node(str(i), shape='circle')

    # draw arrows from BEGIN to that cluster(s) where the start sentences live
    # given that they are above the `s_thresh`
    for i, prob in enumerate(soc_prob):
        if prob > s_thresh:
            dot.edge('BEGIN', str(soc_ixs[i]), label="{:.2f}".format(prob))
            dot.node(str(soc_ixs[i]), fillcolor='cyan', style='filled')

    # draw arrows from cluster(s) where the end sentences live to the END node
    for i, prob in enumerate(eoc_prob):
        if prob > s_thresh:
            dot.edge(str(eoc_ixs[i]), 'END', label="{:.2f}".format(prob))
            dot.node(str(eoc_ixs[i]), fillcolor='pink', style='filled')

    # draw intermediate arcs among the clusters where transitions > thresh

    for row_ix in range(trans.shape[0]):  # row_ix represent FROM, col_ixs represent TO
        row_trans = trans[row_ix, :] / trans[row_ix, :].sum()
        thresh_ixs = np.where(row_trans > thresh)
        if thresh_ixs[0].size > 0:
            # print(row_ix, '->', thresh_ixs)
            for col_ix in thresh_ixs[0]:
                dot.edge(str(row_ix), str(col_ix), label="{:.2f}".format(row_trans[col_ix]))

    # show the image within the notebook
    # dot.view() will open the image in an external window
    return dot

dot = visualize_graph(trans, soc_prob, soc_ixs, eoc_prob, eoc_ixs, s_thresh=0.055, thresh=0.055)
dot.view('images/y_cluster_graph') #change

dot

clu = df.x_clusters.tolist()
clu = [list(i) for i in clu]

utts = df.utterances.tolist()

clu = [j for sub in clu for j in sub]
utts = [j for sub in utts for j in sub]

comb = list(zip(utts,clu))
random.shuffle(comb)
random.shuffle(comb)
random.shuffle(comb)
random.shuffle(comb)

for j in range(20):
  count = 0
  print('Cluster {} samples :'.format(j))
  for i in comb :
    if i[1] == j :
      print(i)
      count+=1
    if count == 10:
      print()
      break

clu = df.y_clusters.tolist()
clu = [list(i) for i in clu]

utts = df.utterances.tolist()

clu = [j for sub in clu for j in sub]
utts = [j for sub in utts for j in sub]

comb = list(zip(utts,clu))
random.shuffle(comb)
random.shuffle(comb)
random.shuffle(comb)
random.shuffle(comb)

for j in range(20):
  count = 0
  print('Cluster {} samples :'.format(j))
  for i in comb :
    if i[1] == j :
      print(i)
      count+=1
    if count == 10:
      print()
      break

#df_new = pd.DataFrame(data = comb, columns = ['utt','clust'])

# words = []

# for i in range(20):
#   utt_list = df_new.loc[df_new.clust == i].utt.tolist()
#   print('Cluster num {} instances : {}.'.format(i,len(utt_list)))
#   to_count = ' '.join(utt_list).lower().split()
#   filtered_words = list(filter(lambda word: word not in set(stopwords.words('english')), to_count))

#   words.append(Counter(filtered_words))
#   print(words[-1])
#   print()

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

df.head()

X = df.y.tolist()  #change

arr = X[0]

for i in range(1,len(X)):
  arr = np.vstack((arr,X[i]))

arr.shape

y = df.y_clusters.tolist()   #change
y = [list(i) for i in y]
y = [j for sub in y for j in sub]

fig = plt.figure(1, figsize=(80, 60))
ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

X_reduced = PCA(n_components=3).fit_transform(arr)
ax.scatter(
    X_reduced[:, 0],
    X_reduced[:, 1],
    X_reduced[:, 2],
    c=y,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40
)

ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.zaxis.set_ticklabels([])
plt.legend('Legend')


plt.show()

fig.savefig('images/y_cluster.png', dpi=fig.dpi) #change