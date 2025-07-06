#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import networkx as nx 
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch_geometric
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphSAGE
from torch_geometric.datasets import EllipticBitcoinDataset


# In[3]:


dataset = EllipticBitcoinDataset(root='data')
data = dataset[0]
# transactions = pd.read_csv("elliptic_txs_classes.csv")
# edges = pd.read_csv("elliptic_txs_edgelist.csv")
# print(transactions)
# print(edges)




# In[4]:


elliptic = EllipticBitcoinDataset(root='data')._data
print(elliptic)
# fraud_dict = dict(zip(transactions['txId'], transactions['class']))


# In[5]:


out_nodes = elliptic.edge_index[0]
print('edges move out of the following nodes:\n', out_nodes)

in_nodes = elliptic.edge_index[1]
print('\nedges move into the following nodes:\n', in_nodes)
# G = nx.from_pandas_edgelist(edges, 'txId1', 'txId2')


# In[6]:


# dense_matrix = torch_geometric.utils.to_dense_adj(elliptic.edge_index)
# print(dense_matrix)

## node_colors = []
## for node in G.nodes():
##     if node in fraud_dict:
##         if fraud_dict[node] == 2:
##             node_colors.append('red')
##         elif fraud_dict[node] == 1:
##             node_colors.append('blue')
##     else:
##         node_colors.append('gray')


# In[71]:


# print("Truth values: ", elliptic.y)
# elliptic_to_x = torch_geometric.utils.to_networkx(elliptic, to_undirected=True)
# plt.figure(figsize=(10, 10))
# nx.draw(elliptic_to_x, with_labels=True, node_color=elliptic.y)



# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# from torch_geometric.data import Data
# import numpy as np

# # Load data SMARTLY
# features = pd.read_csv("elliptic_txs_features.csv", header=None)
# classes = pd.read_csv("elliptic_txs_classes.csv")
# edgelist = pd.read_csv("elliptic_txs_edgelist.csv")

# # Preprocess (keep only labeled nodes)
# labeled_ids = classes[classes['class'] != 'unknown']['txId'].values
# features_labeled = features[features[0].isin(labeled_ids)]
# class_labeled = classes[classes['class'] != 'unknown']

# # Convert to PyG Data format
# x = torch.tensor(features_labeled.iloc[:, 1:].values, dtype=torch.float)
# y = torch.tensor(class_labeled['class'].map({'1':0, '2':1}).values, dtype=torch.long)

# # Create edge indices
# edge_index = torch.tensor(edgelist.values.T, dtype=torch.long)

# # Build graph data object
# data = Data(x=x, edge_index=edge_index, y=y)

# # Visualize clusters with t-SNE (lightning fast)
# tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
# embeddings = tsne.fit_transform(data.x.numpy())

# # Plot
# plt.figure(figsize=(10, 8))
# plt.scatter(
#     embeddings[:, 0], embeddings[:, 1],
#     c=data.y.numpy(), 
#     cmap='coolwarm', 
#     alpha=0.6,
#     s=10
# )
# plt.colorbar(label='Fraud Risk (0=Legit, 1=Fraud)')
# plt.title("t-SNE of Bitcoin Transaction Features")
# plt.savefig("fraud_clusters_tsne.png", dpi=120)


# In[7]:


from torch_geometric.loader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
loader = DataLoader(dataset, batch_size=32, shuffle=False)


# In[8]:


# A = torch_geometric.utils.to_dense_adj(elliptic.edge_index).squeeze()
# A_tilde = A + torch.eye(A.shape())
# sqrt_node_degrees = torch.sqrt(torch.sum(A_tilde, dim=1))
# D_tilde_inv = torch.diag(1/sqrt_node_degrees)

# P = D_tilde_inv @ A_tilde @ D_tilde_inv
# print(P)
# print(P.shape)


# In[9]:


class GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 16)
        self.conv2 = GCNConv(16, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)


        return x


# In[10]:


in_channels = data.num_node_features
out_channels = int(data.y.max().item()) + 1  # handles if labels are not 0-indexed

model = GCN(in_channels, out_channels)


# In[11]:


from sklearn.manifold import TSNE

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap='Set2')
    plt.show()


model = GCN(in_channels, out_channels)
model.eval()
out = model(data)
# visualize(out, color=data.y)


# In[12]:


print(data)


# In[13]:


from IPython.display import Javascript

model = GCN(in_channels, out_channels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss

def test(data):
    model.eval()
    out = model(data)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

for epoch in range(1, 101):
    loss = train(data)
    test_acc = test(data)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Accuracy: {test_acc:.4f}')


# In[14]:


test_acc = test(data)
print(f'Test Accuracy: {test_acc:.4f}')


# In[15]:


model.eval()

out = model(data)
# visualize(out, color=data.y)


# In[16]:


def generate_sar_report(suspicious_transactions):
    """Auto-fills SEC Form SAR with AI findings"""
    from fpdf import FPDF  # pip install fpdf2
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Suspicious Activity Report", ln=1, align='C')
    pdf.multi_cell(0, 10, f"AI detected {len(suspicious_transactions)} high-risk transactions:")
    # Add transaction details...
    pdf.output("SAR_Report.pdf")


# In[ ]:


# train_loader = DataLoader(dataset[:400], batch_size=203769, shuffle=True)
# test_loader = DataLoader(dataset[400:], shuffle=False)


# In[ ]:


# from torch_geometric.nn.pool import global_mean_pool


# In[ ]:


# class PoolingGCN(torch.nn.Module):

#     def __init__(self):
#         super().__init__()

#         self.conv1 = GCNConv(165, 512, catched=False)
#         self.conv2 = GCNConv(512, 2, catched=False)

#     def forward(self, data):
#         x, edge_index = data.x, data.edge_index

#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index)
#         x =  global_mean_pool(x, data.batch)

#         return F.log_softmax(x, dim=1)
    
    


# In[ ]:


# model = PoolingGCN()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# criterion = torch.nn.CrossEntropyLoss()

# def train_step(train_loader):
#     model.train()
#     for batch in train_loader:
#         optimizer.zero_grad()
#         out = model(batch)
#         loss = criterion(out, batch.y)
#         loss.backward()
#         optimizer.step()
#     return loss

# def test_step(test_loader):
#     model.eval()
#     for batch in test_loader:
#         out = model(batch)
#         pred = out.argmax(dim=1)
#         test_correct = pred[batch.test_mask] == batch.y[batch.test_mask]
#         test_acc = int(test_correct.sum()) / int(batch.test_mask.sum())
#     return test_acc


# for epoch in range(1, 101):
#     train_loss = train_step(train_loader)
#     if not epoch%10:
#         print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}')


# In[ ]:




