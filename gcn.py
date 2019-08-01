import os.path as osp
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
#imports
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as F
from tqdm import tqdm_notebook as tqdm

torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


dataset = 'CiteSeer' 
path = osp.join('..', 'data', dataset)
dataset = Planetoid(path, dataset, T.NormalizeFeatures())
data = dataset[0]
class Net(torch.nn.Module):
    def __init__(self,in_features,num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(in_features, 16, cached=True)
        self.conv2 = GCNConv(16, num_classes, cached=True)

    def forward(self,x,edge_index):
            # get the graph data
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features,dataset.num_classes).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
def train():
    
    model.train()
    
    
    optimizer.zero_grad()
    

    prediction = model(data.x, data.edge_index)
    
   
    loss = F.nll_loss(prediction[data.train_mask], data.y[data.train_mask])
 
    loss.backward()
   
    optimizer.step()


def test():
   
    model.eval()
    
   
    logits, accs = model(data.x, data.edge_index), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
    return accs
trends = {'train_accuracy':[],
          'test_accuracy':[],
          'validation_accuracy':[],
         }


best_val_acc = test_acc = 0
for epoch in tqdm(range(1, 201)):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
    
    trends['test_accuracy'].append(tmp_test_acc)
    trends['train_accuracy'].append(train_acc)
    trends['validation_accuracy'].append(val_acc)
    

plt.figure()
plt.plot(trends['train_accuracy'],color='r',label='train accuracy')
plt.plot(trends['test_accuracy'],color='g',label='test accuracy')
plt.plot(trends['validation_accuracy'],color='b',label='validation accuracy')
plt.legend()
plt.show()