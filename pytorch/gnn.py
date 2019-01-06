import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

device = torch.device('cuda')

class GCNLayer(nn.Module):
	def __init__(self, inputDim, outputDim, A, B, C):
		super(GCNLayer, self).__init__()
		self.A = A
		self.B = B
		self.C = C
		self.fc = nn.Linear(inputDim, outputDim)
		self.bn = nn.BatchNorm1d(outputDim)
		# self.fc.data.uniform_(-0.05, 0.05)
	def forward(self, x):
		Cx = torch.mm(torch.sigmoid(self.C), x)
		BCx = torch.mm(torch.sigmoid(self.B), Cx)
		bnBCx = self.bn(self.fc(BCx))
		return F.relu(bnBCx)

class GNN(nn.Module):
	def __init__(self, N, M, inputDim, hiddenDim, outputDim, numLabel):
		super(GNN, self).__init__()
		print("N = ", N)
		self.B = nn.Parameter(data=torch.Tensor(N, M), requires_grad=True)
		self.C = nn.Parameter(data=torch.Tensor(M, N), requires_grad=True)
		self.A = nn.Parameter(data=torch.Tensor(N, N), requires_grad=True)
		self.fc = nn.Linear(outputDim, numLabel)
		self.sm = nn.Softmax(dim = 1)
		self.B.data.uniform_(0, 1.0 / N)
		self.C.data.uniform_(0, 1.0 / N)
		self.A.data.uniform_(0, 1.0 / N)
		# self.fc.data.uniform_(-0.05, 0.05)
		self.gcn1 = GCNLayer(inputDim, hiddenDim, self.A, self.B, self.C)
		self.gcn2 = GCNLayer(hiddenDim, outputDim, self.A, self.B, self.C)
		
	def forward(self, x):
		l1 = self.gcn1(x)
		print('L1: ', l1)
		l2 = self.gcn2(l1)
		print('L2: ', l2)
		l3 = self.fc(l2)
		print('L3: ', l3)
		l4 = self.sm(l3)
		return l4

# Parse configuration
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--lr', type=float, default=0.000001, help='Learning rate')
parser.add_argument('--node_dir', type=str, help='dir to node features')
parser.add_argument('--label_dir', type=str, help = 'dir to labels')
parser.add_argument('--edge_dir', type=str, help = 'dir to edges')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim')
parser.add_argument('--output_dim', type=int, default=64, help='output dim')
args = parser.parse_args()

# Load training data
x = np.loadtxt(args.node_dir, delimiter=',', dtype=np.float)
label = np.loadtxt(args.label_dir, dtype=np.long)
edges = np.loadtxt(args.edge_dir, delimiter=',', dtype=np.long)
# shift label by 1
label = np.add(label, -1)
# shift vertex by 1
edges = np.add(edges, -1)
N = x.shape[0]
inputDim = x.shape[1]
E = np.zeros(shape=(N, N), dtype=np.float)
for e in edges:
	E[e[0], e[1]] = 1

model = GNN(N, N, inputDim, args.hidden_dim, args.output_dim, 55).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)

# Start training
print(x.shape)
print(label.shape)
x = torch.tensor(x, dtype=torch.float).to(device)
label = torch.tensor(label, dtype=torch.long).to(device)
E = torch.tensor(E, dtype=torch.float).to(device)

for epoch in range(args.epochs):
	model.train()
	output = model(x)
	loss = F.cross_entropy(output, label)
	D = E - torch.mm(model.B, model.C)
	loss = loss + 0.00001 * torch.sum(torch.mul(D, D))
	#loss = torch.sum(torch.mul(D, D))
	print(output)
	print(loss)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
