import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import pickle as pkl
import scipy.sparse as sp

device = torch.device('cuda')

def parse_index_file(filename):
	index = []
	for line in open(filename):
		index.append(int(line.strip()))
	return index

def load_data(dataset_str):
	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
			objects.append(pkl.load(f))
	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
	test_idx_range = np.sort(test_idx_reorder)
	if dataset_str == 'citeseer':
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
		tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range-min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range-min(test_idx_range), :] = ty
		ty = ty_extended

	features_s = sp.vstack((allx, tx)).tolil()
	features_s[test_idx_reorder, :] = features_s[test_idx_range, :]
	features_d = np.zeros(shape=features_s.shape, dtype=np.float)
	features_s = features_s.tocoo()
	for i in range(len(features_s.row)):
		features_d[features_s.row[i], features_s.col[i]] = features_s.data[i]

	adj = np.zeros(shape=(len(graph), len(graph)), dtype=np.float)
	for i in range(len(graph)):
		for j in range(len(graph[i])):
			adj[i, graph[i][j]] = 1.0

	labelvec = np.vstack((ally, ty))
	labelvec[test_idx_reorder, :] = labelvec[test_idx_range, :]
	labels = np.zeros(shape=(len(labelvec)), dtype=np.long)
	for i in range(len(labelvec)):
		for j in range(len(labelvec[i])):
			if labelvec[i][j] == 1:
				labels[i] = j
	return features_d, labels, labelvec.sum(0), adj

def normalize_row(x):
	rowsum = x.sum(1)
	r_inv = np.power(rowsum, -1).flatten()
	r_inv[np.isinf(r_inv)] = 0.
	for i in range(len(x)):
		for j in range(len(x[i])):
			x[i,j] = x[i,j] * r_inv[i]
	return x

class GCNLayer(nn.Module):
	def __init__(self, inputDim, outputDim, A, B, C):
		super(GCNLayer, self).__init__()
		self.A = A
		self.B = B
		self.C = C
		self.fc = nn.Linear(inputDim * 3, outputDim)
		self.bn = nn.BatchNorm1d(outputDim)
		self.sm = nn.Softmax(dim = 1)
	def forward(self, x):
		Ax = torch.mm(self.A, x)
		Cx = torch.mm(F.relu(self.C), x)
		BCx = torch.mm(F.relu(self.B), Cx)
		concatX = torch.cat((Ax, x, BCx), 1)
		bnBCx = self.bn(self.fc(concatX))
		return F.relu(bnBCx)

class GNN(nn.Module):
	def __init__(self, N, M, inputDim, hiddenDim, outputDim, numLabel, adj, labelGuess):
		super(GNN, self).__init__()
		print("N = ", N)
		self.A = nn.Parameter(data=torch.Tensor(N, N), requires_grad=False)
		self.B = nn.Parameter(data=torch.Tensor(N, M), requires_grad=True)
		self.C = nn.Parameter(data=torch.Tensor(M, N), requires_grad=True)
		self.fc = nn.Linear(outputDim, numLabel)
		self.sm = nn.Softmax(dim = 1)
		self.A.data = torch.Tensor(adj)
		self.B.data.zero_()
		self.C.data.zero_()
		cnt = np.zeros(shape=(M), dtype=np.long)
		for i in range(N):
			cnt[labelGuess[i]] += 1
		for i in range(N):
			l = labelGuess[i]
			self.B.data[i, l] = 1.0
			self.C.data[l, i] = 1.0 / cnt[l]
		print('selfC.sum(1) = ', self.C.data.sum(1))
		#self.A.data.uniform_(0, 1.0 / N)
		#self.B.data.uniform_(0, 1.0 / N)
		#self.C.data.uniform_(0, 1.0 / N)
		self.gcn1 = GCNLayer(inputDim, hiddenDim, self.A, self.B, self.C)
		self.gcn2 = GCNLayer(hiddenDim, outputDim, self.A, self.B, self.C)
		
	def forward(self, x):
		l1 = self.gcn1(x)
		#print('L1: ', l1)
		l2 = self.gcn2(l1)
		#print('L2: ', l2)
		l3 = self.fc(l2)
		#print('L3: ', l3)
		l4 = self.sm(l3)
		return l4

class GCNBasicLayer(nn.Module):
	def __init__(self, inputDim, outputDim, A):
		super(GCNBasicLayer, self).__init__()
		self.A = A
		self.fc = nn.Linear(inputDim * 2, outputDim)
		self.bn = nn.BatchNorm1d(outputDim)

	def forward(self, x):
		Ax = torch.mm(self.A, x)
		concatAx = torch.cat((Ax, x), dim=1)
		bnAx = self.bn(self.fc(concatAx))
		return F.relu(bnAx)

class GCN(nn.Module):
	def __init__(self, N, inputDim, hiddenDim, outputDim, numLabel, adj):
		super(GCN, self).__init__()
		print("GCN: N = ", N)
		self.A = nn.Parameter(data=torch.Tensor(N, N), requires_grad = False)
		self.fc = nn.Linear(outputDim, numLabel)
		self.sm = nn.Softmax(dim = 1)
		self.A.data = torch.Tensor(adj)
		self.gcn1 = GCNBasicLayer(inputDim, hiddenDim, self.A)
		self.gcn2 = GCNBasicLayer(hiddenDim, outputDim, self.A)

	def forward(self, x):
		l1 = self.gcn1(x)
		l2 = self.gcn2(l1)
		l3 = self.fc(l2)
		l4 = self.sm(l3)
		return l4

# Parse configuration
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Epochs')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
parser.add_argument('--node_dir', type=str, help='dir to node features')
parser.add_argument('--label_dir', type=str, help = 'dir to labels')
parser.add_argument('--edge_dir', type=str, help = 'dir to edges')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim')
parser.add_argument('--output_dim', type=int, default=64, help='output dim')
args = parser.parse_args()

# Load training data
#x = np.loadtxt(args.node_dir, delimiter=',', dtype=np.float)
#label = np.loadtxt(args.label_dir, dtype=np.long)
#edges = np.loadtxt(args.edge_dir, delimiter=',', dtype=np.long)
# shift label and vertex id by 1
#label = np.add(label, -1)
#edges = np.add(edges, -1)
x, label, labelCnt, E = load_data("cora")

# TODO: remove this line
#for i in range(len(E)):
#	E[i, i] = 1.0

E = normalize_row(E)

N = x.shape[0]
inputDim = x.shape[1]
#E = np.zeros(shape=(N, N), dtype=np.float)
#for e in edges:
#	E[e[0], e[1]] = 1

labelGCN = np.loadtxt("kmeans_20_cora.txt", dtype=np.long)
model = GNN(N, 20, inputDim, args.hidden_dim, args.output_dim, len(labelCnt), E, labelGCN).to(device)
#model = GCN(N, inputDim, args.hidden_dim, args.output_dim, len(labelCnt), E).to(device)
optimizer = torch.optim.Adam(model.parameters(), args.lr)

# Start training
print(x.shape)
print(label.shape)
x = torch.tensor(x, dtype=torch.float).to(device)
label = torch.tensor(label, dtype=torch.long).to(device)
E = torch.tensor(E, dtype=torch.float).to(device)

best = 0
oldLoss = 0.0
window = 0
for epoch in range(args.epochs):
	model.train()
	output = model(x)
	split_outputs = torch.split(output, 1357, dim = 0)
	split_labels = torch.split(label, 1357, dim = 0)
	#loss = F.cross_entropy(split_outputs[0], split_labels[0])
	loss = F.nll_loss(split_outputs[0], split_labels[0])
	if loss >= oldLoss:
		windows = windows + 1
		if windows >= 10:
			break
	else:
		windows = 0
	oldLoss = loss
	#D = E - model.A - torch.mm(model.B, model.C)
	#loss += 0.001 * torch.sum(torch.mul(D, D))
	#print(output)
	print(loss)
	_, predicted = torch.max(output.data, 1)
	correct = (predicted == label).sum().item()
	if correct > best:
		best = correct
	print('Correct = ', correct, best)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

#with open('gcn_label.txt', 'w') as f:
#	for i in range(len(predicted)):
#		f.write("%d\n" % predicted[i])
