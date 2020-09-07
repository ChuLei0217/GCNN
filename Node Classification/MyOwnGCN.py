import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T


# load dataset
def get_data(folder="node_classify/cora", data_name="cora"):
    # dataset = Planetoid(root=folder, name=data_name)
    dataset = Planetoid(root=folder, name=data_name,
                        transform=T.NormalizeFeatures())
    return dataset


# create the graph cnn model
class GraphCNN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GraphCNN, self).__init__()
        self.conv1 = pyg_nn.GCNConv(in_channels=in_c, out_channels=hid_c)
        self.conv2 = pyg_nn.GCNConv(in_channels=hid_c, out_channels=out_c)

    def forward(self, data):
        # data.x data.edge_index
        x = data.x  # [N, C]
        edge_index = data.edge_index  # [2 ,E]

        hid = self.conv1(x=x, edge_index=edge_index)  # [N, D]
        hid = F.relu(hid)

        out = self.conv2(x=hid, edge_index=edge_index)  # [N, out_c]

        out = F.log_softmax(out, dim=1)  # [N, out_c]

        return out


class OwnGCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(OwnGCN, self).__init__()
        self.in_ = pyg_nn.SGConv(in_c, hid_c, K=2)

        self.conv1 = pyg_nn.APPNP(K=2, alpha=0.1)
        self.conv2 = pyg_nn.APPNP(K=2, alpha=0.1)

        self.out_ = pyg_nn.SGConv(hid_c, out_c, K=2)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.in_(x, edge_index)
        x = F.dropout(x, p=0.1, training=self.training)

        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)

        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.1, training=self.training)

        x = self.out_(x, edge_index)

        return F.log_softmax(x, dim=1)


def analysis_data(dataset):
    print("Basic Info:      ", dataset[0])
    print("# Nodes:         ", dataset[0].num_nodes)
    print("# Features:      ", dataset[0].num_features)
    print("# Edges:         ", dataset[0].num_edges)
    print("# Classes:       ", dataset.num_classes)
    print("# Train samples: ", dataset[0].train_mask.sum().item())
    print("# Valid samples: ", dataset[0].val_mask.sum().item())
    print("# Test samples:  ", dataset[0].test_mask.sum().item())
    print("Undirected:      ", dataset[0].is_undirected())


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cora_dataset = get_data()

    # todo list
    my_net = OwnGCN(in_c=cora_dataset.num_features, hid_c=300, out_c=cora_dataset.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)
    data = cora_dataset[0].to(device)

    optimizer = torch.optim.Adam(my_net.parameters(), lr=1e-2, weight_decay=1e-3)
    print(my_net)

    # model train
    my_net.train()
    for epoch in range(500):
        optimizer.zero_grad()

        output = my_net(data)
        loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        _, prediction = output.max(dim=1)

        valid_correct = prediction[data.val_mask].eq(data.y[data.val_mask]).sum().item()
        valid_number = data.val_mask.sum().item()

        valid_acc = valid_correct / valid_number
        print("Epoch: {:03d}".format(epoch + 1), "Loss: {:.04f}".format(loss.item()),
              "Valid Accuracy:: {:.4f}".format(valid_acc))

    # model test
    my_net.eval()
    print(my_net)

    _, prediction = my_net(data).max(dim=1)

    target = data.y

    test_correct = prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number = data.test_mask.sum().item()

    train_correct = prediction[data.train_mask].eq(target[data.train_mask]).sum().item()
    train_number = data.train_mask.sum().item()

    print("==" * 20)

    print("Accuracy of Train Samples: {:.04f}".format(train_correct / train_number))

    print("Accuracy of Test  Samples: {:.04f}".format(test_correct / test_number))




if __name__ == '__main__':
    dataset = get_data()
    analysis_data(dataset)
    main()


