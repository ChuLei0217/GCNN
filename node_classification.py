import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid
import torch_geometric.nn as pyg_nn

#load dataset
def get_data(folder="node_calssification/cora",data_name="cora"):
    dataset=Planetoid(root=folder,name=data_name)
    return dataset

#create the graph cnn model
class GraphCNN(nn.Module):
    def __init__(self,in_c,hid_c,out_c):
        super(GraphCNN,self).__init__()
        self.conv1=pyg_nn.GCNConv(in_channels=in_c,out_channels=hid_c)
        self.conv2=pyg_nn.GCNConv(in_channels=hid_c,out_channels=out_c)

    def forward(self,data):
        x=data.x  #[N,C]
        edge_index=data.edge_index  #[2,E]
        hid=self.conv1(x=x,edge_index=edge_index)  #[N,D]
        hid=F.relu(hid)

        out=self.conv2(x=hid,edge_index=edge_index)   #[N,out_c]

        out=F.log_softmax(out)

        return out

def main():
    core_dataset=get_data()
    my_net=GraphCNN(in_c=core_dataset.num_node_features,hid_c=12,out_c=core_dataset.num_classes)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net=my_net.to(device)
    data = core_dataset[0].to(device)

    optimizer=torch.optim.Adam(my_net.parameters(),lr=1e-3)

    # model train
    my_net.train()
    for epoch in range(200):
        output=my_net(data)
        loss=F.nll_loss(output[data.train_mask],data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        print("Epoch", epoch+1, "Loss", loss.item())


    # model test
    my_net.eval()
    _, prediction=my_net(data).max(dim=1)

    target=data.y

    test_correct=prediction[data.test_mask].eq(target[data.test_mask]).sum().item()
    test_number=data.test_mask.sum().item()

    print("Accuracy of Test Samples",test_correct/test_number)


if __name__=='__main__':
    main()