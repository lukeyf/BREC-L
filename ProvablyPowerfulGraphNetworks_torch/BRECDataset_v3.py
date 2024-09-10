import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx, to_networkx
import os
from tqdm import tqdm

torch_geometric.seed_everything(2022)


def graph6_to_pyg(x):
    return from_networkx(nx.from_graph6_bytes(x))


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        name="no_param",
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        line_graph_deg=0,
    ):
        self.root = root
        self.name = name + str(line_graph_deg)
        self.line_graph_deg = line_graph_deg
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])
        

    @property
    def processed_dir(self):
        name = "processed"
        return os.path.join(self.root, self.name, name)

    @property
    def raw_file_names(self):
        return ["brec_v3.npy"]

    @property
    def processed_file_names(self):
        return ["brec_v3_numlabel=0.pt"]

    def process(self):

        data_list = np.load(self.raw_paths[0], allow_pickle=True)
        data = []
        for g in tqdm(data_list):
            g_networkx = nx.from_graph6_bytes(g)
            for _ in range(self.line_graph_deg):
                g_networkx = nx.line_graph(g_networkx)
            edge = np.expand_dims(nx.to_numpy_array(g_networkx), axis=0)
            node = np.expand_dims(np.eye(g_networkx.number_of_nodes()), axis=0)
            data.append(torch.tensor(edge + node, dtype=torch.float32))
            # data.append(
            #     torch.tensor(
            #         np.concatenate((edge, node), axis=0),
            #         dtype=torch.float32,
            #     )
            # )
        self.data = data

        torch.save(self.data, self.processed_paths[0])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data


def main():
    dataset = BRECDataset()
    print(len(dataset))
    print(dataset[0])


if __name__ == "__main__":
    main()
