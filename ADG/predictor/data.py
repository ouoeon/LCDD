import torch
from dgl.data import DGLDataset
from dgl.data.utils import load_graphs

import numpy as np

class ScoreDataset(DGLDataset):
    def __init__(self, paths, train=True):
        super(ScoreDataset, self).__init__(name='ScoreDataset')

        self.paths = paths

    def __getitem__(self, idx):
        path = self.paths[idx]

        graph, labels = load_graphs( path )
        g = graph[0]
        morgan = labels['morgan']
        avalon = labels['avalon']
        erg = labels['erg']
        score = labels['score']

        return g, morgan, avalon, erg, score

    def __len__(self):
        return len(self.paths)