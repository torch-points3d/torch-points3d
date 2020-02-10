from torch.utils.data import Sampler

class BalancedRandomSampler(Sampler):
    def __init__(self, labels, replacement=False):
        self.idx_classes, self.counts = torch.unique(labels, return_counts=True)

        self.indices = {
            idx: (labels == idx).nonzero() for idx in self.idx_classes
        }

        self.probs = self.counts / self.counts.sum()

        self.np_idx_classes = np.asarray(self.idx_classes)

    def __iter__(self):
        while True:
            idx = torch.multinomial(self.probs, 1, self.replacement)
        indices = np.random.choice(self.indices[idx])
        return iter(indices)

    def __len__(self):
        return self.num_samples