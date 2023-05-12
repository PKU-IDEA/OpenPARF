import torch.utils.data
from .base_data_loader import BaseDataLoader


def CreateDataset(opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map):
    dataset = None
    from .aligned_dataset_from_rudy import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt, horizontal_utilization_map, vertical_utilization_map, pin_density_map)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
