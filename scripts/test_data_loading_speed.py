import time

from torch.utils.data import DataLoader

from training.data import OHLCDatasetMmap


def test_validation_dataloading_speed():
    val_dataset = OHLCDatasetMmap(
        "memmap_dataset",
        window_range=(1600, 4096),
        is_train=False,
        first_n=10_00,
        sample_n=10000,
        rank=0,
        world_size=8,
    )
    len(val_dataset)
    val_loader = DataLoader(
        val_dataset, batch_size=16, num_workers=8, collate_fn=val_dataset.collate_fn
    )
    min_index = 1e12
    max_index = 0
    print("rank 0", "len val_loader ", len(val_loader))
    start = time.time()
    for i, data in enumerate(val_loader):
        max_index = max(*data["index"], max_index)
        min_index = min(min_index, *data["index"])
        diff = time.time() - start
        throughput = i / diff
        if i > 100000:
            break
    print(min_index, max_index)
    print("throughput {} iter/s".format(throughput))


def test_train_dataloading_speed():
    val_dataset = OHLCDatasetMmap(
        "memmap_dataset",
        window_range=(1024, 2048),
        is_train=True,
        rank=0,
        world_size=8,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, num_workers=4, collate_fn=val_dataset.collate_fn
    )
    min_index = 1e12
    max_index = 0
    print("rank 0", "len train_loader ", len(val_loader))
    start = time.time()
    for i, data in enumerate(val_loader):
        max_index = max(*data["index"], max_index)
        min_index = min(min_index, *data["index"])
        diff = time.time() - start
        throughput = i / diff
        if i > 10:
            break
    print(min_index, max_index)
    print("throughput {} iter/s".format(throughput))


if __name__ == "__main__":
    # test_validation_dataloading_speed()
    test_train_dataloading_speed()
