import torch


THREAD_MAX_WORKERS = 1
RANDOM_SEED = 0
SOFT_BATCH_SIZE = 8

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.double,
}
