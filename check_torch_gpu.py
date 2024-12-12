import torch

cuda_check = torch.cuda.is_available()

if cuda_check:
    n_gpu = torch.cuda.device_count()
    for i in range(n_gpu):
        print(torch.cuda.device(i), torch.cuda.get_device_name(i))
else:
    print(f"No cuda device found")
