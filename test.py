import torch
import gc
import torch.nn.functional as F
import model

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

input1 = torch.randn(8, 3, 256, 256).to(device)
mean = input1.mean(dim=0, keepdim=True)

test_model = model.Eapp1().to(device)
output = test_model(input1)

