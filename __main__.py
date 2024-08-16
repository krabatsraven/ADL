import torch

if __name__ == "__main__":
    print("hello world")

    x = torch.rand(5, 3)
    print(x)

    print(f"Shape of tensor: {x.shape}")
    print(f"Datatype of tensor: {x.dtype}")
    print(f"Device tensor is stored on: {x.device}")
