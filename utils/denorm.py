import torch


# 用于对图片的反归一化
def denorm(batch, device, mean=[0.1307], std=[0.3081]):
    """
    Convert a batch of tensors to their original scale.

    Args:
        batch (torch.Tensor): Batch of normalized tensors.
        device:
        mean (torch.Tensor or list): Mean used for normalization.
        std (torch.Tensor or list): Standard deviation used for normalization.

    Returns:
        torch.Tensor: batch of tensors without normalization applied to them.
    """
    if isinstance(mean, list):
        mean = torch.tensor(mean, requires_grad=True).to(device)
    if isinstance(std, list):
        std = torch.tensor(std, requires_grad=True).to(device)

    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
