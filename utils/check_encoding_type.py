import torch
def check_encoding_type(labels):
    if labels.ndim == 1:
        # 如果是1维张量，可能是整数编码
        unique_values = torch.unique(labels)
        if torch.all(unique_values == torch.arange(len(unique_values))):
            return "Integer Encoding"
    elif labels.ndim == 2:
        # 如果是2维张量，可能是one-hot编码
        row_sums = labels.sum(dim=1)
        unique_values = torch.unique(labels)
        if labels.size(1) > 1 and torch.all(row_sums == 1) and torch.all((unique_values == 0) | (unique_values == 1)):
            return "One-hot Encoding"
    
    return "Unknown Encoding"