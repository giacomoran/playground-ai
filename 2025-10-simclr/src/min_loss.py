import torch

def min_simclr_loss(batch_size: int, tau: float) -> torch.Tensor:
    """
    Compute the theoretical minimum SimCLR loss for a given batch size and temperature.

    Args:
        batch_size: number of original samples (before augmentation)
        tau: temperature parameter

    Returns:
        Tensor containing the minimum possible loss value
    """
    num_pos = 1  # one positive pair per sample
    num_neg = 2 * batch_size - 2  # total negatives per pair
    numerator = torch.exp(torch.tensor(1.0 / tau))
    denominator = numerator + num_neg * torch.exp(torch.tensor(0.0))
    prob = numerator / denominator
    loss = -torch.log(prob)
    return loss

# Example
for N in [16, 64, 256, 1024]:
    print(f"N={N}, min loss ≈ {min_simclr_loss(N, tau=0.5).item():.5f}")
