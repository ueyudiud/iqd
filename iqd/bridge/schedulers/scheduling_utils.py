import torch


def get_betas(schedule, betas, num_steps, beta_start, beta_end):
    if betas is not None:
        return torch.tensor(betas, dtype=torch.float64)
    elif schedule == 'linear':
        return torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    elif schedule == 'scaled_linear':
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float64) ** 2
    else:
        raise NotImplementedError(f"{schedule} is not implemented.")
