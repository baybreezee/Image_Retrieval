import torch

def compute_rank_and_reward(sim_row: torch.Tensor, true_index: int):
    true_sim = sim_row[true_index]
    rank = (sim_row > true_sim).sum().item() + 1
    reward_value = 1.0 / rank
    reward = torch.tensor(reward_value, device=sim_row.device, dtype=torch.float32)
    return rank, reward