import os
import torch
import numpy as np

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        super(SampleDataset, self).__init__()
        self.sample_list = sample
      
    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        return sample

    def update(self, sample):
        self.sample_list = torch.cat([self.sample_list, sample], dim=0)

    def deque(self, length):
        self.sample_list = self.sample_list[length:]

    def get_seq(self):
        return self.sample_list

    def __len__(self):
        return len(self.sample_list)

    def collate(data_list):
        return torch.stack(data_list)

class RewardDataset(torch.utils.data.Dataset):
    def __init__(self, rewards):
        super(RewardDataset, self).__init__()
        self.rewards = rewards
        self.raw_tsrs = self.rewards

    def __getitem__(self, idx):
        return self.rewards[idx]

    def update(self, rewards):
        new_rewards = rewards

        self.raw_tsrs = torch.cat([self.rewards, new_rewards], dim=0)
        self.rewards = self.raw_tsrs

    def deque(self, length):
        self.raw_tsrs = self.raw_tsrs[length:]
        self.rewards = self.raw_tsrs

    def get_tsrs(self):
        return self.rewards

    def __len__(self):
        return self.rewards.size(0)

    def collate(data_list):
        return torch.stack(data_list)

class ZipDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return len(self.datasets[0])

    def __getitem__(self, idx):
        return [dataset[idx] for dataset in self.datasets]

    def collate(self, data_list):
        return [dataset.collate(data_list) for dataset, data_list in zip(self.datasets, zip(*data_list))]


def collate(data_list):
    sample,rewards  = zip(*data_list)

    sample_data = SampleDataset.collate(sample)
    reward_data = RewardDataset.collate(rewards)

    return sample_data, reward_data



class ReplayBuffer():
    def __init__(self, buffer_size, device, log_reward, batch_size, data_ndim=2, beta=1.0, rank_weight=1e-2, prioritized=None):
        self.buffer_size = buffer_size
        self.prioritized = prioritized
        self.device = device
        self.data_ndim = data_ndim
        self.batch_size = batch_size
        self.reward_dataset = None
        self.buffer_idx = 0
        self.buffer_full = False
        self.log_reward = log_reward
        self.beta = beta
        self.rank_weight = rank_weight
        # self.sampled_rewards = []

    def load_data(self, data_dir):
        samples = np.load(os.path.join(data_dir, 'positions.npy')).reshape(-1, self.data_ndim)
        rewards = np.load(os.path.join(data_dir, 'rewards.npy'))
        samples = torch.tensor(samples)
        rewards = torch.tensor(rewards)
        self.add(samples, rewards)
        log_Z_est = torch.logsumexp(rewards, dim=0) - torch.log(torch.tensor(len(rewards)))
        return log_Z_est
    
    def add(self, samples, log_r):
        if self.reward_dataset is None:
            self.reward_dataset = RewardDataset(log_r.detach())
            self.sample_dataset = SampleDataset(samples.detach())
        else:
            self.sample_dataset.update(samples.detach())
            self.reward_dataset.update(log_r.detach())

        # Keep the buffer size in check
        
        if self.reward_dataset.__len__() > self.buffer_size:
        
            self.reward_dataset.deque(self.reward_dataset.__len__() - self.buffer_size)
            self.sample_dataset.deque(self.sample_dataset.__len__() - self.buffer_size)
            
        if self.prioritized == 'rank':
            self.scores_np = self.reward_dataset.get_tsrs().detach().cpu().view(-1).numpy()
            ranks = np.argsort(np.argsort(-1 * self.scores_np))
            weights = 1.0 / (1e-2 * len(self.scores_np) + ranks)
            self.dataset = ZipDataset(self.sample_dataset,self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(self.scores_np), replacement=True
                    )

            self.loader = torch.utils.data.DataLoader(
                self.dataset, 
                sampler=self.sampler, 
                batch_size=self.batch_size, 
                collate_fn=collate,
                drop_last=True
                )
        else:
            weights = 1.0
            self.dataset = ZipDataset(self.sample_dataset,self.reward_dataset)
            self.sampler = torch.utils.data.WeightedRandomSampler(
                    weights=weights, num_samples=len(self.scores_np), replacement=True
                    )

            self.loader = torch.utils.data.DataLoader(
                self.dataset, 
                sampler=self.sampler, 
                batch_size=self.batch_size, 
                collate_fn=collate,
                drop_last=True
                )
            
        # check if we have any additional samples before updating the buffer and the scorer!


    def sample(self):
        try:
            sample, reward = next(self.data_iter)
        except:
            self.data_iter = iter(self.loader)
            sample, reward = next(self.data_iter)
            
        # current_reward_sum = reward.sum().item()
        # self.sampled_rewards.append(current_reward_sum)
            
        return sample.detach(), reward.detach()
    

    def sample_pos(self, n):
        buffer_size = len(self.sample_dataset)
        if n > buffer_size:
            raise ValueError(f"buffer size is smaller than sample size")
        idx = torch.randperm(buffer_size, device=self.device)[:n]
        sample_list = [self.sample_dataset[i] for i in idx]
        samples = SampleDataset.collate(sample_list)
        return samples.detach()