import torch
import numpy as np
import matplotlib.pyplot as plt
from plot_utils import viz_many_well

def check_mc_buffer(energy, mc_samples, mc_rewards, name):
    
    def draw_energy_histogram(
        ax, log_reward, bins=40, range=(90, 160)
    ):
        log_reward = torch.clamp(log_reward, min=range[0], max=range[1])

        hist, bins = np.histogram(
            log_reward.detach().cpu().numpy(), bins=bins, range=range, density=True
        )

        ax.set_xlabel("log reward")
        ax.set_ylabel("count")
        ax.grid(True)

        return ax.plot(bins[1:], hist, linewidth=3)
    
    def sample_figure(energy, samples, name):
    
        vizualizations = viz_many_well(energy, samples)
        fig_samples_x13, ax_samples_x13, fig_kde_x13, ax_kde_x13, fig_contour_x13, ax_contour_x13, fig_samples_x23, ax_samples_x23, fig_kde_x23, ax_kde_x23, fig_contour_x23, ax_contour_x23 = vizualizations

        fig_contour_x13.savefig(f'{name}_contourx13.pdf', bbox_inches='tight')
        fig_contour_x23.savefig(f'{name}_contourx23.pdf', bbox_inches='tight')
        
        
    sample_figure(energy, mc_samples, name)
    
    fig, ax = plt.subplots()
    draw_energy_histogram(ax, mc_rewards,range=(100, 900))
    draw_energy_histogram(ax, energy.log_reward(energy.sample(batch_size=mc_samples.shape[0])), range=(100, 900))
    fig.savefig(f'{name}_energy_histogram.pdf', bbox_inches='tight')