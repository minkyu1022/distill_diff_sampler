import torch
import numpy as np

def save_eval(name, sample_dict, energy_dict, dist_dict, logging_dict, epochs):
    for k, v in sample_dict.items():
        if logging_dict['epoch'] in epochs and k in ['Teacher', 'GT']:
            continue
        np.save(f'{name}/sample/{k}_{logging_dict["epoch"]}.npy', v.cpu().numpy())
    for k, v in energy_dict.items():
        if logging_dict['epoch'] in epochs and k in ['Teacher', 'GT']:
            continue
        np.save(f'{name}/energy/{k}_{logging_dict["epoch"]}.npy', v)
    for k, v in dist_dict.items():
        if logging_dict['epoch'] in epochs and k in ['Teacher', 'GT']:
            continue
        np.save(f'{name}/dist/{k}_{logging_dict["epoch"]}.npy', v)

def save_checkpoint(name, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer, metrics, logging_dict):
    logging_dict['gfn_losses'].append(metrics['train/gfn_loss'])
    logging_dict['rnd_losses'].append(metrics['train/rnd_loss'])
    logging_dict['energy_call_counts'].append(metrics['train/energy_call_count'])
    
    logging_dict['elbos'].append(metrics['eval/ELBO'].item())
    logging_dict['eubos'].append(metrics['eval/EUBO'].item())
    logging_dict['log_Z_IS'].append(metrics['eval/log_Z_IS'].item())
    logging_dict['mlls'].append(metrics['eval/mean_log_likelihood'].item())
    logging_dict['log_Z_learned'].append(metrics['eval/log_Z_learned'].item())
    for k, v in logging_dict.items():
        np.save(f'{name}/{k}.npy', np.array(v))
    torch.save({
        'epoch': logging_dict['epoch'],
        'gfn_model': gfn_model.state_dict(),
        'rnd_model': rnd_model.state_dict(),
        'gfn_optimizer': gfn_optimizer.state_dict(),
        'rnd_optimizer': rnd_optimizer.state_dict(),
        'logging_dict': logging_dict
    }, f'{name}/ckpt/{logging_dict["epoch"]}.pth')
    
    
def load_checkpoint(path, gfn_model, rnd_model, gfn_optimizer, rnd_optimizer):
    checkpoint = torch.load(path)
    gfn_model.load_state_dict(checkpoint['gfn_model'])
    rnd_model.load_state_dict(checkpoint['rnd_model'])
    gfn_optimizer.load_state_dict(checkpoint['gfn_optimizer'])
    rnd_optimizer.load_state_dict(checkpoint['rnd_optimizer'])
    return checkpoint['logging_dict']
