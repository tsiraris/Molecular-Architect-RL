import torch
import torch.optim as optim
import wandb
import numpy as np
import io
import imageio

from gnn_agent import MoleculeAgent
from chem_env import MoleculeEnvironment
from ppo import PPOBuffer 
from torch.distributions import Categorical
from rdkit import RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog('rdApp.*')

CONFIG = {
    "lr": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.05,
    "value_coef": 0.5,
    "batch_size": 64,
    "ppo_epochs": 10,
    "buffer_size": 2048,
    "hidden_dim": 64,
    "max_steps": 30,
    # SOTA Updates:
    "input_feats": 9,  # C/N/O + Hybridization + Arom + Ring + Focus
    "num_actions": 5   # C/N/O + Stop + Shift
}

def create_gif(mol_list, reward):
    images = []
    for mol in mol_list:
        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            images.append(np.array(img))
        except:
            pass
    if len(images) == 0: return None
    with io.BytesIO() as gif_buffer:
        imageio.mimsave(gif_buffer, images, format='GIF', duration=0.5, loop=0)
        return wandb.Video(io.BytesIO(gif_buffer.getvalue()), format='gif', caption=f"Reward: {reward:.2f}")

def train():
    wandb.init(project="drug-design-rl", config=CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 SOTA PPO Training on {device} (with AMP)")

    env = MoleculeEnvironment(device)
    
    # Initialize Agent with new feature dimensions
    agent = MoleculeAgent(
        num_node_features=CONFIG["input_feats"], 
        num_actions=CONFIG["num_actions"], 
        hidden_dim=CONFIG['hidden_dim']
    ).to(device)
    
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['lr'])
    
    # --- PERFORMANCE: Mixed Precision Scaler ---
    # Fix: Use torch.amp.GradScaler('cuda') to avoid depreciation warning
    scaler = torch.amp.GradScaler('cuda') 
    
    buffer = PPOBuffer(device, gamma=CONFIG['gamma'], gae_lambda=CONFIG['gae_lambda'])

    global_step = 0
    episode_num = 0

    for update in range(500):
        buffer.clear()
        steps_collected = 0
        
        while steps_collected < CONFIG['buffer_size']:
            x, edge_index, batch_vec = env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            mol_snapshots = [env.current_mol.GetMol()]

            while not done and ep_len < CONFIG['max_steps']:
                with torch.no_grad():
                    # Autocast is mostly for training, but good practice here too
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        mask = env.get_action_mask()
                        logits, value = agent(x, edge_index, batch_vec, mask.unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)
                        
                    m = Categorical(probs)
                    action = m.sample()
                    log_prob = m.log_prob(action)

                next_obs, reward, done = env.step(action.item())
                buffer.push((x, edge_index, batch_vec, mask), action.item(), reward, done, log_prob.item(), value.item())
                
                x, edge_index, batch_vec = next_obs
                ep_reward += reward
                ep_len += 1
                steps_collected += 1
                global_step += 1
                mol_snapshots.append(env.current_mol.GetMol())

            episode_num += 1
            if episode_num % 20 == 0:
                print(f"Update {update} | Ep {episode_num} | Reward: {ep_reward:.2f}")
                log_data = {"episode_reward": ep_reward, "global_step": global_step}
                if episode_num % 100 == 0:
                    gif = create_gif(mol_snapshots, ep_reward)
                    if gif: log_data["molecule"] = gif
                wandb.log(log_data)

        # GAE Calculation
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                mask = env.get_action_mask()
                _, last_value = agent(x, edge_index, batch_vec, mask.unsqueeze(0))
        buffer.calculate_gae(last_value.item())

        # PPO Update Loop
        for _ in range(CONFIG['ppo_epochs']):
            for idxs, advantages, returns, actions, old_log_probs in buffer.get_batches(CONFIG['batch_size']):
                
                new_log_probs = []
                new_values = []
                entropies = []
                
                # --- PERFORMANCE: Mixed Precision Training ---
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for j, i in enumerate(idxs):
                        (sx, sedge, sbatch, smask) = buffer.states[i]
                        logits, val = agent(sx, sedge, sbatch, smask.unsqueeze(0))
                        probs = torch.softmax(logits, dim=1)
                        m = Categorical(probs)
                        
                        act = actions[j]
                        new_log_probs.append(m.log_prob(act))
                        new_values.append(val)
                        entropies.append(m.entropy())

                    new_log_probs = torch.stack(new_log_probs).view(-1)
                    new_values = torch.stack(new_values).view(-1)
                    entropies = torch.stack(entropies).mean() 

                    ratio = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - CONFIG['clip_epsilon'], 1 + CONFIG['clip_epsilon']) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = torch.nn.functional.mse_loss(new_values, returns)
                    
                    loss = policy_loss + CONFIG['value_coef'] * value_loss - CONFIG['entropy_coef'] * entropies
                
                # --- PERFORMANCE: Scaled Backward Pass ---
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Unscale before clipping!
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                
                scaler.step(optimizer)
                scaler.update()

    print("✅ SOTA PPO Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    train()