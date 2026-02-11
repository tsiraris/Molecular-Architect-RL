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
    "clip_epsilon": 0.2,   # PPO Clipping range
    "entropy_coef": 0.05,  # Exploration
    "value_coef": 0.5,     # Critic weight
    "batch_size": 64,      # Mini-batch size
    "ppo_epochs": 10,      # Reuse data 10 times
    "buffer_size": 2048,   # Steps to collect before training
    "hidden_dim": 64,
    "max_steps": 30
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
    print(f"🚀 PPO Training on {device}")

    env = MoleculeEnvironment(device)
    agent = MoleculeAgent(num_node_features=3, num_actions=4, hidden_dim=CONFIG['hidden_dim']).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['lr'])
    buffer = PPOBuffer(device, gamma=CONFIG['gamma'], gae_lambda=CONFIG['gae_lambda'])

    # Global counters
    global_step = 0
    episode_num = 0

    # Main PPO Loop: "Collect -> Train -> Repeat"
    # We run for a set number of updates (e.g., 500 updates)
    for update in range(500):
        
        # --- PHASE 1: DATA COLLECTION ---
        buffer.clear()
        steps_collected = 0
        
        while steps_collected < CONFIG['buffer_size']:
            # Start new episode
            x, edge_index, batch_vec = env.reset()
            done = False
            ep_reward = 0
            ep_len = 0
            
            mol_snapshots = [env.current_mol.GetMol()] # For GIF

            while not done and ep_len < CONFIG['max_steps']:
                # No grad needed for collection
                with torch.no_grad():
                    mask = env.get_action_mask()
                    logits, value = agent(x, edge_index, batch_vec, mask.unsqueeze(0))
                    
                    probs = torch.softmax(logits, dim=1)
                    m = Categorical(probs)
                    action = m.sample()
                    log_prob = m.log_prob(action)

                # Step
                next_obs, reward, done = env.step(action.item())
                
                # Store in buffer
                buffer.push((x, edge_index, batch_vec, mask), action.item(), reward, done, log_prob.item(), value.item())
                
                # Update counters
                x, edge_index, batch_vec = next_obs
                ep_reward += reward
                ep_len += 1
                steps_collected += 1
                global_step += 1
                
                mol_snapshots.append(env.current_mol.GetMol())

            # Log Episode Stats
            episode_num += 1
            if episode_num % 20 == 0:
                print(f"Update {update} | Ep {episode_num} | Reward: {ep_reward:.2f}")
                log_data = {"episode_reward": ep_reward, "global_step": global_step}
                
                # Add GIF occasionally
                if episode_num % 100 == 0:
                    gif = create_gif(mol_snapshots, ep_reward)
                    if gif: log_data["molecule"] = gif
                
                wandb.log(log_data)

        # --- PHASE 2: CALCULATE GAE ---
        # Get value of the VERY LAST state to bootstrap the calculation
        with torch.no_grad():
            mask = env.get_action_mask()
            _, last_value = agent(x, edge_index, batch_vec, mask.unsqueeze(0))
            
        buffer.calculate_gae(last_value.item())

        # --- PHASE 3: PPO TRAINING (EPOCHS) ---
        for _ in range(CONFIG['ppo_epochs']):
            # Iterate over mini-batches
            for idxs, advantages, returns, actions, old_log_probs in buffer.get_batches(CONFIG['batch_size']):
                
                # --- THIS WAS THE MISSING BLOCK ---
                new_log_probs = []
                new_values = []
                entropies = []
                # ---------------------------------
                
                # Use enumerate to get the local batch index (j) and the global buffer index (i)
                for j, i in enumerate(idxs):
                    # Retrieve the specific graph snapshot for this step
                    (sx, sedge, sbatch, smask) = buffer.states[i] 
                    
                    logits, val = agent(sx, sedge, sbatch, smask.unsqueeze(0))
                    
                    probs = torch.softmax(logits, dim=1)
                    m = Categorical(probs)
                    
                    # Use 'j' (local) to access the actions tensor
                    act = actions[j] 
                    
                    new_log_probs.append(m.log_prob(act))
                    new_values.append(val)
                    entropies.append(m.entropy())

                # Stack results
                # FIX: Use .view(-1) to handle edge cases where batch_size=1
                new_log_probs = torch.stack(new_log_probs).view(-1)
                new_values = torch.stack(new_values).view(-1)
                entropies = torch.stack(entropies).mean() 

                # --- PPO LOSS MATH ---
                # 1. Ratio
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # 2. Surrogate Loss (Clipped)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - CONFIG['clip_epsilon'], 1 + CONFIG['clip_epsilon']) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 3. Value Loss (MSE)
                value_loss = torch.nn.functional.mse_loss(new_values, returns)
                
                # 4. Total Loss
                loss = policy_loss + CONFIG['value_coef'] * value_loss - CONFIG['entropy_coef'] * entropies
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

    print("✅ PPO Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    train()