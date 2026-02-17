import torch
import torch.optim as optim
import torch.nn.functional as F
import wandb
import numpy as np
import io, imageio, os, datetime
from gnn_agent import MoleculeAgent
from vec_env import VectorMoleculeEnv
from ppo import PPOBuffer 
from torch.distributions import Categorical
from rdkit import RDLogger, Chem, DataStructs
from rdkit.Chem import AllChem, Draw
from torch_geometric.data import Batch, Data

RDLogger.DisableLog('rdApp.*')

CONFIG = {
    "lr": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.05,
    "value_coef": 0.5,
    "batch_size": 256,
    "ppo_epochs": 10,       
    "buffer_size": 2048,   
    "hidden_dim": 64,
    "input_feats": 9,
    "num_actions": 11, 
    "edge_dim": 4,
    "num_envs": 8,
    "curriculum_start": 50,
    "curriculum_end": 400
}

class ResearchLogger:
    def __init__(self, config, log_dir="experiments"):
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.filename = os.path.join(log_dir, f"Run_{ts}.txt")
        with open(self.filename, "w") as f:
            f.write("MOLECULAR RL SOTA LOG\n" + "="*60 + "\n")
            for k, v in config.items(): f.write(f"{k:<20}: {v}\n")
            f.write("="*60 + "\n")
            header = f"{'Update':<6} | {'Reward':<8} | {'Valid%':<6} | {'Unique%':<7} | {'Div':<5} | {'KL':<8} | {'Clip':<6} | {'ExpVar':<8} | {'Best':<6} | {'Best SMILES'}\n"
            f.write(header)
            f.write("-" * len(header) + "\n")

    def log_step(self, up, rew, val, uni, div, kl, clip, ev, best_r, best_s):
        # Truncate string if too long for cleaner table
        s_short = (best_s[:25] + '..') if len(best_s) > 25 else best_s
        row = f"{up:<6} | {rew:<8.3f} | {val:<6.1f} | {uni:<7.1f} | {div:<5.2f} | {kl:<8.4f} | {clip:<6.2f} | {ev:<8.3f} | {best_r:<6.2f} | {s_short}\n"
        with open(self.filename, "a") as f: f.write(row)

class SuccessBuffer:
    def __init__(self, cap=20): 
        self.cap = cap 
        self.buf = [] 
        
    def add(self, smi, rew):
        if rew > 0.1: # Catch any positive progress
            self.buf.append((rew, smi))
            self.buf.sort(key=lambda x: x[0], reverse=True)
            self.buf = self.buf[:self.cap]
            
    def get_best(self): 
        if not self.buf: return "None", 0.0
        # Return (SMILES, Reward)
        return self.buf[0][1], self.buf[0][0]

def calculate_stats(smiles_list):
    mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    valid = [i for i, m in enumerate(mols) if m is not None]
    if not valid: return 0.0, 0.0, 0.0
    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in valid]
    sim = 0
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim += (sum(s) - 1.0) / (len(fps) - 1 + 1e-6)
    uni = len(set([smiles_list[i] for i in valid])) / len(valid)
    return len(valid)/len(smiles_list), uni, 1.0 - (sim / (len(fps) + 1e-6))

def train():
    wandb.init(project="drug-design-rl", config=CONFIG)
    logger = ResearchLogger(CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 SOTA PPO Training on {device} (Fixed Version)")
    
    env = VectorMoleculeEnv(CONFIG['num_envs'], device)
    agent = MoleculeAgent(CONFIG["input_feats"], CONFIG["num_actions"], CONFIG['hidden_dim'], CONFIG['edge_dim']).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['lr'])
    scaler = torch.amp.GradScaler('cuda') 
    buffer = PPOBuffer(device, gamma=CONFIG['gamma'], gae_lambda=CONFIG['gae_lambda'])
    success_buf = SuccessBuffer()

    obs = env.reset()
    for update in range(1000):
        curr_ratio = min(1.0, max(0.0, (update - CONFIG['curriculum_start']) / (CONFIG['curriculum_end'] - CONFIG['curriculum_start'])))
        buffer.clear()
        
        # 1. COLLECTION
        for _ in range(CONFIG['buffer_size'] // CONFIG['num_envs']):
            with torch.no_grad():
                batch_data = Batch.from_data_list(obs)
                with torch.amp.autocast('cuda'):
                    masks = env.get_masks()
                    logits, vals = agent(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, masks)
                    actions = Categorical(logits=logits).sample()
                    log_probs = Categorical(logits=logits).log_prob(actions)
            
            # --- CRITICAL FIX: Unpack 4 values including INFOS ---
            next_obs, rewards, dones, infos = env.step(actions, curr_ratio)
            
            for i in range(CONFIG['num_envs']):
                # FIX: Use terminal_smiles from infos to save the ACTUAL completed molecule
                if dones[i]: 
                    term_smi = infos[i].get('terminal_smiles', 'INVALID')
                    success_buf.add(term_smi, rewards[i].item())
                
                buffer.push(
                    (obs[i].x, obs[i].edge_index, obs[i].edge_attr, None, masks[i]), 
                    actions[i].item(), rewards[i].item(), dones[i].item(), log_probs[i].item(), vals[i].item()
                )
            obs = next_obs

        # 2. GAE
        with torch.no_grad():
             batch_data = Batch.from_data_list(obs)
             with torch.amp.autocast('cuda'):
                 _, last_vals = agent(batch_data.x, batch_data.edge_index, batch_data.edge_attr, batch_data.batch, env.get_masks())
        buffer.calculate_gae(last_vals.mean().item())

        # 3. UPDATE
        approx_kls, clip_fr = [], []
        for _ in range(CONFIG['ppo_epochs']):
            for idxs, advantages, returns, actions, old_log_probs in buffer.get_batches(CONFIG['batch_size']):
                batch_list = [Data(x=buffer.states[i][0], edge_index=buffer.states[i][1], edge_attr=buffer.states[i][2]) for i in idxs]
                batch_masks = torch.stack([buffer.states[i][4] for i in idxs])
                batch_graph = Batch.from_data_list(batch_list).to(device)

                with torch.amp.autocast('cuda'):
                    logits, new_vals = agent(batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch, batch_masks)
                    m = Categorical(logits=logits)
                    log_ratio = m.log_prob(actions) - old_log_probs
                    ratio = torch.exp(log_ratio)
                    
                    with torch.no_grad():
                        approx_kls.append(((ratio - 1) - log_ratio).mean().item())
                        clip_fr.append((torch.abs(ratio - 1.0) > CONFIG['clip_epsilon']).float().mean().item())

                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1 - CONFIG['clip_epsilon'], 1 + CONFIG['clip_epsilon']) * advantages
                    loss = -torch.min(surr1, surr2).mean() + 0.5 * F.mse_loss(new_vals.view(-1), returns) - 0.05 * m.entropy().mean()
                
                optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0); scaler.step(optimizer); scaler.update()

        # 4. LOGGING
        if update % 5 == 0:
            sm = env.get_smiles(); val_r, uni, div = calculate_stats(sm)
            y_pred = torch.tensor(buffer.values, device=device); y_true = buffer.returns
            ev = 1 - torch.var(y_true - y_pred) / (torch.var(y_true) + 1e-8)
            
            # Get correct best molecule
            best_s, best_r = success_buf.get_best()
            
            # FIX: Log BEST molecule to file, not random sample
            logger.log_step(update, np.mean(buffer.rewards), val_r*100, uni*100, div, np.mean(approx_kls), np.mean(clip_fr), ev.item(), best_r, best_s)
            wandb.log({"reward": np.mean(buffer.rewards), "diversity": div, "exp_var": ev, "curriculum": curr_ratio, "best_reward": best_r})
            
            # IMPROVED PRINT: Shows Best Molecule separately from Random Sample
            print(f"Update {update:<4} | R: {np.mean(buffer.rewards):.2f} | Div: {div:.2f} | Best: {best_r:.2f} ({best_s})")
            print(f"Sample: {sm[0]}") # Current snapshot

    wandb.finish()

if __name__ == "__main__": train()