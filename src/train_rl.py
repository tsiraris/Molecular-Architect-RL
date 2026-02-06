import torch
import torch.optim as optim
import wandb
import imageio
import io
import numpy as np
from gnn_agent import MoleculeAgent
from chem_env import MoleculeEnvironment
from torch.distributions import Categorical
from rdkit import RDLogger
from rdkit.Chem import Draw

RDLogger.DisableLog('rdApp.*')  # <--- This kills the noise

# --- CONFIG ---
CONFIG = {
    "episodes": 3000,          
    "lr": 0.001,          
    "gamma": 0.99,
    "hidden_dim": 64,
    "max_steps": 30,
    "log_interval": 20,
    "entropy_coef": 0.05    # Strength of exploration pressure
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
    
    # Save GIF to memory buffer
    with io.BytesIO() as gif_buffer:
        imageio.mimsave(gif_buffer, images, format='GIF', duration=0.5, loop=0)
        
        # --- FIX: Use wandb.Video directly with io.BytesIO ---
        # Wrap the bytes in BytesIO so wandb treats it as a file
        return wandb.Video(io.BytesIO(gif_buffer.getvalue()), format='gif', caption=f"Reward: {reward:.2f}")

def train():
    wandb.init(project="drug-design-rl", config=CONFIG)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on {device}")

    # 1. Initialize
    env = MoleculeEnvironment(device)
    # 3 Features (C,N,O) and 4 Actions (Add C, Add N, Add O, Stop)
    agent = MoleculeAgent(num_node_features=3, num_actions=4, hidden_dim=CONFIG['hidden_dim']).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=CONFIG['lr'])

    # 2. Training Loop
    for episode in range(CONFIG['episodes']):
        x, edge_index, batch = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        # Snapshot storage for GIF (Visuals)
        mol_snapshots = [env.current_mol.GetMol()]

        # Keep track of the game to learn later
        log_probs = []
        values = []
        rewards = []
        entropies = [] # For tracking entropy directly during rollout

        # --- A. PLAY THE GAME (Rollout) ---
        while not done and step_count < CONFIG['max_steps']:
            # Get the mask from the environment
            action_mask = env.get_action_mask() 
            
            # Ask the Agent what to do
            logits, value = agent(x, edge_index, batch, action_mask.unsqueeze(0)) # Unsqueeze to add batch dim to action_mask (from [Num_Actions] to [1, Num_Actions]))
            
            # Sample an action from the probability distribution
            probs = torch.softmax(logits, dim=1)
            
            # SAFETY: Handle NaN in probs if model explodes
            if torch.isnan(probs).any():
                print(" ⚠️  NaN detected in probabilities! Resetting episode.")
                break
            
            m = Categorical(probs)  # Probability distribution (object) over actions
            action = m.sample() # Tensor containing the sampled index
            
            # Execute the action in the environment
            next_obs, reward, done = env.step(action.item())
            
            # Store data for learning
            log_probs.append(m.log_prob(action))
            values.append(value)
            rewards.append(reward)
            entropies.append(m.entropy()) # Calculate entropy of the decision
            
            # Update observation
            mol_snapshots.append(env.current_mol.GetMol()) # Save frame for GIF
            x, edge_index, batch = next_obs
            total_reward += reward
            step_count += 1
        
        # Skip update if episode broke early
        if len(rewards) == 0:  continue

        # --- B. LEARN FROM EXPERIENCE (Update) ---
        # Calculate Returns (Discounted Cumulative Reward)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + CONFIG['gamma'] * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        
        # SAFETY: Normalize only if we have variance (len > 1)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        else:
            returns = returns - returns.mean() # Center arround zero
        
        # Calculate Loss
        policy_loss = []   # Actor Loss: Did action lead to a good return?
        value_loss = []    # Critic Loss: Was agent's prediction of the value accurate?
        entropy_loss = []  # Track entropy
        
        for log_prob, value, R, entropy in zip(log_probs, values, returns, entropies):
            advantage = R - value.item()
            
            # Reinforce algorithm - Policy Gradient: -log(p) * advantage
            # Increase the probability of the action taken if we beat expectations, and decrease if we failed.
            policy_loss.append(-log_prob * advantage)
            
            # Huber Loss: (Predicted - Actual)^2 - Manageable gradients
            value_loss.append(torch.nn.functional.smooth_l1_loss(value, torch.tensor([[R]]).to(device)))

            # Entropy Loss
            entropy_loss.append(-entropy) # We minimize negative entropy to maximize randomness
            
        if len(policy_loss) > 0:
            
            optimizer.zero_grad()
            
            p_loss = torch.stack(policy_loss).sum() # Total policy loss for the entire episode
            v_loss = torch.stack(value_loss).sum()  # Total value loss for the entire episode
            e_loss = torch.stack(entropy_loss).sum() # Sum up entropy penalties
            
            loss = p_loss + v_loss + (CONFIG['entropy_coef'] * e_loss)  # Total loss
            
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=1.0)

            optimizer.step()
            
            # Log only if update happened
            if episode % CONFIG['log_interval'] == 0:
                print(f"Episode {episode}, Reward: {total_reward:.4f}, Steps: {step_count}")
                
                # Create the GIF
                gif_video = create_gif(mol_snapshots, total_reward)
                
                log_dict = {
                    "episode": episode, 
                    "reward": total_reward, 
                    "loss": loss.item(),
                    "steps": step_count,
                    "entropy": -e_loss.item() / step_count # Log average entropy
                }
                
                if gif_video:
                    log_dict["molecule_construction"] = gif_video
                    
                wandb.log(log_dict)
        else:
            # Just log reward if no update
             wandb.log({"episode": episode, "reward": total_reward})

    print(" ✅  Training Complete!")
    wandb.finish()

if __name__ == "__main__":
    train()