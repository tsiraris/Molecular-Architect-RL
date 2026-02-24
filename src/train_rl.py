from __future__ import annotations
import datetime as dt
import os
import random
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.optim as optim
import wandb
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from chem_env import ActionSpec
from gnn_agent import MoleculeAgent
from ppo import PPOBuffer
from vec_env import VectorMoleculeEnv

RDLogger.DisableLog("rdApp.*")

CONFIG = {
    # optimization
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "target_kl": 0.02,
    "entropy_coef_start": 0.08,
    "entropy_coef_end": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 1.0,
    # training
    "batch_size": 256,
    "ppo_epochs": 10,
    "buffer_size": 2048,  # total transitions per update (across envs)
    "num_envs": 8,
    "updates": 800,
    # model
    "hidden_dim": 128,
    "edge_dim": 4,
    # env/action
    "max_atoms": 25,
    "max_steps": 40,
    "min_atoms": 5,
    # curriculum (terminal reward mix)
    "curriculum_start": 25,
    "curriculum_end": 300,
    # eval/logging
    "log_every": 5,
    "save_every": 50,
    "eval_samples": 256,  # molecules per evaluation snapshot
    "project": "molecular-rl-2026",
    "run_name": None,
    "seed": 42,
}

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def tanimoto_diversity(smiles: List[str]) -> Tuple[float, float, float]:
    """ Calculates and returns validity, uniqueness, and diversity (1 - average Tanimoto similarity) of a list of SMILES strings."""
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    valid_idx = [i for i, m in enumerate(mols) if m is not None]    # indices of valid molecules
    if not valid_idx:                                               # if no valid molecules, return 0 for all metrics
        return 0.0, 0.0, 0.0

    fps = [AllChem.GetMorganFingerprintAsBitVect(mols[i], 2, nBits=1024) for i in valid_idx]  # Compute Morgan fingerprints for valid molecules (radius of 2 and nBits of 1024), which are used to calculate Tanimoto similarity and thus diversity.
    sim = 0.0
    for i in range(len(fps)):
        s = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
        sim += (sum(s) - 1.0) / (len(fps) - 1 + 1e-6)        # sum(s) - 1.0 to exclude the similarity of a molecule with itself and average over all other molecules

    uniq = len(set(smiles[i] for i in valid_idx)) / max(1, len(valid_idx))  # Uniqueness is the fraction of unique valid molecules among all valid molecules
    valid_rate = len(valid_idx) / len(smiles)                               # Validity is the fraction of valid molecules among all generated molecules
    diversity = 1.0 - (sim / (len(fps) + 1e-6))                             # Diversity is calculated as 1 minus the average Tanimoto similarity, giving a measure of how structurally diverse the valid molecules are (higher is more diverse).
    return valid_rate, uniq, diversity


def scaffold(smiles: str) -> str:
    """
    Returns the Bemis-Murcko scaffold of a molecule given its SMILES string. 
    Bemis-Murcko scaffolds represent the core structure of a molecule by removing side chains and retaining ring systems and linkers.
    If the SMILES is invalid, returns 'INVALID'. If the molecule has no scaffold (e.g., single atom), returns 'NONE'.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "INVALID"
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(scaf) if scaf is not None else "NONE"
    except Exception:
        return "INVALID"


def mol_props(smiles: str) -> Dict[str, float]:
    """
    Calculates and returns a dictionary of molecular properties for a given SMILES string. 
    If the SMILES is invalid, returns a dictionary with 'valid' set to 0.0 and no other properties.
    If the SMILES is valid, returns 'valid' as 1.0 along with calculated properties.
    """
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return {"valid": 0.0}
    return {
        "valid": 1.0,
        "qed": float(QED.qed(m)),
        "mw": float(Descriptors.MolWt(m)),
        "logp": float(Crippen.MolLogP(m)),
        "tpsa": float(rdMolDescriptors.CalcTPSA(m)),
        "hbd": float(Lipinski.NumHDonors(m)),
        "hba": float(Lipinski.NumHAcceptors(m)),
        "rings": float(rdMolDescriptors.CalcNumRings(m)),
    }


class TopK:
    """
    Maintains a running list of the top-k highest-reward molecules (SMILES) seen during training, along with their rewards. 
    Provides methods to add new candidates, retrieve the best one, and convert the list to a format suitable for logging or saving.
    """
    def __init__(self, k: int = 50): 
        self.k = k
        self.items: List[Tuple[float, str]] = []

    # Add a new molecule and its reward to the top-k list, the list is then sorted by reward in descending order and truncated to keep only the top k items.
    def add(self, reward: float, smiles: str): 
        if smiles is None:
            return
        self.items.append((float(reward), smiles))
        self.items.sort(key=lambda x: x[0], reverse=True)
        self.items = self.items[: self.k]

    # Returns the best reward and corresponding SMILES from the top-k list. If the list is empty, returns 0.0 and "None".
    def best(self) -> Tuple[float, str]:
        if not self.items:
            return 0.0, "None"
        r, s = self.items[0]
        return r, s

    # For logging and saving as a table: Converts the top-k list into a list of dictionaries (reward, SMILES, calculated molecular properties and scaffold) for each molecule.
    def to_rows(self) -> List[Dict]:
        rows = []
        for r, s in self.items:
            p = mol_props(s)
            rows.append({"reward": r, "smiles": s, **p, "scaffold": scaffold(s)})
        return rows


def entropy_coef(update: int) -> float:
    """
    Calculates the entropy coefficient for the current update based on a linear schedule defined by curriculum_start and curriculum_end in the CONFIG. 
    The coefficient starts at entropy_coef_start and linearly decreases to entropy_coef_end as the update progresses from curriculum_start to curriculum_end. 
    Before curriculum_start, it remains at entropy_coef_start, and after curriculum_end, it remains at entropy_coef_end.
    """
    # linear schedule
    u0, u1 = CONFIG["curriculum_start"], CONFIG["curriculum_end"]
    t = 0.0 if update <= u0 else (1.0 if update >= u1 else (update - u0) / max(1, (u1 - u0)))
    return (1.0 - t) * CONFIG["entropy_coef_start"] + t * CONFIG["entropy_coef_end"]


def save_topk_txt(path: str, rows: List[Dict]) -> None:
    """Write top-k molecules as a nicely readable, aligned table."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = list(rows[0].keys())

    # stable column order
    preferred = ["reward", "smiles", "qed", "mw", "logp", "tpsa", "hbd", "hba", "rings", "valid", "scaffold"]
    cols = [c for c in preferred if c in cols] + [c for c in cols if c not in preferred]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4g}"
        return str(v)

    widths = {c: max(len(c), max(len(fmt(r.get(c, ""))) for r in rows)) for c in cols}
    sep = " | "
    header = sep.join([c.ljust(widths[c]) for c in cols])
    rule = "-" * len(header)

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(rule + "\n")
        for r in rows:
            line = sep.join([fmt(r.get(c, "")).ljust(widths[c]) for c in cols])
            f.write(line + "\n")


class ResearchLogger:
    """Plain-text logger that writes a detailed log of training progress and configuration to a timestamped file in the specified output directory."""

    def __init__(self, config: Dict, out_dir: str = "experiments"):
        os.makedirs(out_dir, exist_ok=True)
        ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = os.path.join(out_dir, f"Run_{ts}.txt")

        with open(self.path, "w", encoding="utf-8") as f:
            f.write("MOLECULAR RL (2026 baseline)\n" + "=" * 80 + "\n")
            for k, v in config.items():
                f.write(f"{k:<24}: {v}\n")
            f.write("=" * 80 + "\n")
            header = (
                f"{'Update':<6} | {'R_mean':<8} | {'R_std':<8} | {'Valid%':<7} | {'Unique%':<8} | "
                f"{'Div':<6} | {'KL':<8} | {'Clip':<6} | {'ExpVar':<7} | {'Ent':<6} | {'Curr':<6} | "
                f"{'BestR':<7} | BestSMILES\n"
            )
            f.write(header)
            f.write("-" * (len(header) - 1) + "\n")

    def log_step(
        self,
        update: int,
        r_mean: float,
        r_std: float,
        valid_rate: float,
        unique_rate: float,
        diversity: float,
        kl: float,
        clip_frac: float,
        exp_var: float,
        entropy_coef_val: float,
        curriculum_ratio: float,
        best_reward: float,
        best_smiles: str,
    ) -> None:
        s_short = (best_smiles[:80] + "…") if len(best_smiles) > 80 else best_smiles
        row = (
            f"{update:<6} | {r_mean:<8.3f} | {r_std:<8.3f} | {100*valid_rate:<7.1f} | {100*unique_rate:<8.1f} | "
            f"{diversity:<6.3f} | {kl:<8.4f} | {clip_frac:<6.3f} | {exp_var:<7.3f} | {entropy_coef_val:<6.3f} | "
            f"{curriculum_ratio:<6.3f} | {best_reward:<7.3f} | {s_short}\n"
        )
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(row)


def train():
    set_seed(CONFIG["seed"])     # Set random seeds for reproducibility of the training process across random, numpy, and torch (both CPU and CUDA).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = ActionSpec(max_atoms=CONFIG["max_atoms"])  # Create an action specification for the environment to initialize the environment and determine the action space size.
    num_actions = spec.num_actions
    input_feats = len(spec.atom_types) + 3 + 3  # atom one-hot + hyb(3) + flags(3)

    run_name = CONFIG["run_name"] or f"molrl_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project=CONFIG["project"],
        name=run_name,
        config={**CONFIG, "num_actions": num_actions, "input_feats": input_feats},
    )

    env = VectorMoleculeEnv(CONFIG["num_envs"], device, action_spec=spec)  # Initialize a vectorized environment for molecular generation (parallel simulation of multiple environments/molecules simultaneously).
    agent = MoleculeAgent(input_feats, num_actions, hidden_dim=CONFIG["hidden_dim"], edge_dim=CONFIG["edge_dim"]).to(device)

    optimizer = optim.AdamW(agent.parameters(), lr=CONFIG["lr"])
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
    buffer = PPOBuffer(device, num_envs=CONFIG["num_envs"], gamma=CONFIG["gamma"], gae_lambda=CONFIG["gae_lambda"])
    topk = TopK(k=50)

    obs = env.reset()  # Reset the environment to get the initial observations for each parallel environment.

    out_dir = os.path.join("artifacts", run_name)  # Create an output directory for saving checkpoints and logs, organized under "artifacts" with a subdirectory named after the run.
    os.makedirs(out_dir, exist_ok=True)
    logger = ResearchLogger({**CONFIG, "num_actions": num_actions, "input_feats": input_feats}, out_dir="experiments")

    for update in range(CONFIG["updates"]):
        # Compute the curriculum ratio based on the current update number, which determines how much the reward is influenced by the curriculum (intermediate rewards) versus the final reward.
        curr_ratio = min(1.0, max( 0.0, (update - CONFIG["curriculum_start"])/max(1, (CONFIG["curriculum_end"] - CONFIG["curriculum_start"]))))

        buffer.clear()

        # -------- Collect --------
        steps_per_env = CONFIG["buffer_size"] // CONFIG["num_envs"]
        for _ in range(steps_per_env):
            with torch.no_grad():
                batch = Batch.from_data_list(obs).to(device) # Convert the list of observations (one per environment) into a single batched graph representation suitable for input to the GNN agent.
                masks = env.get_masks()  # shape (num_envs, max_atoms) - bool action masks for each environment (which actions are valid for each environment).
                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits, values = agent(batch.x, batch.edge_index, batch.edge_attr, batch.batch, masks)

                dist = Categorical(logits=logits)   # Create a categorical distribution over actions for each environment based on the logits output by the agent.
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

            next_obs, rewards, dones, infos = env.step(actions, curr_ratio)

            for e in range(CONFIG["num_envs"]):
                # Pack the current state of the environment (graph representation and mask) for environment e to store in the PPO buffer.
                state_pack = (obs[e].x, obs[e].edge_index, obs[e].edge_attr, masks[e]) 
                buffer.push(
                    env_id=e,
                    state=state_pack,
                    action=int(actions[e].item()),
                    reward=float(rewards[e].item()),
                    done=float(dones[e].item()),
                    log_prob=float(log_probs[e].item()),
                    value=float(values[e].item()),
                )

                if dones[e].item() > 0.5:                                   # If the episode for environment e has ended (done)
                    term_smi = infos[e].get("terminal_smiles", "INVALID")   # extract the terminal SMILES from the info dictionary 
                    topk.add(float(rewards[e].item()), term_smi)            # and add it to the top-k list along with its reward

            obs = next_obs

        # Compute value estimates for the last observations to use for advantage estimation in PPO.
        with torch.no_grad():
            batch = Batch.from_data_list(obs).to(device)
            masks = env.get_masks()
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _, last_values = agent(batch.x, batch.edge_index, batch.edge_attr, batch.batch, masks) # bootstrap the value of the final state for any episodes that haven't ended yet.

        buffer.finalize(last_values) # Compute advantages and returns using GAE for each environment's trajectory and flatten the collected transitions into tensors for training.

        # -------- PPO Update --------
        approx_kls, clip_fracs, losses = [], [], []
        ent_coef = entropy_coef(update)

        # Normalize advantages to have mean 0 and std 1 for better training stability.
        adv = buffer.advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _epoch in range(CONFIG["ppo_epochs"]):
            epoch_kls = []
            for idx in buffer.get_batches(CONFIG["batch_size"], shuffle=True):
                idx_list = idx.tolist()

                # Creates a list of Data objects for the current batch of indices, where each Data object contains the node features, edge index, and edge attributes for a single environment's state. 
                batch_list = [     
                    Data(x=buffer.states[i][0], edge_index=buffer.states[i][1], edge_attr=buffer.states[i][2])
                    for i in idx_list
                ] 
                
                batch_masks = torch.stack([buffer.states[i][3] for i in idx_list], dim=0)   # Stack the action masks for the current batch of indices into a single tensor of shape (batch_size, max_atoms).
                batch_graph = Batch.from_data_list(batch_list).to(device)                   # This list is converted into a Batch object that can be processed by the GNN agent in a single forward pass. 

                actions_b = buffer.actions[idx]
                old_logp_b = buffer.log_probs[idx]
                returns_b = buffer.returns[idx]
                adv_b = adv[idx]

                with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    logits, values = agent(
                        batch_graph.x, batch_graph.edge_index, batch_graph.edge_attr, batch_graph.batch, batch_masks
                    )
                    dist = Categorical(logits=logits)

                    logp = dist.log_prob(actions_b)         # Calculate the log probabilities of the actions taken in the batch according to the current policy.
                    ratio = torch.exp(logp - old_logp_b)    # Calculate the probability ratio for PPO (=exponent of the difference between the new log probabilities and the old log probabilities).

                    # PPO policy loss with clipping
                    surr1 = ratio * adv_b
                    surr2 = torch.clamp(ratio, 1.0 - CONFIG["clip_epsilon"], 1.0 + CONFIG["clip_epsilon"]) * adv_b
                    policy_loss = -torch.min(surr1, surr2).mean()

                    # PPO value loss with clipping
                    values_old = buffer.values[idx]
                    values_clipped = values_old + torch.clamp(
                        values - values_old, -CONFIG["clip_epsilon"], CONFIG["clip_epsilon"]
                    )
                    vloss1 = (values - returns_b).pow(2)
                    vloss2 = (values_clipped - returns_b).pow(2)
                    value_loss = 0.5 * torch.max(vloss1, vloss2).mean()

                    entropy = dist.entropy().mean()
                    loss = policy_loss + CONFIG["value_coef"] * value_loss - ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale gradients before clipping to ensure that the clipping is done in the correct scale.
                torch.nn.utils.clip_grad_norm_(agent.parameters(), CONFIG["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()

                with torch.no_grad():
                    kl = (old_logp_b - logp).mean().item()  # Approximate KL divergence between the old and new policy for the current batch, used for monitoring and early stopping of PPO updates if the policy changes too much.
                    epoch_kls.append(kl)
                    clip_frac = (torch.abs(ratio - 1.0) > CONFIG["clip_epsilon"]).float().mean().item()  # Fraction of actions in the batch for which the probability ratio was outside the clipping range, used as a diagnostic metric to understand how much the policy is changing during updates.
                    clip_fracs.append(clip_frac)
                    losses.append(loss.item())

            mean_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            approx_kls.append(mean_kl)
            if mean_kl > CONFIG["target_kl"]:  # If the average KL divergence for this epoch exceeds the target KL specified in the CONFIG, break out of the PPO update loop.
                break

        # -------- Logging / Evaluation --------
        if update % CONFIG["log_every"] == 0:
            smiles_snapshot = env.get_smiles()                              # Get the current batch of generated molecules (SMILES) from the environment for evaluation of validity, uniqueness, and diversity metrics.
            valid_r, uniq_r, div = tanimoto_diversity(smiles_snapshot)      # Calculate validity, uniqueness, and diversity metrics for the current batch of generated molecules.

            best_r, best_s = topk.best()
 
            # Explained variance (EV) of the value function predictions (= how well the value function is fitting the returns with max=1.0).
            with torch.no_grad():
                y_true = buffer.returns
                y_pred = buffer.values
                ev = 1.0 - torch.var(y_true - y_pred) / (torch.var(y_true) + 1e-8)
                ev = float(ev.item())

            wandb.log(
                {
                    "update": update,
                    "curriculum_ratio": curr_ratio,
                    "entropy_coef": ent_coef,
                    "reward/mean": float(buffer.rewards.mean().item()),
                    "reward/std": float(buffer.rewards.std().item()),
                    "eval/valid_rate": valid_r,
                    "eval/unique_rate": uniq_r,
                    "eval/diversity": div,
                    "ppo/kl": float(np.mean(approx_kls)) if approx_kls else 0.0,
                    "ppo/clip_frac": float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                    "ppo/loss": float(np.mean(losses)) if losses else 0.0,
                    "critic/explained_var": ev,
                    "best/reward": best_r,
                }
            )

            logger.log_step(
                update=update,
                r_mean=float(buffer.rewards.mean().item()),
                r_std=float(buffer.rewards.std().item()),
                valid_rate=valid_r,
                unique_rate=uniq_r,
                diversity=div,
                kl=float(np.mean(approx_kls)) if approx_kls else 0.0,
                clip_frac=float(np.mean(clip_fracs)) if clip_fracs else 0.0,
                exp_var=ev,
                entropy_coef_val=ent_coef,
                curriculum_ratio=curr_ratio,
                best_reward=best_r,
                best_smiles=best_s,
            )

            print(
                f"[{update:04d}] "
                f"R={buffer.rewards.mean().item():.3f} "
                f"(std={buffer.rewards.std().item():.3f})  "
                f"valid={valid_r:.3f}  uniq={uniq_r:.3f}  div={div:.3f}  "
                f"kl={(float(np.mean(approx_kls)) if approx_kls else 0.0):.4f}  "
                f"clip={(float(np.mean(clip_fracs)) if clip_fracs else 0.0):.3f}  "
                f"ev={ev:.3f}  ent={ent_coef:.3f}  curr={curr_ratio:.3f}  "
                f"best={best_r:.3f}  {best_s}"
            )

        if update % CONFIG["save_every"] == 0 and update > 0: # Save a checkpoint of the current model, optimizer, scaler state, and top-k molecules every save_every updates.
            save_topk_txt(os.path.join(out_dir, f"topk_update_{update}.txt"), topk.to_rows())
            ckpt = {
                "config": {**CONFIG, "num_actions": num_actions, "input_feats": input_feats},
                "model": agent.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "update": update,
            }
            torch.save(ckpt, os.path.join(out_dir, f"checkpoint_{update}.pt"))

    save_topk_txt(os.path.join(out_dir, "topk_final.txt"), topk.to_rows())
    wandb.finish()


if __name__ == "__main__":
    train()
