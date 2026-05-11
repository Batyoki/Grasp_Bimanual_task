#!/usr/bin/env python3
import sys
import os
import logging
from pathlib import Path
import torch
import torch.optim as optim
import numpy as np

logging.basicConfig(level=logging.INFO, format='[Vision Train] %(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

GPU_ID = int(os.environ.get('GPU_ID', 0))
NUM_ENVS = int(os.environ.get('NUM_ENVS', 4))
TOTAL_TIMESTEPS = int(os.environ.get('TOTAL_TIMESTEPS', 1000000))
RUN_DIR = os.environ.get('RUN_DIR', './logs')

try:
    logger.info("STEP 1: Booting Isaac Engine...")
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    torch.cuda.set_device(GPU_ID)
    device = torch.device(f"cuda:{GPU_ID}")
    
    from isaaclab.app import AppLauncher
    launcher = AppLauncher({"headless": True, "enable_cameras": True})
    sim = launcher.app

    logger.info("STEP 2: Setup Environment and Video Recorder...")
    import gymnasium as gym
    from gymnasium.wrappers import RecordVideo
    from bimanual_vx300s_env.vx300s_vision_env_cfg import VisionVX300sEnvCfg
    from isaaclab.envs import ManagerBasedRLEnv
    from vision_network import DualCameraPPO
    
    checkpoint_dir = Path(RUN_DIR) / "checkpoints"
    video_dir = Path(RUN_DIR) / "videos"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    
    env_cfg = VisionVX300sEnvCfg()
    env_cfg.scene.num_envs = NUM_ENVS
    env_cfg.seed = 42
    
    env = ManagerBasedRLEnv(cfg=env_cfg, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=str(video_dir), step_trigger=lambda step: step % 250 == 0, disable_logger=True)
    
    logger.info("STEP 3: Initialize Custom PPO Network...")
    action_dim = env.action_space.shape[1]
    state_dim = env.observation_space["policy"]["robot_state"].shape[1]
    
    agent = DualCameraPPO(action_dim=action_dim, state_dim=state_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)

    logger.info("STEP 4: Starting Custom Training Loop...")
    NUM_STEPS = 128 
    obs, _ = env.reset()
    num_updates = TOTAL_TIMESTEPS // (NUM_STEPS * NUM_ENVS)

    for update in range(num_updates):
        b_top = torch.zeros((NUM_STEPS, NUM_ENVS, 84, 84, 3), device=device, dtype=torch.uint8)
        b_front = torch.zeros((NUM_STEPS, NUM_ENVS, 84, 84, 3), device=device, dtype=torch.uint8)
        b_states = torch.zeros((NUM_STEPS, NUM_ENVS, state_dim), device=device)
        b_actions = torch.zeros((NUM_STEPS, NUM_ENVS, action_dim), device=device)
        b_logprobs = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_rewards = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_dones = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)
        b_values = torch.zeros((NUM_STEPS, NUM_ENVS), device=device)

        for step in range(NUM_STEPS):
            top = obs["policy"]["top_image"]
            front = obs["policy"]["front_image"]
            state = obs["policy"]["robot_state"]
            
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(top, front, state)
                
            next_obs, reward, term, trunc, _ = env.step(action)
            done = term | trunc
            
            b_top[step] = top; b_front[step] = front; b_states[step] = state
            b_actions[step] = action; b_logprobs[step] = logprob
            b_rewards[step] = reward; b_values[step] = value.flatten()
            b_dones[step] = done
            
            obs = next_obs

        with torch.no_grad():
            next_value = agent.get_value(obs["policy"]["top_image"], obs["policy"]["front_image"], obs["policy"]["robot_state"]).flatten()
            advantages = torch.zeros_like(b_rewards)
            lastgaelam = 0
            for t in reversed(range(NUM_STEPS)):
                nextnonterminal = 1.0 - (done.float() if t == NUM_STEPS - 1 else b_dones[t + 1].float())
                nextvalues = next_value if t == NUM_STEPS - 1 else b_values[t + 1]
                delta = b_rewards[t] + 0.99 * nextvalues * nextnonterminal - b_values[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + b_values

        b_top = b_top.view(-1, 84, 84, 3); b_front = b_front.view(-1, 84, 84, 3)
        b_states = b_states.view(-1, state_dim); b_actions = b_actions.view(-1, action_dim)
        b_logprobs = b_logprobs.view(-1); advantages = advantages.view(-1); returns = returns.view(-1)

        b_inds = np.arange(NUM_STEPS * NUM_ENVS)
        for epoch in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), 64):
                end = start + 64
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_top[mb_inds], b_front[mb_inds], b_states[mb_inds], b_actions[mb_inds])
                
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_adv = advantages[mb_inds]
                
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 0.8, 1.2)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                v_loss = 0.5 * ((newvalue.flatten() - returns[mb_inds]) ** 2).mean()
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

        logger.info(f"Update {update+1}/{num_updates} | Mean Reward: {b_rewards.mean().item():.3f} | Value Loss: {v_loss.item():.3f}")

    torch.save(agent.state_dict(), f"{checkpoint_dir}/vision_model_final.pth")
    env.close()
    sim.close()
    
except Exception as e:
    logger.error(f"\n❌ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)