# 🚀 Mastering PPO: Parallel Environments & Brittle Balance (Hopper-v4)
[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-RL_Environment-112C3E)](https://gymnasium.farama.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-Physics_Engine-1A1A1A?logo=google&logoColor=white)](https://mujoco.org/)
[![Stable Baselines3](https://img.shields.io/badge/Stable_Baselines3-Production_RL-4B8BBE)](https://stable-baselines3.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-Mathematics-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Model_Training-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)

This repository documents the engineering process of training a brittle, one-legged robot (Hopper) to balance and propel itself forward. To solve this unforgiving environment, we transitioned to the **Proximal Policy Optimization (PPO)** algorithm and leveraged **Parallel Vectorized Environments** to massively accelerate data collection.

---

<img width="368" height="305" alt="Screenshot 3" src="https://github.com/user-attachments/assets/37fc39ce-1ccc-49a8-b4ea-7aaae5fce68c" />
<img width="368" height="305" alt="Screenshot 1" src="https://github.com/user-attachments/assets/c774721e-7edb-4199-b783-2bc4ca83a257" />


## 🛤️ The Journey (Project Phases)

Balancing the Hopper required a fundamental shift in algorithm choice and training architecture:

### Phase 1: Environment Anatomy (Unforgiving Physics)
- **Description:** Analyzing the rigid physical constraints of the Hopper.
- **Action Space:** 3 continuous joints (Hip, Knee, Ankle) with a torque range of .
- **Observation Space:** 11-dimensional vector tracking Z-height, Pitch angle, joint angles, and velocities.
- **Termination Conditions:** Unlike HalfCheetah, Hopper is extremely brittle. The episode terminates instantly if the Z-height drops below 0.7 meters (collapsing) or the Pitch angle exceeds ~11 degrees (0.2 rad).

### Phase 2: Algorithm Selection (Why PPO?)
- **Description:** SAC was discarded for this task because its core philosophy relies on Maximum Entropy (high exploration). High exploration in a brittle environment causes the agent to fall and terminate continuously before learning anything.
- **The PPO Advantage:** PPO utilizes **Conservative Updates** (Clipping). When it discovers a movement that keeps it upright, it safely and slowly refines it (max 20% policy change at a time), making it perfect for maintaining fragile equilibrium.
- **Exploration Control:** Entropy was manually zeroed out (`ent_coef=0.0`) to suppress dangerous random kicks.

### Phase 3: Architectural Acceleration (Vectorization)
- **Description:** PPO is an "On-Policy" algorithm, meaning it is extremely data-hungry and discards experiences after learning. Training sequentially takes too long.
- **The Solution:** We spawned **8 Parallel Environments** (`SubprocVecEnv`). This allowed 8 robots to hop and fall simultaneously across multiple CPU cores, feeding massive batches of data (16,384 steps per update) to the neural network in mere seconds.
- **Result:** The agent achieved a breakthrough at ~500,000 timesteps, executing flawless consecutive hops and hitting the 1000-step survival limit.

---

<img width="1185" height="581" alt="Screenshot 2" src="https://github.com/user-attachments/assets/86e24078-8812-4975-82bd-3c8941e2b40f" />


## 🧠 Core Concepts Explored

- **Conservative Updates (Clipping):** Utilizing the PPO `clip_range=0.2` to ensure the neural network doesn't catastrophically forget how to balance while trying to jump faster.
- **Parallel Vectorized Environments:** Bypassing PPO's sample-inefficiency by running `SubprocVecEnv` with multiple workers to aggregate data exponentially faster.
- **High-Frequency Rendering:** Addressing MuJoCo physics sync errors by explicitly forcing `render_fps=125` for evaluation, revealing Hopper's hyper-sensitive `dt=0.008` physics step.

---

## 🛠️ Installation & Repository Structure

### Prerequisites

```bash
pip install torch numpy gymnasium[mujoco] stable-baselines3[extra] imageio
```

### Repository Structure

```text
📦 PPO-MuJoCo-Hopper-Balance
 ┣ 📜 Hopper.py                       # Main script handling 8 parallel envs, PPO training, and rendering
 ┣ 📂 models                          
 ┃ ┗ 📜 ppo_hopper.zip                # Final SB3 Production Model (Trained Weights & Config)
 ┣ 📂 videos                          
 ┃ ┗ 🎬 mujoco_hopper_ppo.mp4         # Rendered cinematic video of the robot hopping seamlessly
 ┗ 📜 README.md
```

---

## 🏋️ Training and Evaluation

To run the full pipeline (initialize 8 parallel workers, train the PPO model, and render the final Hopper performance):

```bash
python Hopper.py
```

## 👨‍💻 Author
**Mohammed Arif Mahyoub Haider**

*Electrical Engineer - Computer and Industrial Control*
