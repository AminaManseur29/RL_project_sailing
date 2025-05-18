import os
import sys
import numpy as np
import pandas as pd
from itertools import product

from src.env_sailing import SailingEnv
from src.initial_windfields import get_initial_windfield
from src.utils.agent_utils import save_qlearning_agent
from src.agents.good_agents.hybrid_smart_qlearning_agent import ExpectedSARSALambdaSmartAgent

sys.path.append(os.path.abspath("src"))


# 🎯 Fonction de reward shaping
def custom_reward(obs, next_obs, reward, done, goal):
    if done:
        return reward
    pos = obs[:2]
    next_pos = next_obs[:2]
    move = next_pos - pos
    goal_vec = goal - pos
    if np.linalg.norm(move) < 1e-2:
        return -1
    progress = np.dot(move, goal_vec) / (np.linalg.norm(move) * np.linalg.norm(goal_vec) + 1e-8)
    return progress * 2


# 🔧 Grille d'hyperparamètres
alphas = [0.05, 0.1]
lambdas = [0.7, 0.8, 0.9]
epsilons = [0.1, 0.2]
gamma = 0.99
num_episodes = 100
max_steps = 1000
goal_position = np.array([17, 31])
results = []

# 🔁 Grid Search
for alpha, lambda_, epsilon in product(alphas, lambdas, epsilons):
    print(f"Training with α={alpha}, λ={lambda_}, ε={epsilon}")
    env = SailingEnv(**get_initial_windfield("training_1"))
    agent = ExpectedSARSALambdaSmartAgent(
        learning_rate=alpha,
        discount_factor=gamma,
        exploration_rate=epsilon,
        lambda_=lambda_,
    )
    agent.set_goal(goal_position)
    agent.seed(42)
    np.random.seed(42)

    total_rewards = []
    total_steps = []
    successes = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        agent.reset()
        state = agent.discretize_state(obs)
        action = agent.act(obs)
        ep_reward = 0

        for step in range(max_steps):
            next_obs, reward, done, truncated, _ = env.step(action)
            next_state = agent.discretize_state(next_obs)
            next_action = agent.act(next_obs)

            shaped_r = custom_reward(obs, next_obs, reward, done, goal_position)
            agent.learn(state, action, shaped_r, next_state)

            obs = next_obs
            state = next_state
            action = next_action
            ep_reward += reward

            if done or truncated:
                break

        total_rewards.append(ep_reward)
        total_steps.append(step + 1)
        successes.append(done)

    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)
    success_rate = 100 * sum(successes) / len(successes)

    results.append({
        "alpha": alpha,
        "lambda": lambda_,
        "epsilon": epsilon,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "success_rate": success_rate
    })

    # ✅ Sauvegarde de l’agent entraîné dans un fichier Python avec sa Q-table
    name = f"expected_sarsa_a{int(alpha*100)}_l{int(lambda_*100)}_e{int(epsilon*100)}"
    save_qlearning_agent(
        agent=agent,
        output_path=f"src/agents/{name}_trained.py"
    )

# 💾 Sauvegarde des résultats
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/grid_search_expected_sarsa.csv", index=False)

print("\n✅ Grid Search Complete. Résultats enregistrés dans outputs/grid_search_expected_sarsa.csv")
print(df_results.sort_values(by="success_rate", ascending=False))
