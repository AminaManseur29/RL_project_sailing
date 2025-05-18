import numpy as np
import pandas as pd
import time
import os
from src.env_sailing import SailingEnv
from src.initial_windfields import get_initial_windfield
from src.agents.my_agents.sarsa_agent import SARSAAgent1

# Grilles de recherche
alphas = [0.05, 0.1]
gammas = [0.9, 0.99]
epsilons = [0.3, 0.9]

# Paramètres d'entraînement
episode_options = [600]  # tu peux ajouter d'autres longueurs
max_steps = 1000
epsilon_decay = 0.98
min_epsilon = 0.05

# Dossier de sortie
os.makedirs("grid_results", exist_ok=True)

for num_episodes in episode_options:
    results = []

    for alpha in alphas:
        for gamma in gammas:
            for epsilon in epsilons:
                print(f"\n🔧 Testing α={alpha}, γ={gamma}, ε={epsilon}, episodes={num_episodes}")

                agent = SARSAAgent1(
                    learning_rate=alpha, discount_factor=gamma, exploration_rate=epsilon
                )
                agent.seed(42)
                np.random.seed(42)

                env = SailingEnv(**get_initial_windfield("training_1"))

                rewards_history = []
                success_history = []

                start_time = time.time()

                for episode in range(num_episodes):
                    obs, info = env.reset(seed=episode)
                    state = agent.discretize_state(obs)
                    action = agent.act(obs)
                    total_reward = 0

                    for step in range(max_steps):
                        next_obs, reward, done, truncated, info = env.step(action)
                        next_state = agent.discretize_state(next_obs)
                        next_action = agent.act(next_obs)

                        agent.learn(state, action, reward, next_state, next_action)

                        state = next_state
                        action = next_action
                        obs = next_obs
                        total_reward += reward

                        if done or truncated:
                            break

                    rewards_history.append(total_reward)
                    success_history.append(done)

                    agent.exploration_rate = max(
                        min_epsilon, agent.exploration_rate * epsilon_decay
                    )

                duration = time.time() - start_time
                success_rate = sum(success_history) / len(success_history) * 100

                results.append({
                    "learning_rate": alpha,
                    "discount_factor": gamma,
                    "epsilon_start": epsilon,
                    "epsilon_final": round(agent.exploration_rate, 3),
                    "episodes": num_episodes,
                    "avg_reward": np.mean(rewards_history),
                    "success_rate": success_rate,
                    "time_sec": round(duration, 2),
                    "q_table_size": len(agent.q_table)
                })

    # Sauvegarde CSV
    df = pd.DataFrame(results)
    filename = f"grid_results/sarsa_grid_search_{num_episodes}_episodes.csv"
    df.to_csv(filename, index=False)
    print(f"\n✅ Résultats sauvegardés dans : {filename}")
