# Rock–Paper–Scissors Reinforcement Learning with Human-in-the-Loop
# Streamlit single-file application

import streamlit as st
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# 1. Problem Decomposition
# -----------------------------
# - Environment: 5-round RPS episode
# - Agent: tabular Q-learning
# - State: opponent last K moves (Markov order K)
# - Action: {rock, paper, scissors}
# - Reward: win +0.2, lose -0.2, draw 0
# - Human provides actions via UI buttons
# =============================

# =============================
# 2. Configuration
# =============================
ACTIONS = ["rock", "paper", "scissors"]
ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}

EMOJI = {
    "rock": "🪨",
    "paper": "📄",
    "scissors": "✂️",
}

EPISODE_LENGTH = 5
MARKOV_ORDER = 2   # how many past human moves define the state
ALPHA = 0.3        # learning rate
GAMMA = 0.9        # discount factor
EPSILON = 0.1      # exploration probability

# =============================
# 3. Utility Functions
# =============================

def rps_result(agent_action, human_action):
    if agent_action == human_action:
        return 0.0
    wins = {
        "rock": "scissors",
        "paper": "rock",
        "scissors": "paper",
    }
    return 0.2 if wins[agent_action] == human_action else -0.2


def init_q_table():
    # state = tuple of last MARKOV_ORDER human actions
    return {}


def get_q(q_table, state):
    if state not in q_table:
        q_table[state] = np.zeros(len(ACTIONS))
    return q_table[state]


def epsilon_greedy(q_values):
    if random.random() < EPSILON:
        return random.randint(0, len(ACTIONS) - 1)
    return int(np.argmax(q_values))


# =============================
# 4. Markov Model (Human Predictor)
# =============================

def update_markov(markov_counts, history):
    if len(history) < 2:
        return
    prev, curr = history[-2], history[-1]
    markov_counts[prev][curr] += 1


def markov_probs(markov_counts, last_action):
    counts = markov_counts[last_action]
    total = sum(counts.values())
    if total == 0:
        return {a: 1 / 3 for a in ACTIONS}
    return {a: counts[a] / total for a in ACTIONS}


# =============================
# 5. Streamlit State Init
# =============================
if "q_table" not in st.session_state:
    st.session_state.q_table = init_q_table()
    st.session_state.episode = 0
    st.session_state.round = 0
    st.session_state.human_score = 0
    st.session_state.agent_score = 0
    st.session_state.total_reward = 0.0
    st.session_state.human_history = []
    st.session_state.episode_rewards = []
    st.session_state.win_rates = []
    st.session_state.markov_counts = {
        a: {b: 0 for b in ACTIONS} for a in ACTIONS
    }

# =============================
# 6. UI Layout
# =============================
st.title("🤖 Rock–Paper–Scissors RL Agent")

st.markdown("""
**Agent**: Tabular Q-learning  
**State**: Last human moves (Markov order = 2)  
**Reward**: Win +0.2 | Lose -0.2 | Draw 0
""")


st.subheader(f"Episode {st.session_state.episode + 1} | Round {st.session_state.round + 1}/5")
st.write(f"**Score:** Human {st.session_state.human_score} : {st.session_state.agent_score} Agent")
if "last_agent_action" in st.session_state:
    st.markdown("### 🎯 Last Round Result")
    st.write(
        f"**Agent played:** {EMOJI[st.session_state.last_agent_action]} "
        f"{st.session_state.last_agent_action.title()}"
    )
    st.write(f"**Outcome:** {st.session_state.last_outcome}")


# =============================
# 7. Action Buttons
# =============================
cols = st.columns(3)
for i, action in enumerate(ACTIONS):
    if cols[i].button(f"{EMOJI[action]} {action.title()}"):
        # --- Agent chooses action ---
        state = tuple(st.session_state.human_history[-MARKOV_ORDER:])
        q_values = get_q(st.session_state.q_table, state)
        agent_action_id = epsilon_greedy(q_values)
        agent_action = ID_TO_ACTION[agent_action_id]

        # --- Environment step ---
        reward = rps_result(agent_action, action)

        # --- Store last outcome for UI ---
        st.session_state.last_agent_action = agent_action
        if reward > 0:
            st.session_state.last_outcome = "Agent Win"
        elif reward < 0:
            st.session_state.last_outcome = "Human Win"
        else:
            st.session_state.last_outcome = "Draw"

        # --- Update scores ---
        if reward > 0:
            st.session_state.agent_score += 1
        elif reward < 0:
            st.session_state.human_score += 1

        st.session_state.total_reward += reward

        # --- Q-learning update ---
        next_state = tuple((st.session_state.human_history + [action])[-MARKOV_ORDER:])
        next_q = get_q(st.session_state.q_table, next_state)

        q_values[agent_action_id] += ALPHA * (
            reward + GAMMA * np.max(next_q) - q_values[agent_action_id]
        )

        # --- Update histories ---
        st.session_state.human_history.append(action)
        update_markov(st.session_state.markov_counts, st.session_state.human_history)

        st.session_state.round += 1

        # --- End of episode ---
        if st.session_state.round == EPISODE_LENGTH:
            st.session_state.episode += 1
            st.session_state.round = 0
            st.session_state.episode_rewards.append(st.session_state.total_reward)

            win_rate = st.session_state.agent_score / max(
                1, st.session_state.agent_score + st.session_state.human_score
            )
            st.session_state.win_rates.append(win_rate)

            st.session_state.total_reward = 0.0
            st.session_state.human_score = 0
            st.session_state.agent_score = 0

        st.rerun()

# =============================
# 8. Markov Prediction Visualization
# =============================
if st.session_state.human_history:
    st.subheader("🔮 Markov Prediction (Human Next Move)")
    last = st.session_state.human_history[-1]
    probs = markov_probs(st.session_state.markov_counts, last)

    df = pd.DataFrame({"Action": probs.keys(), "Probability": probs.values()})
    st.bar_chart(df.set_index("Action"))


# =============================
# 9. Visualizations
# =============================
if st.session_state.episode_rewards:
    st.subheader("📈 Learning Dynamics")

    fig, ax = plt.subplots()
    ax.plot(st.session_state.win_rates)
    ax.set_title("Agent Win Rate per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Win Rate")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    ax.plot(st.session_state.episode_rewards)
    ax.set_title("Total Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    st.pyplot(fig)

