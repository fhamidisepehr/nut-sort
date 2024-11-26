from nutsort.game import NutSortGame
import torch.optim as optim
from nutsort.actor_critic import ActorCriticAgent, ActorCriticNetwork
from sys import argv


def train_policy_gradient(n_colors, n_episodes):
    game = NutSortGame(n_colors)  # Assuming you have the game environment implemented
    input_size = game.state_space_size  # This will be the size of the flattened state (e.g., 12 if there are 12 tubes)
    #temp = np.array(game.tubes).flatten()
    output_size = game.num_tubes ** 2  # The number of valid actions

    policy_network = PolicyNetwork(input_size, game.num_tubes)
    optimizer = optim.Adam(policy_network.parameters(), lr=0.0000001)

    # Initialize the RL agent
    agent = PolicyGradientAgent(game, policy_network, optimizer)

    # Train the agent
    agent.train(episodes=n_episodes)


def train(n_colors, n_episodes):
    game = NutSortGame(n_colors)  # Assuming you have the game environment implemented
    input_size = game.state_space_size  # This will be the size of the flattened state (e.g., 12 if there are 12 tubes)
    #temp = np.array(game.tubes).flatten()
    output_size = game.num_tubes ** 2  # The number of valid actions

    action_critic_network = ActorCriticNetwork(input_size, game.num_tubes)
    optimizer = optim.Adam(action_critic_network.parameters(), lr=0.0000001)

    # Initialize the RL agent
    agent = ActorCriticAgent(game, action_critic_network, optimizer)

    # Train the agent
    agent.train(episodes=n_episodes)

if __name__ == "__main__":
    if len(argv) > 1:
        n_colors, n_episodes = argv[1:]
    else:
        n_colors, n_episodes = 5, 1_000
    train(int(n_colors), int(n_episodes))