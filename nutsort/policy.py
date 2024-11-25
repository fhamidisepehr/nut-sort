import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt
    

# class PolicyNetwork(nn.Module):
#     def __init__(self, input_size, num_tubes):
#         super(PolicyNetwork, self).__init__()
#         hidden_size = 128
#         self.output_size = num_tubes * num_tubes
        
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.bn1 = nn.LayerNorm(hidden_size)
        
#         self.fc2 = nn.Linear(hidden_size, hidden_size)
#         self.bn2 = nn.LayerNorm(hidden_size)
        
#         self.fc3 = nn.Linear(hidden_size, self.output_size)
    
#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = torch.relu(x)
        
#         x = self.fc2(x)
#         x = self.bn2(x)
#         x = torch.relu(x)
        
#         logits = self.fc3(x)
#         return logits

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, num_tubes):
        super().__init__()
        hidden_size = 256
        self.output_size = num_tubes * num_tubes
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size)
            ) for _ in range(3)
        ])
        
        self.output_layer = nn.Linear(hidden_size, self.output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for residual_block in self.residual_blocks:
            identity = x
            x = residual_block(x)
            x = F.relu(x + identity)
            
        return self.output_layer(x)

class PolicyGradientAgent:
    def __init__(self, game, policy_network, optimizer, gamma=0.99):
        self.game = game
        self.policy_network = policy_network
        self.optimizer = optimizer
        self.gamma = gamma

    def get_episode(self,episode_number):
        state = self.game.reset(episode_number)  # Start a new episode
        done = False
        total_reward = 0

        # Reset saved log probabilities and rewards for this episode
        saved_log_probs = []
        rewards = []
        num_step = 0
        history = []
        while not done:
            # Select an action based on the current state
            log_prob, action = self.select_action(state)
            saved_log_probs.append(log_prob)  # Now log_prob is already a tensor
            
            history.append((self.game.tubes,))
            # Take the action and observe the result
            next_state, reward, done, is_won = self.game.step(action)
            history[-1] += (action, self.game.tubes)
            
            # Save the reward for this time step
            rewards.append(reward)
            
            # Update the state
            state = next_state
            
            total_reward += reward
            num_step += 1

        return saved_log_probs, rewards, total_reward, num_step, is_won, history

    def calculate_policy_loss(self, saved_log_probs, rewards):
        """
        Update the policy network using the REINFORCE algorithm.
        
        Args:
            saved_log_probs: List of log probabilities of selected actions
            rewards: List of rewards received at each step
        """
        # Convert rewards to tensor
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # Calculate discounted rewards
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns to reduce variance
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate policy loss
        policy_loss = 0
        for log_prob, R in zip(saved_log_probs, returns):
            policy_loss += -log_prob * R  # Negative because we want to maximize reward
        
        return policy_loss
        
    def select_action(self, state):
        """Select an action based on current state."""
        # Convert state to a tensor and normalize
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = (state_tensor + 1) / (self.game.n_colors + 1)  # Normalize to [0,1]
        
        # Get logits for all possible tube pairs
        action_logits = self.policy_network(state_tensor)
        
        # Get valid moves from the game
        valid_moves = self.game.get_valid_moves()
        
        # If no valid moves, this should be caught by the game's is_lost check
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Create a mask for valid actions (-inf for invalid actions)
        action_mask = torch.full_like(action_logits, float('-inf'))
        
        # Convert valid moves to indices and set mask to 0 for valid actions
        valid_move_indices = [
            from_tube * self.game.num_tubes + to_tube 
            for from_tube, to_tube in valid_moves
        ]
        action_mask[0, valid_move_indices] = 0
        
        # Apply mask and softmax to get probabilities
        masked_logits = action_logits + action_mask
        action_probs = torch.softmax(masked_logits, dim=-1)
        
        # Create distribution and sample
        m = torch.distributions.Categorical(action_probs[0])
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        
        # Convert action index back to (from_tube, to_tube)
        from_tube = action_idx.item() // self.game.num_tubes
        to_tube = action_idx.item() % self.game.num_tubes
        
        return log_prob, (from_tube, to_tube)

    def evaluate_policy(self, episode, num_episodes=64):
        """Evaluate the policy without gradient tracking."""
        total_rewards = []
        num_steps = []
        wins_count = 0
        self.policy_network.eval()
        with torch.no_grad():
            for episode_number in range(num_episodes):
                _, _, total_reward, num_step, is_won, history = self.get_episode(episode_number)
                wins_count += is_won
                total_rewards.append(total_reward)
                num_steps.append(num_step)
        self.policy_network.train()
        eval_reward = np.mean(total_rewards)
        print(f"Episode {episode + 1:4}: Average reward = {eval_reward:.2f}, # steps min, avg, max = {np.min(num_steps)}, {np.mean(num_steps)}, {np.max(num_steps)}, win pct: {wins_count / len(num_steps):.2f}")

    def train(self, episodes=1000):
        self.evaluate_policy(0)
        batch_size = eval_interval = 256 #32
        self.policy_network.train()  # Ensure training mode at start
        
        policy_loss = 0
        for episode in range(episodes):
            saved_log_probs, rewards, *_ = self.get_episode(episode)
            # print(f"Episode {episode + 1:3} finished with total reward: {total_reward:.2f}")
            
            # Calculate the loss for this episode
            episode_loss = self.calculate_policy_loss(saved_log_probs, rewards)
            policy_loss += episode_loss

            if (episode + 1) % batch_size == 0:
                # Back-propagate the loss and update the network
                self.optimizer.zero_grad()
                policy_loss = policy_loss.mean()
                policy_loss.backward()
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), max_norm=1.0)
                # Update the network weights
                self.optimizer.step()
                # Reset the accumulated loss
                policy_loss = 0  

            # eval_rewards = []
            # eval_steps = []
            # Evaluate the policy periodically
            if ((episode + 1) % eval_interval == 0):
                self.evaluate_policy(episode)
                # eval_rewards.append(total_rewards)
                # eval_steps.append(num_steps)
                # print(f"Evaluation at episode {episode + 1}: Average reward = {eval_rewards[-1]:.2f}, #  steps min, avg, max = {np.min(eval_steps[-1])}, {np.mean(eval_steps[-1])}, {np.max(eval_steps[-1])}")
        
        # self.plot_evaluation(eval_rewards)
        
    # def plot_evaluation(self, eval_rewards):
    #     plt.figure(figsize=(10, 6))
    #     plt.plot(list(range(len(eval_rewards))), eval_rewards, 'b-', marker='o')
    #     # plt.xlabel('Episodes')
    #     plt.ylabel('Average Evaluation Reward')
    #     plt.title('Training Progress')
    #     plt.grid(True)
    #     plt.show()

    def update_policy_old(self, saved_log_probs, rewards):
        # Compute the returns (discounted rewards)
        # R = 0
        # policy_loss = []
        # for r in self.rewards[::-1]:  # iterate in reverse to compute discounted rewards
        #     R = r + self.gamma * R  # Apply discount factor
        #     policy_loss.append(R)
        
        # # Convert list of rewards into a tensor and make sure it's properly shaped
        # policy_loss = torch.tensor(policy_loss)

        policy_loss = torch.stack(saved_log_probs) * torch.tensor(rewards)
        # policy_loss = policy_loss.sum()  # Sum over the batch of episodes

        # # Ensure the policy loss is not scalar and handle the computation
        # policy_loss = policy_loss - policy_loss.mean()  # Normalize the rewards (optional)
        # policy_loss = policy_loss / (policy_loss.std() + 1e-5)  # Normalize by standard deviation

        # Now calculate the final loss using the log probabilities and rewards
        # Ensure log probabilities are stored as tensors
        #loss = torch.cat(self.saved_log_probs).sum() * policy_loss.sum()

        # Now calculate the final loss using the log probabilities and rewards
        saved_log_probs = torch.cat(saved_log_probs)  # Concatenate all the log probabilities
        loss = -(saved_log_probs * policy_loss).sum()  # Negative log likelihood times rewards

        # Ensure the loss is connected to the computation graph
        # if not loss.requires_grad:
        #     loss.requires_grad = True

        # Backpropagate and update the policy network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()