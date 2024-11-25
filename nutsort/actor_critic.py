import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, num_tubes):
        super(ActorCriticNetwork, self).__init__()
        hidden_size = 128
        self.output_size = num_tubes * num_tubes
        
        # Shared layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size, self.output_size)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # Shared layers
        x = self.fc1(x)
        x = self.ln1(x)
        x = torch.relu(x)
        
        x = self.fc2(x)
        x = self.ln2(x)
        x = torch.relu(x)
        
        # Actor: output action logits
        action_logits = self.actor(x)
        
        # Critic: output state value
        state_value = self.critic(x)
        
        return action_logits, state_value

class ActorCriticAgent:
    def __init__(self, game, network, optimizer, gamma=0.99):
        self.game = game
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state):
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            state_tensor = (state_tensor + 1) / (self.game.n_colors + 1)
            
            # Get both action logits and state value
            action_logits, state_value = self.network(state_tensor)
            
            # Handle action selection same as before
            valid_moves = self.game.get_valid_moves()
            action_mask = torch.full_like(action_logits, float('-inf'))
            
            valid_move_indices = [
                from_tube * self.game.num_tubes + to_tube 
                for from_tube, to_tube in valid_moves
            ]
            action_mask[0, valid_move_indices] = 0
            
            masked_logits = action_logits + action_mask
            action_probs = torch.softmax(masked_logits, dim=-1)
            
            m = torch.distributions.Categorical(action_probs[0])
            action_idx = m.sample()
            log_prob = m.log_prob(action_idx)
            
            from_tube = action_idx.item() // self.game.num_tubes
            to_tube = action_idx.item() % self.game.num_tubes
            
        self.network.train()
        return log_prob, (from_tube, to_tube), state_value

    def get_episode(self):
        state = self.game.reset()
        done = False
        total_reward = 0

        saved_log_probs = []
        values = []
        rewards = []
        
        while not done:
            # Now also get state value
            log_prob, action, value = self.select_action(state)
            saved_log_probs.append(log_prob)
            values.append(value)
            
            next_state, reward, done = self.game.step(action)
            rewards.append(reward)
            
            state = next_state
            total_reward += reward

        return saved_log_probs, values, rewards, total_reward

    def calculate_returns_and_advantages(self, rewards, values):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Convert values list to tensor
        values = torch.cat(values)
        
        # Calculate advantages
        advantages = returns - values.squeeze()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages

    def calculate_actor_critic_loss(self, saved_log_probs, values, rewards):
        returns, advantages = self.calculate_returns_and_advantages(rewards, values)
        
        # Actor loss
        actor_loss = 0
        for log_prob, advantage in zip(saved_log_probs, advantages):
            actor_loss += -log_prob * advantage.detach()  # Detach advantages
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Combined loss (you can adjust these weights)
        total_loss = actor_loss + 0.5 * critic_loss
        
        return total_loss

    def train(self, episodes=1000, eval_interval=50):
        batch_size = 32
        self.network.train()
        
        total_loss = 0
        eval_rewards = []
        eval_episodes = []
        
        for episode in range(episodes):
            # Get episode experience
            saved_log_probs, values, rewards, total_reward = self.get_episode()
            print(f"Episode {episode + 1:3} finished with total reward: {total_reward:.2f}")
            
            # Calculate loss
            episode_loss = self.calculate_actor_critic_loss(saved_log_probs, 
                                                          torch.cat(values), 
                                                          rewards)
            total_loss += episode_loss

            # Update network
            if (episode + 1) % batch_size == 0:
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
                total_loss = 0
                
            # Evaluate
            if (episode + 1) % eval_interval == 0:
                eval_reward = self.evaluate_policy(num_episodes=5)
                eval_rewards.append(eval_reward)
                eval_episodes.append(episode + 1)
                print(f"Evaluation at episode {episode + 1}: Average reward = {eval_reward:.2f}")