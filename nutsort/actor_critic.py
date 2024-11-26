import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 
import matplotlib.pyplot as plt


class ActorCriticNetwork(nn.Module):
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
        
        # Shared features before heads
        self.features = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_size, self.output_size)
        # Critic (value) head
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.input_layer(x)
        
        for residual_block in self.residual_blocks:
            identity = x
            x = residual_block(x)
            x = F.relu(x + identity)
            
        features = self.features(x)
        
        action_logits = self.actor(features)
        state_value = self.critic(features)
        
        return action_logits, state_value
    
class ActorCriticAgent:
    def __init__(self, game, network, optimizer, gamma=0.99):
        self.game = game
        self.network = network
        self.optimizer = optimizer
        self.gamma = gamma

    def select_action(self, state):
        """Select an action based on current state."""
        # Convert state to a tensor and normalize
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        state_tensor = (state_tensor + 1) / (self.game.n_colors + 1)
        
        # Get both action logits and value from NW
        action_logits, state_value = self.network(state_tensor)
        
        # Get valid moves from the game
        valid_moves = self.game.get_valid_moves()
        
        if not valid_moves:
            raise ValueError("No valid moves available")
        
        # Create mask for valid actions
        action_mask = torch.full_like(action_logits, float('-inf'))
        valid_move_indices = [
            from_tube * self.game.num_tubes + to_tube 
            for from_tube, to_tube in valid_moves
        ]
        action_mask[0, valid_move_indices] = 0
        
        # Apply mask and get probabilities
        masked_logits = action_logits + action_mask
        action_probs = torch.softmax(masked_logits, dim=-1)
        
        # Sample action
        m = torch.distributions.Categorical(action_probs[0])
        action_idx = m.sample()
        log_prob = m.log_prob(action_idx)
        
        # Convert action index back to tube pairs
        from_tube = action_idx.item() // self.game.num_tubes
        to_tube = action_idx.item() % self.game.num_tubes
        
        return log_prob, (from_tube, to_tube), state_value  # Added state_value to return


    def get_episode(self, episode_number):
        state = self.game.reset(episode_number)  # Start a new episode
        done = False
        total_reward = 0

        # Reset saved log probabilities and rewards for this episode
        saved_log_probs = []
        values = []          # Add: store state values
        next_values = []     # Add: store next state values
        rewards = []
        num_step = 0
        history = []

        while not done:
            # Select an action based on the current state
            log_prob, action, value = self.select_action(state)  # Modified: now returns value too
            saved_log_probs.append(log_prob)
            values.append(value)        # Add: save current state value
            
            history.append((self.game.tubes,))
            # Take the action and observe the result
            next_state, reward, done, is_won = self.game.step(action)
            history[-1] += (action, self.game.tubes)
            
            # Get next state value (0 if terminal)
            if done:
                next_value = torch.zeros_like(value)
            else:
                with torch.no_grad():
                    state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
                    state_tensor = (state_tensor + 1) / (self.game.n_colors + 1)
                    _, next_value = self.network(state_tensor)
            next_values.append(next_value)  # Add: save next state value
            
            # Save the reward for this time step
            rewards.append(reward)
            
            # Update the state
            state = next_state
            
            total_reward += reward
            num_step += 1

        # Return everything including new TD learning components
        return saved_log_probs, values, next_values, rewards, total_reward, num_step, is_won, history
    
    def get_episode_MonteCarlo(self, episode_number):
        state = self.game.reset(episode_number)
        done = False
        total_reward = 0
        
        saved_log_probs = []
        values = []
        rewards = []
        num_step = 0
        history = []
        
        while not done:
            log_prob, action, value = self.select_action(state)
            saved_log_probs.append(log_prob)
            values.append(value)
            
            history.append((self.game.tubes,))
            next_state, reward, done, is_won = self.game.step(action)
            history[-1] += (action, self.game.tubes)
            
            rewards.append(reward)
            state = next_state
            total_reward += reward
            num_step += 1
            
        return saved_log_probs, values, rewards, total_reward, num_step, is_won, history

    
    def calculate_actor_critic_loss_MonteCarlo(self, saved_log_probs, values, rewards):
        rewards = torch.tensor(rewards, dtype=torch.float32)
        values = torch.cat(values)
        
        # Calculate returns and advantages
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate advantages (returns - values)
        advantages = returns - values.squeeze()  # A(s,a) = R - V(s)
        
        # Policy (actor) loss
        actor_loss = 0
        for log_prob, advantage in zip(saved_log_probs, advantages):
            actor_loss += -log_prob * advantage.detach()  # Detach advantage
            
        # Value (critic) loss
        critic_loss = F.mse_loss(values.squeeze(), returns)
        
        # Total loss (you can adjust these weights)
        total_loss = actor_loss + 0.5 * critic_loss
        
        return total_loss

    def calculate_loss(self, saved_log_probs, values, rewards, next_values):
        # TD Learning
        values = torch.cat(values)
        next_values = torch.cat(next_values)
        rewards = torch.tensor(rewards)
        
        # Calculate TD targets and errors
        td_targets = rewards + self.gamma * next_values
        td_errors = td_targets - values.squeeze()
        
        # Actor loss using TD error as advantage
        actor_loss = sum(-log_prob * td_error.detach() 
                        for log_prob, td_error in zip(saved_log_probs, td_errors))
        
        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), td_targets.detach())
        
        return actor_loss + 0.5 * critic_loss

    # def calculate_actor_critic_loss(self, saved_log_probs, values, next_values, rewards):
    #     # Convert to tensors if not already
    #     values = torch.cat(values)
    #     next_values = torch.cat(next_values)
    #     rewards = torch.tensor(rewards, dtype=torch.float32)
        
    #     # TD Learning
    #     td_targets = rewards + self.gamma * next_values.squeeze()
    #     td_errors = td_targets - values.squeeze()
    #     normalized_td_errors = (td_errors - td_errors.mean()) / (td_errors.std() + 1e-8)
        
    #     # Actor loss
    #     actor_loss = sum(-log_prob * td_error.detach() 
    #                     for log_prob, td_error in zip(saved_log_probs, normalized_td_errors))
        
    #     # Critic loss
    #     critic_loss = F.mse_loss(values.squeeze(), td_targets.detach())
        
    #     # Combined loss
    #     total_loss = actor_loss + 0.5 * critic_loss
        
    #     return total_loss

    def evaluate_actor_critic(self, episode, num_episodes=64):
        """Evaluate the policy without gradient tracking."""
        total_rewards = []
        num_steps = []
        wins_count = 0
        self.network.eval()
        with torch.no_grad():
            for episode_number in range(num_episodes):
                # _, _, total_reward, num_step, is_won, history = self.get_episode(episode_number)
                _, _, _, _, total_reward, num_step, is_won, history = self.get_episode(episode_number)

                wins_count += is_won
                total_rewards.append(total_reward)
                num_steps.append(num_step)
        self.network.train()
        eval_reward = np.mean(total_rewards)
        print(f"Episode {episode + 1:4}: Average reward = {eval_reward:.2f}, # steps min, avg, max = {np.min(num_steps)}, {np.mean(num_steps)}, {np.max(num_steps)}, win pct: {wins_count / len(num_steps):.2f}")

    
    def calculate_actor_critic_loss(self, saved_log_probs, values, rewards, next_values):
        # Convert lists to tensors
        values = torch.cat(values)
        # next_values = torch.cat(next_values[1:] + [torch.zeros(1)])  # Zero for terminal state
        # next_values = torch.cat(next_values)
        next_values = torch.cat([v if isinstance(v, torch.Tensor) else torch.tensor([v]) for v in next_values])
        rewards = torch.tensor(rewards, dtype=torch.float32)
        
        # TD targets: r + gamma * V(s') - V(s)
        td_targets = rewards + self.gamma * next_values.squeeze()
        td_errors = td_targets - values.squeeze()
        
        # Normalize TD errors
        td_errors = (td_errors - td_errors.mean()) / (td_errors.std() + 1e-8)
        
        # Actor (policy) loss using TD error as advantage
        actor_loss = sum(-log_prob * td_error.detach() 
                        for log_prob, td_error in zip(saved_log_probs, td_errors))
        
        # Critic (value) loss using TD targets
        critic_loss = F.mse_loss(values.squeeze(), td_targets.detach())
    
        return actor_loss + 0.5 * critic_loss  #combined  loss
    
    # Then your training loop becomes cleaner:
    def train(self, episodes=1000):
        self.evaluate_actor_critic(0)
        batch_size = eval_interval = 256
        self.network.train()
        
        total_loss = 0
        for episode in range(episodes):
            saved_log_probs, values, next_values, rewards, total_reward, num_step, is_won, history = self.get_episode(episode)

            episode_loss = self.calculate_actor_critic_loss(
                saved_log_probs, values, next_values, rewards
            )
            total_loss += episode_loss
            
            if (episode + 1) % batch_size == 0:
                total_loss = total_loss / batch_size
                self.optimizer.zero_grad()
                total_loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                # torch.nn.utils.clip_grad_value_(self.network.parameters())
                # Print gradient statistics before clipping
                grad_stats = {}
                for name, param in self.network.named_parameters():
                    if param.grad is not None:
                        grad_stats[name] = {
                            'mean': param.grad.abs().mean().item(),
                            'max': param.grad.abs().max().item(),
                            'min': param.grad.abs().min().item()
                        }
                        print(f"{name}: mean={grad_stats[name]['mean']:.2e}, max={grad_stats[name]['max']:.2e}")
                
                # # Try different clipping values
                # clip_value = 0.5  # Start with this value
                # torch.nn.utils.clip_grad_value_(self.network.parameters(), clip_value)
                
                # Clip critic gradients (larger values)
                critic_params = [p for n, p in self.network.named_parameters() if 'critic' in n]
                torch.nn.utils.clip_grad_value_(critic_params, clip_value=1.0)

                # Clip actor gradients (medium values)
                actor_params = [p for n, p in self.network.named_parameters() if 'actor' in n]
                torch.nn.utils.clip_grad_value_(actor_params, clip_value=0.5)

                # Clip other network gradients (smaller values as they have smaller gradients)
                other_params = [p for n, p in self.network.named_parameters() 
                            if 'actor' not in n and 'critic' not in n]
                torch.nn.utils.clip_grad_value_(other_params, clip_value=0.3)
                
                self.optimizer.step()
                total_loss = 0
                
            if (episode + 1) % eval_interval == 0:
                self.evaluate_actor_critic(episode)
                
    # def train_old(self, episodes=1000):
    #         self.evaluate_policy(0)
    #         batch_size = eval_interval = 256
    #         self.network.train()
            
    #         total_loss = 0
    #         for episode in range(episodes):
    #             saved_log_probs, values, rewards, *_ = self.get_episode(episode)
                
    #             # Calculate loss
    #             episode_loss = self.calculate_actor_critic_loss(saved_log_probs, values, rewards)
    #             total_loss += episode_loss

    #             if (episode + 1) % batch_size == 0:
    #                 self.optimizer.zero_grad()
    #                 total_loss = total_loss.mean()
    #                 total_loss.backward()
    #                 torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
    #                 self.optimizer.step()
    #                 total_loss = 0

    #             if (episode + 1) % eval_interval == 0:
    #                 self.evaluate_policy(episode)