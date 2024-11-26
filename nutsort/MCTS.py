from nutsort.game import NutSortGame
import math
import random
from copy import deepcopy

class MCTSNode:
    def __init__(self, game_state, parent=None, action=None, c=1.414):
        self.game_state = game_state
        self.parent = parent
        self.action = action  # (from_tube, to_tube)
        self.children = {}    # {action: node}
        self.visits = 0
        self.value = 0
        self.c = c
        self.untried_actions = list(self.game_state.get_valid_moves())

    def ucb_value(self, child):
        if child.visits == 0:
            return float('inf')
        exploitation = child.value / child.visits
        exploration = self.c * math.sqrt(math.log(self.visits) / child.visits)
        return exploitation + exploration

    def select_child(self):
        return max(self.children.items(),
                  key=lambda item: self.ucb_value(item[1]))[1]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = deepcopy(self.game_state)
        next_state.move_nut(*action)
        child = MCTSNode(next_state, parent=self, action=action, c=self.c)
        self.children[action] = child
        return child

    def is_terminal(self):
        return self.game_state.is_won() or self.game_state.is_lost()

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

class MCTS:
    def __init__(self, game, episode_number=0):
        self.original_game = game
        self.episode_number = episode_number
        game_state = deepcopy(game)
        game_state.reset(episode_number)  # Using your game's reset with episode_number
        self.root = MCTSNode(game_state)

    def search(self, num_simulations=1000):
        for _ in range(num_simulations):
            node = self._select()
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node)
            self._backpropagate(node, reward)

        # Return best action based on visit count
        return self._best_action()

    def _select(self):
        node = self.root
        while not node.is_terminal():
            if not node.is_fully_expanded():
                return node
            node = node.select_child()
        return node

    def _expand(self, node):
        return node.expand()

    def _simulate(self, node):
        """Simulation using your game's step method"""
        state = deepcopy(node.game_state)
        depth = 0
        max_depth = 100
        total_reward = 0
        
        while depth < max_depth:
            valid_moves = state.get_valid_moves()
            if not valid_moves:
                break
                
            # Prefer moves that don't split color groups
            good_moves = [move for move in valid_moves 
                         if not state.is_split_move(*move)]
            action = random.choice(good_moves if good_moves else valid_moves)
            
            # Using your game's step method
            _, reward, done, is_won = state.step(action)
            total_reward += reward
            
            if done:
                if is_won:
                    total_reward += 10  # Bonus for winning
                break
                
            depth += 1
        
        # Normalize reward to [-1, 1]
        return total_reward / max(depth, 1)

    def _backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _best_action(self):
        if not self.root.children:
            return random.choice(self.root.game_state.get_valid_moves())
        return max(self.root.children.items(),
                  key=lambda item: item[1].visits)[0]

def solve_with_mcts(game, max_moves=100, simulations_per_move=1000):
    """Play the game using MCTS"""
    episode_number = 0  # or pass this as parameter
    moves_history = []
    state = deepcopy(game)
    state.reset(episode_number)
    
    for move_num in range(max_moves):
        if state.is_won() or state.is_lost():
            break
            
        # Create MCTS instance for current state
        mcts = MCTS(state, episode_number)
        action = mcts.search(num_simulations=simulations_per_move)
        
        # Make the move
        moves_history.append(action)
        next_state, reward, done, is_won = state.step(action)
        print(f"Move {move_num + 1}: From tube {action[0]} to {action[1]}, Reward: {reward}")
        
        if done:
            print("Game finished!")
            print("Won!" if is_won else "Lost!")
            break
    
    return moves_history, state.is_won()



# Create game instance
game = NutSortGame(n_colors=4)  # Using your game class

# Solve using MCTS
solution_moves, is_solved = solve_with_mcts(game, simulations_per_move=1000)

if is_solved:
    print(f"Game solved in {len(solution_moves)} moves!")
    for i, (from_tube, to_tube) in enumerate(solution_moves, 1):
        print(f"Move {i}: Tube {from_tube} → Tube {to_tube}")
else:
    print("Failed to solve game")