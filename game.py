import  random
import numpy as np
from IPython.display import clear_output
import torch
import torch.nn as nn
import torch.optim as optim

class NutSortGame:
    def __init__(self, n_colors = 5, tube_capacity = 4, tubes = None):
        """
        Initialize the game board with the given number of tubes and the maximum capacity per tube.
        :param tubes: A list of lists representing the initial arrangement of nuts in each tube.
        :param tube_capacity: The maximum number of nuts that can be in each tube.
        """
        self.tube_capacity = tube_capacity
        self.n_colors = n_colors
        self.num_tubes = n_colors + 2
        self.tubes = tubes or self.init_tubes()
        self.done = False  # Track if the game is done
        self.state_space_size  =  self.num_tubes  * tube_capacity #not_used yet

    def reset(self):
        """Reset the game to the initial state."""
        self.done = False
        #return np.array(self.tubes).flatten()
        tubes_temp = self.init_tubes()
        state_not_flattened = self.pad_tubes(tubes_temp,self.tube_capacity)
        state_flattened = [nut for tube in state_not_flattened for nut in tube]
        #flattened_array = np.concatenate([np.array(tube) for tube in tubes_temp])
        return state_flattened

    def init_tubes(self):
        nuts = [
            color_id
            for color_id in range(self.n_colors)
            for _ in range(self.tube_capacity)
        ]
        random.shuffle(nuts)
        return [
            nuts[n: n + self.tube_capacity]
            for n in range(0, len(nuts), self.tube_capacity)
        ]  + [[], []]

    # def display_board(self):
    #     """Display the current state of the game board."""
    #     # clear_output(wait=False)
    #     colors = "ABCDEFGHIJKL"
    #     for i, tube in enumerate(self.tubes):
    #         print(f"Tube {i + 1}:", " ".join([colors[c] for c in tube]))
    #     print("\n")

    def display_board(self):
        """Display the current state of the game board with colored squares."""
        for i, tube in enumerate(self.tubes):
            tube_str = " ".join(self.get_colored_squares(tube))  # Create a colored string for the tube
            print(f"Tube {i + 1:2}: {tube_str}")
        print("\n")

    def get_colored_squares(self, tube):
        """
        Converts a tube of color indices into colored squares using ANSI escape sequences.
        :param tube: A list of integers representing the color indexes of the nuts.
        :return: A list of colored square strings.
        """
        return [self.color_square(nut) for nut in tube]

    def color_square(self, color_index):
        """
        Returns a string representing a colored square for the given color index.
        :param color_index: The index of the color.
        :return: A string with an ANSI code for colored square.
        """
        color = color_map.get(color_index, '0')  # Default to black if not found
        return f"\033[48;5;{color}m  \033[0m"  # ANSI background color escape sequence
    
    def is_valid_move(self, from_tube, to_tube):
        """
        Check if it's valid to move a nut from `from_tube` to `to_tube`.
        :param from_tube: Index of the source tube.
        :param to_tube: Index of the destination tube.
        :return: True if valid, otherwise False.
        """
        # Check if there's a nut to move
        if not self.tubes[from_tube]:
            return False

        # Check if the destination tube has space
        if len(self.tubes[to_tube]) >= self.tube_capacity:
            return False

        # If the destination tube is empty, any nut can be moved
        if not self.tubes[to_tube]:
            return True

        # Check if the top nut of the destination tube is the same color as the nut to move
        if self.tubes[from_tube][-1] == self.tubes[to_tube][-1]:
            return True
        
        return False

    def get_valid_moves(self):
        """
        Get a list of all valid moves (from_tube, to_tube) that can be made.
        :return: A list of tuples (from_tube, to_tube) representing valid moves.
        """
        valid_moves = []
        
        # Check all pairs of tubes
        for from_tube in range(self.num_tubes):
            for to_tube in range(self.num_tubes):
                # Skip if both tubes are the same
                if from_tube != to_tube and self.is_valid_move(from_tube, to_tube):
                    valid_moves.append((from_tube, to_tube))
        
        return valid_moves
    
    def is_split_move(self, from_tube, to_tube):
        how_many_nuts_to_move = self.how_many_nuts_to_move(from_tube, to_tube)
        from_nuts = self.tubes[from_tube]
        if len(from_nuts)  ==  how_many_nuts_to_move:
            return False
        return from_nuts[-how_many_nuts_to_move - 1] == from_nuts[-1]

    def how_many_nuts_to_move(self, from_tube, to_tube):
        color = self.tubes[from_tube][-1]
        for n in range(
            1, 
            min(
                self.tube_capacity - len(self.tubes[to_tube]),
                len(self.tubes[from_tube])
            ) + 1
        ):
            if self.tubes[from_tube][-n] != color:
                return n - 1
        return n

        # from_nuts = self.tubes[from_tube]
        # num_nuts = 0
        # can_move = True
        # while can_move:
        #     #nut = self.tubes[from_tube].pop()
        #     num_nuts += 1
        #     can_move = (from_nuts[-1] == from_nuts[-num_nuts])
        # num_nuts -= 1 
    
    def move_nut(self, from_tube, to_tube):
        """
        Move a nut from `from_tube` to `to_tube` if valid.
        :param from_tube: Index of the source tube.
        :param to_tube: Index of the destination tube.
        :return: True if move was made, otherwise False.
        """
        if not self.is_valid_move(from_tube, to_tube):
            return False
        
        n_nuts = self.how_many_nuts_to_move(from_tube, to_tube)
        self.tubes[to_tube] += [self.tubes[from_tube][-1]] * n_nuts
        self.tubes[from_tube] = self.tubes[from_tube][:-n_nuts]
        return True

    def is_won(self):
        """
        Check if the game is won.
        :return: True if the game is won, otherwise False.
        """
        for tube in self.tubes:
            if len(tube) > 0 and len(set(tube)) != 1:
                return False
            if sum(not tube for tube in self.tubes) != 2:
                return False
        return True

    def is_lost(self):
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            return True
        if all(self.is_split_move(*move) for move in valid_moves):
            return True
        return False
    
    # def step(self, action):
    #     """Performs a move action and returns the new state and reward."""
    #     from_tube, to_tube = action
    #     self.move_nut(from_tube, to_tube)
        
    #     if self.is_won():
    #         self.done = True
    #         return np.array(self.tubes).flatten(), 1, self.done  # Reward for winning
        
    #     if self.is_lost():
    #         self.done = True
    #         return np.array(self.tubes).flatten(), -1, self.done  # Negative reward for losing
        
    #     return np.array(self.tubes).flatten(), 0, self.done  # No reward

    def get_padded_tube(self, nuts, tube_capacity):
        """
        Pads a tube to the specified capacity, filling with -1 if needed.
        
        :param nuts: List of nuts in the tube.
        :param tube_capacity: The maximum number of nuts a tube can hold.
        :return: A padded tube.
        """
        # Calculate the number of empty spaces needed
        num_empty_spaces = tube_capacity - len(nuts)
        # Pad with -1 if there are empty spaces
        padded_tube = nuts + [-1] * num_empty_spaces
        return padded_tube

    def pad_tubes(self, self_tubes, tube_capacity):
        """
        Pads each tube in self_tubes to the tube_capacity.

        :param self_tubes: List of tubes, each of which is a list of nuts.
        :param tube_capacity: Maximum number of nuts each tube can hold.
        :return: A list of padded tubes.
        """
        return [self.get_padded_tube(tube, tube_capacity) for tube in self_tubes]
    
    def step(self, action):
        """
        Takes an action (i, j), moves a nut from tube i to tube j, and returns:
        - next_state: the updated state after the action
        - reward: 1 if the game is won, -1 if the game is lost, 0 otherwise
        - done: a boolean indicating whether the game is over (either won or lost)
        """
        from_tube, to_tube = action

        # Perform the move if valid
        if self.move_nut(from_tube, to_tube):
            # If the move was successful, update the state
            #next_state = np.concatenate([np.array(tube) for tube in self.tubes])
            next_state_not_flattened = self.pad_tubes(self.tubes,self.tube_capacity)
            next_state = [nut for tube in next_state_not_flattened for nut in tube]

            # Check if the game is won
            if self.is_won():
                self.done = True
                return next_state, 1, self.done  # Reward for winning

            # Check if the game is lost (no valid moves left)
            if self.is_lost():
                self.done = True
                return next_state, -1, self.done  # Negative reward for losing

            # Otherwise, no reward and the game is not done
            return next_state, 0, self.done
        else:
            # If the move was not valid, return the current state and no reward
            # return np.concatenate([np.array(tube) for tube in self.tubes]), 0, self.done
            next_state_not_flattened = self.pad_tubes(self.tubes,self.tube_capacity)
            return [nut for tube in next_state_not_flattened for nut in tube]

    
    def convert_2d_to_1d(self, ij):
        i, j = ij 
        return i * self.tube_capacity + j

    def play(self):
        """Main game loop."""
        while True: 
            self.display_board()
            
            try:
                from_to_tube = input("Comma-separated 1-based index from, to: ")
                from_tube, to_tube = from_to_tube.split(",")

                if not self.move_nut(int(from_tube) - 1, int(to_tube) - 1):
                    print("Invalid move. Try again.")
            except Exception as e:
                print(str(e))
                print("Invalid move. Try again.")
        
        if self.is_won():
            self.display_board()
            print("Congratulations, you've won the game!")
            return
        
        if self.is_lost():
            self.display_board()
            print("Sorry, you lost!")
            return


# This method will generate an entire episode for training
    def create_episode(self, policy_net):
        state = self.reset()
        episode_data = []
        
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            probs = policy_net(state_tensor)
            
            # Select an action based on the probability distribution (sampling from the action space)
            valid_moves = self.get_valid_moves()
            valid_probs = probs[0][valid_moves]
            action = random.choices(valid_moves, valid_probs.detach().numpy())[0]

            # Perform the action and get the new state and reward
            next_state, reward, done = self.step(action)
            
            episode_data.append((state, action, reward, next_state))
            state = next_state
        
        return episode_data
    
