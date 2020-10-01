import numpy as np
import network as net
import tictactoe as game
import MCTS as MCTS

my_net = net.Network([18, 9, 9, 10])
my_board = game.Board()

my_net.feedforward(my_board.network_input())

# training_data = MCTS.play_training_game(my_net, 1, 5)

#for (state, policy, reward) in training_data:
    #state.print_board()
    #print("Policy", policy)
    #print("Reward", reward)
    #print()
