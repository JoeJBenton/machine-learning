import numpy as np
import network as net
import tictactoe as game
import MCTS as MCTS

my_net = net.Network([18, 30, 30, 30, 10])
my_board = game.Board()

training_games = 1000
epochs = 30
c_puct = 5
MCTS_depth = 2

def cost(network_output, true_output):
    total_cost = 0
    for pi, p in zip(true_output[:-1], network_output[0]):
        total_cost -= pi*np.log(p)
    total_cost += (network_output[-1] - true_output[-1]) * \
                        (network_output[-1] - true_output[-1])
    return total_cost

def evaluate(network, test_data):
    total_cost = 0
    for (input, output) in test_data:
        network_output = network.feedforward(input)
        total_cost += cost(network_output, output)
    return total_cost/len(test_data)

def evaulate_value_function(network, test_data):
    total_cost = 0
    for (input, output) in test_data:
        network_output = network.feedforward(input)
        #print("Network output: %s" % network_output[-1])
        #print("Expected output: %s" % output[-1])
        total_cost += (network_output[-1] - output[-1]) * \
                            (network_output[-1] - output[-1])
    #raw_input()
    return total_cost/len(test_data)

testing_data = []
for game_number in range(training_games):
    display = False
    if game_number == 0:
        display = False
    training_game_output = MCTS.play_training_game(my_net, c_puct, MCTS_depth, display)
    for (state, policy, reward) in training_game_output:
        input = state.network_input()
        output = []
        for move in game.move_list:
            if move in policy.keys():
                output.append([policy[move]])
            else:
                output.append([0])
        output.append([reward])
        testing_data.append((input, np.array(output)))

for epoch in range(epochs):
    training_data = []
    for game_number in range(training_games):
        display = False
        if game_number == 0:
            display = False
        training_game_output = MCTS.play_training_game(my_net, c_puct, MCTS_depth, display)
        for (state, policy, reward) in training_game_output:
            input = state.network_input()
            output = []
            for move in game.move_list:
                if move in policy.keys():
                    output.append([policy[move]])
                else:
                    output.append([0])
            output.append([reward])
            training_data.append((input, np.array(output)))



    #print(evaluate(my_net, training_data))
    print(evaulate_value_function(my_net, testing_data))
    my_net.SGD(training_data, 3, 300, 0.02, True)
    #print(evaluate(my_net, training_data))
    print(evaulate_value_function(my_net, testing_data))

#MCTS.play_training_game(my_net, c_puct, MCTS_depth, True)
#MCTS.play_training_game(my_net, c_puct, MCTS_depth, True)
