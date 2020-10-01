import numpy as np
import tictactoe as game
import network as net
import copy

class MCTS_node (object):

    def __init__(self, state, network):
        # A vertex is initialised when it lies just below a visited vertex
        # so there is no need for it to have anything except state and value

        self.state = state
        self.network_value = network.value(state)
        self.visits = 0
        self.has_children = False

    def expand_children(self, network):
        # Expands a new layer of children below the current node

        self.has_children = True
        self.children = {}
        self.Q = {}
        self.N = {}
        self.moves = self.state.valid_moves()
        self.network_policy = network.policy(self.state)

        for move in self.moves:
            child_state = copy.deepcopy(self.state).move(move)
            new_child = MCTS_node(child_state, network)
            self.children[move] = new_child
            self.Q[move] = new_child.network_value
            self.N[move] = 0


def MCTS_tree_search(node, network, c_puct):
    '''
    Takes in a node and a network and performs one iteration of the search down
    from that node, including expanding the tree if necessary.
    '''

    #print("Performing search starting from:")
    #node.state.print_board()

    node.visits += 1

    if node.state.is_end():
        return -node.state.value()*node.state.to_move

    elif node.visits == 1:
        return -node.network_value

    else:
        #Expand another layer of the tree if necessary
        if node.has_children == False:
            node.expand_children(network)

        # Find the move with highest upper bound on value
        #print(node.state.print_board())
        max_U = -np.inf
        best_move = np.nan
        for move in node.moves:
            U = node.Q[move] + c_puct*node.network_policy[move]*np.sqrt(node.visits)/(1+node.N[move])
            #print("Move %s has value %s" %(move, U))
            if U > max_U:
                best_move = move
                max_U = U
        #print("\n")
        #print("Best move is", best_move)

        # Make the best move and iterate down the tree
        reward = MCTS_tree_search(node.children[best_move], network, c_puct)
        node.Q[best_move] = (node.N[best_move]*node.Q[best_move] + reward)/(node.N[best_move] + 1)
        node.N[best_move] += 1
        return -reward


def get_move(policy):
    rnd = np.random.random()
    #print(policy)
    for move in policy.iterkeys():
        #print(rnd)
        if rnd < policy[move]:
            return move
        else:
            rnd -= policy[move]
    return False


def play_training_game(network, c_puct, iterations, display=False):
    initial_state = game.Board()
    current_node = MCTS_node(initial_state, network)

    draft_training_data = []

    while current_node.state.is_end() == False:
        for iter in range(iterations):
            MCTS_tree_search(current_node, network, c_puct)

        improved_policy = {}
        for move in current_node.moves:
            improved_policy[move] = float(current_node.N[move])/(current_node.visits-1)
        draft_training_data.append((current_node.state, improved_policy, 0))

        move = get_move(improved_policy)
        current_node = current_node.children[move]

        if display:
            print("\n\nMOVING FORWARD")
            current_node.state.print_board()
            print("Network value estimate:%s" % current_node.network_value)
            game.print_policy(improved_policy)

    draft_training_data.append((current_node.state, {(0,0): 1}, 0))

    reward = current_node.state.value()
    training_data = []

    for (state, policy, draft_reward) in draft_training_data:
        training_data.append((state, policy, reward))
        reward = -reward
        #training_data[-1][0].print_board()
        #print(training_data[-1][2])
        #raw_input()

    return training_data


