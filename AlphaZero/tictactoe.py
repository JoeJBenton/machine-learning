import numpy as np 
import state

'''
class Move(state.Move):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Move at (%s, %s)" % (self.x, self.y)
'''

class Board(state.State):
    '''
    We denote X by 1, O by -1.
    '''

    def __init__(self):
        self.square = [[0 for x in range(3)] for y in range (3)]
        self.to_move = 1

    def move(self, move, to_move = None):
        if to_move == None:
            to_move = self.to_move

        if move[0] not in [0,1,2]:
            raise Exception("Invalid y value")
        elif move[1] not in [0,1,2]:
            raise Exception("Invalid x value")
        elif to_move not in [-1,1]:
            raise Exception("Invalid player")
        elif self.square[move[0]][move[1]] != 0:
            raise Exception("Playing in an opcupied square")
        else:
            self.square[move[0]][move[1]] = to_move
        self.to_move = -self.to_move
        return self

    def value(self):
        '''Determines whether the board has a winner. Returns:
        1 if X wins
        -1 if O wins
        0 if neither player wins
        Raises an exception if both players win'''
        X_win = False
        O_win = False
        for x in range(3):
            if self.square[x][0] == self.square[x][1] and self.square[x][1] == self.square[x][2] and self.square[x][0] in [-1,1]:
                if self.square[x][0] == 1:
                    X_win = True
                else:
                    O_win = True
        for y in range(3):
            if self.square[0][y] == self.square[1][y] and self.square[1][y] == self.square[2][y] and self.square[0][y] in [-1,1]:
                if self.square[0][y] == 1:
                    X_win = True
                else:
                    O_win = True
        if self.square[0][0] == self.square[1][1] and self.square[1][1] == self.square[2][2] and self.square[0][0] in [-1,1]:
            if self.square[0][0] == 1:
                X_win = True
            else:
                O_win = True
        if self.square[0][2] == self.square[1][1] and self.square[1][1] == self.square[2][0] and self.square[0][2] in [-1,1]:
            if self.square[0][2] == 1:
                X_win = True
            else:
                O_win = True
        if X_win and O_win:
            raise Error("Bad board")
        elif X_win:
            return 1
        elif O_win:
            return -1
        else:
            return 0

    def is_end(self):
        if self.value() == 0:
            for x in range(3):
                for y in range(3):
                    if self.square[x][y] == 0:
                        return False
            return True
        else:
            return True

    def network_input(self):
        input = []
        for x in range(3):
            for y in range(3):
                if self.square[x][y] == 1:
                    input += [[1], [0]]
                elif self.square[x][y] == -1:
                    input += [[0], [1]]
                else:
                    input += [[0], [0]]
        return np.array(input)

    def print_board(self):
        for x in range(3):
            output = ""
            for y in range(3):
                if self.square[x][y] == 1:
                    output += "X"
                elif self.square[x][y] == -1:
                    output += "O"
                else:
                    output += " "
                if y < 2:
                    output += "|"
            print(output)
            if x < 2:
                print("-----")

    def valid_moves(self):
        move_list = []
        for x in range(3):
            for y in range(3):
                if self.square[x][y] == 0:
                    move_list.append((x,y))
        return move_list

def print_policy(policy):
    for x in range(3):
        output = ""
        for y in range(3):
            if (x,y) in policy.keys():
                output += ("%.2f" % policy[(x,y)])
            else:
                output += "0.00"
            if y < 2:
                output += "|"
        print(output)
        if x < 2:
            print("--------------")


raw_move_list = [(0, 0), (0, 1), (0, 2), \
             (1, 0), (1, 1), (1, 2), \
             (2, 0), (2, 1), (2, 2)]
move_list = []
for (x, y) in raw_move_list:
    move_list.append((x, y))
