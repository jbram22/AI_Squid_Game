import numpy as np
import random
import time
import sys
import os 
from BaseAI import BaseAI
from Grid import Grid
import math

# TO BE IMPLEMENTED
#

def manhattan_dist(position, target):
    x_dist = abs(position[0]-target[0])
    y_dist = abs(position[1]-target[1])
    return x_dist + y_dist

class PlayerAI(BaseAI):

    def __init__(self) -> None:
        # You may choose to add attributes to your player - up to you!
        super().__init__()
        self.pos = None
        self.player_num = None
        self.searchDepth = 0
    
    def getPosition(self):
        return self.pos

    def setPosition(self, new_position):
        self.pos = new_position 

    def getPlayerNum(self):
        return self.player_num

    def setPlayerNum(self, num):
        self.player_num = num

    def getMove(self, grid: Grid) -> tuple:
        """ 
        YOUR CODE GOES HERE

        The function should return a tuple of (x,y) coordinates to which the player moves.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Trap* actions, 
        taking into account the probabilities of them landing in the positions you believe they'd throw to.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.
        
        """

        """
        moves = grid.get_neighbors(self.pos, only_available = True)
        move = random.choice(moves)
        return move
        """
        self.searchDepth = 0
        child, util = self.maximize(grid, -100000000, 100000000)
        if child != None:
            move = child.find(self.player_num)
        else:
            move = random.choice(grid.get_neighbors(self.pos, only_available=True))
        return move

    def getTrap(self, grid: Grid) -> tuple:
        """
        The function should return a tuple of (x,y) coordinates to which the player *WANTS* to throw the trap.

        It should be the result of the ExpectiMinimax algorithm, maximizing over the Opponent's *Move* actions,
        taking into account the probabilities of it landing in the positions you want.

        Note that you are not required to account for the probabilities of it landing in a different cell.

        You may adjust the input variables as you wish (though it is not necessary). Output has to be (x,y) coordinates.

        """
        trap = (0, 0)  # Initialize the tuple coordinates of the trap.
        self.depth = 0  # Initialize our current depth.

        # Get the opponent's player number.
        opponent_num = 0  # Initialize opponent's number.
        if self.getPlayerNum() == 1:
            opponent_num = 2
        else:
            opponent_num = 1

        # Find available spaces around the opponent.
        opp_neighbors = grid.get_neighbors(grid.find(opponent_num), only_available=True)

        # Initialize children for parent node for ExpectMiniMax.
        first_children = []
        for cell in opp_neighbors:
            new_grid = grid.clone()
            new_grid.move(cell, opponent_num)
            first_children.append(new_grid)

        # Initialize root node.
        root = Node(utility=0, grid=grid, position=grid.find(opponent_num), parent=None, children=[])

        # Get the ideal move.
        (child, utility) = self.trap_expect_max(root, float('Inf'), float('Inf'), opponent_num)

        # If there are no children, go to some random possible move.
        if child != None:
            trap = child.position
        else:
            trap = random.choice(grid.get_neighbors(grid.find(3-self.player_num), only_available=True))

        return trap

    # maximize util based on where I can move to
    def maximize(self, state, alpha, beta):
        """return (maxChild, maxUtil), which is a board state and its utility"""
        if self.searchDepth == 5:
            self.searchDepth -= 1
            return (None, self.move_heuristic(state))

        maxChild = None
        maxUtil = -1000000

        moves = state.get_neighbors(state.find(self.player_num), only_available=True)
        neighbor_states = []
        for move in moves:
            neighbor_states.append(state.clone().move(move, self.player_num))

        for child in neighbor_states:
            self.searchDepth += 1
            util = self.minimize(child, alpha, beta)[1]

            if util > maxUtil:
                maxChild = child
                maxUtil = util

            if maxUtil >= beta:
                break

            if maxUtil > alpha:
                alpha = maxUtil

        self.searchDepth -= 1
        return (maxChild, maxUtil)

    # minimize my util by throwing to locations on the board
    def minimize(self, state, alpha, beta):
        """return (maxChild, maxUtil), which is a board state and its utility"""
        if self.searchDepth == 5:
            self.searchDepth -= 1
            return (None, self.move_heuristic(state))

        minChild = None
        minUtil = 1000000

        adjacent_spots = state.get_neighbors(state.find(self.player_num), only_available=True)
        neighbor_states = []
        for spot in adjacent_spots:
            neighbor_states.append(state.clone().trap(spot))

        i = 0
        for child in neighbor_states:
            self.searchDepth += 1
            #util = self.maximize(child, alpha, beta)[1]
            # pass through state before trap and intended trap location
            util = self.chance(state, adjacent_spots[i], alpha, beta)
            i += 1

            if util < minUtil:
                minChild = child
                minUtil = util

            if minUtil <= alpha:
                break

            if minUtil < beta:
                beta = minUtil

        self.searchDepth -= 1
        return (minChild, minUtil)

    def chance(self, state, target, alpha, beta):
        all_neighbors = state.get_neighbors(target, only_available=False)
        possible_spots = []
        for each in all_neighbors:
            if each != self.getPosition() and each != state.find(3-self.player_num):
                possible_spots.append(each)

        n = len(possible_spots)
        a = manhattan_dist(state.find(3-self.player_num), target)
        p = 1-0.05*(a-1)
        expectedUtil = p * self.maximize(state.clone().trap(target), alpha, beta)[1]
        for spot in possible_spots:
            expectedUtil += ((1-p)/n) * self.heuristic(state.clone().trap(spot))

        return expectedUtil


    # terminate if we are three moves away, depth of 5
    def terminal_check(self, state):
        y_dist = abs(self.pos[0] - state.find(self.player_num)[0])
        x_dist = abs(self.pos[1] - state.find(self.player_num)[1])
        manhattan_dist = y_dist + x_dist
        moves_away = math.ceil(manhattan_dist/2)
        if moves_away >= 3:
            return True, moves_away
        else:
            return False, moves_away


    def heuristic(self, grid):
        my_neighbors = grid.get_neighbors(self.pos, only_available=True)
        opponent_position = grid.find(3-self.player_num)
        opponent_neighbors = grid.get_neighbors(opponent_position, only_available=True)
        val = len(my_neighbors) - len(opponent_neighbors)
        return val
        #return len(my_neighbors)

    def move_heuristic(self, grid):
        # Find out position, and our possible moves
        my_position = grid.find(self.player_num)
        my_neighbors = grid.get_neighbors(my_position, only_available=True)

        # Create Dictionary with possible move as key,
        # and number of open spaces surrounding that location as the value

        open_count = {}
        for position in my_neighbors:
            # print(f'Moving to {position}')
            manuevers = 0

            for movement in grid.get_neighbors(position, only_available=True):
                # print(f'After moving to {movement}, could move to {grid.get_neighbors(movement, only_available = True)}')
                manuevers += len(grid.get_neighbors(movement, only_available=True))

            # print(f'total of {manuevers} places to go after moving to {position}')
            # print('\n')

            open_count[position] = manuevers

        # Find square with the maximum number of potential places for opponent to lay a trap

        move = max(open_count, key=open_count.get)

        return open_count[move]

    """ Monica's code begins here! """
    """ Code for throwing begins here. """
    """ Below, the "throw_heuristic", though, is actually Jason's code! """

    def throw_heuristic(self, grid):
        # Find out position, and our possible moves
        # print(f'Original Grid:')
        # grid.print_grid()
        # print('\n')
        opponent_position = grid.find(3 - self.player_num)
        opponent_neighbors = grid.get_neighbors((4, 2), only_available=True)

        # Create Dictionary with possible move as key,
        # and number of open spaces surrounding that location as the value
        open_count = {}
        for position in opponent_neighbors:
            # print(f'Placing Trap @ {position}')
            # consider placing trap here & analyze further
            new = grid.trap(position)
            # new.print_grid()

            manuevers = 0
            for movement in new.get_neighbors(opponent_position, only_available=True):
                manuevers += len(new.get_neighbors(movement, only_available=True))
                # print(f'could move to {movement}, where there are {len(new.get_neighbors(movement, only_available = True))} more places to move')
            new.map[position] = 0
            # print(manuevers)
            # print('\n')
            open_count[position] = manuevers

        # Find square which, when we throw a trap in it, will lead to the smallest
        # possible area to which the opponent can move
        throw = min(open_count, key=open_count.get)
        if sum(value == throw for value in open_count.values()) > 0:
            distances = []

        return open_count[throw]

    def trap_getChildren(self, isNode, node, opponent_num):

        if isNode is True:
            the_grid = node.grid
        else:
            the_grid = node

        children = []

        # Get the children by going through possible moves.
        neighbors = the_grid.get_neighbors(the_grid.find(opponent_num), only_available=True)

        if len(neighbors) <= 1:
            return children

        for cell in neighbors:
            new_grid = the_grid.clone()
            new_grid.move(cell, opponent_num)
            new_utility = self.throw_heuristic(new_grid)
            new_node = Node(utility=new_utility, grid=new_grid, position=cell, parent=node, children=[])
            children.append(new_node)

        return children

    # Expected minimize function for getTrap().
    def trap_minimize(self, node, alpha, beta, opponent_num, isNode):

        # A version of a terminal test.
        if (self.depth == 5):
            return (None, self.throw_heuristic(node.grid))

        children = self.trap_getChildren(isNode, node, opponent_num)

        # Initialize what we are going to return.
        (minChild, minUtility) = (None, float('Inf'))

        # If only one valid move, pick it.
        if len(children) == 1:
            minChild = children[0]
            return (minChild, minUtility)

        # Go through and find min.
        for child in children:
            (_, utility) = self.trap_expect_max(child, alpha, beta, opponent_num)
            if (utility < minUtility):
                (minChild, minUtility) = (child, utility)
            if (minUtility <= alpha):
                break
            if (minUtility < beta):
                alpha = minUtility

                # Increment depth of search.
        if (len(children)) > 0:
            self.depth += 1

        return (minChild, minUtility)

    # Get expected value of tossing a trap.
    def trap_chance(self, cur_grid, target_xy, alpha, beta, opponent_num):

        # 9 spots around the current position.
        neighbors = cur_grid.get_neighbors(target_xy, only_available=True)
        neighbors = [neighbor for neighbor in neighbors if cur_grid.getCellValue(neighbor) <= 0]
        n = len(neighbors)

        # Probability of success.
        p_success = 1 - 0.05 * (manhattan_dist(self.getPosition(), target_xy) - 1)

        expectedUtil = p_success * \
                       self.trap_minimize(cur_grid.clone().trap(target_xy), alpha, beta, opponent_num, False)[1]
        for neighbor in neighbors:
            expectedUtil += (1 - p_success) / n * self.heuristic(cur_grid.clone().trap(neighbor))

        return expectedUtil

    # Expected maximize function for getTrap().
    def trap_expect_max(self, node, alpha, beta, opponent_num):

        # A version of a terminal test.
        if (self.depth == 5):
            return (None, self.throw_heuristic(node.grid))

        # Get children of the node we are maximizing.
        children = self.trap_getChildren(True, node, opponent_num)

        # Initialize what we are going to return.
        (maxChild, maxUtility) = (None, float('-Inf'))

        # If only one valid move, pick it.
        if len(children) == 1:
            maxChild = children[0]
            return (maxChild, maxUtility)

        # Go through and find max.
        for child in children:
            expectedUtility = self.trap_chance(node.grid, child.position, alpha, beta, opponent_num)
            if (expectedUtility > maxUtility):
                (maxChild, maxUtility) = (child, expectedUtility)
            if (maxUtility >= beta):
                break
            if (maxUtility > alpha):
                alpha = maxUtility

                # Increment depth of search.
        if (len(children)) > 0:
            self.depth += 1

        return (maxChild, maxUtility)

    # Termination check for getTrap.
    def trap_terminal(self):
        pass


# Node class for the ExpectMiniMax algorithm.
class Node:
    def __init__(self, utility, grid, position, parent=None, children=[]):
        self.utility = utility
        self.grid = grid
        self.position = position
        self.parent = parent
        self.children = children





        

