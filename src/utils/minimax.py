from utils.board import Board
from copy import deepcopy
import numpy as np
import random


def utility(board):
    return np.sum(board)

def heuristic(state): 
    i =  random.randint(0, 1)
    poss = [-1, 1]
    return poss[i]

def action(state, coord, player):
    sucessor = state.copy()

    if(player == 'min'): 
        sucessor[coord[0]][coord[1]] = -1
    elif(player == 'max'): 
        sucessor[coord[0]][coord[1]] = 1

    return sucessor

def result(state, coord, player): 
    return action(state, coord, player)

def min_max_alpha_beta_h(state, player, limit):
    if(player == "max"): 
        v = []
        actions = state.all_legal_moves(state.BLACK)
        actions = list(actions)
        for a in actions: 
            sucessor = result(state.board, a, player)
            temp_board = Board()
            temp_board.board = sucessor
            val = min_value_abh(temp_board, -1000, 1000, 0, limit)
            v.append(val)
        v = np.array(v)
        index = np.argmax(v)
        return actions[index]

    elif player == 'min': 
        v = []
        actions = state.all_legal_moves(state.WHITE)
        actions = list(actions)
        for a in actions.copy(): 
            sucessor = result(state.board, a, player)
            temp_board = Board()
            temp_board.board = sucessor
            val = max_value_abh(temp_board, -1000, 1000, 0, limit)
            v.append(val)
        v = np.array(v)
        index = np.argmin(v)
        return actions[index]
    
def max_value_abh(state, alpha, beta, deep, limit): 
    if(deep == limit): 
        return heuristic(state.board)
    elif(state.check_game_over()): 
        return utility(state.board)
    
    v = -1000000000

    for a in state.all_legal_moves(state.BLACK):   
        sucessor = result(state.board, a, 'max') 
        temp_board = Board()
        temp_board.board = sucessor
        eval_min = min_value_abh(temp_board, alpha, beta, deep + 1, limit)
        v=max(v,eval_min)
        alpha=max(alpha,v)
        if(alpha >= beta): break

    return v

def min_value_abh(state, alpha, beta, deep, limit):
    if(deep == limit): 
        return heuristic(state.board)
    elif(state.check_game_over()): 
        return utility(state.board)

    v = 1000000000
    for a in state.all_legal_moves(state.WHITE): 
        sucessor = result(state.board, a, 'min') 
        temp_board = Board()
        temp_board.board = sucessor
        eval_max = max_value_abh(temp_board, alpha, beta, deep + 1, limit) 
        v = min(v, eval_max)
        beta = min(beta, v)
        if(alpha >= beta): break 

    return v

def minimax(position: Board, depth: int, alpha: int, beta: int, isMaximizingPlayer: bool) -> int:
    if depth == 0 or position.check_game_over() is True:
        return position.evaluate_board()
    
    if isMaximizingPlayer:
        maxEval = float('-inf')
        legal_moves = position.all_legal_moves(Board.BLACK)
        for row, col in legal_moves:
            if position.board[row, col] == Board.EMPTY:

                position_deepcopy = deepcopy(position) 
                position_deepcopy.set_discs(row, col, Board.BLACK)

                opponents_moves = position_deepcopy.all_legal_moves(Board.WHITE)
                eval = minimax(position_deepcopy, depth - 1, alpha, beta, opponents_moves == set())
                maxEval = max(maxEval, eval)

                alpha = max(alpha, eval)
                if beta <= alpha:
                    break

        return maxEval

    # else minimizing player's turn
    minEval = float('+inf')
    legal_moves = position.all_legal_moves(Board.WHITE)
    for row, col in legal_moves:
        if position.board[row, col] == Board.EMPTY:

            position_deepcopy = deepcopy(position) 
            position_deepcopy.set_discs(row, col, Board.WHITE)

            opponents_moves = position_deepcopy.all_legal_moves(Board.BLACK)
            eval = minimax(position_deepcopy, depth - 1, alpha, beta, opponents_moves != set())
            minEval = min(minEval, eval)

            beta = min(beta, eval)
            if beta <= alpha:
                break

    return minEval

def find_best_move(position: Board) -> tuple[int, int]:
    # print(position.board)
    bestMove = (20, 20)
    bestEval = float('+inf')

    legal_moves = position.all_legal_moves(Board.WHITE)
    for row, col in legal_moves:
        if position.board[row, col] == Board.EMPTY:

            position_deepcopy = deepcopy(position) # create a deep copy of the board position
            position_deepcopy.set_discs(row, col, Board.WHITE)

            opponents_moves = position_deepcopy.all_legal_moves(Board.BLACK)
            currentEval = minimax(position_deepcopy, 3, float('-inf'), float('inf'), opponents_moves != set())

            if currentEval <= bestEval:
                bestMove = (row, col)
                bestEval = currentEval
    
    # print(bestMove)
    return bestMove