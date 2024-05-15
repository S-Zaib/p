import random
import time
import copy
import concurrent.futures

# Global Variable
player, opponent = 'x', 'o'
# Nodes Evaluated Counter
nodes_evaluated = 0
nodes_fully_evaluated = 0

def isMovesLeft(board) : 

	for i in range(3) : 
		for j in range(3) : 
			if (board[i][j] == '_') : 
				return True
	return False


def evaluate(b) : 
	# Checking for Rows for X or O victory. 
	for row in range(3) :	 
		if (b[row][0] == b[row][1] and b[row][1] == b[row][2]) :		 
			if (b[row][0] == player) : 
				return 10
			elif (b[row][0] == opponent) : 
				return -10

	# Checking for Columns for X or O victory. 
	for col in range(3) : 
		if (b[0][col] == b[1][col] and b[1][col] == b[2][col]) : 
		
			if (b[0][col] == player) : 
				return 10
			elif (b[0][col] == opponent) : 
				return -10

	# Checking for Diagonals for X or O victory. 
	if (b[0][0] == b[1][1] and b[1][1] == b[2][2]) : 
		if (b[0][0] == player) : 
			return 10
		elif (b[0][0] == opponent) : 
			return -10

	if (b[0][2] == b[1][1] and b[1][1] == b[2][0]) : 
		if (b[0][2] == player) : 
			return 10
		elif (b[0][2] == opponent) : 
			return -10

	# Else if none of them have won then return 0 
	return 0

def minimax(board, depth, isMax, alpha, beta, alpha_beta_pruning = False) : 
	global nodes_evaluated
	global nodes_fully_evaluated
	nodes_evaluated += 1

	score = evaluate(board) 

	# If Maximizer has won the game return his/her 
	# evaluated score 
	if (score == 10) : 
		return score 

	# If Minimizer has won the game return his/her 
	# evaluated score 
	if (score == -10) : 
		return score 

	# If there are no more moves and no winner then 
	# it is a tie 
	if (isMovesLeft(board) == False) : 
		return 0

	# If this maximizer's move 
	if (isMax) :	 
		best = -1000

		# Traverse all cells 
		for i in range(3) :		 
			for j in range(3) : 
			
				# Check if cell is empty 
				if (board[i][j]=='_') : 
				
					# Make the move 
					board[i][j] = player 

					# Call minimax recursively and choose 
					# the maximum value 
					best = max( best, minimax(board, 
											depth + 1, 
											not isMax, alpha, beta, alpha_beta_pruning) ) 
					if alpha_beta_pruning:
						alpha = max(alpha, best)

						if beta <= alpha:
							break
					# Undo the move 
					board[i][j] = '_'
		nodes_fully_evaluated += 1
		return best 

	# If this minimizer's move 
	else : 
		best = 1000

		# Traverse all cells 
		for i in range(3) :		 
			for j in range(3) : 
			
				# Check if cell is empty 
				if (board[i][j] == '_') : 
				
					# Make the move 
					board[i][j] = opponent 

					# Call minimax recursively and choose 
					# the minimum value 
					best = min(best, minimax(board, depth + 1, not isMax, alpha, beta, alpha_beta_pruning)) 
					if alpha_beta_pruning:
						beta = min(beta, best)
						if beta <= alpha:
							break
					# Undo the move 
					board[i][j] = '_'
		return best 
	
# This will return the best possible move for the player 
def findBestMove(board, is_alpha_beta_pruning = False, print_val=True) : 
	bestVal = -1000
	bestMove = (-1, -1) 

	# Traverse all cells, evaluate minimax function for 
	# all empty cells. And return the cell with optimal 
	# value. 
	for i in range(3) :	 
		for j in range(3) : 
		
			# Check if cell is empty 
			if (board[i][j] == '_') : 
			
				# Make the move 
				board[i][j] = player 

				# compute evaluation function for this 
				# move. 
				moveVal = minimax(board, 0, False, -1000, 1000, is_alpha_beta_pruning) 

				# Undo the move 
				board[i][j] = '_'

				# If the value of the current move is 
				# more than the best value, then update 
				# best/ 
				if (moveVal > bestVal) :				 
					bestMove = (i, j) 
					bestVal = moveVal 
	if print_val:
		print("The value of the best Move is :", bestVal) 
		print() 
	return bestMove 


def worker(board, i, j):
    # Make a copy of the board to avoid modifying the same board in different processes
    board_copy = copy.deepcopy(board)

    # Make the move
    board_copy[i][j] = player

    # Compute evaluation function for this move
    counter = [0]
    moveVal = minimax(board_copy, 0, False, -1000, 1000, counter)

    # Return the result
    return ((i, j), moveVal)

def findBestMoveParallel(board):
 
    bestVal = -1000
    bestMove = (-1, -1) 

    # Create a ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Use a list comprehension to create a list of futures
        futures = [executor.submit(worker, board, i, j) for i in range(3) for j in range(3) if board[i][j] == '_']

    # Find the best move
    for future in concurrent.futures.as_completed(futures):
        move, val = future.result()
        if val > bestVal:
            bestMove = move
            bestVal = val

    print("The value of the best Move is :", bestVal) 
    print() 
    return bestMove

# Heuristic function for non-terminal states
def heuristic(board):
    player_lines = 0
    opponent_lines = 0

    # Check rows and columns for potential winning lines
    for i in range(3):
        # Rows
        if board[i][0] == board[i][1] == board[i][2] != '_':
            if board[i][0] == player:
                player_lines += 1
            else:
                opponent_lines += 1
        # Columns
        if board[0][i] == board[1][i] == board[2][i] != '_':
            if board[0][i] == player:
                player_lines += 1
            else:
                opponent_lines += 1

    # Check diagonals for potential winning lines
    if board[0][0] == board[1][1] == board[2][2] != '_':
        if board[0][0] == player:
            player_lines += 1
        else:
            opponent_lines += 1
    if board[0][2] == board[1][1] == board[2][0] != '_':
        if board[0][2] == player:
            player_lines += 1
        else:
            opponent_lines += 1

    return player_lines - opponent_lines


# Update the minimax function to use the heuristic for non-terminal states
def minimax2(board, depth, isMax, alpha, beta, alpha_beta_pruning=False):
	global nodes_evaluated
	global nodes_fully_evaluated
	nodes_evaluated += 1

	score = evaluate(board)

	# If Maximizer has won the game or reached a terminal state, return the score
	if score == 10 or score == -10 or not isMovesLeft(board):
		return score

	# If this maximizer's move
	if isMax:
		best = -1000

		# Sort the available moves based on the heuristic value
		available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '_']
		available_moves.sort(key=lambda move: heuristic(make_move(board, move, player)), reverse=True)

		# Traverse the sorted moves
		for move in available_moves:
			i, j = move

			# Make the move
			make_move(board, move, player)

			# Call minimax recursively and choose the maximum value
			best = max(best, minimax2(board, depth + 1, not isMax, alpha, beta, alpha_beta_pruning))
			if alpha_beta_pruning:
				alpha = max(alpha, best)

				if beta <= alpha:
					break

			# Undo the move
			undo_move(board, move)

		nodes_fully_evaluated += 1
		return best

	# If this minimizer's move
	else:
		best = 1000

		# Sort the available moves based on the heuristic value
		available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '_']
		available_moves.sort(key=lambda move: heuristic(make_move(board, move, opponent)), reverse=True)

		# Traverse the sorted moves
		for move in available_moves:
			i, j = move

			# Make the move
			make_move(board, move, opponent)

			# Call minimax recursively and choose the minimum value
			best = min(best, minimax2(board, depth + 1, not isMax, alpha, beta, alpha_beta_pruning))
			if alpha_beta_pruning:
				beta = min(beta, best)
				if beta <= alpha:
					break

			# Undo the move
			undo_move(board, move)

		return best

# Helper function to make a move on the board
def make_move(board, move, symbol):
	i, j = move
	board[i][j] = symbol
	return board

# Helper function to undo a move on the board
def undo_move(board, move):
	i, j = move
	board[i][j] = '_'

# Update the findBestMove function to use the enhanced alpha-beta pruning
def findBestMove2(board, is_alpha_beta_pruning=False):
	bestVal = -1000
	bestMove = (-1, -1)

	# Sort the available moves based on the heuristic value
	available_moves = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '_']
	available_moves.sort(key=lambda move: heuristic(make_move(board, move, player)), reverse=True)

	# Traverse the sorted moves
	for move in available_moves:
		i, j = move

		# Make the move
		make_move(board, move, player)

		# Compute evaluation function for this move
		moveVal = minimax2(board, 0, False, -1000, 1000, is_alpha_beta_pruning)

		# Undo the move
		undo_move(board, move)

		# If the value of the current move is more than the best value, then update best
		if moveVal > bestVal:
			bestMove = move
			bestVal = moveVal

	print("The value of the best Move is:", bestVal)
	print()
	return bestMove

# Define a heuristic function for the weak player
def weak_player_heuristic(board):
    player_lines = 0
    opponent_lines = 0

    # Check rows and columns for potential winning lines
    for i in range(3):
        # Rows
        if board[i][0] == board[i][1] == board[i][2] != '_':
            if board[i][0] == player:
                player_lines += 1
            elif board[i][0] == opponent:
                opponent_lines += 1
        # Columns
        if board[0][i] == board[1][i] == board[2][i] != '_':
            if board[0][i] == player:
                player_lines += 1
            elif board[0][i] == opponent:
                opponent_lines += 1

    # Check diagonals for potential winning lines
    if board[0][0] == board[1][1] == board[2][2] != '_':
        if board[0][0] == player:
            player_lines += 1
        elif board[0][0] == opponent:
            opponent_lines += 1
    if board[0][2] == board[1][1] == board[2][0] != '_':
        if board[0][2] == player:
            player_lines += 1
        elif board[0][2] == opponent:
            opponent_lines += 1

    return player_lines - opponent_lines

# Updated findBestMove function for the weak player
def findBestMoveWeak(board):
    bestVal = -1000
    bestMove = (-1, -1)

    # Traverse all cells, evaluate the heuristic function for all empty cells.
    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                # Skip the reserved tile
                if (i, j) == reserved_tile:
                    continue

                # Make the move
                board[i][j] = player

                # Compute the heuristic value for this move
                moveVal = weak_player_heuristic(board)

                # Undo the move
                board[i][j] = '_'

                # If the value of the current move is more than the best value, update best
                if moveVal > bestVal:
                    bestMove = (i, j)
                    bestVal = moveVal

    return bestMove

# Updated runCodeWeakPlayer function to allow one-on-one gameplay
def runCodeWeakPlayer():
    global reserved_tile
    board = [['_' for _ in range(3)] for _ in range(3)]
    reserved_tile = random.choice([(i, j) for i in range(3) for j in range(3)])
    print("Weak player will not use tile:", reserved_tile)
    print('Initial Board:')
    for row in board:
        print(row)

    # Game loop
    while True:
        # Player's move
        while True:
            try:
                row, col = map(int, input('Enter your move (row col): ').split())
                if board[row][col] == '_':
                    board[row][col] = opponent
                    break
                else:
                    print('Invalid move. Try again.')
            except (ValueError, IndexError):
                print('Invalid input. Please enter row and col as integers.')

        print('\nYour Move:')
        for row in board:
            print(row)

        # Check if the player wins or the game is a tie
        score = evaluate(board)
        if score == -10:
            print('\nYou Win!')
            break
        elif score == 0 and not isMovesLeft(board):
            print('\nGame Tied!')
            break

        # AI's move
        move = findBestMoveWeak(board)
        board[move[0]][move[1]] = player
        print('\nAI Move:')
        for row in board:
            print(row)

        # Check if the AI wins or the game is a tie
        score = evaluate(board)
        if score == 10:
            print('\nAI Wins!')
            break
        elif score == 0 and not isMovesLeft(board):
            print('\nGame Tied!')
            break


def generate_random_board():
    symbols = ['x', 'o', '_']
    board = [['_' for _ in range(3)] for _ in range(3)] 
    moves = 0  
    while moves < 5:
        row, col = random.randint(0, 2), random.randint(0, 2)
        if board[row][col] == '_':
            board[row][col] = random.choice(symbols[:-1]) 
            moves += 1
	# check if game is over
    if evaluate(board) != 0 or not isMovesLeft(board):
        board = generate_random_board()
    return board

def runCode(alphabeta, board):
	bestMove = findBestMove(board, alphabeta) 
	print("The Optimal Move is :") 
	print("ROW:", bestMove[0], " COL:", bestMove[1]) 
	print("Nodes Evaluated: ", nodes_evaluated)
	print("Nodes Fully Evaluated: ", nodes_fully_evaluated)

def runCodeParallel(board):
	bestMove = findBestMoveParallel(board)
	print("The Optimal Move is :")
	print("ROW:", bestMove[0], " COL:", bestMove[1])
	print("Nodes Evaluated: ", nodes_evaluated)
	print("Nodes Fully Evaluated: ", nodes_fully_evaluated)

def runCodeHeuristic(alphabeta, board):
	bestMove = findBestMove2(board, alphabeta)
	print("The Optimal Move is :")
	print("ROW:", bestMove[0], " COL:", bestMove[1])
	print("Nodes Evaluated: ", nodes_evaluated)
	print("Nodes Fully Evaluated: ", nodes_fully_evaluated)

if __name__ == '__main__':
    board = generate_random_board()
    
    nodes_evaluated = 0
    nodes_fully_evaluated = 0
    board_copy = [row.copy() for row in board]
    print('Initial Board:')
    for row in board_copy:
    	print(row)
    
    print('---------------------------------------')
    print('MiniMax without Alpha-Beta Pruning')
    print('---------------------------------------')
    runCode(False, board_copy)

    nodes_evaluated = 0
    nodes_fully_evaluated = 0
    board_copy = [row.copy() for row in board]
    print('Initial Board:')
    for row in board_copy:
    	print(row)

    print('---------------------------------------')
    print('\nMiniMax with Alpha-Beta Pruning')
    print('---------------------------------------')
    start_time = time.time()
    runCode(True, board_copy)
    print("Time taken: ", time.time() - start_time)

    nodes_evaluated = 0
    nodes_fully_evaluated = 0
    board_copy = [row.copy() for row in board]
    print('Initial Board:')
    for row in board_copy:
    	print(row)

    print('---------------------------------------')
    print('Parallel MiniMax with Alpha-Beta Pruning')
    print('---------------------------------------')
    start_time = time.time()
    runCodeParallel(board_copy)
    print("Time taken: ", time.time() - start_time)

    nodes_evaluated = 0
    nodes_fully_evaluated = 0
    board_copy = [row.copy() for row in board]
    print('Initial Board:')
    for row in board_copy:
    	print(row)

    print('---------------------------------------')
    print('MiniMax with Alpha-Beta Pruning and Heuristic')
    print('---------------------------------------')
    start_time = time.time()
    runCodeHeuristic(True, board_copy)
    print("Time taken: ", time.time() - start_time)

    print('---------------------------------------')
    print('Weak Player')
    print('---------------------------------------')
    runCodeWeakPlayer()