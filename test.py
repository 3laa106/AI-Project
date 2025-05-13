import numpy as np
import sys


class Gomoku:
    def __init__(self, size=15):
        # Validate board size
        if not isinstance(size, int):
            raise ValueError("Board size must be an integer")
        if size < 5:
            raise ValueError("Board size must be at least 5x5")
        if size > 25:
            raise ValueError("Board size must be 25x25 or smaller")

        self.size = size
        self.board = np.zeros((size, size), dtype=np.int8)  # Using int8 instead of int for memory efficiency
        self.current_player = 1  # Player 1 starts (1 for black, 2 for white)
        self.game_over = False
        self.winner = None
        self.frontier = set()
        self.frontier.add((size // 2, size // 2))
        self.move_history = []  # Store move history for efficient undo

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.frontier = set()
        self.frontier.add((self.size // 2, self.size // 2))
        self.move_history = []

    def make_move(self, row, col):
        if self.game_over or not self.is_valid_move(row, col):
            return False

        # Store state before move for potential undo
        self.move_history.append({
            'position': (row, col),
            'player': self.current_player,
            'frontier_before': self.frontier.copy(),
            'game_over': self.game_over,
            'winner': self.winner
        })

        self.board[row, col] = self.current_player

        # Update frontier - remove the current position
        self.frontier.discard((row, col))
        
        # Add surrounding empty cells to frontier
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = row + dx, col + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] == 0:
                    self.frontier.add((nx, ny))

        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif np.count_nonzero(self.board) == self.size * self.size:  # More efficient than checking every cell
            self.game_over = True
        else:
            self.current_player = 3 - self.current_player

        return True

    def undo_move(self):
        """Undo the last move efficiently"""
        if not self.move_history:
            return False
            
        last_move = self.move_history.pop()
        row, col = last_move['position']
        
        # Restore board state
        self.board[row, col] = 0
        
        # Restore other state variables
        self.current_player = last_move['player']
        self.frontier = last_move['frontier_before']
        self.game_over = last_move['game_over']
        self.winner = last_move['winner']
        
        return True

    def is_valid_move(self, row, col):
        return (0 <= row < self.size and 0 <= col < self.size
                and self.board[row, col] == 0)

    def check_win(self, row, col):
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)],  # Diagonal \
            [(1, -1), (-1, 1)]  # Diagonal /
        ]

        for direction_pair in directions:
            count = 1  # The current stone

            for dx, dy in direction_pair:
                x, y = row + dx, col + dy
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player:
                    count += 1
                    x += dx
                    y += dy

            if count >= 5:
                return True

        return False

    def is_board_full(self):
        return np.count_nonzero(self.board) == self.size * self.size

    def print_board(self):
        # Print column headers with width 3
        print("   " + "".join(f"{i:<3}" for i in range(self.size)))

        for i in range(self.size):
            # Print row number with width 2 and padding
            print(f"{i:<2} ", end="")

            for j in range(self.size):
                cell = self.board[i, j]
                if cell == 0:
                    print("·  ", end="")  # Each cell takes 3 spaces
                elif cell == 1:
                    print("●  ", end="")
                else:
                    print("○  ", end="")
            print()  # New line after each row
        print()

    def get_state(self):
        """Get a lightweight representation of the game state"""
        return {
            'board': self.board.copy(),  # Shallow copy is enough for numpy arrays
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'frontier': self.frontier.copy()
        }

    def set_state(self, state):
        """Set the game state from a previously saved state"""
        self.board = state['board'].copy()
        self.current_player = state['current_player']
        self.game_over = state['game_over']
        self.winner = state['winner']
        self.frontier = state['frontier'].copy()


class AIPlayer:
    def __init__(self, player_num, algorithm='minimax', depth=3):
        self.player_num = player_num
        self.algorithm = algorithm
        self.depth = depth

    def get_move(self, game):
        if self.algorithm == 'minimax':
            return self.minimax_move(game)
        elif self.algorithm == 'alphabeta':
            return self.alphabeta_move(game)
        else:
            raise ValueError("Unknown algorithm")

    def minimax_move(self, game):
        best_score = -float('inf')
        best_move = None

        for move in self.get_possible_moves(game):
            # Save current game state
            state_before = game.get_state()
            
            # Make move
            game.make_move(*move)

            # Evaluate
            score = self.minimax(game, self.depth - 1, False)
            
            # Undo move by restoring previous state
            game.set_state(state_before)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, game, depth, is_maximizing):
        if depth == 0 or game.game_over:
            return self.evaluate(game)

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_possible_moves(game):
                # Save state
                state_before = game.get_state()
                
                # Make move
                game.make_move(*move)
                
                # Evaluate
                score = self.minimax(game, depth - 1, False)
                
                # Undo move
                game.set_state(state_before)
                
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                # Save state
                state_before = game.get_state()
                
                # Make move
                game.make_move(*move)
                
                # Evaluate
                score = self.minimax(game, depth - 1, True)
                
                # Undo move
                game.set_state(state_before)
                
                best_score = min(best_score, score)
            return best_score

    def alphabeta_move(self, game):
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')

        for move in self.get_possible_moves(game):
            # Save state
            state_before = game.get_state()
            
            # Make move
            game.make_move(*move)

            # Evaluate
            score = self.alphabeta(game, self.depth - 1, False, alpha, beta)
            
            # Undo move
            game.set_state(state_before)

            if score > best_score:
                best_score = score
                best_move = move

            alpha = max(alpha, best_score)
            if beta <= alpha:
                break

        return best_move

    def alphabeta(self, game, depth, is_maximizing, alpha, beta):
        if depth == 0 or game.game_over:
            return self.evaluate(game)

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_possible_moves(game):
                # Save state
                state_before = game.get_state()
                
                # Make move
                game.make_move(*move)
                
                # Evaluate
                score = self.alphabeta(game, depth - 1, False, alpha, beta)
                
                # Undo move
                game.set_state(state_before)
                
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                # Save state
                state_before = game.get_state()
                
                # Make move
                game.make_move(*move)
                
                # Evaluate
                score = self.alphabeta(game, depth - 1, True, alpha, beta)
                
                # Undo move
                game.set_state(state_before)
                
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score

    def get_possible_moves(self, game):
        # Get all empty cells, prioritizing those near existing stones
        return game.frontier.copy()

    def evaluate(self, game):
        if game.game_over:
            if game.winner == self.player_num:
                return 100000
            elif game.winner is not None:
                return -100000
            else:
                return 0

        score = 0
        directions = [
            (0, 1),  # Horizontal
            (1, 0),  # Vertical
            (1, 1),  # Diagonal \
            (1, -1)  # Diagonal /
        ]

        # Use numpy's vectorized operations for potential patterns
        my_pieces = np.where(game.board == self.player_num)
        opponent_pieces = np.where(game.board == 3 - self.player_num)
        
        # For each of my pieces
        for i, j in zip(my_pieces[0], my_pieces[1]):
            for di, dj in directions:
                score += self.evaluate_direction(game, i, j, di, dj, self.player_num)
                
        # For each opponent piece
        for i, j in zip(opponent_pieces[0], opponent_pieces[1]):
            for di, dj in directions:
                score -= self.evaluate_direction(game, i, j, di, dj, 3 - self.player_num)

        return score

    def evaluate_direction(self, game, row, col, di, dj, player):
        # Check for potential lines in this direction
        count = 0
        empty = 0
        
        # Look ahead up to 4 stones in this direction
        for step in range(1, 5):
            r, c = row + di * step, col + dj * step
            if 0 <= r < game.size and 0 <= c < game.size:
                if game.board[r, c] == player:
                    count += 1
                elif game.board[r, c] == 0:
                    empty += 1
                    break
                else:
                    break
            else:
                break

        # Look behind up to 4 stones in opposite direction
        for step in range(1, 5):
            r, c = row - di * step, col - dj * step
            if 0 <= r < game.size and 0 <= c < game.size:
                if game.board[r, c] == player:
                    count += 1
                elif game.board[r, c] == 0:
                    empty += 1
                    break
                else:
                    break
            else:
                break

        total = count + 1  # +1 for the current stone

        # Evaluate the potential of this line
        if total >= 5:
            return 10000  # Winning move
        elif total == 4 and empty >= 1:
            return 1000  # Open four
        elif total == 4:
            return 500  # Closed four
        elif total == 3 and empty >= 2:
            return 200  # Open three
        elif total == 3 and empty >= 1:
            return 100  # Semi-open three
        elif total == 2 and empty >= 2:
            return 10  # Open two
        else:
            return 1  # Just a stone


def human_vs_ai(board_size=15, ai_algorithm='minimax', ai_depth=3):
    game = Gomoku(board_size)
    ai = AIPlayer(2, ai_algorithm, ai_depth)

    print(f"Human (●) vs AI (○) on {board_size}x{board_size} board")
    print("Enter your moves as 'row col' (e.g., '7 7' for center)")
    print("Type 'exit' to quit\n")

    while not game.game_over:
        game.print_board()

        if game.current_player == 1:  # Human turn
            while True:
                move = input("Your move (row col): ").strip()
                if move.lower() == 'exit':
                    print("Game exited.")
                    return

                try:
                    row, col = map(int, move.split())
                    if game.make_move(row, col):
                        break
                    else:
                        print("Invalid move. Try again.")
                except:
                    print("Invalid input. Please enter two numbers separated by space.")
        else:  # AI turn
            print("AI is thinking...")
            row, col = ai.get_move(game)
            game.make_move(row, col)
            print(f"AI plays: {row} {col}")

    game.print_board()
    if game.winner == 1:
        print("Congratulations! You won!")
    elif game.winner == 2:
        print("AI wins!")
    else:
        print("It's a draw!")


def ai_vs_ai(board_size=15, ai1_algorithm='minimax', ai2_algorithm='alphabeta', depth=3):
    game = Gomoku(board_size)
    ai1 = AIPlayer(1, ai1_algorithm, depth)
    ai2 = AIPlayer(2, ai2_algorithm, depth)

    print(f"AI ({ai1_algorithm}) (●) vs AI ({ai2_algorithm}) (○) on {board_size}x{board_size} board")
    print("Press Enter to continue each move, or type 'exit' to quit\n")

    while not game.game_over:
        game.print_board()

        user_input = input("Press Enter for next move or 'exit' to quit: ").strip()
        if user_input.lower() == 'exit':
            print("Game exited.")
            return

        if game.current_player == 1:
            print(f"AI ({ai1_algorithm}) is thinking...")
            row, col = ai1.get_move(game)
        else:
            print(f"AI ({ai2_algorithm}) is thinking...")
            row, col = ai2.get_move(game)

        game.make_move(row, col)
        print(f"AI plays: {row} {col}")

    game.print_board()
    if game.winner == 1:
        print(f"AI ({ai1_algorithm}) wins!")
    elif game.winner == 2:
        print(f"AI ({ai2_algorithm}) wins!")
    else:
        print("It's a draw!")


def main():
    print("Gomoku (Five in a Row) Game Solver")
    print("1. Human vs AI (Minimax)")
    print("2. Human vs AI (Alpha-Beta)")
    print("3. AI (Minimax) vs AI (Alpha-Beta)")
    print("4. Exit")

    while True:
        choice = input("\nSelect mode (1-4): ").strip()

        if choice in ['1', '2', '3']:
            # Get board size
            while True:
                try:
                    board_size = int(input("Enter board size (5-25): "))
                    if 5 <= board_size <= 25:
                        break
                    else:
                        print("Board size must be between 5 and 25.")
                except ValueError:
                    print("Please enter a valid integer.")

            depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))

            if choice == '1':
                human_vs_ai(board_size, 'minimax', min(max(depth, 1), 5))
            elif choice == '2':
                human_vs_ai(board_size, 'alphabeta', min(max(depth, 1), 5))
            elif choice == '3':
                ai_vs_ai(board_size, 'minimax', 'alphabeta', min(max(depth, 1), 5))
        elif choice == '4':
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please enter 1-4.")


if __name__ == "__main__":
    main()
