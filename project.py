import numpy as np
import sys
from abc import ABC, abstractmethod

class GameEngine:
    def __init__(self):
        self.size = 15
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        self.frontier = set()
        self.frontier.add((self.size // 2, self.size // 2))
        self.move_history = []

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

        self.move_history.append({
            'position': (row, col),
            'player': self.current_player,
            'frontier_before': self.frontier.copy(),
            'game_over': self.game_over,
            'winner': self.winner
        })

        self.board[row, col] = self.current_player
        self.frontier.discard((row, col))
        
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
        elif np.count_nonzero(self.board) == self.size * self.size:
            self.game_over = True
        else:
            self.current_player = 3 - self.current_player

        return True

    def undo_move(self):
        if not self.move_history:
            return False
            
        last_move = self.move_history.pop()
        row, col = last_move['position']
        self.board[row, col] = 0
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
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]

        for direction_pair in directions:
            count = 1
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
        print("   " + "".join(f"{i:<3}" for i in range(self.size)))
        for i in range(self.size):
            print(f"{i:<2} ", end="")
            for j in range(self.size):
                cell = self.board[i, j]
                if cell == 0:
                    print("·  ", end="")
                elif cell == 1:
                    print("●  ", end="")
                else:
                    print("○  ", end="")
            print()
        print()

    def get_state(self):
        return {
            'board': self.board.copy(),
            'current_player': self.current_player,
            'game_over': self.game_over,
            'winner': self.winner,
            'frontier': self.frontier.copy()
        }

    def set_state(self, state):
        self.board = state['board'].copy()
        self.current_player = state['current_player']
        self.game_over = state['game_over']
        self.winner = state['winner']
        self.frontier = state['frontier'].copy()


class SearchAlgorithm(ABC):
    def __init__(self, player_num, depth):
        self.player_num = player_num
        self.depth = depth
        self.evaluator = EvaluatorEngine(player_num)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def search(self, game):
        pass

    def get_possible_moves(self, game):
        return game.frontier.copy()


class Minimax(SearchAlgorithm):
    def name(self):
        return "Minimax"

    def search(self, game):
        best_score = -float('inf')
        best_move = None

        for move in self.get_possible_moves(game):
            state_before = game.get_state()
            game.make_move(*move)
            score = self.minimax(game, self.depth - 1, False)
            game.set_state(state_before)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def minimax(self, game, depth, is_maximizing):
        if depth == 0 or game.game_over:
            return self.evaluator.evaluate(game)

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_possible_moves(game):
                state_before = game.get_state()
                game.make_move(*move)
                score = self.minimax(game, depth - 1, False)
                game.set_state(state_before)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                state_before = game.get_state()
                game.make_move(*move)
                score = self.minimax(game, depth - 1, True)
                game.set_state(state_before)
                best_score = min(best_score, score)
            return best_score


class AlphaBeta(SearchAlgorithm):

    def name(self):
        return "Alphabeta"

    def search(self, game):
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')

        for move in self.get_possible_moves(game):
            state_before = game.get_state()
            game.make_move(*move)
            score = self.alphabeta(game, self.depth - 1, False, alpha, beta)
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
            return self.evaluator.evaluate(game)

        if is_maximizing:
            best_score = -float('inf')
            for move in self.get_possible_moves(game):
                state_before = game.get_state()
                game.make_move(*move)
                score = self.alphabeta(game, depth - 1, False, alpha, beta)
                game.set_state(state_before)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                state_before = game.get_state()
                game.make_move(*move)
                score = self.alphabeta(game, depth - 1, True, alpha, beta)
                game.set_state(state_before)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score


class EvaluatorEngine:
    def __init__(self, player_num):
        self.player_num = player_num

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
            (0, 1),
            (1, 0),
            (1, 1),
            (1, -1)
        ]

        my_pieces = np.where(game.board == self.player_num)
        opponent_pieces = np.where(game.board == 3 - self.player_num)
        
        for i, j in zip(my_pieces[0], my_pieces[1]):
            for di, dj in directions:
                score += self.evaluate_direction(game, i, j, di, dj, self.player_num)
                
        for i, j in zip(opponent_pieces[0], opponent_pieces[1]):
            for di, dj in directions:
                score -= self.evaluate_direction(game, i, j, di, dj, 3 - self.player_num)

        return score

    def evaluate_direction(self, game, row, col, di, dj, player):
        count = 0
        empty = 0
        
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

        total = count + 1

        if total >= 5:
            return 10000
        elif total == 4 and empty >= 1:
            return 1000
        elif total == 4:
            return 500
        elif total == 3 and empty >= 2:
            return 200
        elif total == 3 and empty >= 1:
            return 100
        elif total == 2 and empty >= 2:
            return 10
        else:
            return 1


class AIPlayer:
    def __init__(self, player_num, algorithm, depth=3):
        self.player_num = player_num
        self.depth = depth
        self.algorithm = algorithm

    def get_move(self, game):
        return self.algorithm.search(game)


class GameMode(ABC):
    def __init__(self, game):
        self.game = game
    
    @abstractmethod
    def play(self):
        pass


class HumanVsAI(GameMode):
    def __init__(self, game, ai_algorithm, ai_depth=3):
        super().__init__(game)
        self.ai = AIPlayer(2, ai_algorithm, ai_depth)
        
    def play(self):
        print("Human (●) vs AI (○) on 15x15 board")
        print("Enter your moves as 'row col' (e.g., '7 7' for center)")
        print("Type 'exit' to quit\n")

        while not self.game.game_over:
            self.game.print_board()

            if self.game.current_player == 1:
                self._handle_human_move()
            else:
                self._handle_ai_move()

        self._display_result()

    def _handle_human_move(self):
        while True:
            move = input("Your move (row col): ").strip()
            if move.lower() == 'exit':
                print("Game exited.")
                sys.exit()

            try:
                row, col = map(int, move.split())
                if self.game.make_move(row, col):
                    break
                else:
                    print("Invalid move. Try again.")
            except:
                print("Invalid input. Please enter two numbers separated by space.")

    def _handle_ai_move(self):
        print("AI is thinking...")
        row, col = self.ai.get_move(self.game)
        self.game.make_move(row, col)
        print(f"AI plays: {row} {col}")

    def _display_result(self):
        self.game.print_board()
        if self.game.winner == 1:
            print("Congratulations! You won!")
        elif self.game.winner == 2:
            print("AI wins!")
        else:
            print("It's a draw!")


class AIvsAI(GameMode):
    def __init__(self, game, ai1_algorithm, ai2_algorithm, depth=3):
        super().__init__(game)
        self.ai1 = AIPlayer(1, ai1_algorithm, depth)
        self.ai2 = AIPlayer(2, ai2_algorithm, depth)
        
    def play(self):
        print(f"AI ({self.ai1.algorithm.name()}) (●) vs AI ({self.ai2.algorithm.name()}) (○) on 15x15 board")
        print("Press Enter to continue each move, or type 'exit' to quit\n")

        while not self.game.game_over:
            self.game.print_board()
            if not self._handle_user_input():
                return

            if self.game.current_player == 1:
                self._handle_ai_move(self.ai1, self.ai1.algorithm.name())
            else:
                self._handle_ai_move(self.ai2, self.ai2.algorithm.name())

        self._display_result()

    def _handle_user_input(self):
        user_input = input("Press Enter for next move or 'exit' to quit: ").strip()
        if user_input.lower() == 'exit':
            print("Game exited.")
            return False
        return True

    def _handle_ai_move(self, ai, ai_name):
        print(f"AI ({ai_name}) is thinking...")
        row, col = ai.get_move(self.game)
        self.game.make_move(row, col)
        print(f"AI plays: {row} {col}")

    def _display_result(self):
        self.game.print_board()
        if self.game.winner == 1:
            print(f"AI ({self.ai1.algorithm.name()}) wins!")
        elif self.game.winner == 2:
            print(f"AI ({self.ai2.algorithm.name()}) wins!")
        else:
            print("It's a draw!")


def main():
    print("Gomoku (Five in a Row) Game Solver")
    print("1. Human vs AI (Minimax)")
    print("2. AI (Minimax) vs AI (Alpha-Beta)")
    print("3. Exit")

    while True:
        choice = input("\nSelect mode (1-3): ").strip()

        if choice == '1':
            depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))
            depth = min(max(depth, 1), 5)
            game = GameEngine()
            game_mode = HumanVsAI(game, Minimax(2, depth))
            game_mode.play()
                
        elif choice == '2':
            depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))
            depth = min(max(depth, 1), 5)
            game = GameEngine()
            game_mode = AIvsAI(game, Minimax(1, depth), AlphaBeta(2, depth))
            game_mode.play()
                
        elif choice == '3':
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()