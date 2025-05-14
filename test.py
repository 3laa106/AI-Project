import numpy as np
import sys
from abc import ABC, abstractmethod


class GameEngine:
    def __init__(self):
        self.size = 15
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.currentPlayer = 1
        self.gameOver = False
        self.winner = None
        self.frontier = set()
        self.frontier.add((self.size // 2, self.size // 2))

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=np.int8)
        self.currentPlayer = 1
        self.gameOver = False
        self.winner = None
        self.frontier = set()
        self.frontier.add((self.size // 2, self.size // 2))

    def makeMove(self, row, col):
        if self.gameOver or not self.isValidMove(row, col):
            return False

        self.board[row, col] = self.currentPlayer
        self.frontier.discard((row, col))

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = row + dx, col + dy
                if self.isValidMove(nx, ny):
                    self.frontier.add((nx, ny))

        if self.checkWin(row, col):
            self.gameOver = True
            self.winner = self.currentPlayer
        elif self.isBoardFull():
            self.gameOver = True
        else:
            self.currentPlayer = 3 - self.currentPlayer

        return True

    def undoTest(self, row, col, isApplied):
        if not isApplied:
            return False

        if self.checkWin(row, col):
            self.gameOver = False
            self.winner = None
        elif np.count_nonzero(self.board) == self.size * self.size:
            self.gameOver = None
        else:
            self.currentPlayer = 3 - self.currentPlayer

        self.board[row, col] = 0
        self.frontier.add((row, col))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = row + dx, col + dy
                if self.hasNoOccupiedNeighbors(nx, ny):
                    self.frontier.discard((nx, ny))

        return True

    def isValidMove(self, row, col):
        return (0 <= row < self.size and 0 <= col < self.size and self.board[row, col] == 0)

    def hasNoOccupiedNeighbors(self, row, col):
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = row + dx, col + dy
                if 0 <= nx < self.size and 0 <= ny < self.size and self.board[nx, ny] != 0:
                    return False
        return True

    def checkWin(self, row, col):
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],
            [(1, 0), (-1, 0)],
            [(1, 1), (-1, -1)],
            [(1, -1), (-1, 1)]
        ]

        for directionPair in directions:
            count = 1
            for dx, dy in directionPair:
                x, y = row + dx, col + dy
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x, y] == player:
                    count += 1
                    x += dx
                    y += dy
            if count >= 5:
                return True
        return False

    def isBoardFull(self):
        return np.count_nonzero(self.board) == self.size * self.size

    def printBoard(self):
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


class SearchAlgorithm(ABC):
    def __init__(self, playerNum, depth):
        self.playerNum = playerNum
        self.depth = depth
        self.evaluator = EvaluatorEngine(playerNum)

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def search(self, game):
        pass

    def getPossibleMoves(self, game):
        return game.frontier.copy()

class Minimax(SearchAlgorithm):
    def name(self):
        return "Minimax"

    def search(self, game):
        bestScore = -float('inf')
        bestMove = None

        for move in self.getPossibleMoves(game):
            isApplied = game.makeMove(*move)
            score = self.minimax(game, self.depth - 1, False)
            game.undoTest(*move, isApplied)

            if score > bestScore:
                bestScore = score
                bestMove = move

        return bestMove

    def minimax(self, game, depth, isMaximizing):
        if depth == 0 or game.gameOver:
            return self.evaluator.evaluate(game, depth + 1)

        if isMaximizing:
            bestScore = -float('inf')
            for move in self.getPossibleMoves(game):
                isApplied = game.makeMove(*move)
                score = self.minimax(game, depth - 1, False)
                game.undoTest(*move, isApplied)
                bestScore = max(bestScore, score)
            return bestScore
        else:
            bestScore = float('inf')
            for move in self.getPossibleMoves(game):
                isApplied = game.makeMove(*move)
                score = self.minimax(game, depth - 1, True)
                game.undoTest(*move, isApplied)
                bestScore = min(bestScore, score)
            return bestScore


class AlphaBeta(SearchAlgorithm):
    def name(self):
        return "Alphabeta"

    def search(self, game):
        bestScore = -float('inf')
        bestMove = None
        alpha = -float('inf')
        beta = float('inf')

        for move in self.getPossibleMoves(game):
            isApplied = game.makeMove(*move)
            score = self.alphabeta(game, self.depth - 1, False, alpha, beta)
            game.undoTest(*move, isApplied)

            if score > bestScore:
                bestScore = score
                bestMove = move
            alpha = max(alpha, bestScore)
            if beta <= alpha:
                break

        return bestMove

    def alphabeta(self, game, depth, isMaximizing, alpha, beta):
        if depth == 0 or game.gameOver:
            return self.evaluator.evaluate(game, depth + 1)

        if isMaximizing:
            bestScore = -float('inf')
            for move in self.getPossibleMoves(game):
                isApplied = game.makeMove(*move)
                score = self.alphabeta(game, depth - 1, False, alpha, beta)
                game.undoTest(*move, isApplied)
                bestScore = max(bestScore, score)
                alpha = max(alpha, bestScore)
                if beta <= alpha:
                    break
            return bestScore
        else:
            bestScore = float('inf')
            for move in self.getPossibleMoves(game):
                isApplied = game.makeMove(*move)
                score = self.alphabeta(game, depth - 1, True, alpha, beta)
                game.undoTest(*move, isApplied)
                bestScore = min(bestScore, score)
                beta = min(beta, bestScore)
                if beta <= alpha:
                    break
            return bestScore


class EvaluatorEngine:
    def __init__(self, playerNum):
        self.playerNum = playerNum
        self.patternScores = {
            "FIVE": 100000,
            "OPEN_FOUR": 10000,
            "FOUR": 5000,
            "OPEN_THREE": 1000,
            "THREE": 500,
            "OPEN_TWO": 100,
            "TWO": 50
        }

    def evaluate(self, game, depth):
        if game.gameOver:
            if game.winner == self.playerNum:
                return 100000 * depth
            elif game.winner is not None:
                return -100000 * depth
            else:
                return 0

        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]

        myScore = 0
        opponentScore = 0

        for row in range(game.size):
            for col in range(game.size):
                if game.board[row, col] == self.playerNum:
                    for dr, dc in directions:
                        pattern = self.getPattern(game, row, col, dr, dc, self.playerNum)
                        myScore += self.scorePattern(pattern) * depth

                elif game.board[row, col] == 3 - self.playerNum:
                    for dr, dc in directions:
                        pattern = self.getPattern(game, row, col, dr, dc, 3 - self.playerNum)
                        opponentScore += self.scorePattern(pattern) * depth

        return myScore - opponentScore * 1.1

    def getPattern(self, game, row, col, dr, dc, player):
        count = 1
        r, c = row + dr, col + dc
        while (0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == player):
            count += 1
            r += dr
            c += dc

        openEnds = 0

        if (0 <= r < game.size and 0 <= c < game.size and game.board[r, c] == 0):
            openEnds += 1

        startR, startC = row - dr, col - dc
        if (0 <= startR < game.size and 0 <= startC < game.size and game.board[startR, startC] == 0):
            openEnds += 1

        hasThreat = (count == 4 and openEnds >= 1)

        return (count, openEnds, hasThreat)

    def scorePattern(self, pattern):
        count, openEnds, hasThreat = pattern

        if count >= 5:
            return self.patternScores["FIVE"]
        elif count == 4:
            if openEnds == 2:
                return self.patternScores["OPEN_FOUR"]
            elif openEnds == 1:
                return self.patternScores["FOUR"]
        elif count == 3:
            if openEnds == 2:
                return self.patternScores["OPEN_THREE"]
            elif openEnds == 1:
                return self.patternScores["THREE"]
        elif count == 2:
            if openEnds == 2:
                return self.patternScores["OPEN_TWO"]
            elif openEnds == 1:
                return self.patternScores["TWO"]

        return count


class AIPlayer:
    def __init__(self, playerNum, algorithm, depth=3):
        self.playerNum = playerNum
        self.depth = depth
        self.algorithm = algorithm

    def getMove(self, game):
        return self.algorithm.search(game)


class GameMode(ABC):
    def __init__(self, game):
        self.game = game

    @abstractmethod
    def play(self):
        pass


class HumanVsAI(GameMode):
    def __init__(self, game, aiAlgorithm, aiDepth=3):
        super().__init__(game)
        self.ai = AIPlayer(2, aiAlgorithm, aiDepth)

    def play(self):
        print("Human (●) vs AI (○) on 15x15 board")
        print("Enter your moves as 'row col' (e.g., '7 7' for center)")
        print("Type 'exit' to quit\n")

        while not self.game.gameOver:
            self.game.printBoard()

            if self.game.currentPlayer == 1:
                self.handleHumanMove()
            else:
                self.handleAiMove()

        self.displayResult()

    def handleHumanMove(self):
        while True:
            move = input("Your move (row col): ").strip()
            if move.lower() == 'exit':
                print("Game exited.")
                sys.exit()

            try:
                row, col = map(int, move.split())
                if self.game.makeMove(row, col):
                    break
                else:
                    print("Invalid move. Try again.")
            except:
                print("Invalid input. Please enter two numbers separated by space.")

    def handleAiMove(self):
        print("AI is thinking...")
        row, col = self.ai.getMove(self.game)
        self.game.makeMove(row, col)
        print(f"AI plays: {row} {col}")

    def displayResult(self):
        self.game.printBoard()
        if self.game.winner == 1:
            print("Congratulations! You won!")
        elif self.game.winner == 2:
            print("AI wins!")
        else:
            print("It's a draw!")


class AIvsAI(GameMode):
    def __init__(self, game, ai1Algorithm, ai2Algorithm, depth = 3):
        super().__init__(game)
        self.ai1 = AIPlayer(1, ai1Algorithm, depth)
        self.ai2 = AIPlayer(2, ai2Algorithm, depth)

    def play(self):
        print(f"AI ({self.ai1.algorithm.name()}) (●) vs AI ({self.ai2.algorithm.name()}) (○) on 15x15 board")
        print("Press Enter to continue each move, or type 'exit' to quit\n")

        while not self.game.gameOver:
            self.game.printBoard()
            if not self.handleUserInput():
                return

            if self.game.currentPlayer == 1:
                self.handleAiMove(self.ai1, self.ai1.algorithm.name())
            else:
                self.handleAiMove(self.ai2, self.ai2.algorithm.name())

        self.displayResult()

    def handleUserInput(self):
        userInput = input("Press Enter for next move or 'exit' to quit: ").strip()
        if userInput.lower() == 'exit':
            print("Game exited.")
            return False
        return True

    def handleAiMove(self, ai, aiName):
        print(f"AI ({aiName}) is thinking...")
        row, col = ai.getMove(self.game)
        self.game.makeMove(row, col)
        print(f"AI plays: {row} {col}")

    def displayResult(self):
        self.game.printBoard()
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
            gameMode = HumanVsAI(game, Minimax(2, depth), depth)
            gameMode.play()

        elif choice == '2':
            depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))
            depth = min(max(depth, 1), 5)
            game = GameEngine()
            gameMode = AIvsAI(game, Minimax(1, depth), AlphaBeta(2, depth), depth)
            gameMode.play()

        elif choice == '3':
            print("Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()