import numpy as np
import sys
import tkinter as tk
from tkinter import messagebox, ttk
from copy import deepcopy
from PIL import Image, ImageTk, ImageDraw
from abc import ABC, abstractmethod

class GameEngine:
    def __init__(self, board_size):
        self.size = board_size
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
    def name(self) -> str:
        pass

    @abstractmethod
    def search(self, game):
        pass

    def getPossibleMoves(self, game):
        return game.frontier.copy()

class Minimax(SearchAlgorithm):
    def name(self) -> str:
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
    def name(self) -> str:
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

class GomokuGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Gomoku - Five in a Row")
        self.root.geometry("800x700")
        self.root.minsize(600, 600)
        

        # Game configuration
        self.board_size = 15
        self.ai_depth = 3
        self.game_mode = "human_vs_ai"
        self.ai_algorithm = AlphaBeta(2, self.ai_depth)

        # Colors and styles
        self.bg_color = "#EBC69E"
        self.line_color = "#93776C"
        self.black_stone_color = "#2c2c2c"
        self.white_stone_color = "#f9f9f9"
        
        # Game state
        self.game = GameEngine(self.board_size)
        self.last_move = None
        self.human_player = 1
        self.ai_thinking = False
        self.animation_in_progress = False
        
        # Create UI
        self.create_menu()
        self.create_game_controls()
        self.create_board()
        self.create_status_bar()
        
        # Pre-render stone images
        self.stone_images = self.create_stone_images(30)
        self.canvas.bind("<Motion>", self.on_hover)  # Add this for hover effect
        self.highlight = None  # Add this to track highlight object
        # Start the game
        self.update_ui()
        
    def on_hover(self, event):
        """Show preview stone when hovering over empty cells"""
        if (self.game_mode != "human_vs_ai" or 
            self.game.currentPlayer!= self.human_player or
            self.game.gameOver or self.ai_thinking):
            return
        
        col = int((event.x - self.board_offset_x + (0.5 * self.cell_size)) // self.cell_size)
        row = int((event.y - self.board_offset_y + (0.5 * self.cell_size)) // self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.game.board[row][col] == 0:
            self.highlight_cell(row, col)
        elif not self.highlight == None:
            self.canvas.delete(self.highlight)
            

    def highlight_cell(self, row, col):
        """Show visual feedback when hovering over a cell"""
        if self.highlight:
            self.canvas.delete(self.highlight)
        
        x = self.board_offset_x + col * self.cell_size - (0.5 * self.cell_size)
        y = self.board_offset_y + row * self.cell_size - (0.5 * self.cell_size)
        stone_size = int(self.cell_size * 0.6) 
        
        if self.game.currentPlayer == 1:  # Black
            self.highlight = self.canvas.create_oval(
                x + self.cell_size//2 - stone_size//2,
                y + self.cell_size//2 - stone_size//2,
                x + self.cell_size//2 + stone_size//2,
                y + self.cell_size//2 + stone_size//2,
                fill="#2c2c2c", outline="", width=0, tags="hover"
            )
        else:  # White
            self.highlight = self.canvas.create_oval(
                x + self.cell_size//2 - stone_size//2,
                y + self.cell_size//2 - stone_size//2,
                x + self.cell_size//2 + stone_size//2,
                y + self.cell_size//2 + stone_size//2,
                fill="#f9f9f9", outline="", width=0, tags="hover"
            )    
    def create_menu(self):
        menubar = tk.Menu(self.root)
        
        game_menu = tk.Menu(menubar, tearoff=0)
        game_menu.add_command(label="New Game", command=self.new_game)
        game_menu.add_command(label="Human vs AI", command=lambda: self.set_game_mode("human_vs_ai"))
        game_menu.add_command(label="AI vs AI", command=lambda: self.set_game_mode("ai_vs_ai"))
        game_menu.add_separator()
        game_menu.add_command(label="Exit", command=self.root.quit)
        menubar.add_cascade(label="Game", menu=game_menu)
        
        settings_menu = tk.Menu(menubar, tearoff=0)
        settings_menu.add_command(label="Board Size", command=self.configure_board_size)
        settings_menu.add_command(label="AI Difficulty", command=self.configure_ai_difficulty)
        menubar.add_cascade(label="Settings", menu=settings_menu)
        
        self.root.config(menu=menubar)
    
    def create_game_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.player_indicator = tk.Label(
            control_frame, text="Current: Black (●)", 
            font=("Arial", 10, "bold"), fg="black"
        )
        self.player_indicator.pack(anchor=tk.CENTER)
    
    def create_board(self):
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=5)
        
        self.canvas = tk.Canvas(
            self.board_frame, bg=self.bg_color,
            highlightthickness=0, borderwidth=0
        )
        
        self.canvas.pack(expand=True, fill=tk.BOTH, side=tk.RIGHT)
        
        self.canvas.bind("<Configure>", self.draw_board)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.cell_size = 0
        self.board_offset_x = 0
        self.board_offset_y = 0
    
    def create_status_bar(self):
        self.status_bar = tk.Label(
            self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W,
            font=("Arial", 9), bg="#e0e0e0"
        )
        self.status_bar.pack(fill=tk.X, padx=2, pady=2)
    
    def create_stone_images(self, size):
        stone_images = {
            0: None,
            1: self.create_stone_image(size, self.black_stone_color),
            2: self.create_stone_image(size, self.white_stone_color),
            "last_black": self.create_stone_image(size, self.black_stone_color, highlight=True),
            "last_white": self.create_stone_image(size, self.white_stone_color, highlight=True)
        }
        return stone_images
    
    def create_stone_image(self, size, color, highlight=False):
        img = Image.new("RGBA", (size*2, size*2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        draw.ellipse(
            [(0, 0), (size*2, size*2)],
            fill=color
        )
        
        if highlight:
            highlight_size = size // 3
            draw.ellipse(
                [(size//2, size//2), 
                 (size//2 + highlight_size, size//2 + highlight_size)],
                fill="#322A2888"
            )
        
        return ImageTk.PhotoImage(img.resize((size, size)))
    
    def draw_board(self, event=None):
        if self.game is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.cell_size = min(
            canvas_width // (self.board_size + 4),
            canvas_height // (self.board_size + 4)
        )
        
        self.board_offset_x = (canvas_width - self.board_size * self.cell_size) // 2
        self.board_offset_y = (canvas_height - self.board_size * self.cell_size) // 2
        
        self.canvas.delete("all")
        
        for i in range(self.board_size):
            self.canvas.create_line(
                self.board_offset_x,
                self.board_offset_y + i * self.cell_size,
                self.board_offset_x + (self.board_size - 1) * self.cell_size,
                self.board_offset_y + i * self.cell_size,
                fill=self.line_color, width=1
            )
            
            self.canvas.create_line(
                self.board_offset_x + i * self.cell_size,
                self.board_offset_y,
                self.board_offset_x + i * self.cell_size,
                self.board_offset_y + (self.board_size - 1) * self.cell_size,
                fill=self.line_color, width=1
            )
        
        for i in range(self.board_size):
            self.canvas.create_text(
                self.board_offset_x + i * self.cell_size,
                self.board_offset_y - 35,
                text=str(i), font=("Arial", 10)
            )
            
            self.canvas.create_text(
                self.board_offset_x - 35,
                self.board_offset_y + i * self.cell_size,
                text=str(i), font=("Arial", 10)
            )
        
        stone_size = int(self.cell_size * 0.9)
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.game.board[i][j] != 0:
                    x = self.board_offset_x + j * self.cell_size 
                    y = self.board_offset_y + i * self.cell_size 
                    
                    if self.last_move and self.last_move == (i, j):
                        stone_type = f"last_{'black' if self.game.board[i][j] == 1 else 'white'}"
                    else:
                        stone_type = self.game.board[i][j]
                    
                    if stone_size != self.stone_images[stone_type].width():
                        self.stone_images = self.create_stone_images(stone_size)
                    
                    self.canvas.create_image(
                        x, y,
                        image=self.stone_images[stone_type],
                        tags=f"stone_{i}_{j}"
                    )
        
        self.canvas.configure(scrollregion=(
            0, 0,
            self.board_size * self.cell_size + self.board_offset_x * 2,
            self.board_size * self.cell_size + self.board_offset_y * 2
        ))
    
    def on_click(self, event):
        """Handle mouse clicks on the board with precise placement"""
        if (self.game_mode != "human_vs_ai" or 
            self.game.currentPlayer != self.human_player or
            self.game.gameOver or self.ai_thinking or
            self.animation_in_progress):
            return
        
        # Precise cell calculation using floor division
        col = int((event.x - self.board_offset_x + (0.5 * self.cell_size)) // self.cell_size)
        row = int((event.y - self.board_offset_y + (0.5 * self.cell_size)) // self.cell_size)
        
        # Ensure we're within bounds and cell is empty
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.game.board[row][col] == 0:
            # Remove any hover preview
            if self.highlight:
                self.canvas.delete(self.highlight)
                self.highlight = None
            
            if self.game.makeMove(row, col):
                self.last_move = (row, col)
                self.animate_stone(row, col)

                self.root.after(200, self.update_ui)
                if not self.game.gameOver and self.game_mode == "human_vs_ai":
                    self.root.after(500, self.ai_move)

    def animate_stone(self, row, col):
        if not hasattr(self, 'cell_size') or self.cell_size == 0:
            return
        
        self.animation_in_progress = True
        stone_size = int(self.cell_size * 1)
        x = self.board_offset_x + col * self.cell_size - (0.5 * self.cell_size)
        y = self.board_offset_y + row * self.cell_size - (0.5 * self.cell_size)
        
        temp_stone = self.canvas.create_oval(
            x + stone_size//2 - 5, y + stone_size//2 - 5,
            x + stone_size//2 + 5, y + stone_size//2 + 5,
            fill="black" if self.game.board[row][col] == 1 else "white",
            outline=""
        )
        
        def grow_stone(step):
            current_size = 10 + step * 4
            if current_size >= stone_size:
                self.canvas.delete(temp_stone)
                self.draw_board()
                self.animation_in_progress = False
                return
            
            self.canvas.coords(
                temp_stone,
                x + stone_size//2 - current_size//2,
                y + stone_size//2 - current_size//2,
                x + stone_size//2 + current_size//2,
                y + stone_size//2 + current_size//2
            )
            self.root.after(20, lambda: grow_stone(step + 1))
        
        grow_stone(0)
    
    def update_ui(self):
        if self.game is None:
            return
        
        if self.game.gameOver:
            if self.game.winner == 1:
                self.player_indicator.config(text="Black wins!", fg="black")
            elif self.game.winner == 2:
                self.player_indicator.config(text="White wins!", fg="black")
            else:
                self.player_indicator.config(text="Game ended in draw!", fg="black")
        else:
            player_text = "Black (●)" if self.game.currentPlayer == 1 else "White (○)"
            color = "black" if self.game.currentPlayer == 1 else "white"
            self.player_indicator.config(text=f"Current: {player_text}", fg=color)
        
        if self.ai_thinking:
            self.status_bar.config(text="AI is thinking...")
        elif self.game.gameOver:
            if self.game.winner == 1:
                self.status_bar.config(text="Game over - Black wins!")
            elif self.game.winner == 2:
                self.status_bar.config(text="Game over - White wins!")
            else:
                self.status_bar.config(text="Game over - Draw!")
        else:
            if self.game_mode == "human_vs_ai" and self.game.currentPlayer == self.human_player:
                self.status_bar.config(text="Your turn - click on the board to place a stone")
            else:
                self.status_bar.config(text="Waiting for AI move...")
        
        self.draw_board()
    
    def new_game(self):
        self.game = GameEngine(self.board_size)
        self.last_move = None
        self.ai_thinking = False
        self.animation_in_progress = False
        
        if self.game_mode == "ai_vs_ai":
            self.root.after(500, self.ai_vs_ai_loop)
        
        self.update_ui()
    
    def set_game_mode(self, mode):
        """Set the game mode and properly initialize"""
        self.game_mode = mode
        self.update_game_mode_button()
        self.new_game()
        
        # Start AI vs AI loop if needed
        if mode == "ai_vs_ai":
            self.root.after(500, self.ai_vs_ai_loop)

    def ai_vs_ai_loop(self):
        """Handle AI vs AI game loop with proper updates"""
        if self.game.gameOver or self.game_mode != "ai_vs_ai":
            return
        
        self.ai_move()
        
        # Only continue if still in AI vs AI mode
        if self.game_mode == "ai_vs_ai" and not self.game.gameOver:
            self.root.after(1000, self.ai_vs_ai_loop)

    def ai_move(self):
        """Make AI move with proper UI updates"""
        if self.game.gameOver or (self.game_mode == "human_vs_ai" and self.game.currentPlayer == self.human_player):
            return
        
        self.ai_thinking = True
        self.status_bar.config(text="AI is thinking...")
        self.update_ui()
        self.root.update()  # Force UI update
        
        def run_ai():
            ai = AIPlayer(
                self.game.currentPlayer,
                self.ai_algorithm,
                self.ai_depth
            )
            return ai.getMove(self.game)
        
        def on_ai_complete(result):
            row, col = result
            if self.game.makeMove(row, col):
                self.last_move = (row, col)
                self.animate_stone(row, col)
            
            self.ai_thinking = False
            self.update_ui()
        
        # Use threading to prevent UI freeze
        import threading
        def ai_worker():
            result = run_ai()
            self.root.after(0, lambda: on_ai_complete(result))
        
        threading.Thread(target=ai_worker, daemon=True).start()
        
    def toggle_game_mode(self):
        if self.game_mode == "human_vs_ai":
            self.set_game_mode("ai_vs_ai")
        else:
            self.set_game_mode("human_vs_ai")
    
    def update_game_mode_button(self):
        if self.game_mode == "human_vs_ai":
            self.game_mode_btn.config(text="Switch to AI vs AI")
        else:
            self.game_mode_btn.config(text="Switch to Human vs AI")
    
    def configure_board_size(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure Board Size")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="Board Size:").grid(row=0, column=0, padx=5, pady=5)
        
        size_var = tk.IntVar(value=self.board_size)
        size_spin = tk.Spinbox(
            dialog, from_=5, to=25, textvariable=size_var,
            width=5, font=("Arial", 10)
        )
        size_spin.grid(row=0, column=1, padx=5, pady=5)
        
        def apply_changes():
            new_size = size_var.get()
            if 5 <= new_size <= 25:
                self.board_size = new_size
                self.new_game()
                dialog.destroy()
            else:
                messagebox.showerror("Invalid Size", "Board size must be between 5 and 25")
        
        tk.Button(
            dialog, text="OK", command=apply_changes,
            width=10, bg="#4a7a8c", fg="white"
        ).grid(row=1, column=0, columnspan=2, pady=5)
    
    def configure_ai_difficulty(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Configure AI Difficulty")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()
        
        tk.Label(dialog, text="AI Difficulty:").grid(row=0, column=0, padx=5, pady=5)
        
        depth_var = tk.IntVar(value=self.ai_depth)
        depth_spin = tk.Spinbox(
            dialog, from_=1, to=5, textvariable=depth_var,
            width=5, font=("Arial", 10)
        )
        depth_spin.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(dialog, text="AI Algorithm:").grid(row=1, column=0, padx=5, pady=5)
        
        algo_var = tk.StringVar(value=self.ai_algorithm.name())
        algo_menu = tk.OptionMenu(
            dialog, algo_var, "minimax", "alphabeta"
        )
        algo_menu.config(width=10)
        algo_menu.grid(row=1, column=1, padx=5, pady=5)
        
        def apply_changes():
            self.ai_depth = depth_var.get()
            algo_name = algo_var.get()
            if algo_name == "minimax":
                self.ai_algorithm = Minimax(2,self.ai_depth)
            else:
                self.ai_algorithm = AlphaBeta(2,self.ai_depth)
                
            dialog.destroy()
        
        tk.Button(
            dialog, text="OK", command=apply_changes,
            width=10, bg="#4a7a8c", fg="white"
        ).grid(row=2, column=0, columnspan=2, pady=5)

def main():
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()
    # print("Gomoku (Five in a Row) Game Solver")
    # print("1. Human vs AI (Minimax)")
    # print("2. AI (Minimax) vs AI (Alpha-Beta)")
    # print("3. Exit")

    # while True:
    #     choice = input("\nSelect mode (1-3): ").strip()

    #     if choice == '1':
    #         depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))
    #         depth = min(max(depth, 1), 5)
    #         game = GameEngine()
    #         gameMode = HumanVsAI(game, Minimax(2, depth), depth)
    #         gameMode.play()

    #     elif choice == '2':
    #         depth = int(input("Enter AI depth (1-5, higher is smarter but slower): "))
    #         depth = min(max(depth, 1), 5)
    #         game = GameEngine()
    #         gameMode = AIvsAI(game, Minimax(1, depth), AlphaBeta(2, depth), depth)
    #         gameMode.play()

    #     elif choice == '3':
    #         print("Goodbye!")
    #         sys.exit()
    #     else:
    #         print("Invalid choice. Please enter 1-3.")


if __name__ == "__main__":
    main()
