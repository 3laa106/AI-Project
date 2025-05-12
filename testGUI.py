import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from copy import deepcopy
from PIL import Image, ImageTk, ImageDraw  # Make sure to install Pillow with: pip install Pillow

class Gomoku:
    def __init__(self, size=15):
        self.size = size
        self.board = np.zeros((size, size), dtype=int)
        self.current_player = 1  # Player 1 starts (1 for black, 2 for white)
        self.game_over = False
        self.winner = None
        
    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        self.game_over = False
        self.winner = None
        
    def make_move(self, row, col):
        if self.game_over or not self.is_valid_move(row, col):
            return False
        
        self.board[row][col] = self.current_player
        
        if self.check_win(row, col):
            self.game_over = True
            self.winner = self.current_player
        elif self.is_board_full():
            self.game_over = True
        else:
            self.current_player = 3 - self.current_player  # Switch player (1->2, 2->1)
        
        return True
    
    def is_valid_move(self, row, col):
        return (0 <= row < self.size and 0 <= col < self.size 
                and self.board[row][col] == 0)
    
    def check_win(self, row, col):
        player = self.board[row][col]
        directions = [
            [(0, 1), (0, -1)],  # Horizontal
            [(1, 0), (-1, 0)],  # Vertical
            [(1, 1), (-1, -1)], # Diagonal \
            [(1, -1), (-1, 1)]  # Diagonal /
        ]
        
        for direction_pair in directions:
            count = 1  # The current stone
            
            for dx, dy in direction_pair:
                x, y = row + dx, col + dy
                while 0 <= x < self.size and 0 <= y < self.size and self.board[x][y] == player:
                    count += 1
                    x += dx
                    y += dy
            
            if count >= 5:
                return True
                
        return False
    
    def is_board_full(self):
        return np.all(self.board != 0)

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
            new_game = deepcopy(game)
            new_game.make_move(*move)
            
            score = self.minimax(new_game, self.depth - 1, False)
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
                new_game = deepcopy(game)
                new_game.make_move(*move)
                score = self.minimax(new_game, depth - 1, False)
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                new_game = deepcopy(game)
                new_game.make_move(*move)
                score = self.minimax(new_game, depth - 1, True)
                best_score = min(best_score, score)
            return best_score
    
    def alphabeta_move(self, game):
        best_score = -float('inf')
        best_move = None
        alpha = -float('inf')
        beta = float('inf')
        
        for move in self.get_possible_moves(game):
            new_game = deepcopy(game)
            new_game.make_move(*move)
            
            score = self.alphabeta(new_game, self.depth - 1, False, alpha, beta)
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
                new_game = deepcopy(game)
                new_game.make_move(*move)
                score = self.alphabeta(new_game, depth - 1, False, alpha, beta)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
            return best_score
        else:
            best_score = float('inf')
            for move in self.get_possible_moves(game):
                new_game = deepcopy(game)
                new_game.make_move(*move)
                score = self.alphabeta(new_game, depth - 1, True, alpha, beta)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break
            return best_score
    
    def get_possible_moves(self, game):
        empty_cells = []
        for i in range(game.size):
            for j in range(game.size):
                if game.board[i][j] == 0:
                    has_neighbor = False
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < game.size and 0 <= nj < game.size and game.board[ni][nj] != 0:
                                has_neighbor = True
                                break
                        if has_neighbor:
                            break
                    if has_neighbor or np.all(game.board == 0):
                        empty_cells.append((i, j))
        return empty_cells if empty_cells else [(game.size//2, game.size//2)]
    
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
            (0, 1), (1, 0), (1, 1), (1, -1)
        ]
        
        for i in range(game.size):
            for j in range(game.size):
                if game.board[i][j] == self.player_num:
                    for di, dj in directions:
                        score += self.evaluate_direction(game, i, j, di, dj, self.player_num)
                elif game.board[i][j] != 0:
                    for di, dj in directions:
                        score -= self.evaluate_direction(game, i, j, di, dj, 3 - self.player_num)
        
        return score
    
    def evaluate_direction(self, game, row, col, di, dj, player):
        count = 0
        empty = 0
        
        for step in range(1, 5):
            r, c = row + di * step, col + dj * step
            if 0 <= r < game.size and 0 <= c < game.size:
                if game.board[r][c] == player:
                    count += 1
                elif game.board[r][c] == 0:
                    empty += 1
                    break
                else:
                    break
        
        for step in range(1, 5):
            r, c = row - di * step, col - dj * step
            if 0 <= r < game.size and 0 <= c < game.size:
                if game.board[r][c] == player:
                    count += 1
                elif game.board[r][c] == 0:
                    empty += 1
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
        self.ai_algorithm = "alphabeta"
        
        # Colors and styles
        self.bg_color = "#f0d9b5"
        self.line_color = "#000000"
        self.black_stone_color = "#2c2c2c"
        self.white_stone_color = "#f9f9f9"
        
        # Game state
        self.game = Gomoku(self.board_size)
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
            self.game.current_player != self.human_player or
            self.game.game_over or self.ai_thinking):
            return
        
        col = int((event.x - self.board_offset_x) // self.cell_size)
        row = int((event.y - self.board_offset_y) // self.cell_size)
        
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.game.board[row][col] == 0:
            self.highlight_cell(row, col)

    def highlight_cell(self, row, col):
        """Show visual feedback when hovering over a cell"""
        if self.highlight:
            self.canvas.delete(self.highlight)
        
        x = self.board_offset_x + col * self.cell_size
        y = self.board_offset_y + row * self.cell_size
        stone_size = int(self.cell_size * 0.4)  # Smaller preview stone
        
        if self.game.current_player == 1:  # Black
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
        
        self.new_game_btn = tk.Button(
            control_frame, text="New Game", command=self.new_game,
            bg="#4a7a8c", fg="white", font=("Arial", 10, "bold")
        )
        self.new_game_btn.pack(side=tk.LEFT, padx=5)
        
        self.game_mode_btn = tk.Button(
            control_frame, text="Switch to AI vs AI", 
            command=self.toggle_game_mode,
            bg="#3a5a6c", fg="white", font=("Arial", 10)
        )
        self.game_mode_btn.pack(side=tk.LEFT, padx=5)
        
        self.player_indicator = tk.Label(
            control_frame, text="Current: Black (●)", 
            font=("Arial", 10, "bold"), fg="black"
        )
        self.player_indicator.pack(side=tk.RIGHT, padx=5)
    
    def create_board(self):
        self.board_frame = tk.Frame(self.root)
        self.board_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=5)
        
        self.canvas = tk.Canvas(
            self.board_frame, bg=self.bg_color,
            highlightthickness=0, borderwidth=0
        )
        
        self.v_scroll = ttk.Scrollbar(self.board_frame, orient="vertical", command=self.canvas.yview)
        self.h_scroll = ttk.Scrollbar(self.board_frame, orient="horizontal", command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set, xscrollcommand=self.h_scroll.set)
        
        self.v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.canvas.pack(expand=True, fill=tk.BOTH, side=tk.LEFT)
        
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
        
        shadow_offset = size // 10
        draw.ellipse(
            [(shadow_offset, shadow_offset), 
             (size*2 - shadow_offset, size*2 - shadow_offset)],
            fill="#00000044"
        )
        
        draw.ellipse(
            [(0, 0), (size*2 - shadow_offset, size*2 - shadow_offset)],
            fill=color
        )
        
        if highlight:
            highlight_size = size // 3
            draw.ellipse(
                [(size//2, size//2), 
                 (size//2 + highlight_size, size//2 + highlight_size)],
                fill="#ffffff88"
            )
        
        return ImageTk.PhotoImage(img.resize((size, size), Image.LANCZOS))
    
    def draw_board(self, event=None):
        if self.game is None:
            return
        
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.cell_size = min(
            canvas_width // (self.board_size + 1),
            canvas_height // (self.board_size + 1)
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
                self.board_offset_y - 15,
                text=str(i), font=("Arial", 8)
            )
            
            self.canvas.create_text(
                self.board_offset_x - 15,
                self.board_offset_y + i * self.cell_size,
                text=str(i), font=("Arial", 8)
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
                        anchor=tk.NW, tags=f"stone_{i}_{j}"
                    )
        
        self.canvas.configure(scrollregion=(
            0, 0,
            self.board_size * self.cell_size + self.board_offset_x * 2,
            self.board_size * self.cell_size + self.board_offset_y * 2
        ))
    
    def on_click(self, event):
        """Handle mouse clicks on the board with precise placement"""
        if (self.game_mode != "human_vs_ai" or 
            self.game.current_player != self.human_player or
            self.game.game_over or self.ai_thinking or
            self.animation_in_progress):
            return
        
        # Precise cell calculation using floor division
        col = int((event.x - self.board_offset_x) // self.cell_size)
        row = int((event.y - self.board_offset_y) // self.cell_size)
        
        # Ensure we're within bounds and cell is empty
        if 0 <= row < self.board_size and 0 <= col < self.board_size and self.game.board[row][col] == 0:
            # Remove any hover preview
            if self.highlight:
                self.canvas.delete(self.highlight)
                self.highlight = None
            
            if self.game.make_move(row, col):
                self.last_move = (row, col)
                self.animate_stone(row, col)
                
                if not self.game.game_over and self.game_mode == "human_vs_ai":
                    self.root.after(500, self.ai_move)
    def ai_move(self):
        if self.game.game_over or self.game.current_player == self.human_player:
            return
        
        self.ai_thinking = True
        self.status_bar.config(text="AI is thinking...")
        self.update_ui()
        
        def run_ai():
            ai = AIPlayer(
                self.game.current_player,
                self.ai_algorithm,
                self.ai_depth
            )
            row, col = ai.get_move(self.game)
            return row, col
        
        def on_ai_complete(result):
            row, col = result
            if self.game.make_move(row, col):
                self.last_move = (row, col)
                self.animate_stone(row, col)
            
            self.ai_thinking = False
            self.update_ui()
        
        import threading
        def ai_worker():
            result = run_ai()
            self.root.after(0, lambda: on_ai_complete(result))
        
        threading.Thread(target=ai_worker, daemon=True).start()
    
    def animate_stone(self, row, col):
        if not hasattr(self, 'cell_size') or self.cell_size == 0:
            return
        
        self.animation_in_progress = True
        stone_size = int(self.cell_size * 0.9)
        x = self.board_offset_x + col * self.cell_size
        y = self.board_offset_y + row * self.cell_size
        
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
        
        if self.game.game_over:
            if self.game.winner == 1:
                self.player_indicator.config(text="Black wins!", fg="black")
            elif self.game.winner == 2:
                self.player_indicator.config(text="White wins!", fg="black")
            else:
                self.player_indicator.config(text="Game ended in draw!", fg="black")
        else:
            player_text = "Black (●)" if self.game.current_player == 1 else "White (○)"
            color = "black" if self.game.current_player == 1 else "white"
            self.player_indicator.config(text=f"Current: {player_text}", fg=color)
        
        if self.ai_thinking:
            self.status_bar.config(text="AI is thinking...")
        elif self.game.game_over:
            if self.game.winner == 1:
                self.status_bar.config(text="Game over - Black wins!")
            elif self.game.winner == 2:
                self.status_bar.config(text="Game over - White wins!")
            else:
                self.status_bar.config(text="Game over - Draw!")
        else:
            if self.game_mode == "human_vs_ai" and self.game.current_player == self.human_player:
                self.status_bar.config(text="Your turn - click on the board to place a stone")
            else:
                self.status_bar.config(text="Waiting for AI move...")
        
        self.draw_board()
    
    def new_game(self):
        self.game = Gomoku(self.board_size)
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
        if self.game.game_over or self.game_mode != "ai_vs_ai":
            return
        
        self.ai_move()
        
        # Only continue if still in AI vs AI mode
        if self.game_mode == "ai_vs_ai" and not self.game.game_over:
            self.root.after(1000, self.ai_vs_ai_loop)

    def ai_move(self):
        """Make AI move with proper UI updates"""
        if self.game.game_over or (self.game_mode == "human_vs_ai" and self.game.current_player == self.human_player):
            return
        
        self.ai_thinking = True
        self.status_bar.config(text="AI is thinking...")
        self.update_ui()
        self.root.update()  # Force UI update
        
        def run_ai():
            ai = AIPlayer(
                self.game.current_player,
                self.ai_algorithm,
                self.ai_depth
            )
            return ai.get_move(self.game)
        
        def on_ai_complete(result):
            row, col = result
            if self.game.make_move(row, col):
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
        
        algo_var = tk.StringVar(value=self.ai_algorithm)
        algo_menu = tk.OptionMenu(
            dialog, algo_var, "minimax", "alphabeta"
        )
        algo_menu.config(width=10)
        algo_menu.grid(row=1, column=1, padx=5, pady=5)
        
        def apply_changes():
            self.ai_depth = depth_var.get()
            self.ai_algorithm = algo_var.get()
            dialog.destroy()
        
        tk.Button(
            dialog, text="OK", command=apply_changes,
            width=10, bg="#4a7a8c", fg="white"
        ).grid(row=2, column=0, columnspan=2, pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()