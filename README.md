# Gomoku - Five in a Row Game

![Gomoku Game Screenshot](https://github.com/user-attachments/assets/a5a8f383-df1c-4626-bd56-dabebc1dab09)

## Description

Gomoku, also known as Five in a Row, is a classic strategy board game where two players alternate placing stones on a grid, aiming to be the first to get five stones in a row (horizontally, vertically, or diagonally).

This implementation features:
- Human vs AI gameplay
- AI vs AI simulation
- Configurable board size (5x5 to 25x25)
- Adjustable AI difficulty
- Two AI algorithms (Minimax and Alpha-Beta Pruning)
- Smooth animations and visual feedback

## Requirements

- Python 3.x
- Tkinter (usually included with Python)
- NumPy (`pip install numpy`)
- Pillow (`pip install Pillow`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/3laa106/AI-Project.git
cd AI-Project
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. (Or manually install dependencies if you don't have a requirements file):
```bash
pip install numpy Pillow
```

4. Run the game:
```bash
python gomoku.py
```


## How to Play

### Basic Rules
- ⚫ **Black plays first** (Player 1)
- 🔄 Players alternate placing stones of their color
- 🎯 First player to get **5 stones in a row** (horizontal, vertical, or diagonal) wins
- 🤝 Game ends in a draw if the board fills up without a winner

### Game Modes

#### Human vs AI
🎮 Play against the computer:
- ✨ Click on any empty intersection to place your stone
- 🤖 AI will automatically make its move
- 👁️ Hover preview shows where your stone will be placed

#### AI vs AI
👀 Watch two AI players compete:
- ⚙️ Game plays automatically
- 🔬 Perfect for testing AI strategies
- ⏱️ Adjustable move delay in code

### Controls

#### Menu Options
| Path | Action |
|------|--------|
| `Game > New Game` | Starts fresh game |
| `Game > Human vs AI` | Play against computer |
| `Game > AI vs AI` | Spectate AI battle |
| `Settings > Board Size` | Change grid (5-25) |
| `Settings > AI Difficulty` | Set AI strength (1-5) |

#### Game Board
- 🖱️ **Left-click**: Place stone (Human mode)
- ✨ **Hover**: Preview placement
- 🔄 **Animations**: Stone placement effects


