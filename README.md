\documentclass{article}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{geometry}
\geometry{margin=1in}

\definecolor{codegray}{gray}{0.95}

\lstset{
  backgroundcolor=\color{codegray},
  basicstyle=\ttfamily,
  frame=single,
  breaklines=true
}

\title{Gomoku (Five in a Row) \\ \large Python Project Documentation}
\author{Ahmed Ashraf Attia}
\date{}

\begin{document}

\maketitle

\section*{Overview}
\textbf{Gomoku (Five in a Row)} is a classic strategy board game implemented in Python using Tkinter. This project supports:
\begin{itemize}
    \item Human vs AI gameplay
    \item AI vs AI battles
    \item Smart AI using Minimax and Alpha-Beta pruning
    \item Customizable difficulty and board size
\end{itemize}

\section*{Features}
\begin{itemize}
    \item Interactive GUI with hover preview
    \item Two game modes: Human vs AI, AI vs AI
    \item Adjustable AI depth and algorithm
    \item Visual feedback on moves and game outcome
\end{itemize}

\section*{Getting Started}

\subsection*{Prerequisites}
\begin{lstlisting}[language=bash]
pip install pillow numpy
\end{lstlisting}

\subsection*{Running the Game}
\begin{lstlisting}[language=bash]
python gomoku.py
\end{lstlisting}

\section*{How to Play}
\begin{itemize}
    \item Your goal is to align five stones in a row (vertical, horizontal, or diagonal).
    \item In Human vs AI mode, you play as Black (●).
    \item Click on the board to place your move.
\end{itemize}

\subsection*{Menu Options}
\begin{itemize}
    \item \texttt{Game > New Game} – Restart the game
    \item \texttt{Game > Human vs AI / AI vs AI} – Choose mode
    \item \texttt{Settings > Board Size} – Customize the grid
    \item \texttt{Settings > AI Difficulty} – Set search depth
\end{itemize}

\section*{AI Algorithms}

\subsection*{Minimax}
A standard decision-making algorithm that simulates all possible game outcomes up to a certain depth to determine the optimal move.

\subsection*{Alpha-Beta Pruning}
An optimization of Minimax that skips unnecessary branches, significantly improving performance without affecting the result.

\section*{Project Structure}
\begin{itemize}
    \item \textbf{Gomoku.py}: Board logic and win detection
    \item \textbf{AIPlayer.py}: AI algorithms and move selection
    \item \textbf{GomokuGUI.py}: User interface and interactions
\end{itemize}

\end{document}
