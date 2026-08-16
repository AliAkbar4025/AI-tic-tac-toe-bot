# AI Tic-Tac-Toe Bot

A neural network, built from scratch with just NumPy (no PyTorch/TensorFlow), trained to play optimal tic-tac-toe. Play against it directly in your terminal.

🎥 **Watch the build video:** [YOUTUBE_LINK_HERE](https://youtube.com/your-video-link)

## How it works

1. **Data generation** — every reachable tic-tac-toe board state is enumerated, then labeled with its game-theoretically optimal move using a minimax solver. No external dataset needed — the labels are computed exactly.
2. **Model** — a fully custom feedforward neural network (forward pass, backprop, softmax + cross-entropy, all hand-written) is trained on these (board, best_move) pairs as a 9-class classification problem.
3. **Play** — `main.py` loads the trained weights and lets you play against the model in the terminal, move by move.

## Project structure

```
├── neural_network.py       # NeuralNetwork class (forward/backward pass, save/load)
├── generate_data.py        # Minimax solver + CSV data generation
├── tictactoe_moves.csv     # Generated training data (board state -> best move)
├── model_train.ipynb       # Training notebook
├── tictactoe_model.npz     # Saved trained weights
├── main.py                 # Play the game in your terminal
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train the model** (or skip this and use the included models):
```bash
jupyter notebook model_train.ipynb
```

**Play against it:**
```bash
python main.py
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
