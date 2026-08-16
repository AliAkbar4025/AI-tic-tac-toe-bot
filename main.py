from NeuralNetwork import NeuralNetwork
import numpy as np
import pandas as pd

LINES = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

SYMBOLS = {0: " ", 1: "X", -1: "O"}


def render_game(tiles_list=[0,0,0,0,0,0,0,0,0]):
    t = [SYMBOLS[v] for v in tiles_list]

    print(f"""
            {t[0]} | {t[1]} | {t[2]}
            ----------
            {t[3]} | {t[4]} | {t[5]}
            ----------
            {t[6]} | {t[7]} | {t[8]}
        """)


def check_winner(tiles_list):
    for a, b, c in LINES:
        s = tiles_list[a] + tiles_list[b] + tiles_list[c]

        if s == 3:
            return 1
        
        if s == -3:
            return -1
        
    if 0 not in tiles_list:
        return 0  # draw
    
    return None  # game still going


def get_human_move(tiles_list, player):
    while True:
        raw = input(f"Your move ({SYMBOLS[player]}) — cell 1-9: ")

        if not raw.isdigit() or not (1 <= int(raw) <= 9):
            print("Enter a number 1-9.")
            continue
        idx = int(raw) - 1

        if tiles_list[idx] != 0:
            print("That cell is taken.")
            continue
        return idx


def get_agent_move(tiles_list, player):
    
    model = NeuralNetwork()
    model = model.load("models/easy.npz")
    data = pd.DataFrame({
            "cell0": tiles_list[0], "cell1": tiles_list[1],
            "cell2": tiles_list[2], "cell3": tiles_list[3],
            "cell4": tiles_list[4], "cell5": tiles_list[5],
            "cell6": tiles_list[6], "cell7": tiles_list[7],
            "cell8": tiles_list[8], "player_to_move": player},index=[1])
    move = model.predict(data)
    print(move)
    return move[0]


def game_init():
    choice = input("Choose: O or X: ").strip().upper()
    human = 1 if choice == "X" else -1
    return human


def main():
    human = game_init()
    agent = -human
    tiles_list = [0] * 9
    current_player = 1  # X always starts

    render_game(tiles_list)

    while True:
        if current_player == human:
            move = get_human_move(tiles_list, current_player)
        else:
            move = get_agent_move(tiles_list, current_player)
            print(f"Agent ({SYMBOLS[agent]}) plays {move + 1}")

        tiles_list[move] = current_player
        render_game(tiles_list)

        result = check_winner(tiles_list)
        if result is not None:
            
            if result == 0:
                print("Draw!")
            else:
                winner = "You" if result == human else "Agent"
                print(f"{winner} win!")
            break

        current_player = -current_player


if __name__ == "__main__":
    main()