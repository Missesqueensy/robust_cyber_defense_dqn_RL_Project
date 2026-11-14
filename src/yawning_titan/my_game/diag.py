# diag.py
from make_game_mode import make_game_mode

for p in (0.01, 0.05, 0.3):
    gm = make_game_mode(p_spread=p)
    print(p, gm.red.natural_spreading.capable.value,
          gm.red.natural_spreading.chance.to_connected_node.value)