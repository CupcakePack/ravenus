import helper as unswbc
from helper import Direction, EdgeType
import random

ct: unswbc.Controller
game: unswbc.Game

# Seed so we get the same random generator every time.
random.seed(0)

def execute_turn() -> None:
    ct.output_log("TEST NEW VERSION")
    here = ct.get_position()
    here_tile = ct.get_tile(here)

    directions = Direction.get_direction_list()
    random.shuffle(directions)

    if game.get_round_num() < 350 and ct.get_length() >= 4 and ct.get_unit_count() < 64:
        child_size = ct.get_length() // 2
        if ct.can_split(child_size):
            ct.do_split(child_size)
            return
        
    for direction in directions:
        edge = here_tile.get_edge(direction).get_edge_type()
        if edge == EdgeType.KELP:
            continue

        ahead = ct.get_tile(here.add_dir(direction))

        if ahead.get_dragon() is not None:
            continue

        if ahead.has_pearl():
            ct.make_move(direction)
            return

    safe = []

    for direction in directions:
        edge = here_tile.get_edge(direction).get_edge_type()
        if edge == EdgeType.KELP:
            continue

        ahead = ct.get_tile(here.add_dir(direction))
        if ahead.get_dragon() is not None:
            continue

        safe.append(direction)

    if safe:
        ct.make_move(safe[0])
        return

    if ct.get_length() >= 4 and ct.can_split(ct.get_length() // 2):
        ct.do_split(ct.get_length() // 2)
        return

    ct.make_move(Direction.NORTH)

def main() -> None:
    global ct, game
    ct, game = unswbc.init()

    while unswbc.update(ct, game):
        execute_turn()
        unswbc.end_turn()

if __name__ == "__main__":
    main()
