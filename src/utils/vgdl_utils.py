def state_to_vgdl(state):
    """
    Converte um estado do jogo em texto usando VGDL.
    Exemplo para Sokoban:
    """
    player_pos = state['player_position']
    boxes = state['boxes']
    goals = state['goals']
    return (
        f"Player at ({player_pos[0]}, {player_pos[1]}). "
        f"Boxes: {', '.join([f'({x}, {y})' for x, y in boxes])}. "
        f"Goals: {', '.join([f'({x}, {y})' for x, y in goals])}."
    )