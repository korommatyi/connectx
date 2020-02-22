#!/usr/bin/env python3

from typing import NamedTuple, List, Tuple, Iterable, Callable, Optional
from math import sqrt, log10
from datetime import datetime, timedelta


class State(NamedTuple):
    board: Tuple[int, ...]
    columns: int
    rows: int
    inarow: int


class Action(NamedTuple):
    incolumn: int
    player: int


def reward_in_line(line: Iterable[int], state: State, score: Callable[[int], int], win_score: int):
    line = list(line)
    p1_score = 0
    p2_score = 0
    start = 0
    p1_in_interval = 0
    p2_in_interval = 0
    for i in line[0:start + state.inarow]:
        if state.board[i] == 1:
            p1_in_interval += 1
        elif state.board[i] == 2:
            p2_in_interval += 1

    in_elem = 0
    out_elem = 0
    while start + state.inarow <= len(line):
        if not ((in_elem == out_elem == 1) or (in_elem == out_elem == 2)):
            if p1_in_interval == 0:
                if p2_in_interval == state.inarow:
                    return 0, win_score
                p2_score += score(p2_in_interval)
            elif p2_in_interval == 0:
                if p1_in_interval == state.inarow:
                    return win_score, 0
                p1_score += score(p1_in_interval)

        out_elem = state.board[line[start]]
        if out_elem == 1:
            p1_in_interval -= 1
        elif out_elem == 2:
            p2_in_interval -= 1

        if start + state.inarow < len(line):
            in_elem = state.board[line[start + state.inarow]]
            if in_elem == 1:
                p1_in_interval += 1
            elif in_elem == 2:
                p2_in_interval += 1

        start += 1

    return p1_score, p2_score


def lines(state: State):
    for row in range(state.rows):
        yield range(row*state.columns, (row + 1)*state.columns)
    for col in range(state.columns):
        yield range((state.rows - 1)*state.columns + col, col - 1, -state.columns)
    for lr_diag in range(state.inarow, state.columns + state.rows - state.inarow + 1):
        if lr_diag <= state.rows:
            start_x = 0
            start_y = state.rows - lr_diag
        else:
            start_x = lr_diag - state.rows
            start_y = 0
        yield ((start_y + i) * state.columns + start_x + i
               for i in range(min(state.columns - start_x, state.rows - start_y)))
    for rl_diag in range(state.inarow, state.columns + state.rows - state.inarow + 1):
        if rl_diag < state.columns:
            start_x = rl_diag - 1
            start_y = 0
        else:
            start_x = state.columns - 1
            start_y = rl_diag - state.columns
        yield ((start_y + i) * state.columns + start_x - i
               for i in range(min(start_x + 1, state.rows - start_y)))


def reward_heuristic(state: State, score: Callable[[int], int], win_score: int):
    p1_score = 0
    p2_score = 0
    max_possible = win_score
    for line in lines(state):
        s1, s2 = reward_in_line(line, state, score, win_score)
        if s1 >= max_possible:
            return s1, 0
        elif s2 >= max_possible:
            return 0, s2
        else:
            p1_score += s1
            p2_score += s2
    return p1_score, p2_score


def who_won(state: State):
    win_score = 1
    p_1, p_2 = reward_heuristic(state, score=lambda x: 0, win_score=win_score)
    if p_1 == win_score:
        return 1
    if p_2 == win_score:
        return 2
    return None


def get_possible_moves(state: State):
    return [i for i, p in enumerate(state.board[0:state.columns]) if p == 0]


def state_transition(state: State, action: Action):
    b = list(state.board)
    for row in range(state.rows - 1, -1, -1):
        if b[row*state.columns + action.incolumn] == 0:
            b[row*state.columns + action.incolumn] = action.player
            new_state = State(tuple(b), state.columns, state.rows, state.inarow)
            return new_state
    return None


def f(state: State, player: int, score: Callable[[int], int], win_score: int):
    winner = who_won(state)
    if winner == player:
        return [0 for _ in range(state.columns)], 1
    if winner is not None:
        return [0 for _ in range(state.columns)], 0

    possible_moves = get_possible_moves(state)
    # if no winner and no possible moves => draw
    if not possible_moves:
        return [0 for _ in range(state.columns)], 0

    def win_prob(c):
        if state.board[c] != 0:
            return 0

        new_state = state_transition(state, Action(c, player))
        reward = reward_heuristic(new_state, score, win_score)
        i = player - 1
        other_i = 1 - i
        if reward[i] >= win_score:
            return 1
        if reward[other_i] >= win_score:
            return 0
        return reward[i] / (reward[i] + reward[other_i] + 0.000001)

    win_probs = [win_prob(c) for c in range(state.columns)]
    if 1 in win_probs:
        p = [0 for _ in range(state.columns)]
        p[win_probs.index(1)] = 1
        v = 1
    else:
        norm = sum(win_probs)
        if norm == 0:
            p = [0 for _ in range(state.columns)]
            p[possible_moves[0]] = 1
            v = 0
        else:
            p = [wp/norm for wp in win_probs]
            v = 0
            for wp_i, p_i in zip(win_probs, p):
                v += p_i*wp_i
    return p, v


class Node(NamedTuple):
    N: List[int]
    Q: List[float]
    P: List[float]
    S: List[Optional[State]]
    v: int
    state: State
    whose_turn: int
    end_state: bool
    who_won: Optional[int]
    possible_moves: List[int]


def new_leaf(state: State, whose_turn: int, score: Callable[[int], int], win_score: int):
    p, v = f(state, whose_turn, score, win_score)
    winner = who_won(state)
    possible_moves = get_possible_moves(state)
    return Node(
        N=[0 for _ in range(state.columns)],
        Q=[0 for _ in range(state.columns)],
        P=p,
        S=[None for _ in range(state.columns)],
        v=v,
        state=state,
        whose_turn=whose_turn,
        end_state=winner or not possible_moves,
        who_won=winner,
        possible_moves=possible_moves,
    )


def argmax(l: List[int]):
    # we assume elements are positive
    best = -1
    idx = -1
    for i, v in enumerate(l):
        if v > best:
            best = v
            idx = i
    return idx


def select(node: Node, c_1: float, c_2: int):
    N = node.N
    Q = node.Q
    P = node.P
    sum_N = sum(N)
    sqrt_sum_N = sqrt(sum_N)
    xs = [Q[i] + P[i]*(sqrt_sum_N / (1 + N[i]))*(c_1 + log10((sum_N + c_2 + 1) / c_2))
          for i in node.possible_moves]
    best = argmax(xs)
    return node.possible_moves[best]


def other_player(p: int):
    if p == 1:
        return 2
    else:
        return 1


def mcts_agent(observation, configuration, score_exp: int = 3, c_1: float = 1.25, c_2: int = 20000,
               search_time_in_s: int = 4):
    score = lambda x: pow(x, score_exp)
    win_score = pow(configuration.inarow, score_exp*2)
    me = observation.mark
    stats = {}
    start_state = State(tuple(observation.board), configuration.columns, configuration.rows,
                        configuration.inarow)
    stats[start_state] = new_leaf(start_state, me, score, win_score)

    ts = datetime.now()
    start_ts = ts
    end_ts = start_ts + timedelta(seconds=search_time_in_s)
    avg_sim_time = timedelta(seconds=0)
    has_time_for_one_more = True
    sim_counter = 0
    while has_time_for_one_more:
        state = start_state
        node = stats[state]
        actions = []
        non_leaf_nodes = []
        leaf = False

        while not leaf and not node.end_state:
            non_leaf_nodes.append(node)
            a = select(node, c_1, c_2)
            actions.append(a)

            if node.S[a] is None:
                whose_turn = node.whose_turn
                state = state_transition(node.state, Action(a, whose_turn))
                node.S[a] = state
                if state in stats:
                    node = stats[state]
                else:
                    node = new_leaf(state, other_player(whose_turn), score, win_score)
                    stats[state] = node
                    leaf = True
            else:
                node = stats[node.S[a]]

        last_node = node
        v = last_node.v
        opponent_v = 0 if last_node.end_state and last_node.who_won is None else 1 - v
        for n, a in zip(non_leaf_nodes, actions):
            update = v if n.whose_turn == last_node.whose_turn else opponent_v
            n.Q[a] = (n.N[a] * n.Q[a] + update) / (n.N[a] + 1)
            n.N[a] += 1

        sim_end_ts = datetime.now()
        diff = sim_end_ts - ts
        ts = sim_end_ts
        avg_sim_time *= 0.7
        avg_sim_time += 0.3*diff
        has_time_for_one_more = ts + avg_sim_time < end_ts
        sim_counter += 1

    start_node = stats[start_state]
    if sum(start_node.Q) == 0:
        move = start_node.possible_moves[0]
    else:
        move = argmax(start_node.Q)
    #print(f'{me}: col {move}, win prob {round(start_node.Q[move]*100, 2)}%, sim time {ts - start_ts}, num of simulations {sim_counter}')
    return move


def fine_tuned_rule_based_mcts_agent(observation, configuration):
    return mcts_agent(observation, configuration, score_exp=2, c_1=0.8, c_2=1500, search_time_in_s=4)
