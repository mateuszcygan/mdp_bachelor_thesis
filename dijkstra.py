import sys

import mdp

# Idee fuer spaeter:
# number of states hits einbeziehen, um die Loesung zu entwickeln, die auch selten besuchte Zustaende berueksichtigt

# max_value is in this veriosn the min value - we want to get a path with the biggest probability
max_value = -sys.maxsize - 1


def print_shortest_path_table(shortest_path):
    for state, entry in shortest_path.items():
        probability = entry.probability
        previous_state = entry.previous_state
        executed_action = entry.executed_action_in_prev_state
        print(f"{state} : {probability}, {previous_state}, {executed_action}")


class ShortestPathEntry:
    def __init__(
        self, probability=None, previous_state=None, executed_action_in_prev_state=None
    ):
        self.probability = probability
        self.previous_state = previous_state
        self.executed_action_in_prev_state = executed_action_in_prev_state


# Returns an action with the biggest probability between two states
# Note: After each action it possible to go over to each of the states (each state connected with each state),
# but for some of them the probability is equal to 0 (no connection)
def neighbour_biggest_prob(unvisited_states, approximated_prob, current_visit_state):

    # Create copy of unvisited_states so that modifications of it are not visible outside of this function (shallow copy, but strings in Python unmutable)
    unvisited_states_copy = unvisited_states.copy()

    # We shouldn't consider "the loop" transition - going again to the same state from which we start
    unvisited_states_copy.remove(current_visit_state)

    # neihbour_prob - stores the biggest probability to neighbour (2) and responsible for it action (1)
    # state = { (1)action : (2)probability}
    neighbour_prob = {state: {None: None} for state in unvisited_states_copy}

    # Retreive actions that are possible for curr_visited_state
    poss_actions = mdp.get_possible_actions(approximated_prob, current_visit_state)

    # For each of the possible neighbours find the most probably transition
    for state in unvisited_states_copy:

        # Set initial values for max_action and max_pro
        max_action = poss_actions[0]
        max_prob = 0

        for action in poss_actions:

            new_max_prob = approximated_prob[current_visit_state][action][state]

            if new_max_prob > max_prob:

                max_action = action
                max_prob = new_max_prob

                neighbour_prob[state] = {max_action: max_prob}

    # Check if any value in the dictionary is {None: None} - it is not possible (prob = 0.0) to reach a state through any of available actions
    no_neighbours = [
        state for state, value in neighbour_prob.items() if value == {None: None}
    ]

    # Remove the "no-neighbours" from the dictionary
    for state in no_neighbours:
        del neighbour_prob[state]

    return neighbour_prob


# Updates the values of shortest path
# curr_end_state_prob - if end_state already reached, the probability stored in this variable (for comparisson - necessity of further calculations)
def update_shortest_path(
    current_visit_state, shortest_path, neighbours, curr_end_state_prob
):
    return


def dijkstra_alg(mdp_object, approximated_prob, start_state, end_state):

    S = mdp_object.states

    unvisited_states = list(mdp_object.states)

    shortest_path = {}

    # Initialize the shortest path table
    for state in S:
        shortest_path[state] = ShortestPathEntry(max_value, None, None)
    shortest_path[start_state].probability = 1

    print_shortest_path_table(shortest_path)

    shortest_path_value = -1  # For comparisson if it is required to calculate further

    # previous_nodes - dict that stores the trajectory of the current best known path for each node
    previous_nodes = {}

    # Initialize the current visiting node to node from which we start
    current_visit_state = start_state

    # Execute the algorithm until the end_node is visited
    # while end_state in unvisited_states:
    for i in range(1):

        # Extract probabilities to neighbours (only states (neighbours) that haven't been visited are considered)
        neighbours = neighbour_biggest_prob(
            unvisited_states, approximated_prob, current_visit_state
        )
        update_shortest_path(
            current_visit_state, shortest_path, neighbours, shortest_path_value
        )
    return
