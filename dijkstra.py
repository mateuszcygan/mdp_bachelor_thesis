import sys

import mdp

# Idee fuer spaeter:
# number of states hits einbeziehen, um die Loesung zu entwickeln, die auch selten besuchte Zustaende berueksichtigt


# Returns an action with the biggest probability between two states
# Note: After each action it possible to go over to each of the states (each state connected with each state),
# but for some of them the probability is equal to 0 (no connection)
def neighbour_biggest_prob_value(mdp_states, approximated_prob, curr_visited_state):

    # We shouldn't consider "the loop" transition - going again to the same state from which we start
    mdp_states.remove(curr_visited_state)

    # neihbour_prob - stores the biggest probability to neighbour (2) and responsible for it action (1)
    # state = { (1)action : (2)probability}
    neighbour_prob = {state: {None: None} for state in mdp_states}

    # Retreive actions that are possible for curr_visited_state
    poss_actions = mdp.get_possible_actions(approximated_prob, curr_visited_state)

    # For each of the possible neighbours find the most probably transition
    for state in mdp_states:

        # Set initial values for max_action and max_pro
        max_action = poss_actions[0]
        max_prob = 0

        for action in poss_actions:

            new_max_prob = approximated_prob[curr_visited_state][action][state]

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


def dijkstra_alg(mdp_object, approximated_prob, start_state, end_state):

    unvisited_nodes = list(mdp_object.states)

    shortest_path = {}

    # previous_nodes - dict that stores the trajectory of the current best known path for each node
    previous_nodes = {}

    max_value = sys.maxsize

    for node in unvisited_nodes:
        shortest_path[node] = max_value

    # Initialize the starting node with 0
    shortest_path[start_state] = 0
    print(shortest_path)

    # Execute the algorithm until the end_node is visited
    while end_state in unvisited_nodes:
        return

    return
