import sys

import mdp

# Idee fuer spaeter:
# number of states hits einbeziehen, um die Loesung zu entwickeln, die auch selten besuchte Zustaende berueksichtigt

# max_value is in this version the min value - we want to get a path with the biggest probability
max_value = -sys.maxsize - 1


class ShortestPathEntry:
    def __init__(
        self, probability=None, previous_state=None, executed_action_in_prev_state=None
    ):
        self.probability = probability
        self.previous_state = previous_state
        self.executed_action_in_prev_state = executed_action_in_prev_state


def create_shortest_path_table(states, start_state):
    shortest_path = {}

    # Initialize the shortest path table
    for state in states:
        shortest_path[state] = ShortestPathEntry(max_value, None, None)
    shortest_path[start_state].probability = 1
    return shortest_path


def print_shortest_path_table(shortest_path):
    print("Shortest path table:")
    for state, entry in shortest_path.items():
        probability = entry.probability
        previous_state = entry.previous_state
        executed_action = entry.executed_action_in_prev_state
        print(f"{state} : {probability}, {previous_state}, {executed_action}")
    print("\n")


class NeighbourProbabilityEntry:
    def __init__(self, action=None, probability=None):
        self.action = action
        self.probability = probability


def create_neigh_prob_table(unvisited_states):
    neighbour_prob = {
        state: NeighbourProbabilityEntry(None, None) for state in unvisited_states
    }
    return neighbour_prob


def print_neigh_prob_table(curr_visit_state, neighbour_prob):
    print(curr_visit_state, "neighbours:")
    for state, entry in neighbour_prob.items():
        action = entry.action
        probability = entry.probability
        print(f"{state} : {action}, {probability}")
    print("\n")


# Returns an action with the biggest probability between two states
# Note: After each action it possible to go over to each of the states (each state connected with each state),
# but for some of them the probability is equal to 0 (no connection)
def neighbour_biggest_prob(unvisited_states, approximated_prob, current_visit_state):

    # Create copy of unvisited_states so that modifications of it are not visible outside of this function (shallow copy, but strings in Python unmutable)
    unvisited_states_copy = unvisited_states.copy()

    # We shouldn't consider "the loop" transition - going again to the same state from which we start
    unvisited_states_copy.remove(current_visit_state)

    neighbour_prob = create_neigh_prob_table(unvisited_states_copy)

    # Retreive actions that are possible for curr_visited_state
    poss_actions = mdp.get_possible_actions(approximated_prob, current_visit_state)

    # For each of the possible neighbours find the most probably transition
    for state in unvisited_states_copy:

        # Set initial values for max_action and max_pro
        max_action = poss_actions[0]
        max_prob = 0

        for action in poss_actions:

            new_max_prob = approximated_prob[current_visit_state][action][state]

            # Note: if all probabilities are equal to zero, NeighbourProbabilityEntry values remain None
            if new_max_prob > max_prob:

                max_action = action
                max_prob = new_max_prob

                neighbour_prob[state].action = max_action
                neighbour_prob[state].probability = max_prob

    # Check if both values in NeighbourProbabilityEntry are equals to None
    no_neighbours = [
        state
        for state in unvisited_states_copy
        if neighbour_prob[state].action == None
        and neighbour_prob[state].probability == None
    ]

    # Remove the "no-neighbours" from the dictionary
    for state in no_neighbours:
        del neighbour_prob[state]

    return neighbour_prob


# Updates the values of shortest path
# shortest_path_value - if end_state already reached, the probability stored in this variable (for comparisson - necessity of further calculations)
def update_shortest_path(
    start_state, current_visit_state, shortest_path_value, shortest_path, neighbours
):
    for neighbour, entry in neighbours:

        path_value = entry.probability

        # Neighbour state directly connected to start_state
        if shortest_path[neighbour].previous_state == start_state:
            if entry.probability > shortest_path[neighbour].probability:

                shortest_path[neighbour].probability = entry.probability
                shortest_path[neighbour].previous_state = current_visit_state
                shortest_path[neighbour].executed_action_in_prev_state = entry.action

        # Neighbour state for which no path was found before
        if shortest_path[neighbour].previous_state == None:
            if current_visit_state == start_state:

                shortest_path[neighbour].probability = entry.probability
                shortest_path[neighbour].previous_state = current_visit_state
                shortest_path[neighbour].executed_action_in_prev_state = entry.action

            # Neighbour state for which no path was found before and the path leads NOT DIRECTLY (one transition) from start state
            else:
                # Extract the state from which path leads to curr_visit_state
                predecessor = shortest_path[current_visit_state].previous_state

                while predecessor != start_state:

                    # Extract the state from which path leads to start_state
                    path_value = path_value * shortest_path[predecessor].probability
                    predecessor = shortest_path[predecessor].previous_state

                shortest_path[neighbour].probability = path_value
                shortest_path[neighbour].previous_state = current_visit_state
                shortest_path[neighbour].executed_action_in_prev_state = entry.action


def dijkstra_alg(mdp_object, approximated_prob, start_state, end_state):

    S = mdp_object.states
    unvisited_states = list(mdp_object.states)

    shortest_path = create_shortest_path_table(S, start_state)
    print_shortest_path_table(shortest_path)

    shortest_path_value = (
        -1
    )  # For comparisson if it is required to calculate further (probability of reaching the end state)

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
        print_neigh_prob_table(current_visit_state, neighbours)

        # update_shortest_path(
        #     current_visit_state, shortest_path, neighbours, shortest_path_value
        # )
    return
