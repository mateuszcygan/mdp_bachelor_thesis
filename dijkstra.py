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


def set_shortest_path_entry(
    shortest_path_table,
    neighbour,
    probability,
    previous_state,
    executed_action_in_prev_state,
):

    shortest_path_table[neighbour].probability = probability
    shortest_path_table[neighbour].previous_state = (
        previous_state  # previous_state for neighbour is in the shortest_path table the state, that is visited now (current_visit_state)
    )
    shortest_path_table[neighbour].executed_action_in_prev_state = (
        executed_action_in_prev_state
    )

    return shortest_path_table


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
    start_state,
    end_state,
    current_visit_state,
    shortest_path_value,
    shortest_path,
    neighbours,
):
    for neighbour, entry in neighbours.items():

        new_path_value = entry.probability
        executed_action = entry.action
        old_path_value = shortest_path[neighbour].probability
        previous_state = shortest_path[current_visit_state].previous_state

        # Neighbour state for which no path was found before
        # Only the case when we are in the start_state - after leaving start_state, you leave to connected states that have as previous state "start_state"
        if previous_state == None and current_visit_state == start_state:
            if new_path_value > old_path_value:
                shortest_path = set_shortest_path_entry(
                    shortest_path,
                    neighbour,
                    new_path_value,
                    current_visit_state,
                    executed_action,
                )
        else:
            new_path_value = (
                new_path_value * shortest_path[current_visit_state].probability
            )
            if new_path_value > old_path_value:
                shortest_path = set_shortest_path_entry(
                    shortest_path,
                    neighbour,
                    new_path_value,
                    current_visit_state,
                    executed_action,
                )

        # Check if the value of shortest_path_value should be updated
        if neighbour == end_state:
            if shortest_path_value == -1:
                shortest_path_value = new_path_value
            elif new_path_value > shortest_path_value:
                shortest_path_value = new_path_value

    return shortest_path, shortest_path_value


# Compares if there is at least one entry in shortest_path that is bigger than the current shortest_path_value
# True => True - there is a chance that more probable path can be found
def compare_shortest_path_value(
    shortest_path_table, shortest_path_value, unvisited_states
):
    shortest_path_value_bigger = True
    for state, entry in shortest_path_table.items():
        if state in unvisited_states:
            # There should be at least one value in the table among unvisited states that is bigger than shortest_path_value - chance to find more probable path
            # If shortest_path_value is bigger or equal than each entry in the shortest_path table, there is no need to calculate further - no chance to find more probable path
            shortest_path_value_bigger = shortest_path_value_bigger and (
                shortest_path_value >= entry.probability
            )

    return shortest_path_value_bigger


# Chooses the state with the biggest so far calculated probability in the shortest_path_table
def choose_next_state_to_visit(shortest_path_table, unvisited_states):
    max_state_value = -sys.maxsize - 1
    max_state = None

    for state, entry in shortest_path_table.items():
        if (
            state in unvisited_states
            and entry.probability is not None
            and entry.probability > max_state_value
        ):
            max_state_value = entry.probability
            max_state = state

    return max_state


def get_shortest_path_actions(shortest_path_table, end_state):

    # Get the start state
    for state, entry in shortest_path_table.items():
        if (
            entry.probability == 1
            and entry.previous_state == None
            and entry.executed_action_in_prev_state == None
        ):
            start_state = state

    entry_state = end_state  # State which shortest_path entry we consider
    shortest_path_actions = []

    while entry_state != start_state:

        state = shortest_path_table[entry_state].previous_state
        action_to_execute = shortest_path_table[
            entry_state
        ].executed_action_in_prev_state

        shortest_path_actions_entry = {state: action_to_execute}
        shortest_path_actions.insert(0, shortest_path_actions_entry)

        entry_state = state

    return shortest_path_actions


def dijkstra_alg(mdp_object_states, approximated_prob, start_state, end_state):

    unvisited_states = list(mdp_object_states)

    shortest_path = create_shortest_path_table(mdp_object_states, start_state)
    print_shortest_path_table(shortest_path)

    shortest_path_value = (
        -1
    )  # For comparisson if it is required to calculate further (probability of reaching the end state)

    # Initialize the current visiting node to node from which we start
    current_visit_state = start_state

    # Execute the algorithm until the end_node is visited
    while end_state in unvisited_states:

        # Extract probabilities to neighbours (only states (neighbours) that haven't been visited yet are considered)
        neighbours = neighbour_biggest_prob(
            unvisited_states, approximated_prob, current_visit_state
        )
        print_neigh_prob_table(current_visit_state, neighbours)

        shortest_path, shortest_path_value = update_shortest_path(
            start_state,
            end_state,
            current_visit_state,
            shortest_path_value,
            shortest_path,
            neighbours,
        )
        print_shortest_path_table(shortest_path)
        print("shortest_path_value:", shortest_path_value)

        # Check if so far calculated shortest_path_value is bigger than all entries in the shortest_path table
        shortest_path_value_bigger = compare_shortest_path_value(
            shortest_path, shortest_path_value, unvisited_states
        )
        print(shortest_path_value_bigger)

        # If the shortest_path_value is bigger than all entries of unvisited states in shortest_path table, then there is no more chance to find more probable path
        if shortest_path_value_bigger:
            shortest_path_actions = get_shortest_path_actions(shortest_path, end_state)
            print("Termination condition")
            return shortest_path_actions

        unvisited_states.remove(current_visit_state)
        print("unvisited states:", unvisited_states)
        current_visit_state = choose_next_state_to_visit(
            shortest_path, unvisited_states
        )

    shortest_path_actions = get_shortest_path_actions(shortest_path, end_state)

    return shortest_path_actions
