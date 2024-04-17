import sys

# Idee fuer spaeter:
# number of states hits einbeziehen, um die Loesung zu entwickeln, die auch selten besuchte Zustaende berueksichtigt

# Returns directly connected states to a certain state
def get_neighbour_states(probabilities, current_state):
    return

# Returns value of transition between two states
def get_transition_probability(state1, state2):
    return

def dijkstra_alg(mdp_object, approximated_prob, start_state, end_state):

    unvisited_nodes = list(mdp_object.states)

    shortest_path = {}

    # previous_nodes - dict that stores the trajectory of the current best known path for each node
    previous_nodes = {}

    max_value = sys.maxsize
    print(max_value)

    for node in unvisited_nodes:
        shortest_path[node] = max_value

    # Initialize the starting node with 0
    shortest_path[start_state] = 0

    # Execute the algorithm until the end_node is visited
    while end_state in unvisited_nodes:


    return
