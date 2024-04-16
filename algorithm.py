import copy
import random

import mdp

### ESSENTIAL FUNCTIONS


def assign_initial_approx_probabilities(states, probabilities):
    prob_denominator = len(states)  # Denominator for calculating probabilities

    # Initially all probabilities for transitioning to other state are equal
    initial_prob = 1 / prob_denominator
    approximated_prob = copy.deepcopy(probabilities)

    # Set all transition probabilities to initial probability
    for current_state in approximated_prob:
        for action in approximated_prob[current_state]:
            for following_state in approximated_prob[current_state][action]:
                approximated_prob[current_state][action][following_state] = initial_prob

    return approximated_prob


def execute_action(states, probabilities, current_state, executed_action, states_hits):

    # Get probabilities of transitioning to other states (needed for transition execution)
    states_prob = mdp.get_foll_states_prob_values(
        probabilities, current_state, executed_action
    )

    next_state = random.choices(states, weights=states_prob)[0]

    states_hits[current_state][executed_action][next_state] += 1

    return next_state, states_hits


def update_approx_prob(
    approximated_prob, states_hits, current_state, executed_action, states
):
    # Needed for calculation that provides assigned initial probabilities
    prob_denominator = len(states)

    # Calculate new approximation of probabilities
    hits_list = list(states_hits[current_state][executed_action].values())
    hits_num = sum(
        hits_list
    )  # Sum how many states changes took place for a certain state after executing a certain action

    for state in states:
        approximated_prob[current_state][executed_action][state] = (
            1 + states_hits[current_state][executed_action][state]
        ) / (prob_denominator + hits_num)

    return approximated_prob


### FUNCTIONS RELATED TO 'states_hits' dictionary


def create_states_hits_dictionary(probabilities):
    states_hits = copy.deepcopy(probabilities)

    # Set all values to 0
    for state in states_hits:
        for action in states_hits[state]:
            for next_state in states_hits[state][action]:
                states_hits[state][action][next_state] = 0

    return states_hits


# Calculates number of hits for each of the states
def calculate_states_hits(states_hits):
    states_hits_sum = {}

    # Iterate over each state
    for state, actions in states_hits.items():
        # Initialize sum for current state
        sum_for_state = 0

        # Iterate over actions and sum their values
        for action, transitions in actions.items():
            # Sum values of transitions for current action
            sum_for_state += sum(transitions.values())

        # Add sum to result dictionary
        states_hits_sum[state] = sum_for_state

    return states_hits_sum


# Prints how many hits each state had
def print_states_hits(states_hits_sum):
    print("States hits")
    for state, hit_sum in states_hits_sum.items():
        print(state, ":", hit_sum)


### SYSTEMATIC LEARNING APPROACH


def systematic_learning(
    states, probabilities, approximated_prob, states_hits, current_state, iteration_num
):

    # state_actions - dict that involves data allowing for alternating execution of actions
    # iteration_num - number of transitions to certain state
    # actions_num - number of executable actions for a certain state
    # actions - array that contains executable actions for a certain state
    state_actions = {
        state: {"iteration_num": 0, "actions_num": None, "actions": None}
        for state in states
    }

    # Fill the state_actions dictionary with data
    for state in states:
        state_actions[state]["actions_num"] = len(list(probabilities[state].keys()))
        state_actions[state]["actions"] = list(probabilities[state].keys())

    i = 0
    for _ in range(iteration_num):
        i += 1  # For testing cases

        # Retreive which action should be executed as a next one
        action_to_execute_index = (
            state_actions[current_state]["iteration_num"]
            % state_actions[current_state]["actions_num"]
        )
        executed_action = state_actions[current_state]["actions"][
            action_to_execute_index
        ]

        next_state, states_hits = execute_action(
            states, probabilities, current_state, executed_action, states_hits
        )

        state_actions[current_state][
            "iteration_num"
        ] += 1  # Increase the iteration number of the current state

        approximated_prob = update_approx_prob(
            approximated_prob, states_hits, current_state, executed_action, states
        )

        current_state = next_state

    # Manual tests

    ## 1: the number of all iterations, sum of iterations for each of states and sum of states' hits should be equal
    # states_hits_sum = 0
    # for state, action_foll_state in states_hits.items():
    #     for action, foll_states in action_foll_state.items():
    #         for foll_state, hit in foll_states.items():
    #             states_hits_sum += hit

    # states_iteration_sum = 0
    # for state in states:
    #     states_iteration_sum += state_actions[state]["iteration_num"]

    # print(
    #     "Equality test: iteration_num === states_hits == states_iteration:",
    #     i == states_hits_sum == states_iteration_sum,
    # )
    # mdp.print_mdp_details(states_hits)

    return approximated_prob, current_state


### EXPLORE LEAST KNOWN STATE


# Looks for the least visited state
def get_least_visited_state(states_hits):

    states_hits_sum = calculate_states_hits(states_hits)
    state_with_smallest_hits_num = min(states_hits_sum, key=states_hits_sum.get)

    return state_with_smallest_hits_num


# Returns an action with the highest transition probability for a certain state
def get_max_prob_action(current_state, state_with_smallest_hits, probabilities):
    max_probability = 0
    action_to_execute = None

    # Iterate over the actions for the current state
    for action, transitions in probabilities[current_state].items():
        # Check if the probability for the state with the smallest hits is the maximum so far
        if transitions[state_with_smallest_hits] > max_probability:
            max_probability = transitions[state_with_smallest_hits]
            action_to_execute = action

    return max_probability, action_to_execute


def explore_least_known_states(
    states, probabilities, approximated_prob, iteration_num, states_hits, current_state
):
    for _ in range(iteration_num):
        least_visited_state = get_least_visited_state(states_hits)

        max_probability, action_to_execute = get_max_prob_action(
            current_state, least_visited_state, approximated_prob
        )
        next_state, states_hits = execute_action(
            states, probabilities, current_state, action_to_execute, states_hits
        )

        approximated_prob = update_approx_prob(
            approximated_prob, states_hits, current_state, action_to_execute, states
        )

        current_state = next_state

    return approximated_prob


def learn_probabilities_convergence(mdp_object, threshold):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states
    P = (
        mdp_object.probabilities
    )  # Probabilities needed for changing between transitions

    # Needed for choosing actions that should be executed
    state_actions = {
        s: {"iteration_num": 0, "actions_num": None, "actions": None} for s in S
    }

    # Create dictionary that stores number of possible actions and possible actions in an array
    for state in S:
        state_actions[state]["actions_num"] = len(list(P[state].keys()))
        state_actions[state]["actions"] = list(P[state].keys())

    # for state in S:
    #     print(state)
    #     print("actions_num:", state_actions[state]["actions_num"])
    #     print("actions:", state_actions[state]["actions"])
    #     print("\n")

    # Store how many times a transition to a certain state took place
    states_hits = {s: {a: {s: 0 for s in S} for a in A} for s in S}

    prob_denominator = len(S)  # Denominator for calculating probabilities

    # Initially all probabilities for transitioning to other state are equal
    initial_prob = 1 / prob_denominator
    approximated_prob = copy.deepcopy(P)
    approximated_prob_new = copy.deepcopy(P)

    # Set all transition probabilities to initial probability
    for state in approximated_prob:
        for action in approximated_prob[state]:
            for next_state in approximated_prob[state][action]:
                approximated_prob[state][action][next_state] = initial_prob
                approximated_prob_new[state][action][next_state] = initial_prob

    current_state = "s0"  # Start at state 's0'
    i = 0  # Number of iterations

    # List with possible actions from each state (needed for convergence)
    exe_actions = []
    for state in S:
        exe_actions.append(state_actions[state]["actions"])
    # print(exe_actions)

    while True:
        i += 1  # For testing cases

        # Retreive which action should be executed as a next one
        action_to_execute_index = (
            state_actions[current_state]["iteration_num"]
            % state_actions[current_state]["actions_num"]
        )
        executed_action = state_actions[current_state]["actions"][
            action_to_execute_index
        ]

        state_actions[current_state][
            "iteration_num"
        ] += 1  # Increase the iteration number of the current state

        # Get probabilities of transitioning to other states (needed for transition execution)
        states_prob = mdp.get_foll_states_prob_values(P, current_state, executed_action)
        next_state = random.choices(S, weights=states_prob)[0]

        # Mark state to which transition took place
        states_hits[current_state][executed_action][next_state] += 1

        # Calculate new approximation of probabilities
        hits_list = list(states_hits[current_state][executed_action].values())
        hits_num = sum(
            hits_list
        )  # Sum how many states changes took place for a certain state after executing a certain action

        for state in S:
            approximated_prob_new[current_state][executed_action][state] = (
                1 + states_hits[current_state][executed_action][state]
            ) / (prob_denominator + hits_num)

        # Check convergence
        convergence = True

        for init_state, exe_action_list in zip(S, exe_actions):
            for exe_action in exe_action_list:
                for following_state in S:
                    convergence = convergence and (
                        abs(
                            approximated_prob[init_state][exe_action][following_state]
                            - approximated_prob_new[init_state][exe_action][
                                following_state
                            ]
                        )
                        < threshold
                    )

        if convergence:

            # Manual tests

            ## 1: the number of all iterations, sum of iterations for each of states and sum of states' hits should be equal
            print("Number of iterations: ", i)
            # states_hits_sum = 0
            # for state, action_foll_state in states_hits.items():
            #     for action, foll_states in action_foll_state.items():
            #         for foll_state, hit in foll_states.items():
            #             states_hits_sum += hit

            # states_iteration_sum = 0
            # for state in S:
            #     states_iteration_sum += state_actions[state]["iteration_num"]

            # print("Equality test: iteration_num === states_hits == states_iteration:", i == states_hits_sum == states_iteration_sum)
            return approximated_prob_new

        approximated_prob = copy.deepcopy(approximated_prob_new)
        current_state = next_state


### ALGORITHM


def my_algorithm(mdp_object, sys_learn_iterations, least_known_iterations):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states

    # Probabilities needed for transitioning from one state to another
    P = mdp_object.probabilities

    # Create approximated probabilities
    approximated_prob = assign_initial_approx_probabilities(S, P)
    print("Approximated probabilities:")

    # Store how many times a transition to a certain state took place

    states_hits = create_states_hits_dictionary(P)

    current_state = "s0"  # Start at state 's0'
    next_states = []  # Array that stores sequence of reached states

    approximated_prob, current_state = systematic_learning(
        S, P, approximated_prob, states_hits, current_state, sys_learn_iterations
    )
    # print("Approximated prob after systematic learning")
    # mdp.print_mdp_details(approximated_prob)

    approximated_prob = explore_least_known_states(
        S, P, approximated_prob, least_known_iterations, states_hits, current_state
    )
    # print("Approximated prob after exploring least known states")
    # mdp.print_mdp_details(approximated_prob)

    return approximated_prob
