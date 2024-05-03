import copy
import random

import dijkstra
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


# Note: states_hits sum is increased for current state (for state from which we fired an action)
# in next iteration the sum will be increased in the state to which the transition took place
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

# Calculates number of hits for each action in certain state


# Prints how many hits each state had
def print_states_hits(states_hits_sum):
    print("States hits")
    for state, hit_sum in states_hits_sum.items():
        print(state, ":", hit_sum)


# Checks if a desired number of 'states_hits' took place
def check_desired_states_hits_num(states_hits, desired_states_hits_num):

    states_hits_sum = calculate_states_hits(states_hits)

    min_hits_num_check = True

    for state, hits in states_hits_sum.items():
        min_hits_num_check = min_hits_num_check and (hits >= desired_states_hits_num)

    return min_hits_num_check


### CONVERGENCE
def check_prob_convergence(states, threshold, approximated_prob, approximated_prob_new):

    exe_actions = (
        []
    )  # List with possible actions from each state (needed for convergence)
    for state in states:
        exe_actions.append(mdp.get_possible_actions(approximated_prob, state))

    # Check convergence
    convergence = True

    for init_state, exe_action_list in zip(states, exe_actions):
        for exe_action in exe_action_list:
            for following_state in states:
                convergence = convergence and (
                    abs(
                        approximated_prob[init_state][exe_action][following_state]
                        - approximated_prob_new[init_state][exe_action][following_state]
                    )
                    < threshold
                )
    return convergence


def check_specific_prob_convergence(
    approximated_prob,
    approximated_prob_new,
    prob_to_check,
    threshold,
):
    convergence = True

    for state, action in prob_to_check:

        prob_to_check_old = approximated_prob[state][action].items()
        prob_to_check_new = approximated_prob_new[state][action].items()

        for (k1, v1), (k2, v2) in zip(prob_to_check_old, prob_to_check_new):
            convergence = convergence and (v1 - v2 < threshold)

    return convergence


def convergence(
    approximated_prob,
    approximated_prob_new,
    prob_to_check,
    threshold,
    states_hits,
    desired_states_hits_num,
):
    desired_states_hits_num_check = (
        True  # For cases where number of state's hits is not considered
    )

    if desired_states_hits_num is not None:
        desired_states_hits_num_check = check_desired_states_hits_num(
            states_hits, desired_states_hits_num
        )

    # Check convergence
    convergence = True

    for state, action in prob_to_check:

        prob_to_check_old = approximated_prob[state][action].items()
        prob_to_check_new = approximated_prob_new[state][action].items()

        for (k1, v1), (k2, v2) in zip(prob_to_check_old, prob_to_check_new):
            convergence = convergence and (abs(v1 - v2) < threshold)

    # Check convergence and desired number of states hits
    convergence = desired_states_hits_num_check and convergence

    prob_to_check.clear()

    approximated_prob = copy.deepcopy(approximated_prob_new)

    return convergence, prob_to_check, approximated_prob


### SYSTEMATIC LEARNING APPROACH


def systematic_learning(
    states,
    probabilities,
    approximated_prob,
    states_hits,
    desired_states_hits_num,
    current_state,
    max_iterations,
    convergence_threshold,
    convergence_check_interval,
):

    # No execution wanted
    if max_iterations == 0 and convergence_threshold == None:
        return approximated_prob, states_hits, current_state

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

    iteration_num = 0

    action_details = (
        []
    )  # Array that stores (initial_state, executed_action) after each execution of an action - needed for convergence check
    approximated_prob_new = copy.deepcopy(
        approximated_prob
    )  # Dictionary that stores newly calculated probabilities - needed for convergence check

    while True:
        iteration_num += 1

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

        approximated_prob_new = update_approx_prob(
            approximated_prob_new, states_hits, current_state, executed_action, states
        )

        # Convergence case
        if convergence_check_interval is not None and convergence_threshold is not None:
            # states_hits_num_check = (
            #     True  # For case if states_hits don't have to be considered
            # )

            # Update an array with executed action details
            action_details.insert(1, (current_state, executed_action))

            current_state = next_state

            if iteration_num % convergence_check_interval == 0:

                convergence_check, action_details, approximated_prob = convergence(
                    approximated_prob,
                    approximated_prob_new,
                    action_details,
                    convergence_threshold,
                    states_hits,
                    desired_states_hits_num,
                )

                # if states_hits_num > 0:
                #     states_hits_num_check = check_states_hits(
                #         states_hits, states_hits_num
                #     )

                # Check convergence
                # convergence_check = (
                #     states_hits_num_check  # If 'states_hits_num' is smaller than 1, 'states_hits_num_check' is always True
                #     and check_specific_prob_convergence(
                #         approximated_prob,
                #         approximated_prob_new,
                #         action_details,
                #         convergence_threshold,
                #     )
                # )

                action_details.clear()

                # approximated_prob = copy.deepcopy(approximated_prob_new)

                if convergence_check:
                    break

        current_state = next_state

        # Case with performing a certain number of iterations
        if max_iterations > 0 and iteration_num >= max_iterations:
            approximated_prob = approximated_prob_new
            break

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

    print("systematic_learning iterations:", iteration_num)
    return approximated_prob, states_hits, current_state


### EXPLORE LEAST KNOWN STATE


# Looks for the least visited state
def get_least_visited_state(states_hits, current_visit_state):

    states_hits_sum = calculate_states_hits(states_hits)
    state_with_smallest_hits_num = min(states_hits_sum, key=states_hits_sum.get)

    if state_with_smallest_hits_num == current_visit_state:
        # Remove the first smallest state from the dictionary
        del states_hits_sum[state_with_smallest_hits_num]
        # Find the new smallest state
        second_smallest_state = min(states_hits_sum, key=states_hits_sum.get)
        return second_smallest_state

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
    states,
    probabilities,
    approximated_prob,
    states_hits,
    states_hits_num,
    current_state,
    max_iterations=None,
    convergence_threshold=None,
    convergence_check_interval=None,
):

    # No execution wanted
    if max_iterations == 0 and convergence_threshold == None:
        return approximated_prob, states_hits, current_state

    iteration_num_counter = 0
    approximated_prob_new = copy.deepcopy(approximated_prob)
    action_details = (
        []
    )  # Array that stores (initial_state, executed_action) after each execution of an action - needed for convergence check

    while True:

        iteration_num_counter += 1
        least_visited_state = get_least_visited_state(states_hits, current_state)

        max_probability, action_to_execute = get_max_prob_action(
            current_state, least_visited_state, approximated_prob_new
        )
        next_state, states_hits = execute_action(
            states, probabilities, current_state, action_to_execute, states_hits
        )

        approximated_prob_new = update_approx_prob(
            approximated_prob_new, states_hits, current_state, action_to_execute, states
        )
        # Convergence case
        if convergence_check_interval is not None and convergence_threshold is not None:
            states_hits_num_check = (
                True  # For case if states_hits don't have to be considered
            )

            # Update an array with executed action details
            action_details.insert(1, (current_state, action_to_execute))

            current_state = next_state

            if iteration_num_counter % convergence_check_interval == 0:

                if states_hits_num > 0:
                    states_hits_num_check = check_states_hits(
                        states_hits, states_hits_num
                    )

                # Check convergance
                convergence_check = (
                    states_hits_num_check  # If 'states_hits_num' is smaller than 1, 'states_hits_num_check' is always True
                    and check_specific_prob_convergence(
                        approximated_prob,
                        approximated_prob_new,
                        action_details,
                        convergence_threshold,
                    )
                )

                action_details.clear()

                approximated_prob = copy.deepcopy(approximated_prob_new)

                if convergence_check:
                    break

        current_state = next_state

        # Case with performing a certain number of iterations
        if max_iterations > 0 and iteration_num_counter >= max_iterations:
            approximated_prob = approximated_prob_new
            break

    print("explore_least_know_states iterations:", iteration_num_counter)
    return approximated_prob, states_hits, current_state


### DIJKSTRA ALGORITHM


def explore_least_known_states_dijkstra(
    states,
    probabilities,
    approximated_prob,
    states_hits,
    desired_states_hits_num,
    current_state,
    max_iterations=None,
    convergence_threshold=None,
    convergence_check_interval=None,
):

    # No execution wanted
    if max_iterations == 0 and convergence_threshold == None:
        return approximated_prob, states_hits, current_state

    iteration_num_counter = 0  # Counts how many actions have already been executed

    action_details = (
        []
    )  # Array that stores (initial_state, executed_action) after each execution of an action - needed for convergence check

    approximated_prob_new = copy.deepcopy(
        approximated_prob
    )  # Dictionary that stores newly calculated probabilities - needed for convergence check

    convergence_check = False

    while True:

        if convergence_check:
            break

        # Find least_visited_state and the most probable path for reaching it
        least_visited_state = get_least_visited_state(states_hits, current_state)

        shortest_path_actions = dijkstra.dijkstra_alg(
            states, approximated_prob_new, current_state, least_visited_state
        )

        if len(shortest_path_actions) == 0:
            raise ValueError('Error! "shortest_path_actions" array empty.')

        while True:
            # The shortest path is the path directly leads from current_state to least_visited_state
            if len(shortest_path_actions) == 1:

                # Execute the first and the only action from the shortest_path_actions array
                action_to_execute = shortest_path_actions[0][current_state]

                next_state, states_hits = execute_action(
                    states, probabilities, current_state, action_to_execute, states_hits
                )

                approximated_prob_new = update_approx_prob(
                    approximated_prob_new,
                    states_hits,
                    current_state,
                    action_to_execute,
                    states,
                )

                iteration_num_counter += 1

                if (
                    convergence_check_interval is not None
                    and convergence_threshold is not None
                ):
                    states_hits_num_check = (
                        True  # For case if states_hits don't have to be considered
                    )

                    # Update an array with executed action details
                    action_details.insert(1, (current_state, action_to_execute))

                    if iteration_num_counter % convergence_check_interval == 0:

                        if desired_states_hits_num is not None:
                            desired_states_hits_num_check = (
                                check_desired_states_hits_num(
                                    states_hits, desired_states_hits_num
                                )
                            )

                        # Check convergence
                        convergence_check = (
                            states_hits_num_check  # If 'states_hits_num' is smaller than 1, 'states_hits_num_check' is always True
                            and check_specific_prob_convergence(
                                approximated_prob,
                                approximated_prob_new,
                                action_details,
                                convergence_threshold,
                            )
                        )

                        action_details.clear()

                        approximated_prob = copy.deepcopy(approximated_prob_new)

                        if convergence_check:
                            # Change the value of current_state - needed for return value
                            current_state = next_state
                            break

                current_state = next_state

                break

            # The shortest path leads through more than one state
            else:

                action_to_execute = shortest_path_actions[0][current_state]

                next_state, states_hits = execute_action(
                    states, probabilities, current_state, action_to_execute, states_hits
                )

                approximated_prob_new = update_approx_prob(
                    approximated_prob_new,
                    states_hits,
                    current_state,
                    action_to_execute,
                    states,
                )

                iteration_num_counter += 1

                if (
                    convergence_check_interval is not None
                    and convergence_threshold is not None
                ):
                    states_hits_num_check = (
                        True  # For case if states_hits don't have to be considered
                    )

                    # Update an array with executed action details
                    action_details.insert(1, (current_state, action_to_execute))

                    if iteration_num_counter % convergence_check_interval == 0:

                        if desired_states_hits_num is not None:
                            states_hits_num_check = check_desired_states_hits_num(
                                states_hits, desired_states_hits_num
                            )

                        # Check convergence
                        convergence_check = (
                            states_hits_num_check  # If 'states_hits_num' is smaller than 1, 'states_hits_num_check' is always True
                            and check_specific_prob_convergence(
                                approximated_prob,
                                approximated_prob_new,
                                action_details,
                                convergence_threshold,
                            )
                        )

                        action_details.clear()

                        approximated_prob = copy.deepcopy(approximated_prob_new)

                        if convergence_check:
                            # Change the value of current_state - needed for return value
                            current_state = next_state
                            break

                current_state = next_state

                # Delete first element from shortest_path_actions and check if transition took place to desired state
                shortest_path_actions.pop(0)

                # desired_state - state that dijkstra's algorithm calculated to be the next one
                desired_state = next(iter(shortest_path_actions[0]))

                # Check if the executed action took us to the desired state from shortest_path_actions
                if current_state != desired_state:
                    break

                continue

        # Case with performing a certain number of iterations
        if max_iterations > 0 and iteration_num_counter >= max_iterations:
            approximated_prob = approximated_prob_new
            break

    print("explore_least_known_states_dijkstra iterations:", iteration_num_counter)
    return approximated_prob, states_hits, current_state


### ALGORITHM


def my_algorithm(
    mdp_object,
    # number of iterations
    sys_learn_iterations,
    least_known_iterations,
    dijkstra_iterations,
    # systematic learning convergence
    sys_learn_states_hits_num=None,
    sys_learn_threshold=None,
    sys_learn_interval=None,
    # least known state convergence
    least_known_states_hits_num=None,
    least_known_threshold=None,
    least_known_interval=None,
    # dijkstra convergence
    dijkstra_hits_num=None,
    dijkstra_threshold=None,
    dijkstra_interval=None,
):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states

    # Probabilities needed for transitioning from one state to another
    P = mdp_object.probabilities

    # Create approximated probabilities
    approximated_prob = assign_initial_approx_probabilities(S, P)

    # Store in states_hits how many times a transition to a certain state took place
    states_hits = create_states_hits_dictionary(P)

    current_state = "s0"  # Start at state 's0'

    approximated_prob, states_hits, current_state = systematic_learning(
        S,
        P,
        approximated_prob,
        states_hits,
        sys_learn_states_hits_num,
        current_state,
        sys_learn_iterations,
        sys_learn_threshold,
        sys_learn_interval,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("Systematic learning:")
    # print_states_hits(hits_sum)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)

    approximated_prob, states_hits, current_state = explore_least_known_states(
        S,
        P,
        approximated_prob,
        states_hits,
        least_known_states_hits_num,
        current_state,
        least_known_iterations,
        least_known_threshold,
        least_known_interval,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("Exploring least know states:")
    # print_states_hits(hits_sum)
    # print("Current state after exploring least known state:", current_state)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)

    approximated_prob, states_hits, current_state = explore_least_known_states_dijkstra(
        S,
        P,
        approximated_prob,
        states_hits,
        dijkstra_hits_num,
        current_state,
        dijkstra_iterations,
        dijkstra_threshold,
        dijkstra_interval,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("States hits after exploring least know states with dijkstra:")
    # print_states_hits(hits_sum)
    # print("Current state after exploring least known state:", current_state)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)

    return approximated_prob, states_hits


def my_algo(
    mdp_object,
    # number of iterations
    sys_learn_iterations=0,
    dijkstra_iterations=0,
    # sys_learn convergence
    sys_learn_states_hits_num=None,
    sys_learn_threshold=None,
    sys_learn_interval=None,
    # dijkstra convergence
    dijkstra_states_hits_num=None,
    dijkstra_threshold=None,
    dijkstra_interval=None,
):
    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states

    # Probabilities needed for transitioning from one state to another
    P = mdp_object.probabilities

    # Create approximated probabilities
    approximated_prob = assign_initial_approx_probabilities(S, P)

    # Store in states_hits how many times a transition to a certain state took place
    states_hits = create_states_hits_dictionary(P)

    current_state = "s0"  # Start at state 's0'

    approximated_prob, states_hits, current_state = systematic_learning(
        S,
        P,
        approximated_prob,
        states_hits,
        sys_learn_states_hits_num,
        current_state,
        sys_learn_iterations,
        sys_learn_threshold,
        sys_learn_interval,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("Systematic learning:")
    # print_states_hits(hits_sum)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)

    approximated_prob, states_hits, current_state = explore_least_known_states_dijkstra(
        S,
        P,
        approximated_prob,
        states_hits,
        dijkstra_states_hits_num,
        current_state,
        dijkstra_iterations,
        dijkstra_threshold,
        dijkstra_interval,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("States hits after exploring least know states with dijkstra:")
    # print_states_hits(hits_sum)
    # print("Current state after exploring least known state:", current_state)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)

    return approximated_prob, states_hits

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("Systematic learning:")
    # print_states_hits(hits_sum)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)
