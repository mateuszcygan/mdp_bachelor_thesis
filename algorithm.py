import copy
import math
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


# create dictionary with rewards (all values are set to 0.0)
# 0.0 - double type to distinction between changed and not changed values
def set_values_to_zero(d):
    for key, value in d.items():
        if isinstance(value, dict):
            set_values_to_zero(value)
        else:
            d[key] = 0.0


def create_learned_rewards(rewards):
    learned_rewards = copy.deepcopy(rewards)
    set_values_to_zero(learned_rewards)
    return learned_rewards


# Note: states_hits sum is increased for current state (for state from which we fired an action)
# in next iteration the sum will be increased in the state to which the transition took place
def execute_action(
    states,
    probabilities,
    rewards,
    learned_rewards,
    current_state,
    executed_action,
    states_hits,
):

    # Get probabilities of transitioning to other states (needed for transition execution)
    states_prob = mdp.get_foll_states_prob_values(
        probabilities, current_state, executed_action
    )

    next_state = random.choices(states, weights=states_prob)[0]

    state_hit = states_hits[current_state][executed_action]
    reward = rewards[current_state][executed_action][
        next_state
    ]  # extract the collected reward
    learned_reward = learned_rewards[current_state][executed_action]

    # Check and assign the reward if it hasn't been collected before
    if state_hit[next_state] == 0:
        learned_reward[next_state] = reward

    # Increment the state hits count
    state_hit[next_state] += 1

    return next_state, reward, learned_rewards, states_hits


# Calculates approximated probabilities with uniform distribution initialization
def update_approx_prob_uniform_distribution(
    approximated_prob, states_hits, current_state, executed_action, states
):
    # Needed for calculation that provides assigned initial probabilities
    prob_denominator = len(states)

    # Calculate new approximation of probabilities
    hits_list = list(states_hits[current_state][executed_action].values())
    hits_sum = sum(
        hits_list
    )  # Sum how many states changes took place for a certain state after executing a certain action

    for state in states:
        approximated_prob[current_state][executed_action][state] = (
            1 + states_hits[current_state][executed_action][state]
        ) / (prob_denominator + hits_sum)

    return approximated_prob


# Calculates approximated probabilities based only on states hits
def calculate_approx_prob_states_hits(approximated_prob, states_hits):

    for initial_state, actions in states_hits.items():
        for action, hits in actions.items():
            denominator = sum(hits.values())
            if denominator != 0:
                for following_state, hits_num in hits.items():
                    approximated_prob[initial_state][action][following_state] = (
                        hits_num / denominator
                    )
            else:
                for following_state, hits_num in hits.items():
                    approximated_prob[initial_state][action][following_state] = 0

    return approximated_prob


# Updates approximated probabilities based only on states hits
def update_approx_prob_states_hits(
    approximated_prob, states_hits, current_state, executed_action, states
):

    hits_list = list(states_hits[current_state][executed_action].values())
    hits_sum = sum(hits_list)

    for state in states:
        approximated_prob[current_state][executed_action][state] = (
            states_hits[current_state][executed_action][state] / hits_sum
        )

    return approximated_prob


# Updates approximated probabilities (change of approach dependent on 'update_prob_parameter')
def update_approx_prob(
    update_prob_approach_parameter,
    iterations_num_counter,  # for debugging purposes
    approximated_prob,
    states_hits,
    current_state,
    executed_action,
    states,
):
    global min_iterations_hits_states_approach_reached  # Needed for the approach change - all probabilities are calculated newly

    if not min_iterations_hits_states_approach_reached:

        desired_states_hits_num_check_approach_change = (
            check_desired_state_action_hits_num(
                states_hits, update_prob_approach_parameter
            )
        )

        if not desired_states_hits_num_check_approach_change:
            approximated_prob = update_approx_prob_uniform_distribution(
                approximated_prob, states_hits, current_state, executed_action, states
            )
        else:
            # DEBUG
            # print(
            #     "\n\n\nCalculation approach changed after",
            #     iterations_num_counter,
            #     "iterations.",
            # )

            # state_action_hits_sum = calculate_state_action_hits(states_hits)
            # print(state_action_hits_sum)

            # print("Approximated probabilities (uniform distribution):")
            # mdp.print_mdp_details(approximated_prob)
            # DEBUG

            approximated_prob = calculate_approx_prob_states_hits(
                approximated_prob, states_hits
            )

            # DEBUG
            # print("Approximated probabilities (states hits):")
            # mdp.print_mdp_details(approximated_prob)
            # DEBUG

            min_iterations_hits_states_approach_reached = True
    else:
        approximated_prob = update_approx_prob_states_hits(
            approximated_prob, states_hits, current_state, executed_action, states
        )

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


# Calculates number of hits for each action in cerain state
def calculate_state_action_hits(states_hits):

    state_action_hits = {}
    for state, actions in states_hits.items():
        state_action_hits[state] = {}

        for action, hits_num in actions.items():
            action_hits = sum(hits_num.values())
            state_action_hits[state][action] = action_hits

    return state_action_hits


# Needed for adjustement of 'explore_least_known_state_action_dijkstra'
# Returns the least executed action for a certain state
def get_the_least_executed_action(states_hits, state):

    state_action_hits = calculate_state_action_hits(states_hits)

    # Retreive hits for passed state
    actions_hits = state_action_hits[state]

    # Get an action that was the most rarely executed (the first one is taken if there are more than one)
    min_executed_action_hits = min(actions_hits.values())
    first_least_executed_action = [
        key for key in actions_hits if actions_hits[key] == min_executed_action_hits
    ][0]

    return first_least_executed_action


# Prints how many hits each state had
def print_states_hits(states_hits_sum):
    print("States hits")
    for state, hit_sum in states_hits_sum.items():
        print(state, ":", hit_sum)


# Checks if a desired number of 'states_hits' for (state, action) took place
def check_desired_state_action_hits_num(states_hits, desired_states_hits_num):

    state_action_hits = calculate_state_action_hits(states_hits)

    min_hits_num_check = True

    for state, actions in state_action_hits.items():
        for action, hits_num in actions.items():
            min_hits_num_check = min_hits_num_check and (
                hits_num >= desired_states_hits_num
            )

    # DEBUG
    # print("min_hits_num_check:", min_hits_num_check)

    return min_hits_num_check


# MDP_KNOWLEDGE_STRATEGY
# checks if the agent know the desired percentage of the mdp
def desired_mdp_knowledge_check(
    mdp_knowledge_desired_tuples_termination, mdp_knowledge_states_hits_num, states_hits
):

    state_action_hits = calculate_state_action_hits(states_hits)

    visited_tuple_num = 0  # stores how many (state, action) tupels were already visited

    for state, actions in state_action_hits.items():
        for action, hits in actions.items():
            if hits >= mdp_knowledge_states_hits_num:
                visited_tuple_num += 1

    if visited_tuple_num >= mdp_knowledge_desired_tuples_termination:

        # DEBUG
        print(
            "TERMINATION: visited_tuple_num >= mdp_knowledge_desired_tuples_termination"
        )
        print("state_action_hits:", state_action_hits)
        print("visited_tuple_num:", visited_tuple_num)
        print(
            "strategy_desired_states_hits_termination:",
            mdp_knowledge_desired_tuples_termination,
        )

        return True
    else:
        return False


### CONVERGENCE
def convergence(
    approximated_prob,
    approximated_prob_new,
    prob_to_check,
    threshold,
    states_hits,
    desired_states_hits_num,
):

    desired_states_hits_num_check = check_desired_state_action_hits_num(
        states_hits, desired_states_hits_num
    )
    # DEBUG
    # print("desired_states_hits_num_check:", desired_states_hits_num_check)

    # Check convergence
    convergence = True

    for state, action in prob_to_check:

        prob_to_check_old = approximated_prob[state][action].items()
        prob_to_check_new = approximated_prob_new[state][action].items()

        for (k1, v1), (k2, v2) in zip(prob_to_check_old, prob_to_check_new):
            convergence = convergence and (abs(v1 - v2) < threshold)

    # DEBUG
    # print("convergence:", convergence)

    # Check convergence and desired number of states hits
    convergence = desired_states_hits_num_check and convergence

    prob_to_check.clear()

    approximated_prob = copy.deepcopy(approximated_prob_new)

    return convergence, prob_to_check, approximated_prob


### SYSTEMATIC LEARNING APPROACH


def systematic_learning(
    mdp_object,
    learned_rewards,
    rewards_sum,
    approximated_prob,
    states_hits,
    current_state,
    iterations_num,
    update_prob_approach_parameter,  # needed for approach change in calculating approximated probabilities
    iterations_num_counter,
):

    # DEBUG
    # print("iterations_num_counter systematic_learning:", iterations_num_counter)

    prob_to_check = (
        []
    )  # Array that stores (initial_state, executed_action) after each execution of an action - needed for convergence check outside the function

    # No execution wanted
    if iterations_num == 0:
        return (
            # elements related to rewards
            learned_rewards,
            rewards_sum,
            # elements related to probabilities
            approximated_prob,
            states_hits,
            # elements needed for further execution of alternating_algo
            prob_to_check,
            current_state,
            # variable that stores number of executed iterations (debugging purposes)
            iterations_num_counter,
        )

    # Knowledge about the mdp_object needed for execution of an action
    probabilities = mdp_object.probabilities
    rewards = mdp_object.rewards

    # state_actions - dict that involves data allowing for alternating execution of actions
    # iteration_num - number of transitions to certain state
    # actions_num - number of executable actions for a certain state
    # actions - array that contains executable actions for a certain state
    states = mdp_object.states

    state_actions = {
        state: {"iteration_num": 0, "actions_num": None, "actions": None}
        for state in states
    }

    # Fill the state_actions dictionary with data
    for state in states:
        state_actions[state]["actions_num"] = len(list(probabilities[state].keys()))
        state_actions[state]["actions"] = list(probabilities[state].keys())

    for _ in range(iterations_num):
        iterations_num_counter += 1

        # Retreive which action should be executed as a next one
        action_to_execute_index = (
            state_actions[current_state]["iteration_num"]
            % state_actions[current_state]["actions_num"]
        )
        action_to_execute = state_actions[current_state]["actions"][
            action_to_execute_index
        ]

        next_state, reward, learned_rewards, states_hits = execute_action(
            states,
            probabilities,
            rewards,
            learned_rewards,
            current_state,
            action_to_execute,
            states_hits,
        )

        rewards_sum += reward

        state_actions[current_state][
            "iteration_num"
        ] += 1  # Increase the iteration number of the current state

        approximated_prob = update_approx_prob(
            update_prob_approach_parameter,
            iterations_num_counter,
            approximated_prob,
            states_hits,
            current_state,
            action_to_execute,
            states,
        )

        # Needed for convergence check outside of systematic_learning (which (state, action) values should be compared)
        prob_to_check.insert(1, (current_state, action_to_execute))

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
    #     "Manual equality test in 'systematic_learning': iterations_num === states_hits == states_iteration:",
    #     iterations_num == states_hits_sum == states_iteration_sum,
    # )

    # DEBUG
    # print("systematic_learning iterations:", iterations_num_counter)
    return (
        # elements related to rewards
        learned_rewards,
        rewards_sum,
        # elements related to probabilities
        approximated_prob,
        states_hits,
        # elements needed for further execution of alternating_algo
        prob_to_check,
        current_state,
        # variable that stores number of executed iterations (debugging purposes)
        iterations_num_counter,
    )


### EXPLORE LEAST KNOWN STATE


# Looks for the least visited state
# IMPORTANT!: if the least visited state is the current state then it takes the second least visited state
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


### DIJKSTRA ALGORITHM


# Note:
# 'iterations_num' not always corresponds to number of executed actions:
# 1) the 'shortest_path_actions' can include more than one action to execute for the last iteration and the desired state(s) are reached
# 2) another one additional iteration: we transition to the least_visited_state and then execute the 'least_executed_action'
def explore_least_known_state_action_dijkstra(
    mdp_object,
    learned_rewards,
    rewards_sum,
    approximated_prob,
    states_hits,
    current_state,
    iterations_num,
    update_prob_approach_parameter,  # needed for approach change in calculating approximated probabilities
    iterations_num_counter,
):
    # DEBUG
    # print("iterations_num_counter explore_least_known_state_action_dijkstra:", iterations_num_counter)

    prob_to_check = (
        []
    )  # Array that stores (initial_state, executed_action) after each execution of an action - needed for convergence check outside the function

    i = 0  # variable that counts how many iterations took place (needed for termination condition)

    # No execution wanted
    if iterations_num == 0:
        return (
            # elements related to rewards
            learned_rewards,
            rewards_sum,
            # elements related to probabilities
            approximated_prob,
            states_hits,
            # elements needed for further execution of alternating_algo
            prob_to_check,
            current_state,
            # variable that stores number of executed iterations (debugging purposes)
            iterations_num_counter,
        )

    # Knowledge about the mdp_object needed for execution of an action
    probabilities = mdp_object.probabilities
    rewards = mdp_object.rewards

    states = mdp_object.states

    while True:

        # Find least_visited_state and the most probable path for reaching it
        least_visited_state = get_least_visited_state(states_hits, current_state)

        shortest_path_actions = dijkstra.dijkstra_alg(
            states, approximated_prob, current_state, least_visited_state
        )

        if len(shortest_path_actions) == 0:
            raise ValueError('Error! "shortest_path_actions" array empty.')

        while True:

            # The shortest path is the path that directly leads from current_state to least_visited_state
            if len(shortest_path_actions) == 1:
                iterations_num_counter += 1
                i += 1

                # Execute the first and the only action from the shortest_path_actions array
                action_to_execute = shortest_path_actions[0][current_state]

                next_state, reward, learned_rewards, states_hits = execute_action(
                    states,
                    probabilities,
                    rewards,
                    learned_rewards,
                    current_state,
                    action_to_execute,
                    states_hits,
                )

                rewards_sum += reward

                approximated_prob = update_approx_prob(
                    update_prob_approach_parameter,
                    iterations_num_counter,
                    approximated_prob,
                    states_hits,
                    current_state,
                    action_to_execute,
                    states,
                )

                # Needed for convergence check outside of systematic_learning (which (state, action) values should be compared)
                prob_to_check.insert(1, (current_state, action_to_execute))

                current_state = next_state

                # Fire the least executed action from the least visited state
                if current_state == least_visited_state:
                    iterations_num_counter += 1
                    i += 1

                    least_executed_action = get_the_least_executed_action(
                        states_hits, current_state
                    )

                    next_state, reward, learned_rewards, states_hits = execute_action(
                        states,
                        probabilities,
                        rewards,
                        learned_rewards,
                        current_state,
                        least_executed_action,
                        states_hits,
                    )

                    rewards_sum += reward

                    approximated_prob = update_approx_prob(
                        update_prob_approach_parameter,
                        iterations_num_counter,
                        approximated_prob,
                        states_hits,
                        current_state,
                        least_executed_action,
                        states,
                    )

                break

            # The shortest path leads through more than one state
            else:

                action_to_execute = shortest_path_actions[0][current_state]

                next_state, reward, learned_rewards, states_hits = execute_action(
                    states,
                    probabilities,
                    rewards,
                    learned_rewards,
                    current_state,
                    action_to_execute,
                    states_hits,
                )

                rewards_sum += reward

                iterations_num_counter += 1
                i += 1

                approximated_prob = update_approx_prob(
                    update_prob_approach_parameter,
                    iterations_num_counter,
                    approximated_prob,
                    states_hits,
                    current_state,
                    action_to_execute,
                    states,
                )

                # Needed for convergence check outside of systematic_learning (which (state, action) values should be compared)
                prob_to_check.insert(1, (current_state, action_to_execute))

                current_state = next_state

                # Delete first element from shortest_path_actions and check if transition took place to desired state
                shortest_path_actions.pop(0)

                # desired_state - state that dijkstra's algorithm calculated to be the next one
                desired_state = next(iter(shortest_path_actions[0]))

                # Check if the executed action took us to the desired state from shortest_path_actions
                if current_state != desired_state:
                    break

        # Case with performing a certain number of iterations
        if i >= iterations_num:
            break

    # DEBUG
    # print(
    #     "explore_least_known_state_action_dijkstra iterations:", iterations_num_counter
    # )

    return (
        # elements related to rewards
        learned_rewards,
        rewards_sum,
        # elements related to probabilities
        approximated_prob,
        states_hits,
        # elements needed for further execution of alternating_algo
        prob_to_check,
        current_state,
        # variable that stores number of executed iterations (debugging purposes)
        iterations_num_counter,
    )


### ALGORITHM

min_iterations_hits_states_approach_reached = False  # Variable that stores if the minimum number of iterations needed for change in approach for calculating approximated probabilities is reached


def my_algo_alternating(
    mdp_object,
    # number of alternating iterations
    outer_iterations,
    # "optional" values
    desired_states_hits_update_percentage,
    total_desired_states_hits_num,
    total_threshold=None,
    # number of iterations
    sys_learn_iterations=0,
    dijkstra_iterations=0,
    # number of different states_hits for termination (mdp_knowledge_strategy)
    mdp_knowledge_desired_tuples_termination=None,
    mdp_knowledge_states_hits_num=None,
):

    # The structure of MDP is known
    A = mdp_object.actions
    S = mdp_object.states

    # Probabilities needed for transitioning from one state to another
    P = mdp_object.probabilities

    R = mdp_object.rewards
    rewards_sum = 0  # sum of collected variables is stored

    # Create approximated probabilities
    approximated_prob = assign_initial_approx_probabilities(S, P)
    approximated_prob_new = copy.deepcopy(
        approximated_prob
    )  # needed for convergence check

    # Create rewards dictionary where learned rewards will be stored
    learned_rewards = create_learned_rewards(R)

    # Store in states_hits how many times a transition to a certain state took place
    states_hits = create_states_hits_dictionary(P)

    current_state = "s0"  # Start at state 's0'
    alternate_function = True  # Flag to switch between functions

    # Note:
    # total_desired_states_hits_num - the number of hits that each state should have, so that the algorithm together with convergence terminates
    # desired_states_hits_update_percentage - a parameter that indicates in percantage after how many percent of 'total_desired_states_hits_num' (a certain number),
    # there should be a change in the approach of calculating approximated probabilities
    update_prob_approach_parameter = math.floor(
        desired_states_hits_update_percentage * total_desired_states_hits_num
    )  # indicates how many states_hits each tuple need to have in order to change the approach that calculates the approximated probabilities

    iterations_num = 0  # Counter that stores the number of executed outer iterations

    iterations_num_counter = 0  # Counter that stores the value of already executed iterations inside of systematic_learning and explore_least_known_state_action_dijkstra

    while True:

        iterations_num += 1

        if alternate_function:
            (
                learned_rewards,
                rewards_sum,
                approximated_prob_new,
                states_hits,
                prob_to_check,
                current_state,
                iterations_num_counter,
            ) = systematic_learning(
                mdp_object,
                learned_rewards,
                rewards_sum,
                approximated_prob_new,
                states_hits,
                current_state,
                sys_learn_iterations,
                update_prob_approach_parameter,
                iterations_num_counter,
            )

            # DEBUG
            # print("approximated_prob_new")
            # mdp.print_mdp_details(approximated_prob_new)
            # print("\n")
            # print("approximated_prob")
            # mdp.print_mdp_details(approximated_prob)

            # DEBUG
            # systematic_state_action_hits = calculate_state_action_hits(states_hits)
            # print("Systematic learning:")
            # print(systematic_state_action_hits)
            # print("\n")
            # mdp.print_mdp_details(approximated_prob_new)
            # print("\n\n\n")

        else:
            (
                learned_rewards,
                rewards_sum,
                approximated_prob_new,
                states_hits,
                prob_to_check,
                current_state,
                iterations_num_counter,
            ) = explore_least_known_state_action_dijkstra(
                mdp_object,
                learned_rewards,
                rewards_sum,
                approximated_prob_new,
                states_hits,
                current_state,
                dijkstra_iterations,
                update_prob_approach_parameter,
                iterations_num_counter,
            )

            # DEBUG
            # dijkstra_state_action_hits = calculate_state_action_hits(states_hits)
            # print("Explore least known state action dijkstra:")
            # print(dijkstra_state_action_hits)
            # print("\n")
            # mdp.print_mdp_details(approximated_prob_new)
            # print("\n\n\n")

        # Toggle the flag for the next iteration
        alternate_function = not alternate_function

        if total_threshold is not None:
            convergence_check, prob_to_check, approximated_prob = convergence(
                approximated_prob,
                approximated_prob_new,
                prob_to_check,
                total_threshold,
                states_hits,
                total_desired_states_hits_num,
            )
            if iterations_num >= outer_iterations and convergence_check:
                break
        else:  # total_threshold IS None
            if mdp_knowledge_desired_tuples_termination is None:
                if iterations_num >= outer_iterations:
                    approximated_prob = copy.deepcopy(approximated_prob_new)
                    break
            else:  # total_threshold IS None "and" strategy_desired_states_hits_termination IS NOT None
                desired_mdp_knowledge = desired_mdp_knowledge_check(
                    mdp_knowledge_desired_tuples_termination,
                    mdp_knowledge_states_hits_num,
                    states_hits,
                )
                if iterations_num >= outer_iterations or desired_mdp_knowledge:
                    approximated_prob = copy.deepcopy(approximated_prob_new)

                    break

    # DEBUG
    print("Total outer iterations number:", iterations_num)
    print(
        "Iterations inside 'systematic_learning', 'dijkstra': (iterations_num_counter)",
        iterations_num_counter,
    )

    return (
        approximated_prob,
        learned_rewards,
        rewards_sum,
        states_hits,
        current_state,
        iterations_num_counter,
    )

    # PRINTS FOR DEBUGGING
    # hits_sum = calculate_states_hits(states_hits)
    # print("Systematic learning:")
    # print_states_hits(hits_sum)
    # print("\n")
    # mdp.print_mdp_details(approximated_prob)
