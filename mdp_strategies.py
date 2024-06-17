import copy
import math
import sys

import algorithm
import mdp
import policies

### FUNCTIONS NEEDED AFTER TERMINATION OF 'my_algo_alternating'


# finds the biggest and the smallest reward within the 'learned_rewards' dictionary
def find_rewards_extremes(learned_rewards):

    rewards_extremes = {"min_reward": 0, "max_reward": 0}

    find_rewards_extremes_recursive(learned_rewards, rewards_extremes)

    return rewards_extremes


def find_rewards_extremes_recursive(learned_rewards, rewards_extremes):

    for key, value in learned_rewards.items():
        if isinstance(value, dict):
            find_rewards_extremes_recursive(value, rewards_extremes)
        else:
            if value > rewards_extremes["max_reward"]:
                rewards_extremes["max_reward"] = value
            elif value < rewards_extremes["min_reward"]:
                rewards_extremes["min_reward"] = value


# calculates new value function and policy in the case of change in min_reward or max_reward
def update_value_iteration_policy_rewards(
    overall_iterations_num,
    executed_iterations,
    learned_rewards,
    rewards_extremes,
    mdp_object,
    V,
    policy,
    value_iteration_threshold,
    value_iteration_dis_factor,
):
    old_min_reward = rewards_extremes["min_reward"]
    old_max_reward = rewards_extremes["max_reward"]

    rewards_extremes = find_rewards_extremes(learned_rewards)

    # if the smallest or the biggest reward changed, the value iteration is repeated calculated
    if (
        rewards_extremes["min_reward"] < old_min_reward
        or rewards_extremes["max_reward"] > old_max_reward
    ):
        V, policy = update_value_iteration(
            overall_iterations_num,
            executed_iterations,
            mdp_object,
            value_iteration_threshold,
            value_iteration_dis_factor,
        )

    return V, policy


# checks if there was an enough big change (value_iteration_prob_recalculation_parameter) between two probabilities
# so that value iteration should be recalculated (or - only one such change needed)
def recalculate_value_iteration_prob_difference(
    value_iteration_prob_recalculation_parameter,
    current_state,
    executed_action,
    approximated_prob,
    approximated_prob_new,
):

    recalculate_value_iteration = False
    tolerance = (
        1e-9  # A small tolerance value - inprecision of floating point arithmetic
    )

    prob_to_check_old = approximated_prob[current_state][executed_action].items()
    prob_to_check_new = approximated_prob_new[current_state][executed_action].items()

    for (k1, v1), (k2, v2) in zip(prob_to_check_old, prob_to_check_new):

        prob_difference = abs(v1 - v2)

        is_close = math.isclose(
            prob_difference,
            value_iteration_prob_recalculation_parameter,
            rel_tol=tolerance,
        )

        recalculate_value_iteration = (
            recalculate_value_iteration
            or is_close
            or (prob_difference >= value_iteration_prob_recalculation_parameter)
        )

    return recalculate_value_iteration


# calulates value iteration either with certain convergence value or with finite horizon depending on iterations number left
def update_value_iteration(
    overall_iterations_num,
    executed_iterations,
    mdp_object,
    threshold,
    discount_factor,
):

    iterations_left = overall_iterations_num - executed_iterations

    # value iteration with threshold needs roughly 60 iterations to converge with approximated mdps
    if iterations_left > 60:

        V, policy = policies.value_iteration(mdp_object, threshold, discount_factor)

    else:

        V, policy = policies.value_iteration_finite(mdp_object, iterations_left, 1)
        # finite_horizon = iterations_left, since only "iterations_left" number of actions can be still executed
        # discount_factor = 1, since it should stop after a certain number of steps

    return V, policy


### ITERATIONS_NUM_STRATEGY
# after a certain number of iterations, value iteration is calculated and agent starts to follow calculated strategy
def iterations_num_strategy(
    # needed for 'iterations_number_approach'
    mdp_object,
    overall_iterations_num,
    # needed for my_algo_alternating
    my_algo_alter_percentage,  # percentage of overall iterations that should be performed in 'my_algo_alternating'
    sys_learn_iterations,
    dijkstra_iterations,
    desired_states_hits_update_percentage,
    total_desired_states_hits_num,
    total_threshold,
    # needed for value_iteration
    value_iteration_threshold,
    value_iteration_dis_factor,
    value_iteration_prob_recalculation_parameter,
):

    # Things that are NOT known about the MDP object for agent, but required to execute an action
    states = mdp_object.states
    actions = mdp_object.actions
    probabilities = mdp_object.probabilities
    rewards = mdp_object.rewards

    # Calculations for case when when the number of iterations in different approaches can't be exactly estimated (doubles)
    my_algo_iterations = overall_iterations_num * my_algo_alter_percentage
    outer_iterations = math.floor(
        (my_algo_iterations / (sys_learn_iterations + dijkstra_iterations)) * 2
    )

    (
        approximated_prob,
        learned_rewards,
        rewards_sum,
        states_hits,
        current_state,
        iterations_num_counter,
    ) = algorithm.my_algo_alternating(
        mdp_object,
        # number of alternating iterations
        outer_iterations,
        # "optional" values
        desired_states_hits_update_percentage,
        total_desired_states_hits_num,
        total_threshold,
        # number of iterations
        sys_learn_iterations,
        dijkstra_iterations,
    )

    # DEBUG
    print(states_hits)
    print(current_state)
    print("\n")

    approximated_mdp = mdp.MDP(states, actions, approximated_prob, learned_rewards)

    # extract needed values from approximated mdp (the ones that still will be updated)
    approx_mdp_prob = approximated_mdp.probabilities
    approx_mdp_learned_rewards = approximated_mdp.rewards

    # create copy of approximated probabilities for further checks (essential for value iteration recalculation check)
    approx_mdp_prob_new = copy.deepcopy(approx_mdp_prob)

    V, policy = update_value_iteration(
        overall_iterations_num,
        iterations_num_counter,
        approximated_mdp,
        value_iteration_threshold,
        value_iteration_dis_factor,
    )

    rewards_extremes = find_rewards_extremes(approx_mdp_learned_rewards)

    # start to follow calculated value iteration policy
    while True:
        action_to_execute = policy[current_state]

        next_state, reward, approx_mdp_learned_rewards, states_hits = (
            algorithm.execute_action(
                states,
                probabilities,
                rewards,
                approx_mdp_learned_rewards,
                current_state,
                action_to_execute,
                states_hits,
            )
        )

        approx_mdp_prob_new = algorithm.update_approx_prob_states_hits(
            approx_mdp_prob_new, states_hits, current_state, action_to_execute, states
        )

        # check if the changes that took place in the approximated probabilities are so big that the recalculation of the value iteration is necessary
        recalculate_value_iteration = recalculate_value_iteration_prob_difference(
            value_iteration_prob_recalculation_parameter,
            current_state,
            action_to_execute,
            approx_mdp_prob,
            approx_mdp_prob_new,
        )

        if recalculate_value_iteration:

            # necessary, since 'value_iteration' takes whole mdp as an argument
            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            V, policy = update_value_iteration(
                overall_iterations_num,
                iterations_num_counter,
                approximated_mdp,
                value_iteration_threshold,
                value_iteration_dis_factor,
            )

        # check if the executed action brought new reward that wasn't saved before
        if states_hits[current_state][action_to_execute][next_state] == 1:

            # necessary, since 'value_iteration' takes whole mdp as an argument
            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            # check if the new stored reward is new max/min_reward
            V, policy = update_value_iteration_policy_rewards(
                overall_iterations_num,
                iterations_num_counter,
                approx_mdp_learned_rewards,
                rewards_extremes,
                approximated_mdp,
                V,
                policy,
                value_iteration_threshold,
                value_iteration_dis_factor,
            )

        rewards_sum += reward
        iterations_num_counter += 1
        approx_mdp_prob = copy.deepcopy(approx_mdp_prob_new)
        current_state = next_state

        if iterations_num_counter >= overall_iterations_num:

            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            # DEBUG
            print("policy (after termination)", policy)

            break

    return approximated_mdp, rewards_sum


### MDP_KNOWLEDGE_STRATEGY
def calculate_state_action_tuple_num(probabilities, states):

    possible_states_num = 0

    for state in states:
        possible_states = mdp.get_possible_actions(probabilities, state)
        possible_states_num += len(possible_states)

    return possible_states_num


# after a certain percentage of whole network is known (based on states_hits), agent starts to follow calculated strategy
def mdp_knowledge_strategy(
    # needed for "network_knowledge_strategy"
    mdp_object,
    overall_iterations_num,
    # needed for my_algo_alternating
    mdp_knowledge_percentage,  # after a certain percentage of the network is known, the exploring phase ends and agents start exploition based on calculations from value iteration
    mdp_knowledge_states_hits_num,  # the number of hits needed to clasify a tuple as "known"
    sys_learn_iterations,
    dijkstra_iterations,
    desired_states_hits_update_percentage,
    total_desired_states_hits_num,
    total_threshold,
    # needed for value_iteration
    value_iteration_threshold,
    value_iteration_dis_factor,
    value_iteration_prob_recalculation_parameter,
):
    # Things that are NOT known about the MDP object for agent, but required to execute an action
    states = mdp_object.states
    actions = mdp_object.actions
    probabilities = mdp_object.probabilities
    rewards = mdp_object.rewards

    # calculate how many tuples (state, action) exist for a given mdp
    state_action_num = calculate_state_action_tuple_num(probabilities, states)

    # DEBUG
    print("state_action_num: (number of tuples in mdp)", state_action_num)

    # calculate after how many different state_action hits 'my_algo_alternating' should terminate
    mdp_knowledge_desired_tuples_termination = math.floor(
        state_action_num * mdp_knowledge_percentage
    )

    # DEBUG
    print(
        "strategy_desired_states_hits_termination: (number of tuples for my_algo_alternating to terminate)",
        mdp_knowledge_desired_tuples_termination,
    )

    # calculate how many iterations at the maximum can be performed in 'my_algo_alternating'
    # for the case that the desired number of "mdp_knowledge" can't be achieved
    outer_iterations = math.floor(
        overall_iterations_num / (sys_learn_iterations + dijkstra_iterations)
    )

    (
        approximated_prob,
        learned_rewards,
        rewards_sum,
        states_hits,
        current_state,
        iterations_num_counter,
    ) = algorithm.my_algo_alternating(
        mdp_object,
        # number of alternating iterations
        outer_iterations,
        # "optional" values
        desired_states_hits_update_percentage,
        total_desired_states_hits_num,
        total_threshold,
        # number of iterations
        sys_learn_iterations,
        dijkstra_iterations,
        mdp_knowledge_desired_tuples_termination,
        mdp_knowledge_states_hits_num,
    )

    approximated_mdp = mdp.MDP(states, actions, approximated_prob, learned_rewards)

    # for the case when no desired mdp knowledge achieved - the 'my_algo_alternating' terminated because of outer_iterations (overall_iterations_num)
    if iterations_num_counter >= overall_iterations_num:
        return approximated_mdp, rewards_sum

    # extract needed values from approximated mdp (the ones that still will be updated)
    approx_mdp_prob = approximated_mdp.probabilities
    approx_mdp_learned_rewards = approximated_mdp.rewards

    # create copy of approximated probabilities for further checks (essential for value iteration recalculation check )
    approx_mdp_prob_new = copy.deepcopy(approx_mdp_prob)

    V, policy = update_value_iteration(
        overall_iterations_num,
        iterations_num_counter,
        approximated_mdp,
        value_iteration_threshold,
        value_iteration_dis_factor,
    )

    rewards_extremes = find_rewards_extremes(approx_mdp_learned_rewards)

    # start to follow calculated value iteration policy
    while True:
        action_to_execute = policy[current_state]

        next_state, reward, approx_mdp_learned_rewards, states_hits = (
            algorithm.execute_action(
                states,
                probabilities,
                rewards,
                approx_mdp_learned_rewards,
                current_state,
                action_to_execute,
                states_hits,
            )
        )

        approx_mdp_prob_new = algorithm.update_approx_prob_states_hits(
            approx_mdp_prob_new, states_hits, current_state, action_to_execute, states
        )

        # check if the changes that took place in the approximated probabilities are so big that the recalculation of the value iteration is necessary
        recalculate_value_iteration = recalculate_value_iteration_prob_difference(
            value_iteration_prob_recalculation_parameter,
            current_state,
            action_to_execute,
            approx_mdp_prob,
            approx_mdp_prob_new,
        )

        if recalculate_value_iteration:

            # necessary, since 'value_iteration' takes whole mdp as an argument
            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            V, policy = update_value_iteration(
                overall_iterations_num,
                iterations_num_counter,
                approximated_mdp,
                value_iteration_threshold,
                value_iteration_dis_factor,
            )

        # check if the executed action brought new reward that wasn't saved before
        if states_hits[current_state][action_to_execute][next_state] == 1:

            # necessary, since 'value_iteration' takes whole mdp as an argument
            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            # check if the new stored reward is new max/min_reward
            V, policy = update_value_iteration_policy_rewards(
                overall_iterations_num,
                iterations_num_counter,
                approx_mdp_learned_rewards,
                rewards_extremes,
                approximated_mdp,
                V,
                policy,
                value_iteration_threshold,
                value_iteration_dis_factor,
            )

        rewards_sum += reward
        iterations_num_counter += 1
        approx_mdp_prob = copy.deepcopy(approx_mdp_prob_new)
        current_state = next_state

        if iterations_num_counter >= overall_iterations_num:
            # DEBUG
            print("policy (after termination):", policy)

            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)
            break

    return approximated_mdp, rewards_sum
