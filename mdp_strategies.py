import copy
import math

import algorithm
import mdp
import policies


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

    if (
        rewards_extremes["min_reward"] < old_min_reward
        or rewards_extremes["max_reward"] > old_max_reward
    ):
        V, policy = policies.value_iteration(
            mdp_object, value_iteration_threshold, value_iteration_dis_factor
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

    approximated_mdp = mdp.MDP(states, actions, approximated_prob, learned_rewards)

    # extract needed values from approximated mdp
    approx_mdp_prob = approximated_mdp.probabilities
    approx_mdp_learned_rewards = approximated_mdp.rewards

    # create copy of approximated probabilities for further checks (essential for value iteration recalculation check )
    approx_mdp_prob_new = copy.deepcopy(approx_mdp_prob)

    V, policy = policies.value_iteration(
        approximated_mdp, value_iteration_threshold, value_iteration_dis_factor
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

            V, policy = policies.value_iteration(
                approximated_mdp, value_iteration_threshold, value_iteration_dis_factor
            )

        # check if the executed action brought new reward that wasn't saved before
        if states_hits[current_state][action_to_execute][next_state] == 1:

            # necessary, since 'value_iteration' takes whole mdp as an argument
            approximated_mdp.probabilities = copy.deepcopy(approx_mdp_prob_new)
            approximated_mdp.rewards = copy.deepcopy(approx_mdp_learned_rewards)

            # check if the new stored reward is new max/min_reward
            V, policy = update_value_iteration_policy_rewards(
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
            break

    return approximated_mdp, rewards_sum
