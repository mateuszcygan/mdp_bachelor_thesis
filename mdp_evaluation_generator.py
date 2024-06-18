import os
import random

import generator
import mdp


def save_mdp_in_folder(mdp_obj, folder_name, mdp_number):
    file_name = os.path.join(folder_name, f"mdp_{mdp_number}.pkl")
    generator.save_object(mdp_obj, file_name)


def read_mdps_from_folder(folder_name):
    for x in range(10):
        file_name = os.path.join(folder_name, f"mdp_{x}.pkl")
        current_mdp = generator.read_saved_mdp(file_name)
        print("MDP number ", x)
        print(current_mdp.states)
        print(current_mdp.actions)
        print("probabilities:")
        mdp.print_mdp_details(current_mdp.probabilities)
        mdp.print_mdp_details(current_mdp.rewards)


def generate_dense_mdps(
    folder_name,
    min_states_num,
    max_states_num,
    min_actions_num,
    max_actions_num,
    min_reward,
    max_reward,
    prob_sparse_rate=0,
    rewards_sparse_rate=0,
):
    for x in range(10):

        states_num = random.randint(min_states_num, max_states_num)
        actions_num = random.randint(min_actions_num, max_actions_num)
        mdp_obj = mdp.createMDP(states_num, actions_num, min_reward, max_reward)

        if prob_sparse_rate > 0:
            mdp.sparse_mdp_states(mdp_obj, prob_sparse_rate)

        if rewards_sparse_rate > 0:
            mdp.sparse_mdp_rewards(mdp_obj, 0.1)

        mdp.reduce_actions_number(mdp_obj, (actions_num - 1), actions_num)
        save_mdp_in_folder(mdp_obj, folder_name, x)


def generate_sparse_mdps(
    folder_name,
    min_states_num,
    max_states_num,
    min_actions_num,
    max_actions_num,
    min_reward,
    max_reward,
    prob_sparse_rate,
    rewards_sparse_rate,
):
    for x in range(10):
        states_num = random.randint(min_states_num, max_states_num)
        actions_num = random.randint(min_actions_num, max_actions_num)
        mdp_obj = mdp.createMDP(states_num, actions_num, min_reward, max_reward)
        mdp.sparse_mdp_states(mdp_obj, prob_sparse_rate)
        mdp.sparse_mdp_rewards(mdp_obj, rewards_sparse_rate)
        mdp.reduce_actions_number(mdp_obj, (actions_num - 2), actions_num)
        save_mdp_in_folder(mdp_obj, folder_name, x)
