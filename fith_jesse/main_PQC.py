import numpy as np
# from plot import plot
from tqdm import tqdm
import datetime
import os
import multiprocessing as mp
from itertools import combinations
import argparse

# settings for writing the files, plotting
plotting = False
print_avg = False
save_data = True
print_model_summary = True
print_policy = True
plot_distribution = False
save_length = True
save_reward = True

state_bounds = 1
gamma = 1
averaging_window = 5000

    
anil= 0.25
start = 1




# Start training the agent
# for _ in range(n_reps):
def run(len_state=2, prob_1=3/14, prob_2=11/14, n_episodes = 250000, n_holes = 5, max_steps = 6, batch_size = 10, env_name="FoxInAHolev2",
        lr_in = 0.01, lr_var= 0.01, lr_out = 0.01, n_layers = 2, n_qubits = 2, RxCnot = True, exp_key = "default"):
    from quantumneat.problems.REINFORCE import reinforce_agent
    # from PQC import generate_model_policy, reinforce_update
    from quantumneat.problems.PQC import generate_model_policy, reinforce_update
    import tensorflow as tf
    agent = reinforce_agent(batch_size=batch_size)

    episode_reward_history = []
    episode_length_history = []
    # As the different sets of parameters require different learning rates, create seperate optimizers
    optimizer_in = tf.keras.optimizers.Adam(learning_rate=lr_in, amsgrad=True)
    optimizer_var = tf.keras.optimizers.Adam(learning_rate=lr_var, amsgrad=True)
    optimizer_out = tf.keras.optimizers.Adam(learning_rate=lr_out, amsgrad=True)
    
    # Assign the model parameters to each optimizer
    w_in, w_var, w_out = 1, 0, 2
    # w_in, w_var = 1, 0

    optimizers = [optimizer_in, optimizer_var, optimizer_out]
    ws= [w_in, w_var, w_out]

    model = generate_model_policy(n_qubits= 
    n_qubits, n_layers= n_layers, n_actions= n_qubits, n_inputs = len_state, beta= 1, RxCnot= RxCnot)
    n_batches = n_episodes // batch_size

    # if plot_distribution:

    #     possible_states= np.array(list(combinations(np.arange(-1,n_holes, 1),2)))
    #     possible_states = tf.convert_to_tensor(possible_states)

    # probs_total = np.zeros(n_holes)
    for batch in tqdm(range(n_batches)):
        # # Gather episodes
        # if plot_distribution and (batch==0 or batch/n_batches==0.25 or batch/n_batches==0.5 or batch/n_batches == 0.75 or batch==n_batches-1):
        #     probs = model(possible_states)
        #     probs_avg = np.average(probs, axis=0)
        #     print("the probs at batch {} are {}".format(batch, probs_avg))
        #     pass
        episodes = agent.gather_episodes(state_bounds, n_holes, n_qubits, model, batch_size, env_name, len_state, max_steps= max_steps, prob_1=prob_1, prob_2=prob_2)

        # Group states, actions and returns in numpy arrays
        states = np.concatenate([ep['states'] for ep in episodes])
        actions = np.concatenate([ep['actions'] for ep in episodes])
        rewards = [ep['rewards'] for ep in episodes]
        returns = np.concatenate([agent.compute_returns(ep_rwds, gamma) for ep_rwds in rewards])

        returns = np.array(returns, dtype=np.float32)

        eta = max(start*(1-(batch*batch_size)/(n_episodes*anil)), 0)
        id_action_pairs = np.array([[i, a] for i, a in enumerate(actions)])

        # Update model parameters.
        reinforce_update(states, id_action_pairs, returns, model, ws, optimizers, batch_size=batch_size, eta= eta)

        # Store collected rewards
        for ep_rwds in rewards:
            episode_reward_history.append(np.sum(ep_rwds))
            episode_length_history.append(len(ep_rwds))

        
        if print_avg:
            avg_rewards = np.mean(episode_reward_history[-averaging_window:])
            print('Finished episode', (batch + 1) * batch_size,
            'Average rewards: ', avg_rewards)
            
    # if plotting:
    #     plot(episode_reward_history, "NN", averaging_window)

    if print_policy:
        state = tf.convert_to_tensor([-1*np.ones(len_state)])
        policy = []
        for _ in range(2*(n_holes-2)):
            action = np.random.choice(n_holes, p = model(state).numpy()[0])
            policy.append(action)
            ar_state = state.numpy()
            ar_state[0] = np.roll(ar_state[0], 1)
            ar_state[0][0] = action
            state = tf.convert_to_tensor(ar_state)

        print("Final policy is the following sequence: {}".format(policy))

    if save_data:

        if save_length:

            # the path to where we save the results. we take the first letter of every _ argument block to determine this path

            # ALICE
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f"lrout{lr_out}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            
            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_length/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_length_history)

        if save_reward:

            # the path to where we save the results. we take the first letter of every _ argument block to determine this path
            #ALICE
            directory = f"/home/s2025396/data1/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f"lrout{lr_out}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"

            # # WORKSTATION
            # directory = f"/data1/bosman/resultsQRL/PQC/ep_reward/"+exp_key+f'{n_holes}holes'+f'{n_qubits}qubits'+f'{n_layers}layers'+f'neps{n_episodes}'+f"lrin{lr_in}"+f"lrvar{lr_var}"+f'bsize{batch_size}'+f"gamma{gamma}"+f"start{start}anil{anil}/"
            if not os.path.isdir(directory):
                os.mkdir(directory)

            print(f"Storing results in {directory}")

            # add date and time to filename to create seperate files with the same setting.
            dt = str(datetime.datetime.now()).split()
            time = f";".join(dt[1].split(":"))

            directory += f"-".join((dt[0], time)) + ".npy"


            np.save(directory, episode_reward_history)

    if print_model_summary:
        model.summary()



def test_run():
    for i in tqdm(range(1000)):
        x = i*i

if __name__ == '__main__':   
    argparser = argparse.ArgumentParser()  
    argparser.add_argument("--len_state", "-ls", type = int, default= 2, help="The length of the input state.")
    argparser.add_argument("--prob_1", "-p1", type = float, default= 3/14, help="Probability of split to the right for QFIAH.")
    argparser.add_argument("--n_episodes", "-ne", type = int, default= 250000, help="Amount of episodes to train for.")
    argparser.add_argument("--n_holes", "-nh", type = int, default= 5, help="The amount of holes in the game.")
    argparser.add_argument("--max_steps", "-ms", type = int, default= 6, help="The amount of steps allowed within an episode.")
    argparser.add_argument("--batch_size", "-bs", type = int, default= 10, help="The batch size.")
    argparser.add_argument("--env_name", "-en", type = str, default= "FoxInAHolev2", help="The environment/game version.")
    argparser.add_argument("--lr_in", "-li", type = float, default= 0.01, help="The learning rate for the input scaling.")
    argparser.add_argument("--lr_var", "-lv", type = float, default= 0.01, help="The learning rate for the variational scaling.")
    argparser.add_argument("--lr_out", "-lo", type = float, default= 0.01, help="The learning rate for observable weights.")
    argparser.add_argument("--n_layers", "-nl", type = int, default= 2, help="The amount of layers.")
    argparser.add_argument("--RxCnot", "-Rx", type = int, default= 0, choices= [0,1], help="Use the PQC with RxCnot architecture or not. 0= True, 1= False")
    argparser.add_argument("--n_reps","-nr", type = int, default= 10, help = "The amount of repetitions to run.")

    args = argparser.parse_args()
    len_state = args.len_state
    prob_1= args.prob_1
    prob_2= 1-prob_1
    n_episodes = args.n_episodes
    n_holes = args.n_holes
    max_steps = args.max_steps
    batch_size = args.batch_size
    env_name = args.env_name
    lr_in = args.lr_in
    lr_var = args.lr_var
    lr_out = args.lr_out
    n_layers = args.n_layers
    n_qubits = n_holes
    RxCnot = args.RxCnot
    if RxCnot ==0:
        RxCnot = False
    elif RxCnot == 1:
        RxCnot = True
    n_reps = args.n_reps
 
    try:
        n_cores = os.environ['SLURM_JOB_CPUS_PER_NODE']
        print("cores provided are: "+f"{n_cores}")
        n_cores = int(n_cores)

    except:
        
        n_cores = n_reps

    print("The number of cores available is {}".format(n_cores))
    
    if env_name.lower()== 'qfiahv1'or 'qfiahv2':
        if RxCnot:
            exp_key = f"{len_state}-inp-{env_name}-prob1{round(prob_1,2)}-prob2{round(prob_2,2)}-maxsteps{max_steps}-PQC-v4-RxCNOT"

        else:
            exp_key = f"{len_state}-inp-{env_name}-prob1{round(prob_1,2)}-prob2{round(prob_2,2)}-maxsteps{max_steps}-PQC-v4"
        
    else:
        if RxCnot:
            exp_key = f"{len_state}-inp-{env_name}-maxsteps{max_steps}-PQC-v4-RxCNOT"

        else:
            exp_key = f"{len_state}-inp-{env_name}-maxsteps{max_steps}-PQC-v4"

    print("(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, lr_in, lr_var, lr_out, n_layers, n_qubits, RxCnot, exp_key)")
    print(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, lr_in, lr_var, lr_out, n_layers, n_qubits, RxCnot, exp_key)
    p = mp.Pool(int(n_cores))
    res = p.starmap(run, [(len_state, prob_1, prob_2, n_episodes, n_holes, max_steps, batch_size, env_name, lr_in, lr_var, lr_out, n_layers, n_qubits, RxCnot, exp_key) for _ in range(n_reps)])
    p.close()
    p.join()