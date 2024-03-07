import argparse
import time
import torch
from torch_ac.utils.penv import ParallelEnv
from models.retrieval_module import save_retrieval_data
import numpy as np
import utils
from utils import device


# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--episodes", type=int, default=100,
                    help="number of episodes of evaluation (default: 100)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--procs", type=int, default=16,
                    help="number of processes (default: 16)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="action with highest probability is selected")
parser.add_argument("--worst-episodes-to-show", type=int, default=10,
                    help="how many worst episodes to show")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--embeddings", action="store_true", default=False,
                    help="return and store embeddings for retrieval")


parser.add_argument("--module_dirs", nargs="+", default=None,
                    help="list of directories of modules to be used for the knowledge agents")
parser.add_argument("--module_types", nargs="+", default=None,
                    help="list of moduel types to be used for the knowledge agents")
parser.add_argument('--n_neighbours', type=int, default=1,
                    help='number of neighbours to use for retrieval')
parser.add_argument('-wandb_log', '--wandb_log', action=argparse.BooleanOptionalAction)
parser.add_argument("--agent", required=True,
                    help="agent to use: AMRL | KGRL | AC | KoGuN | A2T | Discrete (REQUIRED)")

if __name__ == "__main__":
    args = parser.parse_args()

    # Set seed for all randomness sources

    utils.seed(args.seed)

    # Set device

    print(f"Device: {device}\n")

    # Load environments

    envs = []
    for i in range(args.procs):
        env = utils.make_env(args.env, args.seed + 10000 * i)
        envs.append(env)
    env = ParallelEnv(envs)
    print("Environments loaded\n")

    # Load agent

    model_dir = utils.get_model_dir(args.model)
    agent = utils.Agent(env.observation_space, 
                        env.action_space, 
                        model_dir,
                        argmax=args.argmax, 
                        num_envs=args.procs,
                        use_memory=args.memory, 
                        use_text=args.text, 
                        return_embedding=args.embeddings,
                        agent_type = args.agent,
                        module_dirs = args.module_dirs,
                        module_types = args.module_types,
                        n_neighbours=args.n_neighbours)
    print("Agent loaded\n")

    # Initialize logs

    logs = {"num_frames_per_episode": [], "return_per_episode": []}

    # Run agent

    start_time = time.time()

    obss = env.reset()

    log_done_counter = 0
    log_episode_return = torch.zeros(args.procs, device=device)
    log_episode_num_frames = torch.zeros(args.procs, device=device)
    state_list = []
    reward_list = []
    action_list = []
    while log_done_counter < args.episodes:
        if agent.return_embedding:
            actions, embeddings = agent.get_actions(obss)
        else:
            actions = agent.get_actions(obss)
        obss, rewards, terminateds, truncateds, _ = env.step(actions)
        dones = tuple(a | b for a, b in zip(terminateds, truncateds))
        agent.analyze_feedbacks(rewards, dones)
        # store for retrieval data
        state_list.append(embeddings)
        reward_list.append(rewards)
        action_list.append(actions)

        log_episode_return += torch.tensor(rewards, device=device, dtype=torch.float)
        log_episode_num_frames += torch.ones(args.procs, device=device)

        for i, done in enumerate(dones):
            if done:
                log_done_counter += 1
                logs["return_per_episode"].append(log_episode_return[i].item())
                logs["num_frames_per_episode"].append(log_episode_num_frames[i].item())

        mask = 1 - torch.tensor(dones, device=device, dtype=torch.float)
        log_episode_return *= mask
        log_episode_num_frames *= mask

    end_time = time.time()
    

    

    if args.embeddings:
        # save retrieval data, concatenate data from parallel environments
        rewards = np.array(reward_list)
        actions = np.array(action_list)
        flattened_rewards = np.reshape(rewards, (rewards.shape[0] * rewards.shape[1]))
        flattened_actions = np.reshape(actions, (actions.shape[0] * actions.shape[1]))
        #  flatten states
        state_array = np.array(state_list)
        state_array = state_array.transpose([1, 0, 2])
        N , M, _ = state_array.shape
        flattened_states = np.reshape(state_array, (N * M, -1))
        save_retrieval_data(flattened_states, flattened_rewards, flattened_actions, model_dir)
    # Print logs

    num_frames = sum(logs["num_frames_per_episode"])
    fps = num_frames / (end_time - start_time)
    duration = int(end_time - start_time)
    return_per_episode = utils.synthesize(logs["return_per_episode"])
    num_frames_per_episode = utils.synthesize(logs["num_frames_per_episode"])

    print("F {} | FPS {:.0f} | D {} | R:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | F:μσmM {:.1f} {:.1f} {} {}"
          .format(num_frames, fps, duration,
                  *return_per_episode.values(),
                  *num_frames_per_episode.values()))

    # Print worst episodes

    n = args.worst_episodes_to_show
    if n > 0:
        print("\n{} worst episodes:".format(n))

        indexes = sorted(range(len(logs["return_per_episode"])), key=lambda k: logs["return_per_episode"][k])
        for i in indexes[:n]:
            print("- episode {}: R={}, F={}".format(i, logs["return_per_episode"][i], logs["num_frames_per_episode"][i]))
