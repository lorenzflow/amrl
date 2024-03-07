import torch
import pickle
import faiss
import torch.nn.functional as F
import utils
from utils.other import device
from models.model import ACModel
import numpy as np
import pickle
from torch.distributions.categorical import Categorical



class Retrieval:
    """An agent.
    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, n_neighbours, module_dir,
                 argmax=False, num_envs=1, use_memory=False, use_text=False):
        """
        Initialize the Retrieval agent.

        Args:
            obs_space (gym.Space): The observation space of the environment.
            action_space (gym.Space): The action space of the environment.
            n_neighbours (int): The number of nearest neighbors to retrieve.
            module_dir (str): The directory where the model is saved.
            data_dir (str): The directory where the retrieval data is stored.
            argmax (bool, optional): Whether to use argmax to select actions. Defaults to False.
            num_envs (int, optional): The number of parallel environments. Defaults to 1.
            use_memory (bool, optional): Whether to use memory in the model. Defaults to False.
            use_text (bool, optional): Whether to use text in the model. Defaults to False.
        """
        # obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        self.module_type = "retrieval"
        # load embedding head from pretrained model
        embedding_model = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        self.num_actions = action_space.n
        try:
            status = utils.get_status(module_dir)
        except OSError:
            ValueError(f"Module {module_dir} not found")
        if "model_state" in status:
            embedding_model.load_state_dict(status["model_state"])
        else:
            ValueError(f"Module {module_dir} has no model_state")
        # embedding_model.load_state_dict(utils.get_model_state(module_dir))
        embedding_model.to(device)
        embedding_model.eval()
        self.embedding_head = embedding_model.image_conv

        self.retrieval_data = self.load_retrieval_data(module_dir)
        self.embedding_dim = self.retrieval_data["states"].shape[1] # might also not work
        print(self.embedding_dim)
        # Build a FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)  # Use the L2 distance metric
        self.index.train(self.retrieval_data["states"])  # Add the database vectors to the index
        print(f"Is retrieval index trained? {self.index.is_trained}")
        self.index.add(self.retrieval_data["states"])  # Add the database vectors to the index
        print(f"Number of vectors in the retrieval db: {self.index.ntotal}")
        self.n_neighbours = n_neighbours

        self.argmax = argmax
        self.num_envs = num_envs


        # if self.acmodel.recurrent:
        #     self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

    def predict(self, obs):
        """
        Get actions for a batch of observations.

        Args:
            obss (list): A list of observations.

        Returns:
            numpy.ndarray: An array of actions.
        """
        x = obs.image.transpose(1, 3).transpose(2, 3)
        # get a batch of queries
        with torch.no_grad():
            queries = self.embedding_head(x).detach()
        # queries = self.embedding_head(x).detach()
        queries = queries.reshape(queries.shape[0], -1)

        # search the index for the k nearest neighbours
        D, I = self.index.search(queries, self.n_neighbours) # D and I are (num_queries, n_neighbours)
        # retrieve the actions and rewards for the k nearest neighbours and for all queries
        # list of length num_queries, each element is a list of length n_neighbours
        retrieved_actions = [[self.retrieval_data["actions"][index] for index in list(I[i,:])] for i in range(I.shape[0])]
        retrieved_rewards = np.array([[self.retrieval_data["rewards"][index] for index in list(I[i,:])] for i in range(I.shape[0])])

        # Convert retrieved actions to one-hot encoded vectors
        one_hot_actions = np.array([[np.eye(self.num_actions)[int(action.item())] for action in actions] for actions in retrieved_actions])
        # multiply by the cumulative rewards to weight the actions

        reward_weighted_actions = one_hot_actions * np.expand_dims(retrieved_rewards, axis=-1)
        # reward_weighted_actions = np.multiply(np.expand_dims(retrieved_rewards, axis=0), one_hot_actions)

        # weight by distance to query
        # weighted_actions = np.multiply(np.expand_dims(1/(D+1), axis=0), reward_weighted_actions)
        weighted_actions = reward_weighted_actions * np.expand_dims(1/(D+1), axis=-1)
   
        # sum across the k nearest neighbours
        action_distribution = np.sum(weighted_actions, axis=1)
        
        # in case only 0 rewards return equal probs for all actions
        zero_indeces = np.where(np.sum(action_distribution, axis=1) == 0)
        action_distribution[zero_indeces,:] = 1/self.num_actions
        # convert to logits with softmax
        probs = action_distribution/np.expand_dims(np.sum(action_distribution,axis=1), 1)
        # logits = np.log(action_distribution/np.expand_dims(np.sum(action_distribution,axis=1), 1))
        probs = torch.tensor(probs)
        probs = F.softmax(probs, dim=1)
        dist = Categorical(probs=probs)
        return dist

    def get_action(self, obs):
        """
        Get an action for a single observation.

        Args:
            obs: The observation.

        Returns:
            numpy.ndarray: The action.
        """
        return self.predict([obs])[0]

      

    
    def update_retrieval_data(self, new_trajectory):
        """
        Update the retrieval data by adding a new trajectory.

        Args:
            new_trajectory (tuple): A tuple containing the states, actions, and rewards of the new trajectory.

        Returns:
            None
        """

        # Unpack the new trajectory
        # Add the new trajectory to the retrieval data
        new_states, new_actions, new_rewards = new_trajectory
        x = new_states.image.transpose(1, 3).transpose(2, 3)
        with torch.no_grad():
            x = self.embedding_head(x)
        x = x.reshape(x.shape[0], -1)
        self.retrieval_data["states"].append(x)
        self.retrieval_data["actions"].append(new_actions)
        self.retrieval_data["rewards"].append(new_rewards)

        # Update the index mechanism with the new trajectory
        self.index.add(new_states)
    
    def load_retrieval_data(self, module_dir):
        """
        Load the retrieval data.

        Args:
            data_dir (str): The directory where the retrieval data is stored.
                            - states: A list of lists where each list contains the states visited during 1 trajectory
                            - actions: A list of lists where each list contains the actions taken during 1 trajectory
                            - rewards: A list of lists where each list contains the rewards received during 1 trajectory

        Returns:
            dict: A dictionary containing the embedded states, actions, and rewards.
        """
        data_dir = module_dir + "/retrieval_data.pickle"
        with open(data_dir, 'rb') as data:
            # what should data look like? Trajectories of state, actions, rewards
            retrieval_data = pickle.load(data) # tuple of (state_lists, action_lists, reward_lists)?
        embedded_states, rewards, actions = retrieval_data
        print(embedded_states.shape)
        
        embedded_states = torch.tensor(embedded_states, device=device) # already embedded by same model
        actions = torch.tensor(actions, device=device)
        rewards = torch.tensor(rewards, device=device)
        
        return {"states":embedded_states, "actions":actions, "rewards":rewards}
    
def save_retrieval_data(states, rewards, actions, model_dir):
    """
    Save the retrieval data.

    Args:
        states, actions, rewards (lists): lists containing the embedded states, actions, and rewards.
        data_dir (str): The directory where the retrieval data is stored.

    Returns:
        None
    """
    data_dir = model_dir + "/retrieval_data.pickle"
    retrieval_data = (states, rewards, actions)
    with open(data_dir, 'wb') as data_dir:
        pickle.dump(retrieval_data, data_dir)
    print(f"Retrieval data saved to {data_dir}")
