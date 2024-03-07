import torch
import utils
from models.model import ACModel
from models.retrieval_module import Retrieval
from models.external_knowledge import expert_behaviors_by_env

def get_max_probs(probs):
    max_probs = torch.zeros_like(probs)
    max_inds = torch.argmax(probs, dim=1)
    max_probs[max_inds] = 1

    return max_probs

def load_module(module_name, module_type, obs_space, action_space, use_memory, use_text, n_neighbours, env_name="all"):
    module_dir = utils.get_model_dir(module_name)
    print(f"Loading module from {module_dir} and module type {module_type}")
    if module_type == "skill":
        module = ACModel(obs_space, action_space, use_memory=use_memory, use_text=use_text)
        try:
            status = utils.get_status(module_dir)
        except OSError:
            ValueError(f"Module {module_dir} not found")
        if "model_state" in status:
            module.load_state_dict(status["model_state"])
        else:
            ValueError(f"Module {module_dir} has no model_state")
        module.eval()
    elif module_type == "retrieval":
        module = Retrieval(obs_space, action_space, n_neighbours, module_dir, argmax=False, num_envs=1, use_memory=False, use_text=False)
    elif module_type == "rules":
        module = expert_behaviors_by_env(env_name)
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    return module


def get_module_dist(obs, module, module_type, probs=True):
    if module_type == "skill":
        probs = module(obs, None)[0].probs
    elif module_type == "retrieval":
        probs = module.predict(obs).probs
    elif module_type == "rules":
        probs =  module(obs)
    else:
        raise ValueError(f"Unknown module type: {module_type}")
    return probs