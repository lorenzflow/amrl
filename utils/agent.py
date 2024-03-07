import torch
import utils
from .other import device
from models.model import ACModel
from models.KoGuN import KoGuN_ACModel
from models.AMRL import AMRL_ACModel
from models.KIAN import KIAN_ACModel

import numpy as np
from torch.distributions.categorical import Categorical

class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, 
                 obs_space, 
                 action_space, 
                 model_dir,
                 agent_type = "AC",
                 module_dirs = None,
                 module_types = None,
                 argmax=False, 
                 num_envs=1, 
                 use_memory=False, 
                 use_text=False, 
                 return_embedding=False,
                 n_neighbours=1):
        
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        
       
        if agent_type == "KoGuN":
            assert module_dirs is not None
            assert module_types is not None
            assert len(module_dirs) == len(module_types)
            self.acmodel = KoGuN_ACModel(obs_space, 
                                    action_space, 
                                    module_dirs, 
                                    module_types,
                                    n_neighbours,
                                    use_memory, 
                                    use_text
                                    )
        elif agent_type == "AMRL":
            assert module_dirs is not None
            assert module_types is not None
            assert len(module_dirs) == len(module_types)
            self.acmodel = AMRL_ACModel(obs_space, 
                                    action_space, 
                                    module_dirs, 
                                    module_types,
                                    n_neighbours,
                                    use_memory, 
                                    use_text
                                    )
        elif agent_type == "KIAN":
            assert module_dirs is not None
            assert module_types is not None
            assert len(module_dirs) == len(module_types)
            self.acmodel = KIAN_ACModel(obs_space, 
                                    action_space, 
                                    module_dirs, 
                                    module_types,
                                    n_neighbours,
                                    use_memory, 
                                    use_text
                                    )
        elif agent_type == "AC":
            self.acmodel = ACModel(obs_space, 
                                   action_space, 
                                   use_memory=use_memory, 
                                   use_text=use_text, 
                                   return_embedding=return_embedding)

            module_dirs = []
        elif agent_type == "AMRL_hard":
            assert module_dirs is not None
            assert module_types is not None
            assert len(module_dirs) == len(module_types)
            self.acmodel = AMRL_ACModel(obs_space, 
                                    action_space, 
                                    module_dirs, 
                                    module_types,
                                    n_neighbours,
                                    use_memory, 
                                    use_text, 
                                    hard_selection=True
                                    )
        else:
            raise ValueError("Incorrect agent name: {}".format(agent_type))
            
        self.argmax = argmax
        self.num_envs = num_envs
        self.return_embedding = return_embedding

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size, device=device)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                if self.return_embedding:
                    dist, _, self.memories, embedding = self.acmodel(preprocessed_obss, self.memories)
                else:
                    dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                if self.acmodel.return_embedding:
                    dist, _, _, embedding = self.acmodel(preprocessed_obss)
                else:
                    dist, _ = self.acmodel(preprocessed_obss)

        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()
        if self.acmodel.return_embedding:
            return actions.cpu().numpy(), embedding.cpu().numpy()
        else:
            return actions.cpu().numpy()

    def get_action(self, obs):
        return self.get_actions([obs])[0]

    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float, device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

