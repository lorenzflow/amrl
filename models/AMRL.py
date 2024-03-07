import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from models.model import init_params
from models.model_utils import get_max_probs, load_module, get_module_dist

class AMRL_ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, 
                 obs_space, 
                 action_space, 
                 module_dirs=None, 
                 module_types=None,
                 n_neighbours=None,
                 use_memory=False, 
                 use_text=False,
                 full_dist=True,
                 hard_selection=False
                 ):
        
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.num_actions = action_space.n
        self.d_k = 8
        self.full_dist = full_dist
        self.hard_selection = hard_selection
        # load modules
        self.modules = []
        self.n_modules = len(module_dirs)
        self.n_neighbours = n_neighbours
        self.module_types = module_types

        if "retrieval" in module_types:
            assert n_neighbours is not None, "n_neighbours must be provided for retrieval modules"

        for module_dir, module_type in zip(module_dirs, module_types): # change this to module names
            module = load_module(module_dir, 
                                 module_type, 
                                 obs_space, 
                                 action_space, 
                                 use_memory, 
                                 use_text, 
                                 n_neighbours, 
                                 env_name="all"
                                 )
            self.modules.append(module)

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model adjust size here to concatenate actions and state
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # additional KoGuN layers
        self.weights = nn.Linear(64, self.n_modules+1)
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)
        x = self.actor(embedding) # B x n_actions
        weights = F.softmax(self.weights(self.actor[:1](embedding)),dim=1) # B x n_modules+1
        
        if self.hard_selection:
            logits=torch.log(weights)
            if torch.isnan(logits).any() or torch.isinf(logits).any():
                raise ValueError("NaN or infinite values in input_data.")

            weights = F.gumbel_softmax(logits, tau=1, hard=True)
            
        # log_softmax version
        fused_dist = torch.softmax(x, dim=1) * weights[:,0].unsqueeze(1) # B x n_actions
        for i, module in enumerate(self.modules):
            probs = get_module_dist(obs, module, self.module_types[i])
            if self.full_dist:
                fused_dist = fused_dist + probs * weights[:,i+1].unsqueeze(1)
            else:
                max_probs = get_max_probs(probs)
                fused_dist = fused_dist + max_probs * weights[:,i+1].unsqueeze(1)
        
        assert torch.allclose(torch.sum(fused_dist, dim=1), torch.ones((fused_dist.shape[0]), dtype=torch.float64), atol=1e-5), "Sum of fused_dist is not equal to 1 for all entries"
        
        dist_logits = torch.log(fused_dist)
        if torch.isnan(dist_logits).any() or torch.isinf(dist_logits).any():
                raise ValueError("NaN or infinite values in dist_log.")
        fused_dist = Categorical(logits=dist_logits)

        x = self.critic(embedding)
        value = x.squeeze(1)
    

        return fused_dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    