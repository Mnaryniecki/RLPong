import torch
import torch.nn as nn
import torch.nn.functional as F
import os


print(torch.version.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

# We will fill these in once we finalize the state vector
STATE_DIM = 8      # placeholder for now (ball pos/vel + paddle pos/dir)
NUM_ACTIONS = 3    # up, stay, down

class PongNet(nn.Module):
    def __init__(self , state_dim=STATE_DIM , num_actions=NUM_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(state_dim , 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, num_actions)

    def forward(self, x):
        # x expected tensor of shape
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x) # Raw output not normalized

class PongAgent:
    def __init__(self, device =None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"PongAgent initialized on: {self.device}")
        self.model = PongNet().to(self.device)

        # Load RL weights if they exist, otherwise fallback to teacher, then random
        if os.path.exists("pong_rl.pth"):
            weights_path = "pong_rl.pth"
            print(f"Loading RL weights: {weights_path}")
        elif os.path.exists("pong_pretrained_teacher.pth"):
            weights_path = "pong_pretrained_teacher.pth"
            print(f"Loading Teacher weights: {weights_path}")
        else:
            weights_path = None
            print("No weights found, using random initialization")

        if weights_path:
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def act(self, state, stochastic=False , temperature=1):
        # Converting python list state into a tensor
        state_t= torch.tensor(state , dtype=torch.float , device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self.model(state_t)

        if stochastic:
            probs = torch.softmax(logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample().item()
        else:
            # Greedy algorithm otherwise
            action = torch.argmax(logits , dim=-1 ).item()


        # DEBUG comment out if unused
        #print("STATE:",state )
        #print("LOGITS:",logits.squeeze(0).cpu().numpy())
        #print("PROBS:",torch.softmax(logits / temperature, dim=-1).squeeze(0).cpu().numpy())
        #print("ACTION:",action)


        return action

    def act_batch(self, state, stochastic=False , temperature=1):
        # Converting python list state into a tensor
        state_t= torch.tensor(state , dtype=torch.float , device=self.device)

        with torch.no_grad():
            logits = self.model(state_t)

        if stochastic:
            probs = torch.softmax(logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
        else:
            # Greedy algorithm otherwise
            action = torch.argmax(logits , dim=-1 )

        return action