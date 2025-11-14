import torch
import torch.nn as nn
import torch.optim as optim

from agent import PongNet, STATE_DIM, NUM_ACTIONS

N_SAMPLES = 20_000
MARGIN = 0.05

def generate_synth_data(n_samples:int):

    cube_x= torch.rand(n_samples)
    cube_y= torch.rand(n_samples)
    cube_vx= torch.empty(n_samples).uniform_(-1.0,1.0)
    cube_vy= torch.empty(n_samples).uniform_(-1.0,1.0)

    right_paddle_y = torch.rand(n_samples)
    left_paddle_y = torch.rand(n_samples)

    right_paddle_dir = torch.zeros(n_samples)
    left_paddle_dir = torch.zeros(n_samples)

    states = torch.stack(
        [
            cube_x,
            cube_y,
            cube_vx,
            cube_vy,
            right_paddle_y,
            right_paddle_dir,
            left_paddle_y,
            left_paddle_dir,
        ],
        dim=1
    )

    diff = cube_y - right_paddle_y

    actions = torch.empty(n_samples, dtype=torch.long)

    actions[diff < -MARGIN] = 0                         #up
    actions[(diff >= -MARGIN) & (diff <= MARGIN)] = 1   #stay
    actions[diff > MARGIN] = 2                          #down

    return states, actions


def main():
    states, actions = generate_synth_data(N_SAMPLES)

    model = PongNet(state_dim=STATE_DIM, num_actions=NUM_ACTIONS)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(100):
        optimizer.zero_grad()
        logits = model(states)
        loss = criterion(logits, actions)
        loss.backward()
        optimizer.step()
        print(f"epoch {epoch+1}, loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "./pong_pretrained_teacher.pth")
    print("Saved pretrained weights to pong_pretrained_teacher.pth")

if __name__ == "__main__":
    main()
