import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def finetune_adapt(model, clips, labels, device, steps=30, lr=5e-4):
    """
    Plain fine-tune on the labelled adaptation clips - the no-meta baseline.
    Adapts and returns the model.
    """
    model.to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    clips, labels = clips.to(device), labels.to(device)
    for step in range(steps):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(clips), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def reptile_adapt(model, clips, labels, device, episodes=30, inner_steps=10,
                  inner_lr=5e-4, epsilon=0.20):
    """
    Reptile adaptation: each episode take inner_steps of SGD, then move the weights a
    fraction epsilon toward the adapted weights. The with-meta arm. Returns the model.
    """
    model.to(device)
    clips, labels = clips.to(device), labels.to(device)

    for ep in range(episodes):
        start = {n: p.detach().clone() for n, p in model.named_parameters()}
        optimizer = torch.optim.SGD(model.parameters(), lr=inner_lr, momentum=0.9)
        model.train()
        for _ in range(inner_steps):
            optimizer.zero_grad()
            F.cross_entropy(model(clips), labels).backward()
            optimizer.step()
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.copy_(start[n] + epsilon * (p - start[n]))
    model.eval()
    return model
