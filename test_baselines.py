import torch
ckpt = torch.load("pretrained_weights/neurostorm.ckpt", map_location="cpu")
state = None
for k in ["state_dict", "model", "backbone", "encoder"]:
    if isinstance(ckpt, dict) and isinstance(ckpt.get(k), dict):
        state = ckpt[k]
        print("using:", k)
        break
if state is None and isinstance(ckpt, dict):
    state = ckpt
for i, key in enumerate(state.keys()):
    print(key)
    if i >= 29:
        break
