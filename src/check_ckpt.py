import torch
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu')
print("Config:", ckpt['config'])
print("\nState dict keys:")
for k, v in ckpt['model_state'].items():
    print(f"  {k}: {v.shape}")