import torch
from src.utils.config import Config
from src.model.gpt import CausalLM
from src.training.optimizer import create_optimizer

config = Config.from_yaml("configs/model.yaml")
model = CausalLM(config)

print("=== PARAMETER COUNT BREAKDOWN ===")
unique_params_dict = {}
for name, param in model.named_parameters():
    unique_params_dict[id(param)] = (name, param)

total_unique_params = sum(p.numel() for name, p in unique_params_dict.values())
total_named_params = sum(p.numel() for p in model.parameters())

print(f"Total Unique Parameters: {total_unique_params:,}")
print(f"Total Named Parameters (with duplicates): {total_named_params:,}")

print(f"Number of unique parameter tensors: {len(unique_params_dict)}")
print(f"Number of named parameter entries: {len(list(model.named_parameters()))}")

print("\nNamed parameters detail:")
for name, param in model.named_parameters():
    print(f"  {name}: shape={list(param.shape)}, numel={param.numel():,}, id={id(param)}")

print("\n=== OPTIMIZER PARAMETER GROUP INSPECTION ===")
optimizer = create_optimizer(model)

group_ids_0 = set(id(p) for p in optimizer.param_groups[0]['params'])
group_ids_1 = set(id(p) for p in optimizer.param_groups[1]['params'])

intersection = group_ids_0.intersection(group_ids_1)
print(f"\nParameter tensor IDs present in BOTH decay and no-decay groups: {len(intersection)}")
for p_id in intersection:
    names = [name for name, p in model.named_parameters() if id(p) == p_id]
    print(f"  Tensor ID {p_id} has names: {names}")
