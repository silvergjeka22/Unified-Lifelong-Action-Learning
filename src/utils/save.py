from src.meta_learning.training import evaluate
from src.imports import device

# recording meta-learning results
results_t1, results_t2 = {}, {}

def record(name, model, results_dict, old_loader, new_loader, all_loader):
    _, old     = evaluate(model, old_loader, device)
    _, new_    = evaluate(model, new_loader, device)
    _, all_acc = evaluate(model, all_loader, device)
    results_dict[name] = {"Old": old, "New": new_, "All": all_acc}
    print(f" {name:<36} Old:{old:.4f}  New:{new_:.4f}  All:{all_acc:.4f}")