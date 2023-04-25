from torch.nn import Module;

def get_model_num_classes(model: Module):
    last_layer = list(model.children())[-1]
    last_layer_outputs = list(last_layer.parameters())[-1]
    return len(last_layer_outputs)