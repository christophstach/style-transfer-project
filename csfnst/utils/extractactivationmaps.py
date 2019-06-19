def extract_activation_maps(image_batch, model, layers, detach=False):
    class SaveActionMap:
        def __init__(self):
            self.activation = []

        def hook(self, model, inpt, outpt):
            self.activation.append(outpt.detach() if detach else outpt)

    sam = SaveActionMap()

    handles = []
    for layer in layers:
        i = list(dict(model.named_children()).keys()).index(str(layer))
        handles.append(model[i].register_forward_hook(sam.hook))
    
    model(image_batch)

    for handle in handles:
        handle.remove()

    return sam.activation
