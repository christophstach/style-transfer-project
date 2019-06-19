def extract_activation_map(image, model, layer):
    class SaveActionMap:
        def __init__(self):
            self.activation = None

        def hook(self, model, inpt, outpt):
            self.activation = outpt

    sam = SaveActionMap()

    handle = model[layer].register_forward_hook(sam.hook)
    model(image.unsqueeze(0))
    handle.remove()

    return sam.activation.squeeze()
