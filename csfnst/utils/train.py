import os

import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from tqdm import tnrange, tqdm_notebook


def train(
        model,
        dataset,
        epochs=10,
        optimizer=None,
        criterion=nn.CrossEntropyLoss(),
        checkpoint_name=None
):
    path = os.path.realpath(
        f'{os.path.dirname(os.path.realpath(__file__))}/../../checkpoints/{checkpoint_name}.pkl'
    )

    running_loss_avg = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not optimizer:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    if checkpoint_name and os.path.exists(path) and os.path.isfile(path):
        print(f'Loading checkpoint from "{path}"')
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=1
    )

    model.train()
    for epoch in tnrange(epochs, desc='Epoch:'):  # loop over the dataset multiple times
        running_loss = 0.0
        itr = tqdm_notebook(enumerate(loader, 0), total=len(loader))
        for i, data in itr:

            # get the inputs
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % running_loss_avg == running_loss_avg - 1:
                itr.set_postfix_str(f'Loss: {round(running_loss / running_loss_avg, 3)}')
                running_loss = 0.0

    if checkpoint_name:
        print(f'Saving checkpoint to "{path}"')

        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, path)
