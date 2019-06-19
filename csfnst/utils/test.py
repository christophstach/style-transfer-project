import torch
import torch.utils.data


def test(network, dataset):
    network.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
