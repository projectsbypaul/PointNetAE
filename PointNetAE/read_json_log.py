import json

json_strings = []

epochs = []
losses = []
val_losses = []

for line in open(r'versions/loss_log.json'):
    json_strings.append(line)
    dictionary = eval(line)

    epochs.append(dictionary['epoch'])
    losses.append(dictionary['loss'])
    val_losses.append(dictionary['val_loss'])

print(epochs, '\n', losses, '\n', val_losses)
