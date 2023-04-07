def simple_train(model, loss_fn, optimizer, train_loader, num_epochs=5, device='gpu'):
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()

            if (i + 1) % 2000 == 0:
                print(
                    f'epoch {epoch + 1}/{num_epochs}, step {i + 1}/{n_total_steps}, loss = {loss.item():.4f}')

    print('Finished training')


