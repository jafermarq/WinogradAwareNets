import torch
from tqdm import tqdm

from src import datasets, utils
from src import models
from src import configurator as config


# This code allows:
#  1. Using a pre-trained model with standard convolutions to initialize a winograd-aware model
#  2. Resume training given a checkpoint 

def main():

    device, args, WA_args, Q_args, M_args = config.adaptConfiguration()   

    train_loader, val_loader, test_loader = datasets.getCIFAR10(batch_size=args.batch_size, use_cuda=(True if device =='cuda' else False))
    dummy_input = torch.zeros(1, 3, 32, 32).to(device)
    num_classes = 10

    # here we create the network as described in by the new arguments (winograd/normal, F, Q, bits,...) and some of those from the model that would be loaded (e.g. mult, dataset)
    model = models.ResNet18(mult=args.mult, num_classes=num_classes, winogradArgs=WA_args, quantArgs=Q_args, miscArgs=M_args)

    model.to(device)
    target_model_dict = model.state_dict()

    print("loading source dict")
    # 0. loading pre-trained dict
    source_dict = torch.load(args.modelDir + '/' + args.modelToLoad)
    # 1. filter out unnecessary keys
    source_dict = {k: v for k, v in source_dict.items() if k in target_model_dict}
    # 2. overwrite entries in the existing state dict
    target_model_dict.update(source_dict) 
    # 3. load the new state dict
    print("Loading data to new model")
    #! If you're training a WA model from a pre-trained model using standard convolutions, this will likely output several warning due to 'non found keys'. 
    # This is entirely normal since the WinograAware pipeline adds several elements that are not present in default convolutions
    not_found = model.load_state_dict(source_dict, strict=False)
    print(not_found)

    # now we initialize the model
    dir, writer = utils.init(args, model, dummy_input)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=4e-5)

    # lr decay steps
    milestones = [args.epochs * 3.0/5, args.epochs * 4.0/5]
    print("Learning rate decay at: ", milestones)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    best_val = 0
    epoch = 0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion, writer)
        acc, loss = test(args, model, device, val_loader, epoch, criterion, writer)

        if acc > best_val:
            print("New best model found!")
            print("Saving to: ", dir + "/best_model.pt")
            torch.save(model.state_dict(), dir+"/best_model.pt")
            best_val = acc

    # evaluating test set
    test(args, model, device, test_loader, epoch, criterion, writer, is_test=True)

    # saving model
    torch.save(model.state_dict(), dir+"/model.pt")

    # now we load best model and evaluate test set
    if epoch > 0:
        model.load_state_dict(torch.load(dir+"/best_model.pt"))
        test(args, model, device, test_loader, epoch+1, criterion, writer, is_test=True)

    writer.close()


def train(args, model, device, train_loader, optimizer, scheduler, epoch, criterion, writer):
    model.train()
    loss_avg = utils.RunningAvg()
    acc_avg = utils.RunningAvg()

    with tqdm(total=len(train_loader.dataset), desc='Train Epoch #' + str(epoch) + "/" + str(args.epochs)) as t:
        for i , (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output, 1)
            correct = (predicted == target).sum()
            acc_avg.update(correct.item()/float(data.shape[0]))

            loss_avg.update(loss.item())
            t.set_postfix({'avgAcc':'{:05.3f}'.format(acc_avg()), 'avgLoss':'{:05.3f}'.format(loss_avg())})
            t.update(data.shape[0])

    writer.add_scalar('Train/Loss', loss_avg(), epoch)
    writer.add_scalar('Train/Accuracy', acc_avg(), epoch)
    writer.add_scalar('Train/lr', scheduler.get_lr()[0], epoch)

    scheduler.step()


def test(args, model, device, test_loader, epoch, criterion, writer, is_test: bool = False):
    model.eval()
    test_loss = 0
    correct = 0

    with tqdm(total=len(test_loader.dataset)) as t:
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item() # sum up batch loss

                _, predicted = output.max(1)
                correct += predicted.eq(target).sum().item()
                t.update(data.shape[0])

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    if not(is_test):
        writer.add_scalar('Val/Loss', test_loss, epoch)
        writer.add_scalar('Val/Accuracy', acc, epoch)
    else:
        writer.add_scalar('Test/Accuracy', acc, epoch)


    print(('Test' if is_test else 'Val' )+ 'set: Average loss: {:.4f}, Accuracy: {}/{} ({:.1f}%)'.format( test_loss, correct, len(test_loader.dataset), acc))

    return acc, test_loss

if __name__ == '__main__':
    main()
