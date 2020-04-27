from torch.utils.data import DataLoader
from src.model import SSD
from src.multibox_loss import MultiBoxLoss
from src.prior_box import PriorBox
from src.data.datasets import VOCxx
from src.utils import *
from argparser import get_train_argument, set_cuda_dev

def main(agrs):
    start_epoch = 0

    # Initialize model or load trained checkpoint
    if args.resume:
        start_epoch, model, optimizer = load_checkpoint(args.trained_model)
    else:
        model = SSD('train', args)
        optimizer = init_optimizer(model, args)

    # Move to default device and set 'train' mode
    model = model.cuda()
    model.train()

    # Create multibox loss
    criterion = MultiBoxLoss(PriorBox().forward().cuda(), args.overlap_threshold, args.negpos_ratio, args.alpha)

    # VOC dataloaders
    train_dataset = VOCxx('train', args.dataroot, args.datayears, args.datanames, discard_difficult=args.discard_difficult, use_augment=args.use_augment)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=1, pin_memory=True)
    
    # Loop and decay params
    epochs = args.iterations // (len(train_dataset) // args.batch_size)
    decay_iters = [int(it) for it in args.lr_decay.split(',')]
    decay_lr_at = [it // (len(train_dataset) // args.batch_size) for it in decay_iters]
    print('total length of dataset : ', len(train_dataset))
    print('total epochs : ', epochs)
    print('decay lr at : ', decay_lr_at)

    # Epochs
    loc_losses, conf_losses = [], []
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            optimizer = adjust_lr(optimizer)

        for i, (images, targets) in enumerate(train_loader):
            # Move to default device
            images = images.cuda()
            targets = [t.cuda() for t in targets]

            # Forward prop
            preds = model(images) 

            # Loss
            loc_loss, conf_loss = criterion(preds, targets)
            loss = loc_loss + conf_loss

            # Backward prop
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients if necessary
            if args.clip_grad:
                clip_gradient(model.parameters(), args.clip_grad)

            # Update model
            optimizer.step()

            # Print status
            if i % 200 == 0:
                print('Epoch: [{0}][{1}/{2}]\t' 'Loss : {loss:.4f}\t'.format(epoch, i, len(train_loader), loss=loss.item()))
                loc_losses.append(loc_loss.item())
                conf_losses.append(conf_loss.item())

        # Plot losses
        plot_losses(loc_losses, 'regression', args.model_save_name)
        plot_losses(conf_losses, 'classification', args.model_save_name)

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, args.model_save_name)

if __name__ == '__main__':
    # Create the default required directory
    if not os.path.exists(os.getcwd()+'/results/images/'):
        os.makedirs(os.getcwd()+'/results/images/')
    if not os.path.exists(os.getcwd()+'/results/loss/'):
        os.makedirs(os.getcwd()+'/results/loss/')
        print('./results/loss/ folder is created. ')
    # Get arguments for train
    args = get_train_argument()
    print('Arguments of train : ', args)

    # Set cuda device
    set_cuda_dev(args.ngpu)

    # Train
    main(args)
