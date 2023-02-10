import logging
import numpy as np
import torch
import warnings
from tqdm import tqdm
import os

from spaformer.utils import (
    build_args,
    create_optimizer,
    set_random_seed,
    Logger,
    get_current_lr,
    load_best_configs,
    load_dataset,
)
from spaformer.models import build_model


logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


def pretrain(model, graphs, optimizer, max_epoch, device, scheduler, num_classes, loginfo, logger=None, mask=None, test_all=False, test_pe=False):
    test_id = -1
    logging.info("start training..")
    epoch_iter = tqdm(range(max_epoch))
    graphs = [g.to(device) for g in graphs]
    val_list = []
    for epoch in epoch_iter:
        model.train()
        
        for i in range(len(graphs)):
            graph = graphs[i]
            x = graph.ndata['input']

            loss = model(graph, x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2)
            optimizer.step()
            
        if scheduler is not None:
            scheduler.step()

        epoch_iter.set_description(f"# Epoch {epoch}: train_loss: {loss.item():.4f}")

        if (epoch + 1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                val_list.append(model.evaluate(graphs[test_id], graphs[test_id].ndata['input']))
                print('Validation Loss(RMSE):', val_list[-1])
                
                if test_pe:
                    print('Test Loss(RMSE) with PE:', model.evaluate(graphs[test_id], graphs[test_id].ndata['input'], mode='test', mask=mask[test_id]))
                    print('Test Loss(RMSE) w.o PE:', model.evaluate(graphs[test_id], graphs[test_id].ndata['input'], mode='test', mask=mask[test_id], npe=True))
                else:
                    print('Test Loss(RMSE):', model.evaluate(graphs[test_id], graphs[test_id].ndata['input'], mode='test', mask=mask[test_id]))
                
                if epoch>10 and min(val_list[:-1]) > val_list[-1]:
                    if not os.path.exists('ckpt'):
                        os.mkdir('ckpt')
                    torch.save(model.state_dict(), f"ckpt/{loginfo}.ckpt")
                elif min(val_list[-10:]) != min(val_list):
                    logging.info('Early Stop.')
                    break
                    
                if test_all:
                    perflist = []
                    for j in range(len(graphs)):
                        perflist.append(model.evaluate(graphs[j], graphs[j].ndata['input'], mode='test', mask=mask[j]))
                        print(f"fov {j} {perflist[-1]}")
                    print(sum(perflist)/len(perflist))
                    
    model.load_state_dict(torch.load(f"ckpt/{loginfo}.ckpt"))
    return model

def main(args):
    device = args.device if args.device >= 0 else "cpu"
    seed = args.seed
    dataset_name = args.dataset
    max_epoch = args.max_epoch
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    encoder_type = args.encoder
    decoder_type = args.decoder
    optim_type = args.optimizer 
    loss_fn = args.loss_fn
    lr = args.lr
    weight_decay = args.weight_decay
    load_model = args.load_model
    use_scheduler = args.scheduler
    standardscale = args.standardscale
    if args.pe == 'None': args.pe = None
    pe = str(args.pe)
    objective = args.objective
    data_path = args.data_path
    pc = args.noise_pc
    test_pe = args.test_pe
    cache_path = args.cache_path

    graphs, num_features, num_classes, mask, label = load_dataset(dataset_name, pc, standardscale, data_path, cache_path=cache_path)
    args.num_features = num_features

    acc_list = []
    estp_acc_list = []

    loginfo = f"{dataset_name}_pc_{pc}_nh_{num_hidden}_nl_{num_layers}_pe_{pe}_obj_{objective}_{encoder_type}_{decoder_type}_seed_{seed}"
    logger = Logger(args=args)
    logging.info(f"####### seed {seed}")
    set_random_seed(seed)

    model = build_model(args)
    model.to(device)
    optimizer = create_optimizer(optim_type, model, lr, weight_decay)

    if use_scheduler:
        logging.info("Use schedular")
        scheduler = lambda epoch :( 1 + np.cos((epoch) * np.pi / max_epoch) ) * 0.5
        # scheduler = lambda epoch: epoch / warmup_steps if epoch < warmup_steps \
                # else ( 1 + np.cos((epoch - warmup_steps) * np.pi / (max_epoch - warmup_steps))) * 0.5
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler)
    else:
        scheduler = None

    if not load_model:
        torch.set_num_threads(1) 
        model = pretrain(model, graphs, optimizer, max_epoch, device, scheduler, num_classes, loginfo, logger, mask, test_pe=test_pe)
        torch.set_num_threads(8) 
        model = model.cpu()

    if load_model:
        logging.info("Loading Model ... ")
        model.load_state_dict(torch.load("checkpoint.pt"))

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        model.batch_test(graphs, mask, label, logger)

# Press the green button in the gutter to run the script.
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    warnings.filterwarnings("ignore", category=DeprecationWarning) 
    np.seterr(divide='ignore', invalid='ignore')
    args = build_args()
    print(args)
    main(args)
