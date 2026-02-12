import argparse
import numpy as np
import pdb
from tqdm import tqdm
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from myfolder.code.pipelines.pretrain.vqvae_skeleton.models.MAE.model.SkeletonMAE import SkeletonMAE
from datasets.mabe_mouse import MabeMouseDataset

def get_args_parser():

    parser = argparse.ArgumentParser("STTF Training & Compute Representation", add_help=False)

    """STTFormer Model Hyperparameters"""
    parser.add_argument('--dim_in', default=2, type=int, help='input dimension')
    parser.add_argument('--dim_feat', default=128, type=int, help='feature dimension')
    parser.add_argument('--decoder_dim_feat', default=128, type=int, help='decoder feature dimension')
    parser.add_argument('--depth', default=5, type=int, help='number of layers in the encoder')
    parser.add_argument('--decoder_depth', default=2, type=int, help='number of layers in the decoder')
    parser.add_argument('--num_heads', default=8,  type=int, help='number of attention heads')
    parser.add_argument('--mlp_ratio', default=4, type=int, help='ratio of mlp hidden dim to embedding dim')
    parser.add_argument('--num_frames', default=300, type=int, help='number of frames in the input skeleton sequence')
    parser.add_argument('--num_joints', default=12, type=int, help='number of joints in the input skeleton sequence')
    parser.add_argument('--patch_size', default=1, type=int, help='spatial patch size (number of joints per patch)')
    parser.add_argument('--t_patch_size', default=3, type=int, help='temporal patch size (number of frames per patch)')
    parser.add_argument('--qkv_bias', action='store_true', help='if True, add a learnable bias to query, key, value')
    parser.add_argument('--qk_scale', default=None, type=float, help='override default qk scale of head_dim ** -0.5 if set')
    parser.add_argument('--drop_rate', default=0., type=float, help='dropout rate')
    parser.add_argument('--attn_drop_rate', default=0., type=float, help='attention dropout rate')
    parser.add_argument('--drop_path_rate', default=0., type=float, help='stochastic depth decay rate')
    parser.add_argument('--norm_layer', default=nn.LayerNorm, type=type, help='normalization layer')
    parser.add_argument('--norm_skes_loss', action='store_true', help='if True, normalize skeletons before computing loss')
    
    
    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mabe_mouse')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/MaBe/mouse/mouse_triplet_train.npy')
    parser.add_argument("--sliding_window", default=300, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--if_fill_holes", default=False, type=str2bool)
    parser.add_argument("--cache_path", type=str, default='./data/tmp/mabe_mouse_train.pkl')
    parser.add_argument("--cache", default=False, type=str2bool) # if true cache processed data or load from cache
    #parser.add_argument("--compression_factor", type=int, default=24)
    # in foward function of STTFormer
    parser.add_argument('--mask_ratio', default=0.90, type=float, help='Masking ratio (percentage of removed patches).')
    
    """Dataset augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=True, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_testdata", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)
    
    """Training parameters"""
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')

    """Saving and logging"""
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default=None) # checkpoint path to for train model
    parser.add_argument("--model_path", type=str, default="./outputs/models/vqvae_model.pth") # path to save model after training

    """Type of job"""
    parser.add_argument("--job", type=str, choices=["train", "compute_representations"])

    return parser.parse_args()



# Train funtion
def train(model, loader_train, optimizer, device, writer, timestamp, args):
    # load checkpoint if exists
    if args.ckpt_path is not None:
        print(f"Loading checkpoint from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        print("No checkpoints found, starting training from scratch.")
        os.makedirs(os.path.join(args.save_dir, 'checkpoints'), exist_ok=True)
        start_epoch = 0
    
    model = model.to(device)
    num_epochs = args.epochs - start_epoch
    print('Number of epochs to train:', num_epochs)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        results = {'embedding_loss': 0, 
                   'recon_errors': 0, 
                   'total_loss': 0,
                   'perplexities': 0}
        
        for batch_idx, (x, _)  in enumerate(tqdm(loader_train, total=len(loader_train))):
        #for i, (x, _) in enumerate(tqdm(islice(loader_train, 100), total=100)):
            x = x.to(device)
            optimizer.zero_grad()
            loss, pred, mask = model(x, mask_ratio=args.mask_ratio)
            loss.backward()
            optimizer.step()
            """
            embedding_loss, x_hat, perplexity, _, _, _ = model(x, mask_ratio=args.mask_ratio)
            recon_loss = torch.mean((x_hat - x)**2)
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()
            results["embedding_loss"] += embedding_loss.item()
            results["recon_errors"] += recon_loss.item()
            results["total_loss"]   += loss.item()
            results["perplexities"] += perplexity.item()
            """
            results["total_loss"]  += loss.item()
            
            if (batch_idx + 1) % args.log_interval == 0:
                avg_loss = results["total_loss"] / (batch_idx + 1)
                print(f"Epoch [{epoch}/{args.epochs}], Step [{batch_idx+1}/{len(loader_train)}], Loss: {avg_loss:.4f}")
                #writer.add_scalar('train/loss', avg_loss, epoch * len(loader_train) + batch_idx)
                #results["total_loss"] = 0.0
        """
        avg_embed_error = results["embedding_loss"] / len(loader_train)
        avg_recon_error = results["recon_errors"] / len(loader_train)
        avg_perplexity = results["perplexities"] / len(loader_train)
        """
        avg_total_loss = results["total_loss"] / len(loader_train)
        
        print(f'Epoch {epoch}/{args.epochs} - Loss: {avg_total_loss:.4f},')
              #f'Recon: {avg_recon_error:.4f}, Embed: {avg_embed_error:.4f}, Perplexity: {avg_perplexity:.2f}')
        

        # Save checkpoint
        if args.save_dir and (epoch % 10 == 0 or epoch + 1 == args.epochs):
            checkpoint_path = os.path.join(args.save_dir, 'checkpoints', f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
    
    save_model(model, optimizer, args)
    print(f"Model saved at {args.save_dir}/models/")
    save_results(results, args, timestamp)




if __name__ == "__main__":

    timestamp = readable_timestamp()
    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    """Set up model"""
    model = SkeletonMAE(
        dim_in=args.dim_in,
        dim_feat=args.dim_feat,
        decoder_dim_feat=args.decoder_dim_feat,
        depth=args.depth,
        decoder_depth=args.decoder_depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,  
        num_frames=args.num_frames,
        num_joints=args.num_joints,
        patch_size=args.patch_size,
        t_patch_size=args.t_patch_size,
        qkv_bias=args.qkv_bias,
        qk_scale=args.qk_scale,
        drop_rate=args.drop_rate,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path_rate, 
        norm_layer=args.norm_layer, 
        norm_skes_loss=args.norm_skes_loss
    )
    total_params = sum(p.numel() for p in  model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    dataset_train = MabeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                     sampling_rate=args.sampling_rate,
                                     num_frames=args.num_frames, 
                                     sliding_window=args.sliding_window,
                                     if_fill=args.if_fill_holes,
                                     #patch_size=args.patch_size,
                                     cache_path=args.cache_path, cache=args.cache,
                                     augmentations=args.data_augment, #centeralign=args.centeralign,
                                     include_testdata=args.include_testdata,)

    loader_train = DataLoader(dataset_train, #sampler=sampler_train,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_mem, drop_last=True,)

    """Set up optimizer and training loop"""
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    train(model, loader_train, optimizer, device, writer=None, timestamp=timestamp, args=args)
    
    