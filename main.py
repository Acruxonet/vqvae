import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
import time
from models.vqvae import VQVAE

parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--dataset",  type=str, default='CIFAR10')

# whether or not to save model
parser.add_argument("-save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
    args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)

model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}


def train():
    model.train()
    # 【核心修改 1】在循环外面创建迭代器，只创建这一次！
    data_iter = iter(training_loader)
    
    print(f"Model is on: {next(model.parameters()).device}")

    for i in range(args.n_updates):
        t0 = time.time()
        
        # 【核心修改 2】从同一个迭代器里拿数据，不重新开始
        try:
            (x, _) = next(data_iter)
        except StopIteration:
            # 如果一轮（Epoch）跑完了，重置迭代器继续跑
            data_iter = iter(training_loader)
            (x, _) = next(data_iter)
            
        t_data = time.time() - t0 # 记录取图时间

        t1 = time.time()
        x = x.to(device)
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss

        loss.backward()
        optimizer.step()
        
        t_gpu = time.time() - t1 # 记录显卡干活时间

        # 记录结果 (保持不变)
        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())

        # 每 50 次 Update 打印一次真相
        if i % 50 == 0:
            print(f"Update {i} | 读图: {t_data:.4f}s | 显卡算力: {t_gpu:.4f}s | 显卡利用率估算: {t_gpu/(t_data+t_gpu):.1%}")
            print(f"Recon Error: {np.mean(results['recon_errors'][-50:]):.4f}")


if __name__ == "__main__":
    train()
