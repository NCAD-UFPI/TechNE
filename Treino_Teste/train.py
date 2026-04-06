"""
train.py - Script de Treino Distribuído com PyTorch DDP
========================================================
Treina uma ResNet-18 no dataset CIFAR-10 usando DistributedDataParallel (DDP)
para aproveitar múltiplas GPUs em paralelo.

Uso:
    torchrun --nproc_per_node=2 train.py
    torchrun --nproc_per_node=2 train.py --target-seconds 540
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms


def parse_args():
    p = argparse.ArgumentParser(description="Treino DDP (CIFAR-10) com opcional carga extra na GPU.")
    p.add_argument("--epochs", type=int, default=5, help="Numero de epocas de treino (default: 5).")
    p.add_argument(
        "--target-seconds",
        type=int,
        default=0,
        help="Se > 0, garante execucao aproximada ate esse tempo total (s), adicionando carga extra na GPU no final.",
    )
    p.add_argument(
        "--burn-mat-dim",
        type=int,
        default=4096,
        help="Dimensao da matriz (quadrada) usada na carga extra de GPU (default: 4096).",
    )
    p.add_argument(
        "--burn-log-every",
        type=int,
        default=15,
        help="Intervalo (s) para imprimir progresso durante a carga extra (default: 15).",
    )
    return p.parse_args()


def setup_distributed():
    """Inicializa o processo distribuído usando variáveis de ambiente do torchrun."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    """Finaliza o grupo de processos distribuídos."""
    dist.destroy_process_group()


def get_dataloaders(world_size, rank, batch_size=64):
    """
    Cria DataLoaders para treino e teste do CIFAR-10.
    Usa DistributedSampler para dividir os dados entre as GPUs.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download do CIFAR-10 (apenas no rank 0 para evitar conflitos)
    if rank == 0:
        torchvision.datasets.CIFAR10(root="./data", train=True, download=True)
        torchvision.datasets.CIFAR10(root="./data", train=False, download=True)
    dist.barrier()  # Aguarda o rank 0 terminar o download

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=False, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=False, transform=transform_test
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=test_sampler,
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, train_sampler


def create_model(local_rank):
    """Cria o modelo ResNet-18 adaptado para CIFAR-10 e envolve com DDP."""
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    # Adaptar a primeira camada conv para imagens 32x32 do CIFAR-10
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()  # Remove maxpool (desnecessário para 32x32)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])
    return model


def train_one_epoch(model, train_loader, criterion, optimizer, local_rank, epoch):
    """Treina uma época completa."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(local_rank), targets.to(local_rank)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Log a cada 50 batches (apenas no rank 0)
        if local_rank == 0 and (batch_idx + 1) % 50 == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            print(f"  [Época {epoch+1}] Batch {batch_idx+1}/{len(train_loader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    return running_loss / len(train_loader), 100.0 * correct / total


def evaluate(model, test_loader, criterion, local_rank):
    """Avalia o modelo no conjunto de teste."""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(local_rank), targets.to(local_rank)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return test_loss / len(test_loader), 100.0 * correct / total


def burn_gpu(seconds, local_rank, rank, mat_dim=4096, log_every=15):
    """
    Mantem a GPU ocupada por ~seconds, para permitir avaliar logs/monitoramento.
    Cada processo (uma GPU) executa multiplicacoes de matrizes no proprio device.
    """
    if seconds <= 0:
        return

    device = torch.device(f"cuda:{local_rank}")
    dtype = torch.float16

    # Mantem alocacao pequena o bastante para caber com folga em 32G.
    with torch.no_grad():
        a = torch.randn((mat_dim, mat_dim), device=device, dtype=dtype)
        b = torch.randn((mat_dim, mat_dim), device=device, dtype=dtype)

        start = time.time()
        next_log = start
        it = 0
        while True:
            # Matmul usa Tensor Cores (fp16) e consome GPU de forma previsivel.
            c = a @ b
            a = c  # reusa para evitar alocar infinitamente
            it += 1

            # Evita que o loop seja apenas enfileiramento de kernels.
            torch.cuda.synchronize(device)
            now = time.time()
            if now - start >= seconds:
                break
            if rank == 0 and now >= next_log:
                elapsed = now - start
                print(f"🔥 burn_gpu: {elapsed:.0f}/{seconds}s (iters={it}, dim={mat_dim})", flush=True)
                next_log = now + max(1, log_every)


def main():
    args = parse_args()

    # ─── Configuração Distribuída ───────────────────────────────────────
    local_rank = setup_distributed()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        print("=" * 60)
        print("  TREINO DISTRIBUÍDO - ResNet-18 + CIFAR-10")
        print(f"  GPUs em uso: {world_size}")
        print(f"  Dispositivos: {[f'cuda:{i}' for i in range(world_size)]}")
        print("=" * 60)

    # ─── Hiperparâmetros ────────────────────────────────────────────────
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = 128
    LEARNING_RATE = 0.01
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5e-4

    if rank == 0:
        print(f"\n📋 Hiperparâmetros:")
        print(f"   Épocas: {NUM_EPOCHS}")
        print(f"   Batch Size: {BATCH_SIZE} (por GPU)")
        print(f"   Learning Rate: {LEARNING_RATE}")
        print(f"   Momentum: {MOMENTUM}")
        print(f"   Weight Decay: {WEIGHT_DECAY}\n")

    # ─── Dados ──────────────────────────────────────────────────────────
    train_loader, test_loader, train_sampler = get_dataloaders(
        world_size, rank, BATCH_SIZE
    )

    if rank == 0:
        print(f"📦 Dataset CIFAR-10 carregado:")
        print(f"   Treino: {len(train_loader.dataset)} amostras")
        print(f"   Teste:  {len(test_loader.dataset)} amostras\n")

    # ─── Modelo, Loss e Optimizer ───────────────────────────────────────
    model = create_model(local_rank)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=LEARNING_RATE,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(f"🧠 Modelo: ResNet-18 (adaptado para CIFAR-10)")
        print(f"   Parâmetros: {total_params:,}\n")

    # ─── Loop de Treino ─────────────────────────────────────────────────
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_sampler.set_epoch(epoch)  # Essencial para DDP: embaralha dados por época
        epoch_start = time.time()

        if rank == 0:
            print(f"{'─' * 60}")
            print(f"🚀 Época {epoch + 1}/{NUM_EPOCHS} (LR: {scheduler.get_last_lr()[0]:.6f})")

        # Treino
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, local_rank, epoch
        )

        # Avaliação
        test_loss, test_acc = evaluate(model, test_loader, criterion, local_rank)

        scheduler.step()

        epoch_time = time.time() - epoch_start

        if rank == 0:
            print(f"\n  📊 Resultados da Época {epoch + 1}:")
            print(f"     Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"     Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.2f}%")
            print(f"     ⏱️  Tempo: {epoch_time:.1f}s")

            # Salvar melhor modelo (apenas rank 0)
            if test_acc > best_acc:
                best_acc = test_acc
                checkpoint = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_acc": test_acc,
                    "test_loss": test_loss,
                }
                os.makedirs("outputs", exist_ok=True)
                torch.save(checkpoint, "outputs/melhor_modelo.pth")
                print(f"     💾 Melhor modelo salvo! (Acc: {best_acc:.2f}%)")

    # ─── Resumo Final ───────────────────────────────────────────────────
    total_time = time.time() - start_time

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  ✅ TREINO FINALIZADO!")
        print(f"  ⏱️  Tempo total: {total_time:.1f}s")
        print(f"  🏆 Melhor Accuracy: {best_acc:.2f}%")
        print(f"  💾 Checkpoint salvo em: outputs/melhor_modelo.pth")
        print(f"{'=' * 60}")

    # Se a execucao terminou "rapido demais", adiciona carga extra na GPU para
    # permitir observar logs/monitoramento durante alguns minutos.
    if args.target_seconds and args.target_seconds > 0:
        remaining = int(args.target_seconds - (time.time() - start_time))
        if remaining > 0:
            if rank == 0:
                print(f"\n⏳ Mantendo carga extra na GPU por ~{remaining}s para avaliacao...", flush=True)
            burn_gpu(
                remaining,
                local_rank=local_rank,
                rank=rank,
                mat_dim=args.burn_mat_dim,
                log_every=args.burn_log_every,
            )
            dist.barrier()

    cleanup()


if __name__ == "__main__":
    main()
