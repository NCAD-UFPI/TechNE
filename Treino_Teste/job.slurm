#!/bin/bash
###############################################################################
#                   JOB SLURM - Treino Distribuído Multi-GPU
#                      Cluster TechNE - UFPI
###############################################################################

# ─── IDENTIFICAÇÃO DO JOB ───────────────────────────────────────────────────
# Define o nome do job que aparecerá nos comandos de monitoramento (squeue, sacct)
#SBATCH --job-name=treino-nn

# ─── PARTIÇÃO ───────────────────────────────────────────────────────────────
# Seleciona a fila/partição onde o job será executado.
# A partição "debug" tem tempo máximo de 30 minutos e é destinada a testes rápidos.
# Ela é acessível apenas para a conta "ncad" e tem prioridade alta (PriorityTier=100).
#SBATCH --partition=debug

# ─── NÓ DE EXECUÇÃO ────────────────────────────────────────────────────────
# Especifica exatamente em qual nó o job deve rodar.
# A gpunode01 possui: 16 CPUs, ~63GB RAM, 2x GPU NVIDIA L4
#SBATCH --nodelist=gpunode01

# ─── RECURSOS DE GPU ───────────────────────────────────────────────────────
# Solicita 2 GPUs genéricas para o job.
# O SLURM aloca automaticamente as GPUs disponíveis no nó selecionado.
# Isso é essencial para o treinamento distribuído com DDP (uma GPU por processo).
#SBATCH --gres=gpu:2

# ─── MEMÓRIA ────────────────────────────────────────────────────────────────
# Reserva 32 GB de memória RAM para o job.
# A gpunode01 tem ~63 GB total, então 32 GB é suficiente para nosso treino
# sem monopolizar toda a memória do nó.
#SBATCH --mem=32G

# ─── TEMPO MÁXIMO ──────────────────────────────────────────────────────────
# Define o tempo limite de execução (wall-time) no formato HH:MM:SS.
# Se o job não terminar nesse tempo, ele será cancelado automaticamente pelo SLURM.
# 10 minutos é suficiente para ~5 épocas do CIFAR-10 com 2 GPUs.
#SBATCH --time=00:10:00

# ─── SAÍDAS E LOGS ─────────────────────────────────────────────────────────
# Redireciona a saída padrão (stdout) para um arquivo de log.
# %j é substituído pelo ID do job, gerando um arquivo único por execução.
#SBATCH --output=logs/treino_%j.out

# Redireciona a saída de erros (stderr) para um arquivo separado.
# Isso facilita a depuração caso algo dê errado durante a execução.
#SBATCH --error=logs/treino_%j.err

###############################################################################
#                       COMANDOS DE EXECUÇÃO
###############################################################################

# Exibe informações do ambiente para fins de depuração
echo "========================================"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nó: $SLURM_NODELIST"
echo "  GPUs alocadas: $CUDA_VISIBLE_DEVICES"
echo "  Data de início: $(date)"
echo "========================================"

# Garante que os diretórios de saída existam
mkdir -p outputs logs

# Mostra informações das GPUs disponíveis
nvidia-smi

# ─── EXECUÇÃO DO TREINO ────────────────────────────────────────────────────
# Usa 'torchrun' para lançar o treinamento distribuído.
#
# --nproc_per_node=2 → Lança 2 processos no nó (1 por GPU)
#   O torchrun configura automaticamente as variáveis de ambiente:
#   - RANK: identificador global do processo (0 ou 1)
#   - LOCAL_RANK: identificador local no nó (0 ou 1, usado para selecionar GPU)
#   - WORLD_SIZE: número total de processos (2)
#   - MASTER_ADDR e MASTER_PORT: endereço para comunicação entre processos
#
# train.py → Script principal de treino que utiliza PyTorch DDP
torchrun --nproc_per_node=2 train.py

echo ""
echo "========================================"
echo "  Job finalizado: $(date)"
echo "========================================"
