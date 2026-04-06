#!/bin/bash
###############################################################################
#                   JOB SLURM - Treino Distribuído Multi-GPU
#                      Cluster TechNE - UFPI
###############################################################################

set -euo pipefail

# ─── IDENTIFICAÇÃO DO JOB ───────────────────────────────────────────────────
# Define o nome do job que aparecerá nos comandos de monitoramento (squeue, sacct)
#SBATCH --job-name=treino-nn

# ─── PARTIÇÃO ───────────────────────────────────────────────────────────────
# Seleciona a fila/partição onde o job será executado.
# A partição "debug" tem tempo máximo de 30 minutos e é destinada a testes rápidos.
# Ela é acessível apenas para a conta "ncad" e tem prioridade alta (PriorityTier=100).
#SBATCH --partition=debug

# ─── CONTA/ACCOUNT ───────────────────────────────────────────────────────────
# A particao debug no TechNE e restrita a conta ncad.
# Sem este parametro o sbatch pode falhar antes mesmo de entrar na fila.
#SBATCH --account=ncad

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
#SBATCH --output=treino_%j.out

# Redireciona a saída de erros (stderr) para um arquivo separado.
# Isso facilita a depuração caso algo dê errado durante a execução.
#SBATCH --error=treino_%j.err

###############################################################################
#                       COMANDOS DE EXECUÇÃO
###############################################################################

# Exibe informações do ambiente para fins de depuração
echo "========================================"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nó: $SLURM_NODELIST"
echo "  GPUs alocadas: ${CUDA_VISIBLE_DEVICES:-indefinido}"
echo "  Diretório de submissão: $SLURM_SUBMIT_DIR"
echo "  Data de início: $(date)"
echo "========================================"

# Garante execucao a partir do diretorio de onde o sbatch foi chamado
cd "$SLURM_SUBMIT_DIR"

# Garante que os diretórios de saída existam
mkdir -p outputs logs

# Move logs principais para a pasta logs/ sem depender dela existir no momento do sbatch
exec > >(tee -a "logs/treino_${SLURM_JOB_ID}.out") 2> >(tee -a "logs/treino_${SLURM_JOB_ID}.err" >&2)

# Mostra informações das GPUs disponíveis
nvidia-smi

# Garante log "ao vivo" (tail -f) sem buffering excessivo do Python.
export PYTHONUNBUFFERED=1

# ─── CORREÇÃO IMPORTANTE PARA CUDA ─────────────────────────────────────────
# Sem isso, o PyTorch desse cluster não consegue encontrar as bibliotecas do CUDA!
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/lib/x86_64-linux-gnu"

# Mantem o job vivo por ~N segundos com logs periodicos (fallback de demonstracao).
keepalive_heartbeat() {
	local seconds="$1"
	local start now elapsed
	start=$(date +%s)
	echo "[keepalive] iniciando por ~${seconds}s"
	while true; do
		now=$(date +%s)
		elapsed=$((now - start))
		if (( elapsed >= seconds )); then
			break
		fi
		echo "[keepalive] ${elapsed}s/${seconds}s - $(date '+%H:%M:%S')"
		if command -v nvidia-smi >/dev/null 2>&1; then
			nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader || true
		fi
		sleep 10
	done
	echo "[keepalive] finalizado"
}

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
#
# --target-seconds=570 -> Mantem o job ocupado por ~9min30s mesmo se o treino
# for muito rapido (ajuda a avaliar monitoramento/logs). O wall-time do SLURM
# continua em 10 minutos.
TARGET_SECONDS=570
START_TS=$(date +%s)

if command -v torchrun >/dev/null 2>&1; then
	set +e
	torchrun --nproc_per_node=2 train.py --epochs 5 --target-seconds "$TARGET_SECONDS"
	TRAIN_RC=$?
	set -e

	if [ "$TRAIN_RC" -ne 0 ]; then
		echo "[aviso] torchrun falhou com codigo $TRAIN_RC. Entrando em fallback keepalive."
		NOW_TS=$(date +%s)
		ELAPSED=$((NOW_TS - START_TS))
		REMAINING=$((TARGET_SECONDS - ELAPSED))
		if (( REMAINING > 0 )); then
			keepalive_heartbeat "$REMAINING"
		fi
	fi
else
	echo "[aviso] torchrun nao encontrado no ambiente. Rodando modo demonstracao de 10 minutos."
	keepalive_heartbeat "$TARGET_SECONDS"
fi

echo ""
echo "========================================"
echo "  Job finalizado: $(date)"
echo "========================================"
