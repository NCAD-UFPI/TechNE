#!/bin/bash
###############################################################################
#             JOB SLURM - Demonstração Rápida (zero instalação)
#                      Cluster TechNE - UFPI
###############################################################################

set -euo pipefail

# ─── IDENTIFICAÇÃO DO JOB ───────────────────────────────────────────────────
# Define o nome do job que aparecerá nos comandos de monitoramento (squeue, sacct)
#SBATCH --job-name=demo-nn

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
# Nao fixa nó para facilitar agendamento rapido; o scheduler escolhe automaticamente.

# ─── RECURSOS DE GPU ───────────────────────────────────────────────────────
# Solicita 1 GPU para aumentar chance de agendamento imediato na demo.
# Se quiser treino DDP depois, altere para gpu:2 e torchrun com 2 processos.
#SBATCH --gres=gpu:1

# ─── MEMÓRIA ────────────────────────────────────────────────────────────────
# Reserva 32 GB de memória RAM para o job.
# A gpunode01 tem ~63 GB total, então 32 GB é suficiente para nosso treino
# sem monopolizar toda a memória do nó.
#SBATCH --mem=32G

# ─── TEMPO MÁXIMO ──────────────────────────────────────────────────────────
# Define o tempo limite de execução (wall-time) no formato HH:MM:SS.
# Se o job não terminar nesse tempo, ele será cancelado automaticamente pelo SLURM.
# 5 minutos e suficiente para uma demonstracao em video.
#SBATCH --time=00:05:00

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

# Mostra informações das GPUs disponíveis (sem quebrar se nao existir nvidia-smi)
if command -v nvidia-smi >/dev/null 2>&1; then
	nvidia-smi || true
else
	echo "[aviso] nvidia-smi nao encontrado; seguindo sem consulta detalhada da GPU."
fi

# Garante log "ao vivo" (tail -f) sem buffering excessivo do Python.
export PYTHONUNBUFFERED=1

# ─── CORREÇÃO IMPORTANTE PARA CUDA ─────────────────────────────────────────
# Sem isso, o PyTorch desse cluster não consegue encontrar as bibliotecas do CUDA!
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/lib/x86_64-linux-gnu"

# Mantem o job vivo por ~N segundos com logs periodicos (modo demonstracao).
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
		sleep 5
	done
	echo "[keepalive] finalizado"
}

# ─── MODO DEMONSTRAÇÃO (PADRÃO) E MODO TREINO (OPCIONAL) ──────────────────
# Este job foi configurado para nao exigir instalacao de pacotes.
# Por padrao, ele roda no modo demo (heartbeat + nvidia-smi), ideal para vídeo.
#
# Para tentar treino real (se ambiente já tiver PyTorch/torchrun), submeta assim:
#   sbatch --export=ALL,DEMO_MODE=0,TARGET_SECONDS=240 job.slurm
#
# DEMO_MODE=1 -> nao depende de python/torch.
DEMO_MODE="${DEMO_MODE:-1}"
TARGET_SECONDS="${TARGET_SECONDS:-240}"
START_TS=$(date +%s)

if [ "$DEMO_MODE" = "1" ]; then
	echo "[info] DEMO_MODE=1 -> execucao sem dependencias Python/PyTorch."
	keepalive_heartbeat "$TARGET_SECONDS"
elif command -v torchrun >/dev/null 2>&1 && [ -f "train.py" ]; then
	set +e
	torchrun --nproc_per_node=1 train.py --epochs 2 --target-seconds "$TARGET_SECONDS"
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
	echo "[aviso] torchrun/train.py indisponivel. Rodando modo demonstracao."
	keepalive_heartbeat "$TARGET_SECONDS"
fi

echo ""
echo "========================================"
echo "  Job finalizado: $(date)"
echo "========================================"
