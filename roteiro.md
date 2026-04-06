# 🖥️ Roteiro de Apresentação — Cluster TechNE (UFPI)

**Treino Distribuído de Rede Neural com SLURM + PyTorch DDP**

| Arquivo            | Descrição                                            |
| ------------------ | ---------------------------------------------------- |
| `train.py`         | Treino distribuído ResNet-18 + CIFAR-10 (PyTorch DDP)|
| `job.slurm`        | Job SLURM: 2 GPUs, 32GB, 10min, partition debug      |
| `requirements.txt` | Dependências Python                                  |
| `outputs/`         | Checkpoints do modelo                                |
| `logs/`            | Logs de saída/erro do SLURM                          |

---

## Etapa 1 — Login no Cluster

> Acessamos o nó de login via SSH.

```bash
ssh aluno_cielio@10.94.80.10
```

---

## Etapa 2 — Conferência do Diretório

> Confirmar que estamos no home do usuário.

```bash
pwd
```

```bash
ls -la
```

---

## Etapa 3 — Obtendo os Arquivos no Cluster

> Clonar o repositório direto no cluster (já logado via SSH):

```bash
git clone https://github.com/NCAD-UFPI/TechNE.git
```

```bash
cd TechNE/Treino_Teste
```

```bash
ls -la
```

> **Alternativa** (se preferir enviar da máquina local via `rsync`, abrir outro terminal):

```bash
# (Opcional) apagar a pasta antiga no cluster antes de reenviar
ssh aluno_cielio@10.94.80.10 "rm -rf /home/aluno_cielio/Treino_Teste"

# Envio da pasta (foi o comando usado)
rsync -avz /Users/francieliocastro/Developer/Cluster/Treino_Teste/ aluno_cielio@10.94.80.10:/home/aluno_cielio/Treino_Teste
```

---

## Etapa 4 — Verificação dos Nós

> Ver estado geral das partições e nós.

```bash
sinfo
```

> Saída mostra: PARTITION (filas), STATE (idle/alloc/drain), TIMELIMIT, NODELIST.

```bash
scontrol show nodes
```

> Detalhes de cada nó: CPUs, RAM, GPUs, estado, carga.

> Para ver apenas a gpunode01:

```bash
scontrol show node gpunode01
```

---

## Etapa 5 — Submissão do Job (Sem Instalar Nada)

> Submeter em **modo demo rapido** (padrao atual), sem instalar pacote nenhum.

```bash
# Garantir pastas de saida antes da submissao
mkdir -p logs outputs
```

```bash
# Validacao rapida do script sem submeter (mostra erro de account/particao)
sbatch --test-only job.slurm
```

```bash
sbatch job.slurm
```

> Saída esperada: `Submitted batch job <JOB_ID>`

> O `job.slurm` atual ja vem com `DEMO_MODE=1` por padrao, entao ele roda mesmo sem Python/PyTorch instalado.

> Dica: para já capturar o `JOB_ID` em uma variável e facilitar o tail:

```bash
JOB_ID=$(sbatch job.slurm | awk '{print $4}')
echo "JOB_ID=$JOB_ID"
```

> Opcional (se voce quiser tentar treino real e o ambiente ja tiver torchrun):

```bash
JOB_ID=$(sbatch --export=ALL,DEMO_MODE=0,TARGET_SECONDS=240 job.slurm | awk '{print $4}')
echo "JOB_ID=$JOB_ID"
```

### Resumo do `job.slurm` (versao para demo agil):

| Diretiva                       | Função                                         |
| ------------------------------ | ---------------------------------------------- |
| `--job-name=demo-nn`           | Nome de identificação do job                   |
| `--partition=debug`            | Fila de execução (max 30min, prioridade alta)  |
| `--account=ncad`               | Conta exigida para usar a partição `debug`     |
| `--nodelist` omitido           | Scheduler escolhe o nó automaticamente         |
| `--gres=gpu:1`                 | Solicita 1 GPU (mais fácil de agendar)         |
| `--mem=32G`                    | Reserva 32 GB de RAM                           |
| `--time=00:05:00`              | Limite de 5 minutos                            |
| `exec ... tee logs/treino_...` | Log principal em `logs/` para acompanhar com `tail -F` |
| `DEMO_MODE=1` (padrão)         | Roda heartbeat + status de GPU, sem dependências |
| `DEMO_MODE=0`                  | Tenta `torchrun` se o ambiente já estiver pronto |

---

## Etapa 6 — Monitoramento

> Ver fila de jobs:

```bash
squeue
```

> Filtrar só nossos jobs:

```bash
squeue -u aluno_cielio
```

> Se o job sumir rapido do `squeue`, verificar estado final imediatamente:

```bash
sacct -j ${JOB_ID} --format=JobID,JobName,Partition,State,ExitCode,Elapsed,NodeList -P
```

> Acompanhar logs em tempo real (Ctrl+C para sair):

```bash
tail -F logs/treino_${JOB_ID}.out
```

> Ver erros (se houver):

```bash
tail -F logs/treino_${JOB_ID}.err
```

> O `job.slurm` atual foi feito para demo: por padrao ele fica em keepalive com heartbeat e `nvidia-smi`, sem exigir instalacao. Se tentar treino e falhar, volta automaticamente para keepalive.

> Se o `sbatch` falhar antes de entrar na fila, diagnosticar motivo:

```bash
sbatch --test-only job.slurm
```

```bash
sacctmgr show assoc user=aluno_cielio format=Account,User,Cluster
```

---

## Etapa 7 — Análise dos Resultados

> Ver log completo:

```bash
cat logs/treino_*.out
```

> Verificar checkpoint salvo:

```bash
ls -la outputs/
```

> Detalhes do job finalizado (substituir JOB_ID):

```bash
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS
```

> Cancelar um job se necessário:

```bash
scancel <JOB_ID>
```

---

## Etapa 8 — Entendendo as Partições do Cluster TechNE

### Infraestrutura

| Componente          | Hostname     | IP           | Hardware                             |
| ------------------- | ------------ | ------------ | ------------------------------------ |
| **Nó de Login / Master** | slurm-master | 10.94.80.10  | Gerencia agendamento de jobs e acesso SSH |
| **GPU Node 01**     | gpunode01    | 10.94.80.11  | 16 CPUs, ~63 GB RAM, **2x L4**      |
| **GPU Node 02**     | gpunode02    | 10.94.80.12  | 12 CPUs, ~31 GB RAM, **1x L4**      |

### Partições

| Partição  | Tempo Max | Acesso (Accounts)                          | Prioridade     | Uso                                |
| --------- | --------- | ------------------------------------------ | -------------- | ---------------------------------- |
| **debug** | 30 min       | `ncad`                                     | 🔴 100 (Alta)  | Testes rápidos e depuração         |
| **gpu**   | 2 dias       | `lab_andre`, `lab_gavelino`, `lab_romuere`  | 🟡 50 (Média)  | Treinos regulares dos laboratórios |
| **long**  | 7 dias       | Grupo `professores`                        | 🟢 10 (Baixa)  | Treinos longos de professores      |

### Diagrama de Agendamento

```
                    ┌─────────────────────┐
                    │    slurm-master      │
                    │   (10.94.80.10)      │
                    │   Gerencia filas     │
                    │   e agendamento      │
                    └────────┬────────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │   debug    │  │    gpu     │  │    long    │
     │  ≤ 30min   │  │  ≤ 2 dias  │  │  ≤ 7 dias  │
     │ Prior: 100 │  │ Prior: 50  │  │ Prior: 10  │
     └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
           │               │               │
           ▼               ▼               ▼
     ┌──────────────────────────────────────────┐
     │  gpunode01 (2x L4)  │  gpunode02 (1x L4) │
     └──────────────────────────────────────────┘
```

### Detalhes do `slurm.conf`

- **`sched/backfill`** — Preenche buracos na fila com jobs menores, maximizando utilização
- **`select/cons_tres`** — Seleção por recursos consumíveis (CPU, RAM, GPU), permitindo múltiplos jobs por nó
- **`GresTypes=gpu`** — GPUs como recurso genérico solicitável via `--gres`
- **`AccountingStorageEnforce=limits,qos,safe`** — Aplica limites por conta, nenhum grupo ultrapassa a cota
- **`PriorityTier`** — Maior valor = maior prioridade; debug (100) passa na frente de gpu (50)
- **`AllowAccounts` / `AllowGroups`** — Controla quais contas/grupos acessam cada partição

---

## 📋 Cola Rápida — Todos os Comandos

```bash
# 1. Login
ssh aluno_cielio@10.94.80.10

# 2. Conferir diretório
pwd
ls -la

# 3. Clonar repositório (dentro do SSH)
git clone https://github.com/NCAD-UFPI/TechNE.git
cd TechNE/Treino_Teste
ls -la

# 3b. (Alternativa) reenviar da maquina local via rsync
# ssh aluno_cielio@10.94.80.10 "rm -rf /home/aluno_cielio/Treino_Teste"
# rsync -avz /Users/francieliocastro/Developer/Cluster/Treino_Teste/ aluno_cielio@10.94.80.10:/home/aluno_cielio/Treino_Teste

# 4. Verificar nós
sinfo
scontrol show nodes
scontrol show node gpunode01

# 5. Submeter job
mkdir -p logs outputs
sbatch --test-only job.slurm
JOB_ID=$(sbatch job.slurm | awk '{print $4}')
echo "JOB_ID=$JOB_ID"

# 5b. Opcional: tentar treino real (somente se o ambiente ja tiver torchrun)
# JOB_ID=$(sbatch --export=ALL,DEMO_MODE=0,TARGET_SECONDS=240 job.slurm | awk '{print $4}')
# echo "JOB_ID=$JOB_ID"

# 6. Monitorar
squeue
squeue -u aluno_cielio
tail -F logs/treino_${JOB_ID}.out
tail -F logs/treino_${JOB_ID}.err

# 7. Resultados
cat logs/treino_*.out
ls -la outputs/
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS

# Extra: cancelar job
scancel <JOB_ID>
```
