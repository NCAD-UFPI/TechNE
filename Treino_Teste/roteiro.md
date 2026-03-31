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
ssh aluno_cielio@10.94.80.13
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

## Etapa 3 — Envio dos Arquivos (executar na máquina local)

> Abrir **outro terminal local** e enviar a pasta para o cluster.

```bash
scp -r ./Treino_Teste aluno_cielio@10.94.80.13:/home/aluno_cielio/
```

> Voltar ao terminal SSH e verificar:

```bash
cd Treino_Teste
```

```bash
ls -la
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

## Etapa 5 — Submissão do Job

> Submeter o treino distribuído.

```bash
sbatch job.slurm
```

> Saída esperada: `Submitted batch job <JOB_ID>`

### Resumo do `job.slurm`:

| Diretiva                       | Função                                         |
| ------------------------------ | ---------------------------------------------- |
| `--job-name=treino-nn`         | Nome de identificação do job                   |
| `--partition=debug`            | Fila de execução (max 30min, prioridade alta)  |
| `--nodelist=gpunode01`         | Nó com 2x GPU NVIDIA L4                        |
| `--gres=gpu:2`                 | Solicita 2 GPUs                                |
| `--mem=32G`                    | Reserva 32 GB de RAM                           |
| `--time=00:10:00`              | Limite de 10 minutos                           |
| `--output=logs/treino_%j.out`  | Log stdout (%j = Job ID)                       |
| `--error=logs/treino_%j.err`   | Log stderr (%j = Job ID)                       |
| `torchrun --nproc_per_node=2`  | Lança 2 processos (1 por GPU) com DDP          |

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

> Acompanhar logs em tempo real (Ctrl+C para sair):

```bash
tail -f logs/treino_*.out
```

> Ver erros (se houver):

```bash
tail -f logs/treino_*.err
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
| **Nó de Controle**  | slurm-master | 10.94.80.10  | Gerencia agendamento de jobs         |
| **GPU Node 01**     | gpunode01    | 10.94.80.11  | 16 CPUs, ~63 GB RAM, **2x L4**      |
| **GPU Node 02**     | gpunode02    | 10.94.80.12  | 12 CPUs, ~31 GB RAM, **1x L4**      |
| **Nó de Login**     | (llmnode01)  | 10.94.80.13  | Ponto de acesso SSH                  |

### Partições

| Partição  | Tempo Max | Acesso (Accounts)                          | Prioridade     | Uso                                |
| --------- | --------- | ------------------------------------------ | -------------- | ---------------------------------- |
| **debug** | 30 min    | `ncad`                                     | 🔴 100 (Alta)  | Testes rápidos e depuração         |
| **gpu**   | 2 dias    | `lab_andre`, `lab_gavelino`, `lab_romuere`  | 🟡 50 (Média)  | Treinos regulares dos laboratórios |
| **long**  | 7 dias    | Grupo `professores`                        | 🟢 10 (Baixa)  | Treinos longos de professores      |

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
ssh aluno_cielio@10.94.80.13

# 2. Conferir diretório
pwd
ls -la

# 3. Envio (executar na máquina LOCAL)
scp -r ./Treino_Teste aluno_cielio@10.94.80.13:/home/aluno_cielio/

# 3b. Voltar ao SSH e entrar na pasta
cd Treino_Teste
ls -la

# 4. Verificar nós
sinfo
scontrol show nodes
scontrol show node gpunode01

# 5. Submeter job
sbatch job.slurm

# 6. Monitorar
squeue
squeue -u aluno_cielio
tail -f logs/treino_*.out
tail -f logs/treino_*.err

# 7. Resultados
cat logs/treino_*.out
ls -la outputs/
sacct -j <JOB_ID> --format=JobID,JobName,Partition,State,ExitCode,Elapsed,MaxRSS

# Extra: cancelar job
scancel <JOB_ID>
```
