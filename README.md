# 🚀 Cluster HPC TECHNE — Documentação Técnica Completa

Este repositório documenta a arquitetura, configuração e infraestrutura do **Cluster HPC TECHNE**, utilizado para processamento de alto desempenho (HPC) com gerenciamento via **Slurm**.

---

## 📌 1. Visão Geral e Arquitetura

O cluster TECHNE é composto por um nó controlador e múltiplos nós de execução heterogêneos, com suporte a GPUs NVIDIA L4, armazenamento compartilhado e monitoramento centralizado.

### 🔧 Componentes Principais

| Componente | Detalhes Técnicos |
|-----------|-------------------|
| **Controlador / Master** | `slurm-master` — IP: `10.xx.yy.zz`<br>Serviços: Slurmctld, Slurmdbd, PostgreSQL/MariaDB, Munge |
| **Nó de Execução 1** | `gpunode01` — 16 Cores, 62.9 GB RAM<br>2x GPUs NVIDIA L4 |
| **Nó de Execução 2** | `gpunode02` — 12 Cores, 31.0 GB RAM<br>1x GPU NVIDIA L4 |
| **Sistema Operacional** | Linux Ubuntu/Debian — Kernel 6.8.x |
| **Armazenamento** | NFS em `/data/` + LVM no disco principal |

### 🖥️ Configuração de Hardware (via `lshw`)

- **CPU:** Intel® Xeon® Gold 6526Y (2 sockets lógicos)  
- **RAM Total:** 32 GiB (62.9 GiB disponíveis ao Slurm via `RealMemory`)  
- **GPUs:** 2× NVIDIA L4 (AD104GL) — 23 GB VRAM cada  
- **Controladoras:** Virtio SCSI e Virtio Network  

---

## 📡 2. Configuração do Agendador Slurm

O Slurm é configurado de modo centralizado e replicado em todos os nós,
utilizando Munge para autenticação.

### 📄 2.1. `slurm.conf` (Trecho Principal)

``` ini

# Configuração principal do SLURM
ClusterName=techne

# --- CONFIGURAÇÃO CENTRAL DE CONTROLE (IDÊNTICO EM TODOS OS NÓS) ---
ControlMachine=slurm-master             # Nome do Nó de Controle
ControlAddr=10.94.80.10                 # IP do Nó de Controle (Crucial para comunicação)
SlurmUser=slurm

SlurmctldPort=6817
SlurmdPort=6818
AuthType=auth/munge

# --- DIRETIVAS ESSENCIAIS DO AGENDADOR E PROCESSOS ---
SchedulerType=sched/backfill
SelectType=select/cons_tres
ProctrackType=proctrack/cgroup
SwitchType=switch/none
MpiDefault=none
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd

# Plugins de recursos genéricos
GresTypes=gpu

# --- CONFIGURAÇÃO DOS NÓS DE EXECUÇÃO ---
NodeName=gpunode01 NodeAddr=10.94.80.11 CPUs=16 Sockets=2 CoresPerSocket=8 ThreadsPerCore=1 RealMemory=62900 Gres=gpu:l4:2 State=IDLE
NodeName=gpunode02 NodeAddr=10.94.80.12 CPUs=12 Sockets=2 CoresPerSocket=6 ThreadsPerCore=1 RealMemory=31000 Gres=gpu:l4:1 State=IDLE
#NodeName=llmnode01 NodeAddr=10.94.80.13 CPUs=12 Sockets=2 CoresPerSocket=6 ThreadsPerCore=1 RealMemory=31000 Gres=gpu:l4:1 State=IDLE

# Partição
PartitionName=debug Nodes=gpunode01,gpunode02 Default=NO MaxTime=00:30:00 AllowAccounts=ncad PriorityTier=100 State=UP
PartitionName=gpu Nodes=gpunode01,gpunode02 Default=YES MaxTime=2-00:00:00 AllowAccounts=lab_andre,lab_gavelino,lab_romuere PriorityTier=50 State=UP
PartitionName=long Nodes=gpunode01,gpunode02 Default=NO MaxTime=7-00:00:00 AllowGroups=professores PriorityTier=10 State=UP

# --- Slurm Accounting ---
AccountingStorageType=accounting_storage/slurmdbd
AccountingStorageHost=storagecluster
AccountingStoragePort=6819
AccountingStorageUser=slurm
AccountingStorageEnforce=limits,qos,safe
JobCompType=jobcomp/none
JobAcctGatherType=jobacct_gather/cgroup
```

### 📊 2.2. Contabilidade e Logs

-   **slurmdbd** executando no controlador.
-   Bancos:
    -   **MariaDB** → Slurm Accounting\
    -   **PostgreSQL** → Métricas do monitoramento / Grafana
-   Usuário **manager** com `AdminLevel=Manager` no `sacctmgr`.

------------------------------------------------------------------------

## 📈 3. Pipeline de Monitoramento (Customizado)

O cluster possui um pipeline próprio de coleta e visualização de
métricas via Python + PostgreSQL + Grafana.

### 🐍 3.1. Agente Python (`collect_metrics.py`)

-   Local: `/opt/cluster_monitor/collect_metrics.py`
-   Execução: a cada **1 minuto** (via `cron`)
-   Funções:
    -   coleta completa do estado do cluster (jobs, GPU, CPU, RAM)
    -   normalização dos dados
    -   envio ao PostgreSQL

#### ✔ Correção Importante

Para compatibilidade com CUDA:

``` bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/lib/x86_64-linux-gnu
```

Sem isso, PyTorch não encontra as bibliotecas CUDA.

### 🗄️ 3.2. Estrutura do Banco (PostgreSQL)

  -----------------------------------------------------------------------
  Tabela                   Armazena                       Uso
  ------------------------ ------------------------------ ---------------
  **gpu_log**              Utilização, memória usada,     Gráficos
                           temperatura                    temporais de
                                                          GPU

  **job_log**              Histórico de jobs (JobID,      Auditoria e
                           runtime, state)                estatísticas

  **queue_state**          Contagem de jobs por estado    Status da fila
                                                          no Grafana

  **utilization**          Uso de CPU (%) e RAM (%) por   Painéis de
                           nó                             ocupação
  -----------------------------------------------------------------------

### 📊 3.3. Dashboards no Grafana

-   **Jobs por Estado** → Gráfico de barras
-   **Uso da GPU** → Time series (GPU 0 / GPU 1)
-   **Uso de Disco** → Gauge (percentual)

------------------------------------------------------------------------

## 🧩 4. Tecnologias Utilizadas (Active Stack)

  Categoria             Tecnologias
  --------------------- -------------------------------------------------
  **Gerenciamento**     Slurm 23.11.4, Munge, systemd
  **Aceleração**        NVIDIA Drivers 570.x, CUDA 12.0/12.8, cuDNN 8.9
  **Desenvolvimento**   Python 3.12, venv
  **Bibliotecas**       psycopg2, psutil, subprocess, re, python-dotenv
  **Bancos de Dados**   PostgreSQL, MariaDB
  **Rede**              SSH/SCP, ufw

------------------------------------------------------------------------

## 📚 Licença

Este documento faz parte da infraestrutura interna do Cluster TECHNE e
pode ser reutilizado para estudos, configuração e expansão do ambiente.

------------------------------------------------------------------------

## ✨ Contato

**INFRA NCAD / UFPI**\
Gerenciamento e Desenvolvimento do Cluster HPC TECHNE

