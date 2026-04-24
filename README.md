# F1RL

Simulador multiagente de Fórmula 1 com aprendizado por reforço, organizado em estrutura `src/` profissional, com ambiente contínuo, grid temático de equipes da F1 e stack de treino baseado em self-play.

O projeto implementa como algoritmo principal um **MAPPO-style centralized-critic PPO com parameter sharing**, não GRPO. A escolha é deliberada: o domínio possui ação contínua, parcial observabilidade local por carro, dinâmica multiagente simultânea e necessidade de estabilidade on-policy na presença de forte não-estacionariedade induzida por self-play.

## Algoritmo Implementado

### Algoritmo padrão

**Self-Play MAPPO (parameter-sharing PPO + centralized critic)**

Arquitetura efetivamente usada no código:

- **ator compartilhado** para todos os carros, reduzindo variância e aumentando eficiência amostral;
- **crítico centralizado** condicionado em `local_obs_i + global_state`, o que melhora o baseline em ambiente competitivo;
- **política contínua squashed Gaussian** (`tanh`) para controlar throttle, brake, compromisso em curva, deploy de ERS, fuel mix, racecraft e pit call;
- **GAE-Lambda** para estimativa de vantagem;
- **clipped surrogate objective** no estilo PPO;
- **value clipping** no crítico;
- **entropy regularization** para evitar colapso prematuro da política.

Formulação do update:

```math
L^{clip}(\theta) = \mathbb{E}\left[\min(r_t(\theta)\hat{A}_t,\ \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t)\right]
```

com:

```math
r_t(\theta) = \frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{old}}(a_t|o_t)}
```

e vantagem via GAE:

```math
\hat{A}_t = \sum_{l=0}^{T-t-1}(\gamma\lambda)^l \delta_{t+l}
```

### Por que não GRPO?

GRPO é mais natural em cenários com avaliação por grupos, ranking comparativo explícito ou sinais de preferência agregados. Aqui, o problema é mais bem modelado como **controle contínuo multiagente com crédito temporal fino**, então PPO com crítico centralizado é uma escolha mais apropriada e tecnicamente mais defensável.

## Características do Ambiente

O ambiente implementado em [src/f1rl/envs/race.py](/home/lucas/Formula-1-RL/src/f1rl/envs/race.py) não é um `toy env` de progresso linear puro. Ele inclui:

- pista segmentada por setores com curvatura, grip, DRS e regiões de pit;
- dinâmica longitudinal contínua;
- degradação de pneus dependente de composto, velocidade, curvatura, temperatura, clima e estilo do carro;
- massa variável por consumo de combustível;
- gerenciamento de ERS com deploy e regeneração sob frenagem;
- janelas de DRS baseadas em gap;
- lógica de ultrapassagem com racecraft ofensivo/defensivo;
- incidentes, dano e safety car simplificado;
- pit stop com troca automática de composto orientada pelo restante da corrida;
- grid com equipes reais da F1 no config default:
  `McLaren`, `Ferrari`, `Red Bull Racing`, `Mercedes`, `Aston Martin`, `Alpine`, `Williams`, `Haas`, `Sauber`, `Racing Bulls`.

### Observação local

Cada agente recebe uma observação local fixa com:

- estado do próprio carro;
- progresso na pista;
- recursos internos: combustível, ERS, temperatura, dano, composto;
- contexto da pista: curvatura, grip, DRS, wetness, safety car;
- atributos do carro/equipe;
- features dos adversários mais próximos.

No config padrão:

- `obs_dim = 42`
- `action_dim = 7`
- `state_dim = 106`

### Espaço de ação

Cada agente escolhe vetores contínuos em `[-1, 1]`:

1. throttle
2. brake
3. corner commitment
4. ERS deploy
5. fuel mix
6. racecraft bias
7. pit request

### Recompensa

A recompensa combina:

- progresso longitudinal;
- ganho/perda de posição;
- bônus por completar a corrida;
- regularização leve para preservação de pneus/ERS;
- penalidade por dano incremental;
- penalidade por incidente;
- penalidade por excursion/off-track;
- custo de pit stop.

Isso produz um sinal híbrido entre **pace management**, **race outcome** e **resource strategy**.

## Estrutura do Projeto

```text
Formula-1-RL/
├── configs/
│   └── experiments/
│       └── interlagos_mappo.yaml
├── src/
│   └── f1rl/
│       ├── baselines/
│       │   └── heuristic.py
│       ├── cli/
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── simulate.py
│       ├── config/
│       │   ├── loader.py
│       │   └── schema.py
│       ├── envs/
│       │   ├── car.py
│       │   ├── dynamics.py
│       │   ├── race.py
│       │   └── track.py
│       ├── rl/
│       │   ├── buffer.py
│       │   ├── mappo.py
│       │   └── networks.py
│       └── utils/
├── tests/
├── pyproject.toml
└── README.md
```

### Responsabilidade dos módulos

- [src/f1rl/envs/race.py](/home/lucas/Formula-1-RL/src/f1rl/envs/race.py): ambiente principal e regras de corrida.
- [src/f1rl/envs/dynamics.py](/home/lucas/Formula-1-RL/src/f1rl/envs/dynamics.py): dinâmica longitudinal, grip e desgaste.
- [src/f1rl/rl/networks.py](/home/lucas/Formula-1-RL/src/f1rl/rl/networks.py): ator squashed Gaussian e crítico centralizado.
- [src/f1rl/rl/buffer.py](/home/lucas/Formula-1-RL/src/f1rl/rl/buffer.py): armazenamento de rollouts multiagente.
- [src/f1rl/rl/mappo.py](/home/lucas/Formula-1-RL/src/f1rl/rl/mappo.py): coleta, GAE, PPO update, checkpoint e avaliação.
- [src/f1rl/baselines/heuristic.py](/home/lucas/Formula-1-RL/src/f1rl/baselines/heuristic.py): baseline determinístico para benchmark/debug.
- [configs/experiments/interlagos_mappo.yaml](/home/lucas/Formula-1-RL/configs/experiments/interlagos_mappo.yaml): experimento default.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Dependências centrais:

- `torch`
- `numpy`
- `PyYAML`
- `pytest` para desenvolvimento

## Como Executar

### 1. Simular uma corrida com baseline heurístico

```bash
f1rl-simulate --config configs/experiments/interlagos_mappo.yaml --policy heuristic
```

### 2. Treinar a política MAPPO

```bash
f1rl-train --config configs/experiments/interlagos_mappo.yaml
```

### 3. Avaliar um checkpoint

```bash
f1rl-evaluate \
  --config configs/experiments/interlagos_mappo.yaml \
  --checkpoint artifacts/interlagos-mappo-f1-teams/checkpoint_iter_010.pt \
  --episodes 5
```

## Design de Treino

### Coleta

O trainer coleta rollouts multiagente síncronos:

- todas as ações são amostradas simultaneamente;
- `active_mask` evita treinar em agentes já terminados/crashados;
- o crítico consome o estado global repetido por agente;
- `parameter sharing` reduz o custo de otimização em grids maiores.

### Política

O ator usa uma distribuição Gaussiana com squash via `tanh`, permitindo:

- ação naturalmente contínua;
- amostragem reparametrizável;
- cálculo consistente de `log_prob` com correção do Jacobiano.

### Crítico

O crítico recebe `obs_local + estado_global`, o que dá contexto suficiente para:

- pace relativo;
- tráfego;
- efeito de safety car;
- ordenação do grid;
- disponibilidade sistêmica de recursos.

## Configuração Experimental

O experimento default está em [configs/experiments/interlagos_mappo.yaml](/home/lucas/Formula-1-RL/configs/experiments/interlagos_mappo.yaml).

Pontos importantes:

- pista: `Interlagos`
- grid: 10 equipes
- corrida: 15 voltas
- rollout: 192 steps
- epochs PPO: 6
- `hidden_dim`: 256
- `gamma = 0.995`
- `gae_lambda = 0.95`
- `clip_ratio = 0.20`

Para criar novos experimentos, o caminho recomendado é duplicar o YAML e alterar:

- número de agentes;
- segmentação da pista;
- nível de aleatoriedade climática;
- horizonte de rollout;
- hiperparâmetros do PPO.

## Artefatos

O trainer escreve em `artifacts/<run_name>/`:

- `metrics.jsonl`
- `training_summary.json`
- `checkpoint_iter_XXX.pt`

Isso facilita integração com pipelines externos de análise, sweep e tracking.

## Testes

```bash
PYTHONPATH=src pytest -q
```

Cobertura atual de regressão:

- parsing de config;
- invariantes básicos do ambiente;
- simulação curta completa;
- smoke test de treino MAPPO.

## Considerações de Pesquisa

Este projeto é uma base séria para iteração, mas não tenta ser um simulador aerodinâmico de alta fidelidade. Ele está posicionado como **research environment for multi-agent continuous-control RL**. Evoluções naturais:

- action masking explícito para pit windows;
- grid completo de 20 carros;
- curriculum por pista/clima;
- population-based training;
- self-play com pools históricos de políticas;
- modelagem mais fiel de undercut/overcut e delta real de pit lane;
- integração com trackers como Weights & Biases ou MLflow.

