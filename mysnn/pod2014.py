# Copyright (C) 2024 Björn A. Lindqvist <bjourne@gmail.com>
from humanize import metric
import numpy as np
from pathlib import Path

# Build parameters
SCALE_N = 1.0
SCALE_K = 1.0

NEST_SEED = 100
NUMPY_SEED = 100

# Number of NEST threads to use
NEST_N_THREADS = 1

# parameters
C_M = 250
TAU_SYN = 0.5
TAU_M = 10
BG_RATE = 8

# Simulation time
N_MS = 1500
DT = 0.1
N_TICS = int(N_MS / DT)
N_TICS_PER_MS = int(1.0 / DT)

# Mean synaptic strengths
W_EXC_MU = 0.15
W_INH_MU = -4 * W_EXC_MU

# w_f
d = TAU_SYN - TAU_M
p = TAU_SYN * TAU_M
q = TAU_M / TAU_SYN
W_F = (C_M * d) / \
    (np.sqrt(SCALE_K) * p *(q**(TAU_M/d) - q**(TAU_SYN/d)))

# Other computed constants from NEST
BETA = (TAU_SYN * TAU_M) / (TAU_M - TAU_SYN)
GAMMA = BETA / C_M
P21 = np.exp(-DT / TAU_SYN) * GAMMA * (np.exp(DT/BETA) - 1)
P22 = np.exp(-DT / TAU_M)
P20 = TAU_M / C_M * (1 - P22)
P11 = np.exp(-DT / TAU_SYN)

# Weight of Poisson spikes
PSN_W = W_F * W_EXC_MU

# Voltage levels
V_THR = -50
V_R = -65

# Refractory period
T_REF = 2.0

# Refractory period in tics
T_REF_TICS = int(T_REF * N_TICS_PER_MS + 0.5)

# Population counts
N = np.array([20683, 5834, 21915, 5479, 4850, 1065, 14395, 2948])
LEN_N = len(N)

# Names
NAMES = [
    '23E', '23I',
    '4E', '4I',
    '5E', '5I',
    '6E', '6I'
]

# Initial potentials
N0_MU = np.array([
    -68.28,
    -63.16,
    -63.33,
    -63.45,
    -63.11, -61.66, -66.72, -61.43
])
N0_SIG = np.array([
    5.36, 4.57, 4.74, 4.94,
    4.94, 4.55, 5.46, 4.48
])

# Mean synaptic delays
D_EXC_MU = 1.5
D_INH_MU = 0.75

D = np.zeros((LEN_N, LEN_N))
D[:, 0:LEN_N:2] = D_EXC_MU
D[:, 1:LEN_N:2] = D_INH_MU

# Connection probabilities
P = np.array([
    # L23
    [0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0., 0.0076, 0.],
    [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0., 0.0042, 0.],
    # L4
    [0.0077, 0.0059, 0.0497, 0.135, 0.0067, 0.0003, 0.0453, 0.],
    [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0., 0.1057, 0.],
    # L5
    [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.],
    [0.0548, 0.0269, 0.0257, 0.0022, 0.06, 0.3158, 0.0086, 0.],
    # L6
    [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
    [0.0364, 0.001, 0.0034, 0.0005, 0.0277, 0.008, 0.0658, 0.1443]
])

# Target spiking rates
R = [0.903, 2.965, 4.414, 5.876, 7.569, 8.633, 1.105, 7.829]

# External synapses
K = np.array([1600, 1500, 2100, 1900, 2000, 1900, 2900, 2100])

# Synapse counts
prod = np.outer(N, N)
SC = np.log(1.0 - P) / np.log((prod - 1.0) / prod)

# Scaled counts
NS = np.round(N * SCALE_N).astype(int)
SCS = np.round(SC * SCALE_N * SCALE_K).astype(int)
KS = np.round(K * SCALE_K).astype(int)
OS = np.cumsum(np.concatenate(([0], NS)))

N_NEURONS = np.sum(NS)
N_SYNAPSES = np.sum(SCS)

# Rates of Poisson input
PSN_RATE = 8.0 * KS

# Synaptic mean weights in mV
A = np.zeros((LEN_N, LEN_N))
A[:, 0:LEN_N:2] = W_EXC_MU
A[:, 1:LEN_N:2] = W_INH_MU
A[0, 2] = 0.3

# Scaled mean weights in pA
WS = A * W_F

# Constant input per population. Set to zero for large network...
DC = W_EXC_MU * K * 8 + np.sum(A*(SC/N[:, np.newaxis])*R, axis=1)
DC *= 0.001 * TAU_SYN * (1.0 - np.sqrt(SCALE_K))
DC *= np.sqrt(SCALE_K) * W_F

# Metrics for prettier printing
MET_N_NEURONS = metric(N_NEURONS)
MET_N_SYNAPSES = metric(N_SYNAPSES)
MET_N_PSN_SAMPLES = metric(N_TICS * N_NEURONS)

# Directories
DIR_NETWORK_BASE = Path('networks')
DIR_MMAP_TMP = Path('mmap-tmp')

# Files for disk storage
FILE_PSN_SPIKES = 'poisson_spikes.npy'

FILE_SYNAPSE_WEIGHT = 'synapse_weight.npy'
FILE_SYNAPSE_DELAY = 'synapse_delay.npy'

FILE_SYNAPSE_DST = 'synapse_dst.npy'
FILE_SYNAPSE_OFFSET = 'synapse_offset.npy'
FILE_SYNAPSE_OFFSET_DELAY = 'synapse_offset_delay.npy'

FILE_NEURONS_CURRENT_INITIAL = 'neurons_current.npy'

FILE_LAYER_PARAMS = 'layer_params.npy'
FILE_NETWORK_PARAMS = 'network_params.npy'
