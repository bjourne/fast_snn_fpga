# Copyright (C) 2023-2024 Björn A. Lindqvist <bjourne@gmail.com>
from humanize import metric
from mysnn import MyProgress
from mysnn.pod2014 import *
from pathlib import Path
from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn

import numpy as np

def save_layer_and_network_params(dir):
    # Save layers and network params
    params = np.array([NS, DC]).astype(np.float64)
    np.save(dir / FILE_LAYER_PARAMS, params)

    # Save network params
    params = np.array([
        [b'v_r', V_R],
        [b'v_thr', V_THR],
        [b'p11', P11],
        [b'p20', P20],
        [b'p21', P21],
        [b'p22', P22],
        [b'psn_w', PSN_W],
        [b't_ref_tics', T_REF_TICS]
    ])
    np.save(dir / FILE_NETWORK_PARAMS, params)

def limited_normal(mu, sig, size, lo = -np.inf, hi = np.inf):
    A = np.random.normal(mu, sig, size)
    bad = np.where((A < lo) | (A > hi))
    n_bad = len(bad[0])
    if n_bad:
        A[bad] = limited_normal(mu, sig, n_bad, lo, hi)
    return A

def delete_duplicate_synapses(bprg, src, dst, delay, weight):
    n_dup_chunk = 10 * 1000
    bprg.next_phase('Finding duplicates among %s synapses', MET_N_SYNAPSES)
    bprg.set_current_task(N_SYNAPSES / n_dup_chunk, 'Finding duplicates')

    deletes = []
    src0 = -1
    for i in range(N_SYNAPSES):
        src1 = src[i]
        if src1 != src0:
            src0 = src1
            seen = {}
        key = dst[i], delay[i]
        j = seen.get(key)
        if j is not None:
            weight[j] += weight[i]
            deletes.append(i)
        else:
            seen[key] = i
        if i % n_dup_chunk == 0:
            bprg.next_steps(1, '%s / %s', metric(i), MET_N_SYNAPSES)
    bprg.next_step('Deleting %s synapses', metric(len(deletes)))

    dst = np.delete(dst, deletes)
    weight = np.delete(weight, deletes)
    src = np.delete(src, deletes)
    delay = np.delete(delay, deletes)

    return src, dst, delay, weight

def write_synapses_and_index(bprg, network_dir,
                             src, dst, delay, weight):
    new_n = metric(len(src))

    bprg.next_phase('Writing %s synapses', new_n)
    np.save(network_dir / FILE_SYNAPSE_DST, dst)
    np.save(network_dir / FILE_SYNAPSE_WEIGHT, weight)
    np.save(network_dir / FILE_SYNAPSE_DELAY, delay)

    # Offsets
    ofs = np.where(src[:-1] != src[1:])[0] + 1
    ofs = np.concatenate(([0], ofs, [len(src)]))
    ofs = ofs.astype(np.uint32)
    np.save(network_dir / FILE_SYNAPSE_OFFSET, ofs)

    # Also write a second-level index.
    delay_ofs = []
    for o0, o1 in zip(ofs, ofs[1:]):
        cnts = np.histogram(delay[o0:o1],
                            range = (0, 63), bins = 63)[0]
        this = np.concatenate(([0], np.cumsum(cnts))) + o0
        delay_ofs.append(this)

    # Last element is missing
    delay_ofs = np.concatenate(delay_ofs).astype(np.int32)
    np.save(network_dir / FILE_SYNAPSE_OFFSET_DELAY, delay_ofs)

def generate_synapses(network_dir, mmap_dir, bprg):
    bprg.set_current_task(2 * LEN_N + 7 * LEN_N**2, 'Generating synapses')
    src = np.memmap(mmap_dir / 'src.memmap', np.uint32, 'w+',
                    0, (N_SYNAPSES,))
    dst = np.memmap(mmap_dir / 'dst.memmap', np.uint32, 'w+',
                    0, (N_SYNAPSES,))
    weight = np.memmap(mmap_dir / 'weight.memmap', np.float32, 'w+',
                       0, (N_SYNAPSES,))
    delay = np.memmap(mmap_dir / 'delay.memmap', np.uint8, 'w+',
                      0, (N_SYNAPSES,))

    at = 0
    for j in range(LEN_N):
        row_cnt = np.sum(SCS[:,j])
        met_row_cnt = metric(row_cnt)
        src_row = np.empty(row_cnt, np.uint32)
        dst_row = np.empty(row_cnt, np.uint32)
        weight_row = np.empty(row_cnt, np.float32)
        delay_row = np.empty(row_cnt, np.uint8)
        row_at = 0
        for i in range(LEN_N):

            cell_cnt = SCS[i][j]
            met_cell_cnt = metric(cell_cnt)

            bprg.next_step('Generating %s src/dst pairs', met_cell_cnt)
            src_n0, src_n1 = OS[j], OS[j + 1]
            dst_n0, dst_n1 = OS[i], OS[i + 1]
            src_cell = np.random.randint(src_n0, src_n1, cell_cnt)
            dst_cell = np.random.randint(dst_n0, dst_n1, cell_cnt)

            bprg.next_step('Generating %s weights', met_cell_cnt)
            w_mu = WS[i][j]
            w_sig = np.abs(w_mu) * 0.1
            lo = 0 if w_mu > 0 else -np.inf
            hi = 0 if w_mu < 0 else np.inf
            weight_cell = limited_normal(w_mu, w_sig, cell_cnt, lo, hi)

            bprg.next_step('Generating %s delays', met_cell_cnt)
            d_mu = D[i][j]
            d_sig = d_mu * 0.5
            lo = DT - 0.5 * DT
            delay_cell = limited_normal(d_mu, d_sig, cell_cnt, lo, np.inf)
            # No banker's rounding
            delay_cell = delay_cell * N_TICS_PER_MS + 0.5
            if cell_cnt > 0:
                assert np.max(delay_cell) < 256

            met_at = metric(at + row_at)

            bprg.next_step('Writing %s at %s', 'src', met_at)
            src_row[row_at:row_at+cell_cnt] = src_cell

            bprg.next_step('Writing %s at %s', 'dst', met_at)
            dst_row[row_at:row_at+cell_cnt] = dst_cell

            bprg.next_step('Writing %s at %s', 'weight', met_at)
            weight_row[row_at:row_at+cell_cnt] = weight_cell

            bprg.next_step('Writing %s at %s', 'delay', met_at)
            delay_row[row_at:row_at+cell_cnt] = delay_cell
            row_at += cell_cnt
        assert row_at == row_cnt

        bprg.next_step('Sorting %s synapes', met_row_cnt)
        perm = np.lexsort((dst_row, delay_row, src_row))

        bprg.next_step('Writing %s synapses', met_row_cnt)
        src[at:at + row_cnt] = src_row[perm]
        dst[at:at + row_cnt] = dst_row[perm]
        weight[at:at + row_cnt] = weight_row[perm]
        delay[at:at + row_cnt] = delay_row[perm]
        at += row_cnt
    assert at == N_SYNAPSES

    src, dst, delay, weight = delete_duplicate_synapses(
        bprg, src, dst, delay, weight)
    write_synapses_and_index(bprg, network_dir, src, dst, delay, weight)

########################################################################

def build_numpy(bprg):
    bprg.next_phase('Creating directories')
    network_dir = DIR_NETWORK_BASE / 'numpy'
    mmap_dir = DIR_MMAP_TMP / 'pod2014_numpy'
    network_dir.mkdir(exist_ok = True, parents = True)
    mmap_dir.mkdir(exist_ok = True, parents = True)

    bprg.next_phase('Generating initial potentials')
    lst = [np.random.normal(N0_MU[i], N0_SIG[i], size = NS[i])
           for i in range(LEN_N)]
    V_m = np.concatenate(lst, dtype = np.float64)
    np.save(network_dir / FILE_NEURONS_CURRENT_INITIAL, V_m)

    # Save layers and network params
    save_layer_and_network_params(network_dir)

    # Handle synapes
    bprg.next_phase('Generating %s synapses', MET_N_SYNAPSES)
    generate_synapses(network_dir, mmap_dir, bprg)

    bprg.next_phase('Drawing %s Poisson samples', MET_N_PSN_SAMPLES)
    spikes = []
    for i in range(LEN_N):
        n_samples = N_TICS * NS[i]
        bprg.set_current_task(NS[i], 'Spikes for pop. %s', i)
        rate = PSN_RATE[i] * 0.001 * DT
        met_total = metric(NS[i] * N_TICS)
        for lo in range(0, NS[i], 1_000):
            hi = min(lo + 1_000, NS[i])
            cnt = hi - lo
            spikes0 = np.random.poisson(rate, (N_TICS, cnt))
            spikes0 = spikes0.astype(np.uint8)
            spikes.append(spikes0)
            met_at = metric(hi * N_TICS)
            bprg.next_steps(cnt, '%s / %s', met_at, met_total)
        bprg.next_phase('')
    spikes = np.hstack(spikes)
    np.save(network_dir / FILE_PSN_SPIKES, spikes)

    bprg.next_phase('%s Poisson spikes generated', metric(np.sum(spikes)))

def build(args):
    tp = args[0]
    np.random.seed(NUMPY_SEED)
    prg = Progress(
        TextColumn('{task.fields[name]}'),
        BarColumn(), TimeElapsedColumn(),
        TextColumn("{task.description}"),
        refresh_per_second = 2
    )
    with prg:
        if tp == 'numpy':
            bprg = MyProgress(prg, 14)
            build_numpy(bprg)
        else:
            assert False
