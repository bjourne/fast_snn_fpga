#include "synced_queue.h"
#include "simd.h"

#define N_SENDERS               1
#define DELIVER_SPIKE 0x00

// c1 = p22
// c2 = p21
// c3 = z
// Too slow but good to save...
uint32_t
update_vm_avx2(uint32_t cnt,
               double * restrict x_ptr,
               double * restrict y_ptr,
               int8_t *c_ptr,
               double c1, double c2, double c3,
               double thr, int8_t rst,
               synced_queue **queues, uint32_t at) {
    uint32_t n_chunks = cnt / 4;
    uint32_t n_rem = cnt % 4;
    uint32_t n_spikes = 0;

    double4 r_c1 = d4_set_1x(c1);
    double4 r_c2 = d4_set_1x(c2);
    double4 r_c3 = d4_set_1x(c3);
    double4 r_thr = d4_set_1x(thr);
    long4 r_rst = l4_set_1x(rst);
    for (uint32_t i = 0; i < n_chunks; i++) {
        long4 c = l4_set_4x(c_ptr[0],
                            c_ptr[1],
                            c_ptr[2],
                            c_ptr[3]);
        c = l4_sub(c, l4_1());

        double4 x = d4_load(x_ptr);
        double4 y = d4_load(y_ptr);
        x = d4_add(d4_add(d4_mul(x, r_c1), r_c3),
                   d4_mul(y, r_c2));

        double4 ev = d4_cmp_gte(x, r_thr);
        x = d4_tern((double4)c, d4_andnot(ev, x), d4_0());
        c = l4_tern(c, l4_and((long4)ev, r_rst), c);
        d4_store(x, x_ptr);

        uint32_t msk = d4_movemask(ev);
        if (msk) {
            for (uint32_t j = 0; j < 4; j++) {
                if ((msk >> j) & 1) {
                    uint32_t data = (DELIVER_SPIKE << 24) | (at + j);
                    for (uint32_t k = 0; k < N_SENDERS; k++) {
                        synced_queue_add(queues[k], data);
                    }
                    n_spikes++;
                }
            }
        }



        c_ptr[0] = c[0];
        c_ptr[1] = c[1];
        c_ptr[2] = c[2];
        c_ptr[3] = c[3];

        c_ptr += 4;
        x_ptr += 4;
        y_ptr += 4;
        at += 4;
    }
    for (uint32_t i = 0; i < n_rem; i++) {
        int8_t c = *c_ptr;
        if (!c) {
            double x = *x_ptr * c1 + c3 + *y_ptr * c2;
            if (x >= thr) {
                uint32_t data = (DELIVER_SPIKE << 24) | at;
                for (uint32_t k = 0; k < N_SENDERS; k++) {
                    synced_queue_add(queues[k], data);
                }
                n_spikes++;
                *x_ptr = 0.0;
                *c_ptr = rst;
            } else {
                *x_ptr = x;
            }
        } else {
            *c_ptr = c - 1;
        }
        c_ptr++;
        x_ptr++;
        y_ptr++;
        at++;
    }
    return n_spikes;
}



void
psc_update_avx2(uint32_t cnt, double c1, double psn_w,
                double * restrict psc,
                uint8_t * restrict psn,
                float * restrict ibuf) {
    uint32_t n_chunks = cnt / 4;
    uint32_t n_rem = cnt % 4;

    double4 r_c1 = d4_set_1x(c1);
    double4 r_psn_w = d4_set_1x(psn_w);

    for (uint32_t i = 0; i < n_chunks; i++) {
        double4 r_psc = _mm256_load_pd(psc);
        r_psc = d4_mul(r_psc, r_c1);
        double4 r_psn = d4_set_4x((double)psn[0],
                                  (double)psn[1],
                                  (double)psn[2],
                                  (double)psn[3]);
        r_psn = d4_mul(r_psn, r_psn_w);

        double4 r_ibuf = d4_set_1x_f4(f4_load(ibuf));
        r_psc = d4_add(d4_add(r_psc, r_psn), r_ibuf);
        d4_store(r_psc, psc);
        psc += 4;
        psn += 4;
        ibuf += 4;
    }
    for (uint32_t i = 0; i < n_rem; i++) {
        *psc = *psc * c1 + *psn * psn_w + *ibuf;
        psc++;
        psn++;
        ibuf++;
    }
}
