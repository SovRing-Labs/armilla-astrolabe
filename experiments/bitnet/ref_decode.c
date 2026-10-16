/* bitnet.cpp reference greedy decode over the same token ids (OUTREACH-P1-DECODE). Links the existing bitnet.cpp libllama.
 * usage: ref_decode MODEL.gguf NGEN THREADS tok... ; env REF_LOGITS=file dumps the logits of each generated step. */
#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
int main(int argc, char **argv) {
    int ngen = atoi(argv[2]), nt = atoi(argv[3]), np = argc - 4; llama_token toks[512];
    for (int i = 0; i < np; i++) toks[i] = atoi(argv[4 + i]);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params(); mp.n_gpu_layers = 0;
    struct llama_model *m = llama_model_load_from_file(argv[1], mp); if (!m) return 2;
    struct llama_context_params cp = llama_context_default_params(); cp.n_ctx = 512; cp.n_batch = 512; cp.n_threads = nt; cp.n_threads_batch = nt;
    struct llama_context *c = llama_init_from_model(m, cp); if (!c) return 3;
    int nv = llama_vocab_n_tokens(llama_model_get_vocab(m));
    FILE *lf = getenv("REF_LOGITS") ? fopen(getenv("REF_LOGITS"), "wb") : NULL;
    if (getenv("REF_PPL")) {                /* same protocol as armilla_bitnet ABN_PPL: n_ctx 512, BOS first per chunk, score second half */
        FILE *pf = fopen(getenv("REF_PPL"), "r"); static int ids[1 << 16]; int nid = 0; if (!pf) return 6;
        while (nid < (1 << 16) && fscanf(pf, "%d", &ids[nid]) == 1) nid++; fclose(pf);
        double nll = 0; long cnt = 0; int nchunk = nid / 512; struct llama_batch bt = llama_batch_init(512, 0, 1);
        for (int ch = 0; ch < nchunk; ch++) {
            int *ct = ids + ch * 512; int save = ct[0]; ct[0] = llama_vocab_bos(llama_model_get_vocab(m));
            llama_memory_clear(llama_get_memory(c), true); bt.n_tokens = 512;
            for (int j = 0; j < 512; j++) { bt.token[j] = ct[j]; bt.pos[j] = j; bt.n_seq_id[j] = 1; bt.seq_id[j][0] = 0; bt.logits[j] = j >= 256 && j < 511; }
            if (llama_decode(c, bt)) return 7;
            double cn = 0; long cc = 0;
            for (int j = 256; j < 511; j++) {
                float *lg = llama_get_logits_ith(c, j); double mx = lg[0]; for (int i = 1; i < nv; i++) if (lg[i] > mx) mx = lg[i];
                double z = 0; for (int i = 0; i < nv; i++) z += exp((double)lg[i] - mx);
                cn += -((double)lg[ct[j + 1]] - mx - log(z)); cc++;
            }
            ct[0] = save; nll += cn; cnt += cc; fprintf(stderr, "ppl chunk %d: %.4f (running %.4f)\n", ch, exp(cn / cc), exp(nll / cnt));
        }
        printf("PPL %.4f over %ld tokens in %d chunks (bitnet.cpp libllama)\n", exp(nll / cnt), cnt, nchunk);
        return 0;
    }
    if (np > 1 && llama_decode(c, llama_batch_get_one(toks, np - 1))) return 4;      /* prefill, same split as armilla_bitnet */
    llama_token cur = toks[np - 1]; double tg = 0; int n = 0;
    for (int g = 0; g < ngen; g++) {
        struct timespec a, b; clock_gettime(CLOCK_MONOTONIC, &a);
        if (llama_decode(c, llama_batch_get_one(&cur, 1))) return 5;
        clock_gettime(CLOCK_MONOTONIC, &b); tg += (b.tv_sec - a.tv_sec) + 1e-9 * (b.tv_nsec - a.tv_nsec);
        float *lg = llama_get_logits_ith(c, -1); int best = 0; for (int i = 1; i < nv; i++) if (lg[i] > lg[best]) best = i;
        if (lf) fwrite(lg, 4, nv, lf);
        printf("%d ", best); fflush(stdout); cur = best; n++;
    }
    printf("\n"); fprintf(stderr, "decode %d tokens in %.3fs = %.3f tok/s (%d threads)\n", n, tg, n / tg, nt);
    return 0;
}
