Held-out text for the SiLU-vs-ReLU² perplexity check (OUTREACH-P1-FOLLOWUP).
- `pride-and-prejudice-ch1-slice.txt`: 16,000 characters from "It is a truth universally acknowledged" in Jane Austen, *Pride and Prejudice* (1813, public domain),
  Project Gutenberg #1342 (https://www.gutenberg.org/cache/epub/1342/pg1342.txt, file sha256 3f6bb9d6f78e0293b56acd4714dd68cb7d6d1d293402031ce9d5a216bcaf9d75).
- slice sha256 c9118a2703dc9625b515203c0bdeaa037aecb0448f0c9b09135930a1688581d0; ids sha256 dc2ddc2a5ec4630a49923e0d29762c9706d0cf0ed4e088989a32985972cb2be2.
- `ppl-ids.txt`: HF tokenizer ids (BitNet 2B4T tokenizer.json), 6 chunks × 512 (BOS + 511 text tokens); protocol = llama.cpp perplexity (n_ctx 512, second half scored → 255 tokens/chunk, 1,530 scored).
