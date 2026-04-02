"""
Downloads, tokenises, and memory-maps the full English Wikipedia corpus.

Outputs two files (paths are constants exported from this module):
  wiki_tokens.dat      -- raw int32 token IDs as a flat binary memmap
  wiki_tokens_meta.npy -- single-element array storing the exact token count

Tokenisation is parallelised across NUM_WORKERS processes.  Progress is
checkpointed every CHECKPOINT_EVERY articles so the run can be interrupted
and resumed without losing work.

Run directly to build the cache:
    python projects/latent_ar/wiki_data.py

Both latent_ar.py and scan_layers.py import load_tokens() from here.
"""

import os
import sys
from multiprocessing import Pool

import numpy as np

_DIR               = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_PATH    = os.path.join(_DIR, 'wiki_tokens.dat')
DATA_META_PATH     = os.path.join(_DIR, 'wiki_tokens_meta.npy')
PROGRESS_PATH      = os.path.join(_DIR, 'wiki_tokens_progress.npy')
NUM_WORKERS      = 12
CHUNK_SIZE       = 200    # articles per worker task (imap chunksize)
CHECKPOINT_EVERY = 50_000  # articles between progress saves
LOG_EVERY        = 10_000  # articles between progress prints

TRAIN_PATH      = os.path.join(_DIR, 'wiki_tokens_train.dat')
TRAIN_META_PATH = os.path.join(_DIR, 'wiki_tokens_train_meta.npy')
TEST_PATH       = os.path.join(_DIR, 'wiki_tokens_test.dat')
TEST_META_PATH  = os.path.join(_DIR, 'wiki_tokens_test_meta.npy')
SPLIT_CHUNK     = 2048  # tokens per split unit (2× block_size)
TEST_FRACTION   = 0.01       # ~1% of chunks go to test
SPLIT_SEED      = 1337       # reproducible, fixed split


# ---------------------------------------------------------------------------
# Worker-process helpers (must be module-level for pickling)

def _worker_init():
    """Initialise one GPT-2 tokenizer per worker process."""
    global _tok
    import os as _os
    _os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import GPT2Tokenizer
    _tok = GPT2Tokenizer.from_pretrained('gpt2')


def _tokenize(text):
    return _tok.encode(text)


# ---------------------------------------------------------------------------

def prepare_data():
    """
    Download the full English Wikipedia dataset, tokenise every article with
    GPT-2 BPE using a multiprocessing pool, and write the token IDs to a
    memory-mapped binary file so training can randomly sample from it without
    loading everything into RAM.

    Checkpoints progress every CHECKPOINT_EVERY articles so the run can be
    stopped and resumed cleanly.
    """
    # Already complete
    if os.path.exists(DATA_CACHE_PATH) and os.path.exists(DATA_META_PATH):
        n_tokens = int(np.load(DATA_META_PATH)[0])
        tokens = np.memmap(DATA_CACHE_PATH, dtype=np.int32, mode='r',
                           shape=(n_tokens,))
        print(f"Loaded {n_tokens:,} tokens from {DATA_CACHE_PATH}")
        return tokens

    from datasets import load_dataset
    from transformers import GPT2Tokenizer

    print("Loading Wikipedia dataset (parquet shards cached by HuggingFace)...")
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    n_articles = len(ds)
    print(f"Dataset: {n_articles:,} articles")

    # Resume from checkpoint only if both the progress file and the dat file
    # exist — if the dat was deleted, discard the stale progress and restart.
    if os.path.exists(PROGRESS_PATH) and os.path.exists(DATA_CACHE_PATH):
        progress      = np.load(PROGRESS_PATH)
        articles_done = int(progress[0])
        idx           = int(progress[1])
        print(f"Resuming from checkpoint: {articles_done:,} articles done, "
              f"{idx:,} tokens written")
        fp = np.memmap(DATA_CACHE_PATH, dtype=np.int32, mode='r+')
    else:
        if os.path.exists(PROGRESS_PATH):
            os.remove(PROGRESS_PATH)
            print("Removed stale progress file — starting fresh.")
        articles_done = 0
        idx           = 0

        # Estimate total tokens from a stratified sample of 2 000 articles
        print("Estimating corpus size from a 2 000-article sample...")
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        step       = max(1, n_articles // 2_000)
        sample_ids = list(range(0, n_articles, step))[:2_000]
        sample_toks = sum(
            len(tokenizer.encode(ds[i]['text'])) for i in sample_ids
        )
        avg   = sample_toks / len(sample_ids)
        total = int(avg * n_articles * 1.1)   # 10 % headroom
        print(f"  ~{avg:.0f} tokens/article, "
              f"estimated {total:,} tokens ({total * 4 / 1e9:.1f} GB)")

        fp = np.memmap(DATA_CACHE_PATH, dtype=np.int32, mode='w+',
                       shape=(total,))

    # Tokenise in parallel using imap so workers stay fed continuously.
    # pool.imap lazily pulls from the text generator and dispatches chunks to
    # workers as they become free — no manual batching needed and no idle time
    # while the main process collects a batch or writes results.
    import time
    remaining = ds.select(range(articles_done, n_articles))

    def _texts(dataset):
        for article in dataset:
            yield article['text']

    with Pool(processes=NUM_WORKERS, initializer=_worker_init) as pool:
        t_window = time.time()   # time at the start of the current log window
        n_window = 0             # articles processed in the current window

        for ids in pool.imap(_tokenize, _texts(remaining),
                             chunksize=CHUNK_SIZE):
            n = len(ids)
            if idx + n > len(fp):
                print(f"Warning: hit preallocated limit; stopping early "
                      f"({idx:,} tokens written).")
                _finish(fp, idx, articles_done)
                return _open_readonly(idx)
            fp[idx : idx + n] = ids
            idx           += n
            articles_done += 1
            n_window      += 1

            if articles_done % LOG_EVERY == 0:
                now  = time.time()
                rate = n_window / (now - t_window)
                pct  = articles_done / n_articles * 100
                print(f"  {articles_done:,}/{n_articles:,} ({pct:.1f}%)  "
                      f"{idx:,} tokens  {rate:.0f} articles/sec")
                t_window = now
                n_window = 0

            if articles_done % CHECKPOINT_EVERY == 0:
                fp.flush()
                np.save(PROGRESS_PATH,
                        np.array([articles_done, idx], dtype=np.int64))
                print(f"  [checkpoint saved]")

    _finish(fp, idx, n_articles)
    return _open_readonly(idx)


def _finish(fp, idx, articles_done):
    fp.flush()
    del fp
    np.save(DATA_META_PATH, np.array([idx], dtype=np.int64))
    if os.path.exists(PROGRESS_PATH):
        os.remove(PROGRESS_PATH)
    print(f"Tokenisation complete: {articles_done:,} articles, "
          f"{idx:,} tokens -> {DATA_CACHE_PATH}")


def _open_readonly(idx):
    return np.memmap(DATA_CACHE_PATH, dtype=np.int32, mode='r', shape=(idx,))


# ---------------------------------------------------------------------------

def load_tokens():
    """
    Load the pre-built token memmap.  Exits with a clear error message if the
    cache files don't exist yet (i.e. prepare_data() hasn't been run).
    """
    if not (os.path.exists(DATA_CACHE_PATH) and os.path.exists(DATA_META_PATH)):
        print("ERROR: wiki_tokens.dat not found.")
        print("Run `python projects/latent_ar/wiki_data.py` first to download")
        print("and tokenise the Wikipedia corpus.")
        sys.exit(1)
    n_tokens = int(np.load(DATA_META_PATH)[0])
    tokens   = np.memmap(DATA_CACHE_PATH, dtype=np.int32, mode='r',
                         shape=(n_tokens,))
    print(f"Loaded {n_tokens:,} tokens from {DATA_CACHE_PATH}")
    return tokens


def split_data(tokens=None):
    """
    Split wiki_tokens.dat into train and test sets by streaming through in
    SPLIT_CHUNK-sized chunks and randomly assigning each chunk to train or test.

    The split is seeded for reproducibility — the same chunk boundaries and
    assignments are produced every time.  Skips if output files already exist.

    Accepts an optional pre-loaded tokens array to avoid loading twice when
    called immediately after prepare_data().
    """
    if (os.path.exists(TRAIN_PATH) and os.path.exists(TRAIN_META_PATH) and
            os.path.exists(TEST_PATH) and os.path.exists(TEST_META_PATH)):
        train_n = int(np.load(TRAIN_META_PATH)[0])
        test_n  = int(np.load(TEST_META_PATH)[0])
        print(f"Split already exists: {train_n:,} train tokens, {test_n:,} test tokens")
        return

    if tokens is None:
        tokens = load_tokens()
    n_tokens = len(tokens)
    rng      = np.random.default_rng(SPLIT_SEED)

    print(f"Splitting {n_tokens:,} tokens "
          f"(chunk={SPLIT_CHUNK:,}, test_fraction={TEST_FRACTION})...")

    train_n = 0
    test_n  = 0
    n_chunks = 0

    with open(TRAIN_PATH, 'wb') as f_train, open(TEST_PATH, 'wb') as f_test:
        for start in range(0, n_tokens, SPLIT_CHUNK):
            chunk = tokens[start : start + SPLIT_CHUNK]
            if rng.random() < TEST_FRACTION:
                f_test.write(chunk.tobytes())
                test_n += len(chunk)
            else:
                f_train.write(chunk.tobytes())
                train_n += len(chunk)
            n_chunks += 1
            if n_chunks % 50_000 == 0:
                pct = (start + len(chunk)) / n_tokens * 100
                print(f"  {pct:.1f}%  train={train_n:,}  test={test_n:,}")

    np.save(TRAIN_META_PATH, np.array([train_n], dtype=np.int64))
    np.save(TEST_META_PATH,  np.array([test_n],  dtype=np.int64))
    print(f"Split complete: {train_n:,} train tokens ({train_n/n_tokens*100:.1f}%), "
          f"{test_n:,} test tokens ({test_n/n_tokens*100:.1f}%)")


def load_train_tokens():
    """Load the train split.  Run wiki_data.py first if not yet split."""
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TRAIN_META_PATH)):
        print("ERROR: wiki_tokens_train.dat not found.")
        print("Run `python projects/latent_ar/wiki_data.py` to build and split the corpus.")
        sys.exit(1)
    n = int(np.load(TRAIN_META_PATH)[0])
    tokens = np.memmap(TRAIN_PATH, dtype=np.int32, mode='r', shape=(n,))
    print(f"Loaded {n:,} train tokens from {TRAIN_PATH}")
    return tokens


def load_test_tokens():
    """Load the test split.  Run wiki_data.py first if not yet split."""
    if not (os.path.exists(TEST_PATH) and os.path.exists(TEST_META_PATH)):
        print("ERROR: wiki_tokens_test.dat not found.")
        print("Run `python projects/latent_ar/wiki_data.py` to build and split the corpus.")
        sys.exit(1)
    n = int(np.load(TEST_META_PATH)[0])
    tokens = np.memmap(TEST_PATH, dtype=np.int32, mode='r', shape=(n,))
    print(f"Loaded {n:,} test tokens from {TEST_PATH}")
    return tokens


if __name__ == '__main__':
    tokens = prepare_data()
    split_data(tokens)
