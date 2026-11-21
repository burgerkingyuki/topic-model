# LiSA Reproduction

Run with:

```powershell
python run_lisa.py --config config.json
```

Set API settings in `.env` and pipeline settings in `config.json`.

Strict paper defaults:

- LLM: Llama-3-8B-Instruct
- Embedder: `sentence-transformers/all-MiniLM-L6-v2`
- Clustering: KMeans
- Document input: full text from the `text` field, truncated only for LLM prompts

