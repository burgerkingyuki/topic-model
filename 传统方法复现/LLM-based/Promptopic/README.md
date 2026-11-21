# PromptTopic Reproduction

Run with:

```powershell
python run_promptopic.py --config config.json
```

Set API settings in `.env` and pipeline settings in `config.json`.

Strict notes from the paper:

- The method has three stages: topic generation, topic collapse, and topic representation generation.
- Topic collapse supports Prompt-Based Matching (PBM), Word Similarity Matching (WSM), and PBM-then-WSM.
- The paper reports `N=4` demonstrations as best, but the PDF only visibly discloses part of the demonstration prompt. Configure demonstrations in `config.json` for exact experiment control.

