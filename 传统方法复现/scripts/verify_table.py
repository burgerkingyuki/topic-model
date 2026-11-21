# -*- coding: utf-8 -*-
"""Cross-check every parameter row in the generated docx against its config.json."""

import json
import os
import re
import sys

from docx import Document

BASE = r"E:\主题建模\LSTFM\传统方法复现"
DOC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "复现基线模型参数设置表.docx")

CFG = {
    "LDA": r"传统方法\新版本\lda\config.json",
    "NMF": r"传统方法\新版本\nmf\config.json",
    "BERTopic": r"传统方法\新版本\bertopic\config.json",
    "PromptTopic": r"LLM-based\Promptopic\config.json",
    "TopicGPT": r"LLM-based\Topicgpt\config.json",
    "LISA": r"LLM-based\LISA\config.json",
}

# Values the table deliberately reports differently from the config file.
# demonstrations: the config holds a list; the table reports its length.
OVERRIDES = {
    ("LDA", "num_topics"): "40",
    ("PromptTopic", "demonstrations"): "2",
}

doc = Document(DOC)
table = doc.tables[0]

current = None
param_rows = 0
problems = []
per_model = {}
for row in table.rows[1:]:
    cells = row.cells
    model = cells[0].text.strip().replace("\n", "")
    if model:
        current = model
    key_cell = cells[3].text.strip()
    value = cells[4].text.strip()
    match = re.match(r"^([A-Za-z0-9_]+)", key_cell)
    if not match:
        problems.append(("no key parsed", key_cell))
        continue
    key = match.group(1)
    param_rows += 1
    per_model[current] = per_model.get(current, 0) + 1

    cfg_path = os.path.join(BASE, CFG[current])
    with open(cfg_path, encoding="utf-8") as handle:
        cfg = json.load(handle)

    if (current, key) in OVERRIDES:
        expected = OVERRIDES[(current, key)]
    elif key not in cfg:
        problems.append(("key not in config", current, key))
        continue
    else:
        raw = cfg[key]
        if isinstance(raw, bool):
            expected = "是" if raw else "否"
        elif isinstance(raw, list):
            expected = str(raw)
        else:
            expected = str(raw)

    if expected != value:
        problems.append(("value mismatch", current, key, "docx=" + value, "config=" + expected))

print("models:", len(per_model))
for name, n in per_model.items():
    print("  %-24s %d params" % (name, n))
print("param rows checked:", param_rows)
print("problems:", len(problems))
for item in problems:
    print("   ", item)
