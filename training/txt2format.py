'''
alpaca-chatbot-format:
[
    {
        "instruction": "Give three tips for staying healthy.",
        "input": "",
        "output": "..."
    },
]
'''

import os
import json

path = "/home/orion/文档/rawDatasets/HNovels"
files = os.listdir(path)

inputFormats = []
instruction = "写一篇小黄文"
i = 0
for file in files:
    position = path + '/' + file
    with open(position, "r") as f:
        data = f.read()
    tags = file[:-13]
    input = tags
    output = data
    inputFormat = {"instruction": instruction, "input": input, "output": output}
    inputFormats.append(inputFormat)
    i += 1
    if i >= 1:
        break

with open("/home/orion/AIplayground/text-generation-webui/training/datasets/HNovels.json", "w", encoding="utf-8") as f:
    json.dump(inputFormats, f, ensure_ascii=False)
