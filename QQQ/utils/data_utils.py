from transformers import AutoTokenizer
from datasets import load_dataset
import torch
import random
import json


def get_pile(nsamples, seed, seqlen, tokenizer_path):
    print("get_pile")
    traindata = load_dataset(
        "json",
        data_files="/mnt/dolphinfs/hdd_pool/docker/share/1/zhangying/datasets/pile/val.jsonl.zst",
        split="train",
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"][:1000]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, None


def get_wikitext2(nsamples, seed, seqlen, tokenizer_path):
    print("get_wikitext2")
    traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_ptb(nsamples, seed, seqlen, tokenizer_path):
    print("get_ptb")
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    valdata = load_dataset("ptb_text_only", "penn_treebank", split="validation")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    trainenc = tokenizer("\n\n".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(valdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4(nsamples, seed, seqlen, tokenizer_path):
    print("get_c4")
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]["text"], return_tensors="pt")
            if tmp.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    return trainloader, valenc


def get_ptb_new(nsamples, seed, seqlen, tokenizer_path):
    print("get_ptb_new")
    traindata = load_dataset("ptb_text_only", "penn_treebank", split="train")
    testdata = load_dataset("ptb_text_only", "penn_treebank", split="test")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    trainenc = tokenizer(" ".join(traindata["sentence"]), return_tensors="pt")
    testenc = tokenizer(" ".join(testdata["sentence"]), return_tensors="pt")

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


def get_c4_new(nsamples, seed, seqlen, tokenizer_path):
    print("get_c4_new")
    traindata = load_dataset(
        "allenai/c4",
        data_files={"train": "en/c4-train.00000-of-01024.json.gz"},
        split="train",
    )
    valdata = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]["text"], return_tensors="pt")
            if trainenc.input_ids.shape[1] >= seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(" ".join(valdata[:1100]["text"]), return_tensors="pt")
    valenc = valenc.input_ids[:, : (256 * seqlen)]
    return trainloader, valenc


def get_custom_data(nsamples, seed, seqlen, tokenizer_path, data_path):
    raise NotImplementedError(
        "You should implentment the function to load your own dataset!"
    )


def get_loaders(
    name="",
    nsamples=128,
    seed=0,
    seqlen=2048,
    tokenizer_path="",
    custom_data_path="",
):
    if custom_data_path != "":
        return get_custom_data(nsamples, seed, seqlen, tokenizer_path, custom_data_path)
    if "wikitext2" in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer_path)
    if "pile" in name:
        return get_pile(nsamples, seed, seqlen, tokenizer_path)
    if "ptb" in name:
        if "new" in name:
            return get_ptb_new(nsamples, seed, seqlen, tokenizer_path)
        return get_ptb(nsamples, seed, seqlen, tokenizer_path)
    if "c4" in name:
        if "new" in name:
            return get_c4_new(nsamples, seed, seqlen, tokenizer_path)
        return get_c4(nsamples, seed, seqlen, tokenizer_path)
    if "mix" in name:
        wiki_train, wiki_val = get_wikitext2(
            nsamples // 3, seed, seqlen, tokenizer_path
        )
        ptb_train, ptb_val = get_ptb(nsamples // 3, seed, seqlen, tokenizer_path)
        c4_train, c4_val = get_c4(nsamples // 3, seed, seqlen, tokenizer_path)
        train = wiki_train + ptb_train + c4_train
        val = None
        return train, val
