#coding:utf-8
# @Time : 2021/6/19
# @Author : Han Fang
# @File: dataloader.py
# @Version: version 1.0
import sys
import os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')

import torch
from torch.utils.data import DataLoader
from dataloaders.dataloader_msvd_frame import MSVD_multi_sentence_dataLoader


def dataloader_msvd_train(args, tokenizer):
    """return dataloader for training msvd
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msvd_dataset): length
        train_sampler: sampler for distributed training
    """

    msvd_dataset = MSVD_multi_sentence_dataLoader(
        subset="train",
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msvd_dataset)

    dataloader = DataLoader(
        msvd_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )

    return dataloader, len(msvd_dataset), train_sampler

def dataloader_msvd_test(args, tokenizer, subset="test"):
    """return dataloader for testing msvd in multi-sentence captions
    Args:
        args: hyper-parameters
        tokenizer: tokenizer
    Returns:
        dataloader: dataloader
        len(msvd_dataset): length
    """

    msvd_test_set = MSVD_multi_sentence_dataLoader(
        subset=subset,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
        myreplace=args.myreplace
    )

    dataloader = DataLoader(
        msvd_test_set,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        shuffle=False,
        drop_last=False,
    )
    return dataloader, len(msvd_test_set)


