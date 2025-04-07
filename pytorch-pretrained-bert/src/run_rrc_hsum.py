#!/usr/bin/env python
# coding=utf-8
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This code is modified to use the HSUM fusion technique for question answering.

import os
import logging
import argparse
import random
import json
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.optim import AdamW

from transformers import BertTokenizer, BertModel, BertPreTrainedModel, get_linear_schedule_with_warmup
import squad_data_utils as data_utils
import modelconfig

# Import BertLayer – adjust the import based on your transformers version.
from transformers.models.bert.modeling_bert import BertLayer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # Loop over model parameters and add perturbation to embedding parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # Save original parameter values
                self.backup[name] = param.data.clone()
                # compute perturbation
                if param.grad is None:
                    continue
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class HSUMQA(torch.nn.Module):
    def __init__(self, count, config):
        """
        Args:
          count: number of layers to fuse (e.g. 4)
          config: BERT configuration
        """
        super(HSUMQA, self).__init__()
        self.count = count
        # Create fusion layers (one for each fused layer)
        self.pre_layers = torch.nn.ModuleList([BertLayer(config) for _ in range(count)])
        # Define two classifiers for start and end positions
        self.start_classifier = torch.nn.Linear(config.hidden_size, 1)
        self.end_classifier = torch.nn.Linear(config.hidden_size, 1)
        # Loss function used for training
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, layers, attention_mask, start_positions=None, end_positions=None):
        # Ensure attention_mask is float and has the correct dimensions
        if attention_mask.dtype != torch.float:
            attention_mask = attention_mask.float()
        if attention_mask.dim() == 2:
            # Expand to [batch_size, 1, 1, seq_len] for use in BertLayer
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        else:
            extended_attention_mask = attention_mask

        fused = torch.zeros_like(layers[0])
        start_logits_list = []
        end_logits_list = []

        for i in range(self.count):
            fused = fused + layers[-i - 1]  # residual sum
            # Use the extended attention mask here:
            layer_output = self.pre_layers[i](fused, extended_attention_mask)
            # If the output is a tuple, take the first element (hidden states)
            if isinstance(layer_output, tuple):
                fused = layer_output[0]
            else:
                fused = layer_output
            start_logits = self.start_classifier(fused).squeeze(-1)  # [B, T]
            end_logits = self.end_classifier(fused).squeeze(-1)
            start_logits_list.append(start_logits)
            end_logits_list.append(end_logits)

        avg_start_logits = torch.stack(start_logits_list, dim=0).mean(dim=0)
        avg_end_logits = torch.stack(end_logits_list, dim=0).mean(dim=0)

        if start_positions is not None and end_positions is not None:
            seq_len = avg_start_logits.size(1)
            if (start_positions >= seq_len).any() or (start_positions < -1).any():
                print("Invalid start_positions detected:", start_positions)
                print("Sequence length:", seq_len)
                raise ValueError("start_positions out of bounds")
            if (end_positions >= seq_len).any() or (end_positions < -1).any():
                print("Invalid end_positions detected:", end_positions)
                print("Sequence length:", seq_len)
                raise ValueError("end_positions out of bounds")

            loss_start = self.loss_fct(avg_start_logits, start_positions)
            loss_end = self.loss_fct(avg_end_logits, end_positions)
            loss = (loss_start + loss_end) / 2
            return loss, avg_start_logits, avg_end_logits
        else:
            return avg_start_logits, avg_end_logits



class BertForQAWithHSUM(BertPreTrainedModel):
    def __init__(self, config, hs_count=4):
        super(BertForQAWithHSUM, self).__init__(config)
        self.bert = BertModel(config)
        self.hsumqa = HSUMQA(hs_count, config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                start_positions=None, end_positions=None):
        # Convert attention_mask to float if needed
        if attention_mask is not None and attention_mask.dtype != torch.float:
            attention_mask = attention_mask.float()

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=True)
        hidden_states = outputs.hidden_states
        if start_positions is not None and end_positions is not None:
            loss, start_logits, end_logits = self.hsumqa(hidden_states, attention_mask,
                                                         start_positions, end_positions)
            return {"loss": loss, "start_logits": start_logits, "end_logits": end_logits}
        else:
            start_logits, end_logits = self.hsumqa(hidden_states, attention_mask)
            return {"start_logits": start_logits, "end_logits": end_logits}


def train(args):
    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)
    tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    train_examples = data_utils.read_squad_examples(os.path.join(args.data_dir, "train.json"), is_training=True)
    num_train_steps = int(len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

    train_features = data_utils.convert_examples_to_features(
        train_examples, tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, is_training=True)

    logger.info("***** Running training *****")
    logger.info("  Num orig examples = %d", len(train_examples))
    logger.info("  Num split examples = %d", len(train_features))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
    all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_start_positions, all_end_positions)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # Setup validation if needed
    if args.do_valid:
        valid_examples = data_utils.read_squad_examples(os.path.join(args.data_dir, "dev.json"), is_training=True)
        valid_features = data_utils.convert_examples_to_features(
            valid_examples, tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, is_training=True)
        valid_all_input_ids = torch.tensor([f.input_ids for f in valid_features], dtype=torch.long)
        valid_all_segment_ids = torch.tensor([f.segment_ids for f in valid_features], dtype=torch.long)
        valid_all_input_mask = torch.tensor([f.input_mask for f in valid_features], dtype=torch.long)
        valid_all_start_positions = torch.tensor([f.start_position for f in valid_features], dtype=torch.long)
        valid_all_end_positions = torch.tensor([f.end_position for f in valid_features], dtype=torch.long)
        valid_data = TensorDataset(valid_all_input_ids, valid_all_segment_ids, valid_all_input_mask,
                                   valid_all_start_positions, valid_all_end_positions)
        logger.info("***** Running validations *****")
        logger.info("  Num orig examples = %d", len(valid_examples))
        logger.info("  Num split examples = %d", len(valid_features))
        logger.info("  Batch size = %d", args.train_batch_size)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.train_batch_size)
        best_valid_loss = float('inf')
        valid_losses = []

    # Instantiate custom model instead of the standard BertForQuestionAnswering
    model = BertForQAWithHSUM.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    if args.fp16:
        model.half()
    model.cuda()

    # Prepare optimizer
    param_optimizer = [(k, v) for k, v in model.named_parameters() if v.requires_grad]
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(args.warmup_proportion * num_train_steps),
                                                num_training_steps=num_train_steps)

    global_step = 0

    if args.do_adv:
        fgm = FGM(model)

    model.train()
    for _ in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, segment_ids, input_mask, start_positions, end_positions = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                start_positions=start_positions,
                end_positions=end_positions,
            )
            loss = outputs["loss"]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()

            if args.do_adv:
                # Apply adversarial perturbation to word embeddings
                fgm.attack(epsilon=args.adv_epsilon, emb_name='word_embeddings')
                # Forward pass on adversarial examples

                adv_output = model(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    start_positions=start_positions,
                    end_positions=end_positions,
                )

                adv_loss = adv_output["loss"]
                if args.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(adv_loss)
                else:
                    adv_loss.backward()
                # Restore the original parameters
                fgm.restore(emb_name='word_embeddings')
            # ------------------------------------

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        if args.do_valid:
            model.eval()
            with torch.no_grad():
                losses = []
                valid_size = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = tuple(t.cuda() for t in batch)
                    input_ids, segment_ids, input_mask, start_positions, end_positions = batch
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        start_positions=start_positions,
                        end_positions=end_positions,
                    )
                    loss = outputs["loss"]
                    losses.append(loss.item() * input_ids.size(0))
                    valid_size += input_ids.size(0)
                valid_loss = sum(losses) / valid_size
                logger.info("Validation loss: %f", valid_loss)
                if valid_loss < best_valid_loss:
                    torch.save(model, os.path.join(args.output_dir, "model.pt"))
                    best_valid_loss = valid_loss
            model.train()

    if args.do_valid:
        with open(os.path.join(args.output_dir, "valid.json"), "w") as fw:
            json.dump({"valid_losses": valid_losses}, fw)
    else:
        torch.save(model, os.path.join(args.output_dir, "model.pt"))



def test(args):
    tokenizer = BertTokenizer.from_pretrained(modelconfig.MODEL_ARCHIVE_MAP[args.bert_model])
    eval_examples = data_utils.read_squad_examples(os.path.join(args.data_dir, "test.json"), is_training=False)
    eval_features = data_utils.convert_examples_to_features(
        eval_examples, tokenizer, args.max_seq_length, args.doc_stride, args.max_query_length, is_training=False)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_segment_ids, all_input_mask, all_example_index)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    model = torch.load(os.path.join(args.output_dir, "model.pt"), weights_only=False)
    model.cuda()
    model.eval()
    all_results = []
    for step, batch in enumerate(eval_dataloader):
        example_indices = batch[-1]
        batch = tuple(t.cuda() for t in batch[:-1])
        input_ids, segment_ids, input_mask = batch

        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=input_mask,
                            token_type_ids=segment_ids)
            batch_start_logits = outputs["start_logits"]
            batch_end_logits = outputs["end_logits"]

        for i, example_index in enumerate(example_indices):
            start_logits = batch_start_logits[i].detach().cpu().tolist()
            end_logits = batch_end_logits[i].detach().cpu().tolist()
            eval_feature = eval_features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            all_results.append(data_utils.RawResult(unique_id=unique_id,
                                                    start_logits=start_logits,
                                                    end_logits=end_logits))
    output_prediction_file = os.path.join(args.output_dir, "predictions.json")
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
    data_utils.write_predictions(eval_examples, eval_features, all_results,
                                 args.n_best_size, args.max_answer_length,
                                 True, output_prediction_file, output_nbest_file, False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--bert_model", default='bert-base', type=str)
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir containing json files.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--max_seq_length", default=320, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--do_train", default=False, action='store_true', help="Whether to run training.")
    parser.add_argument("--do_valid", default=False, action='store_true', help="Whether to run validation.")
    parser.add_argument("--do_eval", default=False, action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for eval.")
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=6, type=int, help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for initialization")
    parser.add_argument('--doc_stride', type=int, default=128)
    parser.add_argument('--max_query_length', type=int, default=30)
    parser.add_argument('--max_answer_length', type=int, default=30)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--fp16', default=False, action='store_true',
                        help="Whether to use 16-bit precision instead of 32-bit")
    parser.add_argument('--loss_scale', type=float, default=0, help="Loss scaling for fp16 training")
    parser.add_argument('--n_best_size', type=int, default=20)
    parser.add_argument("--do_adv",
                        default=True,
                        action='store_true',
                        help="Whether to perform adversarial training.")
    parser.add_argument("--adv_epsilon",
                        default=0.2,
                        type=float,
                        help="Magnitude of adversarial perturbation.")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.do_train:
        train(args)
    if args.do_eval:
        test(args)


if __name__ == "__main__":
    main()
