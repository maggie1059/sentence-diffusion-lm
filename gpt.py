import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, \
get_linear_schedule_with_warmup, GPT2Config, GPT2Model
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import csv
import os
import math
from typing import Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

def noise_tensor(x_start):
    noise = torch.randn_like(x_start)
    t = torch.tensor([0], device="cuda")

    betas = betas_for_alpha_bar(
            2000,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    sqrt_alphas_cumprod = np.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - alphas_cumprod)

    output = (
        _extract_into_tensor(sqrt_alphas_cumprod, t, x_start.shape) * x_start
        + _extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        * noise
    )

    return output

class Dialog(Dataset):  
    def __init__(self, split='train'):

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.text = []
        self.labels = []
        self.attention = []

        file_name = 'dialogues_' + split + '.txt'

        # max_seq_len = 200
        with open(file_name, encoding="utf-8") as f:
            for idx, row1 in enumerate(f):
                if row1.strip():
                    row = row1.split('__eou__')[:-1]
                    for utterance in row:
                      # utter = utterance + " <|endoftext|>"
                      # print("utterance: ", utterance)
                      # utter_tok = self.tokenizer.encode(utter)
                      encoding = self.tokenizer(utterance, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
                      utter_tok = torch.LongTensor(encoding['input_ids'])
                      attention_mask = torch.LongTensor(encoding['attention_mask'])
                      # print("utter tok: ", utter_tok)
                      # print("attention mask: ", attention_mask)

                      # self.labels.append(torch.tensor(utter_tok))
                      self.labels.append(utter_tok)
                      self.attention.append(attention_mask)

                      utter_encode = [utterance] + ['[PAD]' for _ in range(len(utter_tok[0])-1)]
                      # print("utter_encode: ", utter_encode)

                      embed = torch.tensor(model.encode(utter_encode), device="cuda")
                      noise_embed = noise_tensor(embed)
                      # print("noise embed: ", noise_embed.shape)

                      self.text.append(noise_embed)

        self.text_count = len(self.labels)
        
    def __len__(self):
        return self.text_count

    def __getitem__(self, item):
        return self.text[item], self.labels[item], self.attention[item]

    # def pad_data(self, data):
    #     sents = [x[0] for x in data]
    #     sent_ids = [x[1] for x in data]

    #     encoding = self.tokenizer(sents, return_tensors='pt', padding=True, truncation=True)
    #     token_ids = torch.LongTensor(encoding['input_ids'])
    #     attention_mask = torch.LongTensor(encoding['attention_mask'])

    #     return token_ids, attention_mask, sents, sent_ids

    def collate_fn(self, data):
        # token_ids, attention_mask, sents, sent_ids= self.pad_data(all_data)
        # print("data: ", data)
        input_embeds = [x[0] for x in data]
        labels = [x[1] for x in data]
        # print("labels: ", labels)
        am = [x[2] for x in data]

        batched_data = {
                'input_embeds': torch.stack(input_embeds),
                'attention_mask': torch.stack(am),
                'labels': torch.stack(labels)
            }

        return batched_data
    
train_dataset = Dialog('train')
# val_dataset = train_dataset
val_dataset = Dialog('valid')

class GPT2SentModel(GPT2Model):
    _keys_to_ignore_on_load_missing = ["attn.masked_bias"]

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.convert_sent = nn.Linear(384, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        inputs_embeds = self.convert_sent(inputs_embeds)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

class GPT2LMHeadSentModel(GPT2LMHeadModel):
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]
    _keys_to_ignore_on_load_unexpected = [r"h\.\d+\.attn\.masked_bias", r"h\.\d+\.attn\.bias"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = GPT2SentModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Model parallel
        self.model_parallel = False
        self.device_map = None

        # Initialize weights and apply final processing
        self.post_init()
    
#Get the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadSentModel.from_pretrained('gpt2')
model.load_state_dict(torch.load('./ckpt-0.pt'))

#Accumulated batch size (since GPT2 is so big)
def pack_tensor(new_tensor, packed_tensor, max_seq_len):
    if packed_tensor is None:
        return new_tensor, True, None
    if new_tensor.size()[1] + packed_tensor.size()[1] > max_seq_len:
        return packed_tensor, False, new_tensor
    else:
        packed_tensor = torch.cat([new_tensor, packed_tensor[:, 1:]], dim=1)
        return packed_tensor, True, None
    
from google.colab import files

def train(
    train_dataset, val_dataset, model, tokenizer,
    batch_size=1, epochs=5000, lr=2e-5,
    max_seq_len=400, warmup_steps=200,
    gpt2_type="gpt2", output_dir=".", output_prefix="ckpt",
    test_mode=False,save_model_on_epoch=True,
):
    acc_steps = 100
    eval_every = 1
    device=torch.device("cuda")
    model = model.cuda()
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=-1
    )

    train_dataloader = DataLoader(train_dataset, batch_size=2, collate_fn=train_dataset.collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=val_dataset.collate_fn, shuffle=True)
    loss=0
    best_loss = float('inf')
    accumulating_batch_count = 0
    input_tensor = None

    for epoch in range(epochs):

        print(f"Training epoch {epoch}")
        print(loss)
        for idx, entry in tqdm(enumerate(train_dataloader)):

            input_tensor, label, attention_mask = (entry['input_embeds'], entry['labels'], entry['attention_mask'])
            input_tensor = input_tensor.to(device)
            label = label.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(inputs_embeds=input_tensor, labels=label, attention_mask=attention_mask)
            loss = outputs[0]
            loss.backward()

            if (accumulating_batch_count % batch_size) == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                model.zero_grad()

            accumulating_batch_count += 1
            input_tensor = None
        if save_model_on_epoch:
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, f"{output_prefix}-{epoch}.pt"),
            )
        # files.download(f"{output_prefix}-{epoch}.pt")

        if epoch % eval_every == 0:
            with torch.no_grad():
                avg_loss = 0
                count = 0
                for idx, entry in tqdm(enumerate(val_dataloader)):
                  if idx == 10: break
                  input_tensor, label, attention_mask = (entry['input_embeds'], entry['labels'], entry['attention_mask'])
                  input_tensor = input_tensor.to(device)
                  label = label.to(device)
                  attention_mask = attention_mask.to(device)

                  outputs = model(inputs_embeds=input_tensor, labels=label, attention_mask=attention_mask)
                  avg_loss += outputs[0]
                  count += 1
                  # output = model.generate(inputs_embeds=input_tensor)
                  # decoded = tokenizer.decode(output[0])
                  logits = outputs[1]
                  softmax = nn.Softmax(dim=-1)
                  softmax_l = softmax(logits)
                  # print("softmax shape: ", softmax_l.shape)
                  argmax = torch.argmax(softmax_l, axis=-1)
                  # print("argmax shape: ", argmax.shape)
                  print("decoded:")
                  for word in argmax:
                    print(tokenizer.decode(word))
                  print("label:")
                  for word in label[0]:
                    print(tokenizer.decode(word))
                  # print("decoded: ", decoded)
                avg_loss /= count
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    print("avg loss: ", avg_loss)
                    torch.save(
                        model.state_dict(),
                        os.path.join(output_dir, "ckpt-best.pt"),
                    )
                
    return model

model = train(train_dataset, val_dataset, model, tokenizer)