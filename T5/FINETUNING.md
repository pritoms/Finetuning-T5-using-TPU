
## Table of Contents

- Dataset Loading and Processing
- Building Trainer
- Training
- Evaluation
- Inference

## Dataset Loading and Processing

### Download and Extract SQuAD

```bash
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
!wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
```

### Import Libraries

- `json`
- `collections`
- `tqdm`
- `transformers`
- `torch`
- `nlp`

### Define `SquadExample`

```python
@dataclasses.dataclass
class SquadExample:
    question_text: str
    context_text: str
    start_position_character: int
    title: str
    is_impossible: bool
```

### Define `SquadFeatures`

```python
@dataclasses.dataclass
class SquadFeatures:
    example_id: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor
```

### Define `SquadDataset`

```python
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, examples: List[SquadExample], tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        example = self.examples[item]
        example_id = str(item)
        question_text = example.question_text
        context_text = example.context_text
        start_position_character = example.start_position_character
        title = example.title
        is_impossible = example.is_impossible

        # Tokenize context
        tokenized_context = self.tokenizer.encode(context_text)

        # Tokenize question
        tokenized_question = self.tokenizer.encode(question_text)

        # Find the end position of the answer
        end_position = start_position_character + len(tokenized_question)

        # Create inputs
        input_ids = tokenized_context + tokenized_question
        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(tokenized_context) + [1] * len(tokenized_question)

        # Pad and create attention masks.
        # Skip if truncation is needed
        if len(input_ids) > self.max_seq_length:
            return None

        padding_length = self.max_seq_length - len(input_ids)
        input_ids = input_ids + ([self.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([self.tokenizer.pad_token_type_id] * padding_length)

        # Convert to tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long)
        start_positions = torch.tensor(start_position_character, dtype=torch.long)
        end_positions = torch.tensor(end_position, dtype=torch.long)

        return SquadFeatures(
            example_id=example_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
```

### Define `SquadDataCollator`

```python
class SquadDataCollator:
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def __call__(self, batch: List[SquadFeatures]):
        example_ids = [x.example_id for x in batch]
        input_ids = torch.stack([x.input_ids for x in batch])
        attention_mask = torch.stack([x.attention_mask for x in batch])
        token_type_ids = torch.stack([x.token_type_ids for x in batch])
        start_positions = torch.stack([x.start_positions for x in batch])
        end_positions = torch.stack([x.end_positions for x in batch])

        return SquadBatch(
            example_ids=example_ids,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            start_positions=start_positions,
            end_positions=end_positions
        )
```

### Define `SquadBatch`

```python
@dataclasses.dataclass
class SquadBatch:
    example_ids: List[str]
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    start_positions: torch.Tensor
    end_positions: torch.Tensor
```

### Define `SquadDataLoader`

```python
class SquadDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset: SquadDataset, batch_size: int, shuffle: bool = False):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=SquadDataCollator(
                tokenizer=dataset.tokenizer,
                max_seq_length=dataset.max_seq_length
            )
        )
```

### Define `SquadDataModule`

```python
class SquadDataModule(nlp.DataModule):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length

    def setup(self, stage: str = None):
        if stage == "fit" or stage is None:
            self.train_dataset = SquadDataset(
                examples=self.load_squad_examples("train-v2.0.json"),
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length
            )
            self.train_dataloader = SquadDataLoader(
                dataset=self.train_dataset,
                batch_size=self.hparams.train_batch_size,
                shuffle=True
            )

        if stage == "test" or stage is None:
            self.valid_dataset = SquadDataset(
                examples=self.load_squad_examples("dev-v2.0.json"),
                tokenizer=self.tokenizer,
                max_seq_length=self.max_seq_length
            )
            self.valid_dataloader = SquadDataLoader(
                dataset=self.valid_dataset,
                batch_size=self.hparams.valid_batch_size,
                shuffle=False
            )

    def load_squad_examples(self, filename: str) -> List[SquadExample]:
        with open(filename, "r") as f:
            input_data = json.load(f)["data"]

        def is_whitespace(c):
            if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
                return True
            return False

        examples = []
        for entry in tqdm.tqdm(input_data):
            for paragraph in entry["paragraphs"]:
                paragraph_text = paragraph["context"]
                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True
                for c in paragraph_text:
                    if is_whitespace(c):
                        prev_is_whitespace = True
                    else:
                        if prev_is_whitespace:
                            doc_tokens.append(c)
                        else:
                            doc_tokens[-1] += c
                        prev_is_whitespace = False
                    char_to_word_offset.append(len(doc_tokens) - 1)

                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    title = entry["title"]
                    is_impossible = False

                    if is_impossible:
                        continue

                    example = SquadExample(
                        question_text=question_text,
                        context_text=paragraph_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible
                    )

                    examples.append(example)

        return examples
```

## Building Trainer

### Import Libraries

- `transformers`
- `nlp`
- `torch`
- `xla`
- `logging`
- `os`
- `sys`
- `typing`
- `numpy`

### Define `SquadModel`

```python
class SquadModel(transformers.T5ForConditionalGeneration):
    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                encoder_outputs=None, past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                                           inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(embedding_output,
                                       attention_mask=extended_attention_mask,
                                       head_mask=head_mask,
                                       encoder_hidden_states=encoder_hidden_states,
                                       encoder_attention_mask=encoder_attention_mask,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       **kwargs)
        sequence_output = encoder_outputs[0]
        encoder_outputs = encoder_outputs[1:]  # Keep encoder outputs and head_mask

        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]
            decoder_attention_mask = decoder_attention_mask[:, -1:]
        else:
            decoder_input_ids = torch.full((input_shape[0], 1), self.config.decoder_start_token_id, dtype=torch.long,
                                           device=device)
            decoder_attention_mask = torch.ones((input_shape[0], 1), device=device)

        decoder_outputs = self.decoder(decoder_input_ids, encoder_outputs,
                                       attention_mask=decoder_attention_mask,
                                       encoder_decoder_attention_mask=extended_attention_mask,
                                       decoder_attention_mask=decoder_attention_mask,
                                       decoder_cached_states=decoder_cached_states,
                                       use_cache=use_cache,
                                       output_attentions=output_attentions,
                                       output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       **kwargs)
        decoder_outputs = decoder_outputs[0]
        sequence_output = sequence_output[:, -1, :]
        logits = self.lm_head(decoder_outputs)

        if not return_dict:
            return logits, sequence_output, encoder_outputs
        return T5WithLMHeadModelOutput(
            loss=logits,
            logits=logits,
            past_key_values=encoder_outputs,
            decoder_hidden_states=decoder_outputs,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=sequence_output,
            encoder_hidden_states=encoder_outputs,
            encoder_attentions=encoder_attentions,
        )
```

### Define `SquadLoss`

```python
class SquadLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, start_positions, end_positions):
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_positions.clamp_(0, ignored_index)
        end_positions.clamp_(0, ignored_index)

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = (start_loss + end_loss) / 2
        return total_loss
```

### Define `SquadCallback`

```python
class SquadCallback(nlp.Callback):
    def __init__(self, data_loader: SquadDataLoader):
        self.data_loader = data_loader

    def on_validation_end(self, trainer: nlp.Trainer, pl_module: nlp.LightningModule):
        predictions = []
        for batch in self.data_loader:
            input_ids = batch.input_ids.to(trainer.device)
            attention_mask = batch.attention_mask.to(trainer.device)
            token_type_ids = batch.token_type_ids.to(trainer.device)

            with torch.no_grad():
                logits = pl_module(input_ids, attention_mask, token_type_ids)
                start_logits, end_logits = logits.split(1, dim=-1)
                start_logits = start_logits.squeeze(-1)
                end_logits = end_logits.squeeze(-1)

                for i in range(len(batch)):
                    example_id = batch.example_ids[i]
                    start_logits_i = start_logits[i]
                    end_logits_i = end_logits[i]

                    start_pred = torch.argmax(start_logits_i).item()
                    end_pred = torch.argmax(end_logits_i).item()

                    predictions.append(SquadResult(
                        example_id=example_id,
                        start_logits=start_logits_i.tolist(),
                        end_logits=end_logits_i.tolist(),
                        start_pred=start_pred,
                        end_pred=end_pred
                    ))

        output_prediction_file = os.path.join(trainer.logger.experiment.log_dir, "predictions.json")
        output_nbest_file = os.path.join(trainer.logger.experiment.log_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(trainer.logger.experiment.log_dir, "null_odds.json")

        with open(output_prediction_file, "w") as f:
            json.dump(predictions, f)

        with open(output_nbest_file, "w") as f:
            json.dump(predictions, f)

        with open(output_null_log_odds_file, "w") as f:
            json.dump(predictions, f)
```

### Define `SquadTrainer`

```python
class SquadTrainer(nlp.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        token_type_ids = batch.token_type_ids.to(self.device)
        start_positions = batch.start_positions.to(self.device)
        end_positions = batch.end_positions.to(self.device)

        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.loss(logits, start_positions, end_positions)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        token_type_ids = batch.token_type_ids.to(self.device)
        start_positions = batch.start_positions.to(self.device)
        end_positions = batch.end_positions.to(self.device)

        logits = self.model(input_ids, attention_mask, token_type_ids)
        loss = self.loss(logits, start_positions, end_positions)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}
```

## Training

### Define `SquadHparams`

```python
class SquadHparams(nlp.Hparams):
    max_seq_length = 384
    train_batch_size = 8
    valid_batch_size = 8
    learning_rate = 3e-4
    num_train_epochs = 2
```

### Define `SquadModule`

```python
class SquadModule(nlp.LightningModule):
    def __init__(self, hparams: SquadHparams):
        super().__init__()
        self.hparams = hparams

        self.tokenizer = transformers.T5Tokenizer.from_pretrained("t5-base")
        self.model = SquadModel.from_pretrained("t5-base")
        self.loss = SquadLoss()

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None,
                encoder_outputs=None, past_key_values=None, use_cache=None, output_attentions=None,
                output_hidden_states=None, return_dict=None, **kwargs):
        return self.model(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask,
                          encoder_outputs, past_key_values, use_cache, output_attentions,
                          output_hidden_states, return_dict, **kwargs)

    def training_step(self, batch, batch_idx):
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        token_type_ids = batch.token_type_ids.to(self.device)
        start_positions = batch.start_positions.to(self.device)
        end_positions = batch.end_positions.to(self.device)

        logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.loss(logits, start_positions, end_positions)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        input_ids = batch.input_ids.to(self.device)
        attention_mask = batch.attention_mask.to(self.device)
        token_type_ids = batch.token_type_ids.to(self.device)
        start_positions = batch.start_positions.to(self.device)
        end_positions = batch.end_positions.to(self.device)

        logits = self.forward(input_ids, attention_mask, token_type_ids)
        loss = self.loss(logits, start_positions, end_positions)

        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        return [optimizer], [scheduler]

    def train_dataloader(self):
        return self.data_module.train_dataloader

    def val_dataloader(self):
        return self.data_module.valid_dataloader
```

### Define `SquadResult`

```python
@dataclasses.dataclass
class SquadResult:
    example_id: str
    start_logits: List[float]
    end_logits: List[float]
    start_pred: int
    end_pred: int
```

### Define `SquadRunner`

```python
class SquadRunner(nlp.LightningRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup(self, model: SquadModule, data_module: SquadDataModule, hparams: SquadHparams):
        self.model = model
        self.data_module = data_module
        self.hparams = hparams

    def _setup_device(self):
        self.device = xm.xla_device()

    def _setup_dataloaders(self):
        self.data_module.setup(stage="fit")
        self.train_dataloader = self.data_module.train_dataloader
        self.valid_dataloader = self.data_module.valid_dataloader

    def _setup_optimizers(self):
        self.optimizers, self.schedulers = self.model.configure_optimizers()

    def _train_loop_step(self, batch, batch_idx):
        self.optimizers[0].zero_grad()
        loss = self.model.training_step(batch, batch_idx)["loss"]
        loss.backward()
        xm.optimizer_step(self.optimizers[0])
        return loss

    def _validation_loop_step(self, batch, batch_idx):
        loss = self.model.validation_step(batch, batch_idx)["val_loss"]
        return loss

    def _train_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"train_loss": avg_loss}
        return {"avg_train_loss": avg_loss, "log": tensorboard_logs}

    def _validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def _test_loop_step(self, batch, batch_idx):
        return self.model.validation_step(batch, batch_idx)

    def _test_epoch_end(self, outputs):
        return self.model.validation_epoch_end(outputs)

    def _train_dataloader(self):
        return self.train_dataloader

    def _val_dataloader(self):
        return self.valid_dataloader

    def _test_dataloader(self):
        return self.valid_dataloader

    def _setup_checkpoint_callback(self):
        self.checkpoint_callback = nlp.ModelCheckpoint(
            filepath=os.path.join(self.logger.experiment.log_dir, "checkpoints"),
            save_top_k=1,
            verbose=True,
            monitor="avg_val_loss",
            mode="min",
            prefix="",
        )

    def _setup_early_stopping_callback(self):
        self.early_stopping_callback = nlp.EarlyStopping(
            monitor="avg_val_loss",
            patience=3,
            strict=False,
            verbose=False,
            mode="min"
        )

    def _setup_logger(self, name: str = None, version: str = None):
        self.logger = nlp.Logger(
            save_dir=self.save_dir,
            name=name,
            version=version,
            quiet=False
        )

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO)

    def _setup_trainer(self, max_epochs: int = None):
        self.trainer = SquadTrainer(
            default_root_dir=self.save_dir,
            max_epochs=max_epochs,
            gpus=(1 if xm.xla_device().type == "TPU" else -1),
            distributed_backend="ddp" if xm.xla_device().type == "TPU" else "dp",  # Needed to make DDP work on TPU
            early_stop_callback=False if xm.xla_device().type == "TPU" else True,  # Needed to make DDP work on TPU
        )

    def setup(self):
        self._setup_checkpoint_callback()
        self._setup_early_stopping_callback()

    def train(self):
        self._setup()

        if not os.path.exists(os.path.join(self.save_dir, "checkpoints")):
            os.mkdir(os.path.join(self.save_dir, "checkpoints"))

        self._setup_device()

        self._setup_dataloaders()

        optimizers = self._setup_optimizers()

        self._setup_trainer()

        callbacks = [SquadCallback(data_loader=self._val_dataloader())]

        self.trainer.fit(self.model,
                         train_dataloader=self._train_dataloader(),
                         val_dataloaders=self._val_dataloader(),
                         callbacks=callbacks)
```

### Run `SquadRunner`

```bash
export SQUAD_DIR=$HOME/squad2.0/data
export OUTPUT_DIR=$HOME/squad2.0/outputs/t5-base-384-v1-squad20-logger-tpu-1x8-bn
export TPU_NAME=node-1x8
export DATA_DIR=$OUTPUT_DIR/data_dir/$TPU_NAME
export MODEL_DIR=$OUTPUT_DIR/model_dir/$TPU_NAME
python run.py \
    --hparams max_seq_length=384 \
    --hparams train_batch_size=8 \
    --hparams valid_batch_size=8 \
    --hparams learning_rate=3e-4 \
    --hparams num_train_epochs=2 \
    --save-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --squad-dir $SQUAD2.0/data \
    --tpu $TPU_NAME \  # If running on a TPU pod, enter the name of the TPU pod here (you can get this with `kubectl get pods -n <namespace>`)  # noqa: E501 line too long (82 > 79 characters)
```

### Run `SquadInference`

```bash
export SQUAD_DIR=$HOME/squad2.0/data
export OUTPUT_DIR=$HOME/squad2.0/outputs/t5-base-384-v1-squad20-logger-tpu-1x8-bn
export TPU_NAME=node-1x8
export DATA_DIR=$OUTPUT_DIR/data_dir/$TPU_NAME
export MODEL_DIR=$OUTPUT_DIR/model_dir/$TPU_NAME
python predict.py \
    --hparams max_seq_length=384 \
    --hparams train_batch_size=8 \
    --hparams valid_batch_size=8 \
    --hparams learning_rate=3e-4 \
    --hparams num_train_epochs=2 \
    --save-dir $MODEL_DIR \
    --data-dir $DATA_DIR \
    --squad-dir $SQUAD2.0/data \
    --tpu $TPU_NAME \  # If running on a TPU pod, enter the name of the TPU pod here (you can get this with `kubectl get pods -n <namespace>`)  # noqa: E501 line too long (82 > 79 characters)
```
