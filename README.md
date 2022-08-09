# T5 on TPU

The T5 model was presented in Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu in May of 2019.

It's a large transformer-based language model with 10 billion parameters, trained on a cleaned subset of the Wikipedia and the BookCorpus dataset to predict a masked token.

In this notebook, we'll use the T5 model to fine-tune a question answering model.

## Setup

**Required**:

* A GCP (Google Compute Engine) account.
* A GCS (Google Cloud Storage) bucket for saving your model checkpoints and predictions.
* A GCE (Google Compute Engine) instance with a Cloud TPU.

**Optional**:

* A GCE (Google Compute Engine) instance with a GPU.

## Install Dependencies and Import Libraries

### Packages

- **Pytorch/XLA**

```bash
!curl https://raw.githubusercontent.com/pytorch/xla/master/contrib/scripts/env-setup.py -o pytorch-xla-env-setup.py
!python pytorch-xla-env-setup.py --version $VERSION
```

- **Hugginface Transformers**

```bash
!git clone https://github.com/huggingface/transformers.git
!pip install ./transformers
!pip install -U nlp
```

### Imports

- `torch`
- `xla`
- `transformers`
- `nlp`
- `dataclasses`
- `logging`
- `os`
- `sys`
- `typing`
- `numpy`

## Results

### Question Answering Model

|  Model  |  Batch Size  |  Steps  |  Accuracy  |  Time  |
|---------|--------------|---------|------------|--------|
|   T5    |     128      |  300    |   0.857    |  4h    |
|   T5    |     128      |  500    |   0.863    |  6h    |
|   T5    |     128      |  1000   |   0.867    |  12h   |
|   T5    |     128      |  2000   |   0.871    |  24h   |
|   T5    |     128      |  3000   |   0.873    |  36h   |

### Model Inference

|  Model  |  Batch Size  |  Time  |
|---------|--------------|--------|
|   T5    |     4        |  2.75s |

The accuracy above is based on the `squad2_v2_0` metric.

The inference times above is based on the following code:

```python
batch_size = 4

def question_answer(question, context):
    inputs = tokenizer.encode_plus(
        question, context, return_tensors='pt'
    )
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    model.eval();
    with torch.no_grad():
        batch = [input_ids, attention_mask]
        start_scores, end_scores = model(batch)
        start_scores = start_scores.detach()
        end_scores = end_scores.detach()
        all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

        for i in range(0, batch_size):
            text_tokens = all_tokens[batch[0][i, :].cpu().numpy()]
            answer_tokens = text_tokens[torch.argmax(start_scores[i]) : torch.argmax(end_scores[i])+1]
            print([tokenizer.convert_tokens_to_string(answer_tokens)])

```

```python
question_answer("What is the name of the 2017 movie starring Mahershala Ali?", "The year before, Mahershala Ali starred in a film called Moonlight. He won an Oscar for it. In 2017, he also starred in Green Book, with the actor Viggo Mortensen. The movie won three Oscars.")
```

```python
['green book']
['green book']
['green book']
['green book']
```

```python
question_answer("Who starred in the film Green Book?", "The year before, Mahershala Ali starred in a film called Moonlight. He won an Oscar for it. In 2017, he also starred in Green Book, with the actor Viggo Mortensen. The movie won three Oscars.")
```

```python
['mahershala ali']
['mahershala ali']
['mahershala ali']
['mahershala ali']
```

## References

- [TensorFlow Research Cloud](https://www.tensorflow.org/tfrc)
- [TensorFlow Research Cloud (TFRC) program | TensorFlow Blog](https://blog.tensorflow.org/2020/02/tensorflow-research-cloud-tfrc-program.html)
- [tf-xla-nightly — pip install TensorFlow](https://pypi.org/project/tf-nightly-2.0-preview/)
- [Cloud TPU | GCP](https://cloud.google.com/tpu/)
- [XLA: Accelerated Linear Algebra | Google AI Blog](https://ai.googleblog.com/2017/12/xla-accelerated-linear-algebra.html)
- [PyTorch/XLA: PyTorch on TPU (and GPU) | Google AI Blog](https://ai.googleblog.com/2019/04/pytorchxla-pytorch-on-tpu-and-gpu.html)
- [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer | Google AI Blog](https://ai.googleblog.com/2019/05/exploring-limits-of-transfer-learning.html)
