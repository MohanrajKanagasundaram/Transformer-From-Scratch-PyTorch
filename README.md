# Transformer From Scratch — PyTorch

A from-scratch implementation of a Transformer in PyTorch, built while studying how Transformers work internally.

The goal of this project is not to build a production-ready Transformer, but to understand the architecture by implementing the components myself and running experiments around training, attention, normalization, depth, width, and regularization.

---

## What I am learning

* Transformer architecture
* Self-attention
* Multi-head attention
* Causal / masked attention
* Encoder-decoder architecture
* Decoder-only architecture
* Positional embeddings
* Autoregressive training and inference
* Teacher forcing
* Layer normalization
* Dropout
* Gradient flow
* Model depth vs width
* Pre-LN vs Post-LN

---

# Experiments

## 1. Experiment: Depth — 2 vs 4 vs 8 vs 12 Layers

### Setup

The goal was to understand how increasing the number of decoder layers affects training and validation performance.

* Same dataset
* Same model configuration
* Same training setup
* Only the number of decoder layers changes
* Compared 2, 4, 8 and 12 layers

### What I observed

The gradient norms changed as the model became deeper.

The 2-layer model showed higher gradient norms. This does not necessarily mean that it is better.

The 4-layer model had lower gradient norms, and the 8-layer model had lower gradient norms compared with both the 2-layer and 4-layer models.

However, the models remained training-stable. I did not observe gradients exploding midway through training.

### Validation Loss

The 4-layer model achieved a better validation loss than the 2-layer model.

My interpretation was that the 2-layer model has less representational capacity, while adding depth allows the model to learn more complex representations.

The 8-layer model showed further improvement in learning compared with the shallower models.

### Initial takeaway

Depth appears to provide additional representational capacity, but increasing depth also makes gradient flow more important.

I want to investigate the depth/width trade-off further instead of assuming that simply making the model deeper is always better.

---

# 2. Depth vs Width

One question I wanted to understand was:

> Is it better to increase model depth or model width?

Increasing depth means gradients have to flow through more layers, which can make optimization more difficult in deeper networks.

Increasing width increases the size of the matrix multiplications and therefore can increase training compute, but it also increases the model's representational capacity.

My current understanding:

* **More depth** → more sequential transformations and potentially better hierarchical representations
* **More width** → larger representations and more computation per layer
* Both increase model capacity, but they have different effects on optimization and compute

This is something I want to experiment with more systematically.

---

# 3. Experiment: Pre-LN vs Post-LN

I experimented with three configurations in a 12-layer decoder-only Transformer:

* No LayerNorm
* Post-LN
* Pre-LN

The experiment was run for only 5 epochs.

### Results

| Experiment | Best Val Loss | Improvement % |    Tok/s |  GPU MB | Params (M) |
| ---------- | ------------: | ------------: | -------: | ------: | ---------: |
| No LN      |        4.3102 |             — | 23131.84 | 2646.98 |       21.3 |
| Post-LN    |        4.0537 |         5.95% | 22348.16 | 2744.93 |       21.3 |
| Pre-LN     |        4.0184 |         0.87% | 22410.02 | 2745.25 |       21.3 |

### Validation Loss Improvements

```text
No LN → Post-LN
4.3102 → 4.0537
Improvement: 5.95%

Post-LN → Pre-LN
4.0537 → 4.0184
Improvement: 0.87%
```

### What I observed

Pre-LN converged faster than Post-LN in my experiment.

My current intuition is that Pre-LN provides a more direct path for gradients through the residual connections, which can make optimization easier, especially as the network gets deeper.

In Post-LN, LayerNorm is applied after the residual addition, so the gradient path is different and can make optimization more sensitive in deeper networks.

I also found that learning-rate warmup may be important for Post-LN training and want to investigate this further.

---

# 4. Multi-Head Attention

I implemented the multi-head attention logic from scratch.

The main idea I am trying to understand is why multiple attention heads are useful.

My current understanding is that each head can learn different attention patterns during training.

For example, different heads might learn to focus on different relationships between tokens.

These are **not hardcoded features**. The attention patterns are learned during training.

The original Transformer used multiple heads, and I am currently working on understanding the intuition behind why splitting the representation into multiple heads works better than using a single attention operation.

### Implementation

I initially implemented the multi-head attention logic without the batch dimension.

This helped me understand the PyTorch tensor dimensions and how the matrix operations work.

Next steps:

* Implement the batched version
* Better understand why `Attention @ V` produces the final representation
* Trace every tensor dimension through the operation

---

# 5. Why is Softmax Applied Over the Key Dimension?

One thing I initially did not understand was:

> Why do we apply softmax over the keys rather than the values?

For a given query token, the attention scores represent how important each key is.

Applying softmax over the keys gives us a distribution of attention weights across the available tokens.

So softmax is doing more than just normalization:

1. Converts attention scores into a probability-like distribution
2. Gives a relative weight to each key
3. Creates competition between keys — if one key receives more attention, the others receive less

This helped me understand attention as:

> For this query token, how much should I use information from each other token?

---

# 6. Autoregressive Training and Inference

I am currently working through how autoregressive Transformers actually generate tokens.

Some areas I initially found confusing:

* Teacher forcing
* Validation loops
* Predicting tokens one at a time
* Decoder input/output shifting
* Causal masking
* Positional embeddings

The training loop itself is becoming clearer, but I am still working on building a strong mental model of how training differs from inference.

### Teacher Forcing

My current understanding is that during training, the model can receive the ground-truth previous tokens rather than having to generate the entire sequence one token at a time.

During inference, however, the model generates autoregressively:

```text
Input
  ↓
Predict next token
  ↓
Append predicted token
  ↓
Predict next token
  ↓
Append predicted token
  ↓
...
```

I am continuing to investigate the exact relationship between teacher forcing during training and autoregressive generation during inference.

---

# 7. Positional Embeddings

I am also studying sinusoidal positional embeddings.

The main question I am trying to understand is:

> How does the Transformer know where a token occurs in the sequence when attention itself does not inherently contain sequence order?

I am working through the intuition behind the sine/cosine formulation rather than treating it as just something to implement.

---

# 8. Dropout

Dropout is a regularization technique where some activations are randomly set to zero during training.

The remaining activations are scaled so that the expected magnitude stays consistent.

My current understanding is that dropout introduces noise during training and can help prevent the model from relying too heavily on particular neurons or representations, reducing overfitting.

---

# What I want to implement next

* [ ] Batched multi-head attention
* [ ] Encoder-decoder architecture
* [ ] Masked self-attention
* [ ] Complete autoregressive training loop
* [ ] Teacher forcing
* [ ] Autoregressive validation/inference
* [ ] Sinusoidal positional embeddings
* [ ] More systematic depth experiments
* [ ] Depth vs width experiments
* [ ] Learning-rate warmup experiments
* [ ] Better attention visualization/instrumentation

---

# Why I am building this

I wanted to go beyond using Transformer libraries and actually understand what is happening inside the model.

The project is intentionally experimental. Some sections represent things I understand well, while others represent concepts I am still investigating.

The goal is to learn by implementing, measuring, debugging, and experimenting rather than treating the Transformer as a black box.

