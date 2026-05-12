# Transfer-Ready EEGNet

## 1. Why build EEGNet this way?

 The original `EEGNet` in this repository was structured as a **single-task EEG classifier**: it took an EEG window as input and directly returned class predictions for one fixed label space.

That design is fine when the goal is only to solve one dataset-specific classification problem, but it becomes limiting as soon as we want to do any of the following:

- reuse a trained model on a new EEG task,
- keep the learned EEG representation while changing the label space,
- extract features for downstream classifiers or analysis,
- adapt the model to a new dataset without retraining everything from scratch.

For transfer learning, the more useful mental model is not “one model = one task,” but rather:

> **EEG backbone + task-specific head**

In that design:

- the **backbone** learns general EEG representations,
- the **head** maps those representations to the output required by one specific task.

This separation gives us a much cleaner workflow:

1. train the backbone on a source task,
2. keep the backbone weights,
3. replace the head for a new task,
4. fine-tune only what is necessary.

That is the main reason `EEGNet` was refactored into a transfer-ready version.

---

## 2. What problem does the refactor solve?

Before the refactor, the model logic was effectively:

```text
input EEG -> convolution blocks -> flatten -> dense -> softmax
```

This creates three practical problems.

### 2.1 The model head was tied to one task

The final linear layer was defined using:

- `nb_classes`: number of target classes,
- `Samples`: number of time points after temporal reduction.

That means the output layer was tightly coupled to the current task definition.

### 2.2 Feature extraction was not a first-class API

Even though the model clearly learned intermediate EEG features, there was no explicit interface for getting them. If someone wanted to use EEGNet as a feature extractor, they had to manually edit the model or bypass the last layer in ad hoc ways.

### 2.3 Training semantics were not ideal for PyTorch

The old `forward()` returned softmax probabilities. In standard PyTorch training, `nn.CrossEntropyLoss()` expects **logits**, not post-softmax probabilities. Returning logits in `forward()` makes the training interface more standard and more flexible.

---

## 3. The design idea: backbone and head

The transfer-ready version of EEGNet separates the model into two conceptual parts.

### 3.1 Backbone

The backbone includes:

- temporal convolution,
- depthwise spatial convolution,
- separable convolution,
- pooling,
- dropout,
- flattening into a feature vector.

Its job is to transform raw EEG windows into a compact feature representation.

### 3.2 Classifier head

The classifier head is just the last task-specific mapping:

```text
features -> classifier -> logits
```

Because this part is isolated, it can be replaced when the new task has a different number of output classes.

---

## 4. Input and output conventions

### 4.1 Input

The transfer-ready `EEGNet` accepts EEG windows in either of these formats:

- `(batch, Chans, Samples, 1)`
- `(batch, Chans, Samples)`

Internally, the model converts them into the PyTorch convolution format:

- `(batch, 1, Chans, Samples)`

This means one input example is still:

> a multi-channel EEG time window with `Chans` electrodes and `Samples` time points.

### 4.2 Output

The transfer-ready version exposes **three** meaningful outputs depending on the API you call.

#### `forward(x)`

Returns:

- **logits** of shape `(batch, nb_classes)`

Use this for training with `nn.CrossEntropyLoss()`.

#### `predict_proba(x)`

Returns:

- **probabilities** of shape `(batch, nb_classes)`

Use this for inference when you explicitly want normalized class probabilities.

#### `forward_features(x)`

Returns:

- **flattened feature vectors** of shape `(batch, feature_dim)`

Use this when EEGNet is acting as a representation learner rather than a fixed classifier.

---

## 5. How the transfer-ready EEGNet is built

The refactored implementation follows this structure.

### Step 1: Keep the original EEGNet convolution blocks

We preserve the two-block EEGNet structure because it already captures the main inductive biases of the architecture:

- temporal filtering,
- spatial filtering across channels,
- compact feature fusion.

This keeps the backbone behavior consistent with the original model.

### Step 2: Add an explicit input preparation step

The helper method `_prepare_input(x)` standardizes supported input layouts.

Its job is simple:

- if input is `(B, C, T, 1)`, permute it to `(B, 1, C, T)`;
- if input is `(B, C, T)`, unsqueeze it to `(B, 1, C, T)`.

This avoids duplicating shape-handling logic across the model.

### Step 3: Define `forward_features(x)`

This method runs the EEG window through the backbone only:

```text
input -> block1 -> block2 -> flatten
```

It returns the feature vector before classification.

This is the most important addition for transfer learning, because it makes feature extraction an official part of the model interface.

### Step 4: Isolate the classifier head

Instead of treating the last linear layer as a hidden implementation detail, we make it explicit:

```python
self.classifier = nn.Linear(self.feature_dim, nb_classes)
```

and then use:

```python
def forward_classifier(self, features):
    return self.classifier(features)
```

This keeps the last task-specific mapping replaceable.

### Step 5: Make `forward(x)` return logits

The new `forward()` is intentionally simple:

```text
features = forward_features(x)
logits = forward_classifier(features)
return logits
```

This is the standard PyTorch design for classification.

### Step 6: Add `predict_proba(x)` for inference

Because users sometimes want probabilities directly, `predict_proba(x)` was added as a convenience wrapper:

```text
logits -> softmax -> probabilities
```

This keeps training and inference semantics cleanly separated.

### Step 7: Add `reset_classifier(nb_classes)`

This method rebuilds the task head for a new output space while keeping the backbone intact.

That gives a clean transfer workflow:

1. load a pretrained EEGNet,
2. keep the backbone weights,
3. replace the head with a new number of classes,
4. train on the target task.

The implementation also preserves the current head’s device and dtype, so replacing the classifier after moving the model to GPU or a different dtype does not create mismatched tensors.

---

## 6. What exactly changed in the model API?

The transfer-ready version introduces these methods:

### `forward(x)`

- returns logits
- preferred for training

### `forward_features(x)`

- returns backbone features
- preferred for feature extraction and transfer learning

### `forward_classifier(features)`

- maps feature vectors to task logits
- useful when features are already computed

### `predict_proba(x)`

- returns probabilities
- preferred for explicit inference workflows

### `reset_classifier(nb_classes)`

- replaces the task head
- preferred when adapting to a new task label space

---

## 7. How to use the model

### 7.1 Standard supervised training

```python
import torch
import torch.nn as nn
from EEGModels_PyTorch import EEGNet

model = EEGNet(nb_classes=4, Chans=64, Samples=128)
criterion = nn.CrossEntropyLoss()

x = torch.randn(8, 64, 128, 1)
y = torch.randint(0, 4, (8,))

logits = model(x)
loss = criterion(logits, y)
loss.backward()
```

Here, `model(x)` returns logits, which is what `CrossEntropyLoss` expects.

### 7.2 Get EEG features directly

```python
features = model.forward_features(x)
print(features.shape)
```

This is useful when:

- you want to train another classifier on top of EEGNet features,
- you want to visualize learned representations,
- you want to use EEGNet as a generic encoder.

### 7.3 Get class probabilities for inference

```python
probs = model.predict_proba(x)
preds = torch.argmax(probs, dim=1)
```

This is useful when you need confidence-like outputs or probability distributions.

### 7.4 Transfer to a new classification task

```python
model = EEGNet(nb_classes=4, Chans=64, Samples=128)

# load pretrained weights here if available
# model.load_state_dict(...)

# replace the head for a new 6-class task
model.reset_classifier(nb_classes=6)
```

Now the model keeps the backbone but uses a new classifier head.

---

## 8. Recommended transfer-learning workflow

The typical workflow is:

### Option A: linear probing

Freeze the backbone and only train the new head.

```python
for name, param in model.named_parameters():
    if not name.startswith("classifier") and not name.startswith("dense"):
        param.requires_grad = False

model.reset_classifier(nb_classes=6)
```

This tests whether the pretrained EEG representation is already useful for the new task.

### Option B: head-first, then fine-tune

1. replace the head,
2. train only the head for a few epochs,
3. unfreeze some or all backbone layers,
4. fine-tune the full model.

This is often a better choice when the source and target tasks are related but not identical.

---

## 9. What still depends on the task?

Even after the refactor, some parts of EEGNet are still tied to the dataset definition.

### 9.1 Number of classes

This is handled by the classifier head and can now be changed cleanly with `reset_classifier()`.

### 9.2 Number of channels

The spatial depthwise convolution depends on `Chans`, because it spans the channel axis.

That means if the target dataset has different electrodes or a different electrode ordering, you may need to:

- restrict both datasets to a common channel set,
- reorder channels consistently,
- or rebuild spatial layers if the channel structure is different enough.

### 9.3 Number of time samples

The feature dimension depends on the temporal reduction path and therefore on `Samples`.

If the target task uses a different window length, the head may need rebuilding, and the temporal hyperparameters may also need adjustment.

### 9.4 Sampling rate

If sampling rate changes substantially, `kernLength` and pooling settings may also need to change, because EEGNet’s temporal filters are tied to the time scale of the input.

---

## 10. Why this design is a good compromise

This refactor was intentionally minimal.

It does **not** try to redesign the whole repository or create an abstract training framework. Instead, it adds the smallest useful transfer-learning surface on top of the existing EEGNet implementation.

That is a good compromise because it:

- preserves the familiar EEGNet architecture,
- keeps the original code easy to read,
- improves correctness for PyTorch training,
- exposes reusable EEG features,
- makes downstream task adaptation much cleaner.

In other words, the goal was not to make EEGNet more complicated.

The goal was to make its learned EEG representation **reusable**.

---

## 11. Summary

The transfer-ready EEGNet is built around one core idea:

> separate **representation learning** from **task-specific prediction**.

That is why the model now provides:

- `forward()` for logits,
- `predict_proba()` for probabilities,
- `forward_features()` for reusable EEG features,
- `reset_classifier()` for transferring to new tasks.

This makes the model easier to train correctly, easier to analyze, and much easier to adapt to new EEG classification problems.
