# Feed‑Forward Neural Network (FNN/MLP) on **Fashion‑MNIST**

Fashion‑MNIST = 28×28 grayscale images (pixel values 0–255) classified into **10 classes**: 0 T‑shirt/top, 1 Trouser, 2 Pullover, 3 Dress, 4 Coat, 5 Sandal, 6 Shirt, 7 Sneaker, 8 Bag, 9 Ankle boot.

---

## 1) What an FNN/MLP is

A **Feed‑Forward Neural Network** (also called **MLP**) is a stack of **fully‑connected (Dense) layers**.

- “Feed‑forward” means data flows **only forward** (no loops like RNN).
- A “Dense/Fully‑Connected” layer means **every input connects to every neuron** in the next layer.

### Layer equation (core formula)

For one layer:

- Input activations: **a** (vector)
- Weights: **W** (matrix)
- Bias: **b** (vector)
- Pre‑activation: **z**
- Activation function: **g(·)**

$$
\boxed{z = aW + b}\quad \text{and}\quad \boxed{a_{next} = g(z)}
$$

> Shapes example: if a has shape (1×784) and the next layer has 128 neurons, then W is (784×128), b is (1×128), z is (1×128).

---

## 2) Fashion‑MNIST data flow (end‑to‑end)

### Step A — Load

- Training set: **60,000** images
- Test set: **10,000** images

### Step B — Preprocess (most common)

1. **Flatten** each image from 28×28 to a vector of length **784**.
2. **Normalize** pixels: $x_i = \frac{pixel_i}{255}$ so $x_i\in[0,1]$.
3. Label $y$ is an integer 0–9.
4. For math, we often convert label to **one‑hot** vector (length 10):
   - If true label is 1 → $y=[0,1,0,0,0,0,0,0,0,0]$

---

## 3) Typical MLP architecture for Fashion‑MNIST

Example architecture (very common):

- Input: 784
- Hidden1: 128 (ReLU)
- Hidden2: 64 (ReLU)
- Output: 10 (Softmax)

### Layer diagram (with shapes)

```text
x (1×784)  ──Dense(W1:784×128,b1:128)──>  z1 (1×128) ─ReLU─> a1 (1×128)
                 |
                 v
             Dense(W2:128×64,b2:64)──>  z2 (1×64)  ─ReLU─> a2 (1×64)
                 |
                 v
             Dense(W3:64×10,b3:10)──>  z3 (1×10)  ─Softmax─> p (1×10)

p = [p0, p1, …, p9] = predicted class probabilities
predicted class = argmax(p)
```

---

## 4) Forward pass (step‑by‑step math)

Assume one sample $x$ (flattened + normalized) and label $y$.

### Hidden layer 1

$$
z_1 = xW_1 + b_1\qquad a_1 = \text{ReLU}(z_1)
$$

ReLU definition:

$$
\text{ReLU}(t)=\max(0,t)
$$

### Hidden layer 2

$$
z_2 = a_1W_2 + b_2\qquad a_2 = \text{ReLU}(z_2)
$$

### Output layer (logits → probabilities)

Logits (raw scores):

$$
z_3 = a_2W_3 + b_3
$$

Softmax:

$$
\boxed{p_k = \frac{e^{z_{3k}}}{\sum_{j=0}^{9} e^{z_{3j}}}}\quad (k=0..9)
$$

---

## 5) Loss function (Cross‑Entropy)

For one sample with one‑hot label $y$ and predicted probabilities $p$:

$$
\boxed{\mathcal{L} = -\sum_{k=0}^{9} y_k\log(p_k)}
$$

If the true class is $c$, then only $y_c=1$:

$$
\boxed{\mathcal{L} = -\log(p_c)}
$$

---

## 6) Backpropagation (how training learns)

Goal: adjust weights/biases to reduce loss.

### Key terms

- **Gradient**: how much loss changes if a parameter changes.
- **Learning rate (η)**: step size for updates.
- **Gradient Descent update**:

$$
\boxed{W \leftarrow W - \eta\,\frac{\partial\mathcal{L}}{\partial W}},\qquad
\boxed{b \leftarrow b - \eta\,\frac{\partial\mathcal{L}}{\partial b}}
$$

### Very important simplification (Softmax + Cross‑Entropy)

If $p=\text{softmax}(z_3)$, then:

$$
\boxed{\frac{\partial\mathcal{L}}{\partial z_3} = p - y}
$$

That is the starting point for backprop.

### Backprop steps (vector/matrix form)

Let $\boxed{\delta_3 = \frac{\partial\mathcal{L}}{\partial z_3} = p-y}$.

**Output layer (Layer 3):**

$$
\boxed{\frac{\partial\mathcal{L}}{\partial W_3} = a_2^T\delta_3}\qquad
\boxed{\frac{\partial\mathcal{L}}{\partial b_3} = \delta_3}
$$

$$
\boxed{\delta_2^{(act)} = \delta_3 W_3^T}
$$

**ReLU gate (Layer 2):**

$$
\boxed{\delta_2 = \delta_2^{(act)} \odot \text{ReLU}'(z_2)}
$$

Where $\odot$ is element‑wise multiply and

$$
\text{ReLU}'(t)=\begin{cases}1,&t>0\\0,&t\le0\end{cases}
$$

**Hidden layer 2 weights:**

$$
\boxed{\frac{\partial\mathcal{L}}{\partial W_2} = a_1^T\delta_2}\qquad
\boxed{\frac{\partial\mathcal{L}}{\partial b_2} = \delta_2}
$$

Propagate further:

$$
\delta_1^{(act)} = \delta_2 W_2^T\qquad
\delta_1 = \delta_1^{(act)} \odot \text{ReLU}'(z_1)
$$

**Hidden layer 1 weights:**

$$
\boxed{\frac{\partial\mathcal{L}}{\partial W_1} = x^T\delta_1}\qquad
\boxed{\frac{\partial\mathcal{L}}{\partial b_1} = \delta_1}
$$

> In mini‑batch training (batch size B), you compute these gradients for each sample and then **average** them before the update.

---

## 7) One complete numeric example (single sample)

Doing all 784 inputs by hand is huge, so here’s the **exact same process** with **4 selected pixels** (think of them as 4 positions chosen from the 784). The math scales identically to 784.

### Example setup

- Input (4 pixels after normalization):

$$
\boxed{x=[0.0,\;0.2,\;0.9,\;0.4]}
$$

- One hidden layer with 3 neurons (ReLU)
- Output layer with 10 neurons (Softmax)
- True label: class **1** → $y=[0,1,0,0,0,0,0,0,0,0]$

### Layer 1 parameters (4 → 3)

$$
W_1=
\begin{bmatrix}
0.1 & -0.2 & 0.0\\
0.0 & 0.1 & 0.3\\
0.2 & 0.0 & -0.1\\
-0.1 & 0.2 & 0.1
\end{bmatrix},
\quad
b_1=[0.0,\;0.1,\;-0.1]
$$

Compute each component:

- $z_{11} = 0.0(0.1)+0.2(0.0)+0.9(0.2)+0.4(-0.1)+0.0 = 0.18-0.04=\boxed{0.14}$
- $z_{12} = 0.0(-0.2)+0.2(0.1)+0.9(0.0)+0.4(0.2)+0.1 = 0.02+0.08+0.1=\boxed{0.20}$
- $z_{13} = 0.0(0.0)+0.2(0.3)+0.9(-0.1)+0.4(0.1)-0.1 = 0.06-0.09+0.04-0.1=\boxed{-0.09}$

So:

$$
\boxed{z_1=[0.14,\;0.20,\;-0.09]}
$$

$$
\boxed{a_1=[0.14,\;0.20,\;0]}
$$

### Layer 2 parameters (3 → 10)

Bias $b_2=0$ for simplicity. Columns of $W_2$ (each column is one class weight vector):

- class0:  [0.1,  0.0, 0]
- class1:  [0.0,  0.1, 0]
- class2:  [-0.1, 0.1, 0]
- class3:  [0.05,-0.05,0]
- class4:  [0.2,  0.1, 0]
- class5:  [-0.2, 0.0, 0]
- class6:  [0.0, -0.1, 0]
- class7:  [0.1,  0.1, 0]
- class8:  [0.0,  0.0, 0]
- class9:  [-0.05,0.05,0]

Using dot products:

- $z_{20}=0.14(0.1)+0.20(0.0)=\boxed{0.014}$
- $z_{21}=0.14(0.0)+0.20(0.1)=\boxed{0.020}$
- $z_{22}=0.14(-0.1)+0.20(0.1)=\boxed{0.006}$
- $z_{23}=0.14(0.05)+0.20(-0.05)=\boxed{-0.003}$
- $z_{24}=0.14(0.2)+0.20(0.1)=\boxed{0.048}$
- $z_{25}=0.14(-0.2)+0.20(0)=\boxed{-0.028}$
- $z_{26}=0.14(0)+0.20(-0.1)=\boxed{-0.020}$
- $z_{27}=0.14(0.1)+0.20(0.1)=\boxed{0.034}$
- $z_{28}=0$
- $z_{29}=0.14(-0.05)+0.20(0.05)=\boxed{0.003}$

So:

$$
\boxed{z_2=[0.014,\;0.020,\;0.006,\;-0.003,\;0.048,\;-0.028,\;-0.020,\;0.034,\;0,\;0.003]}
$$

$$
\boxed{p\approx[0.1006,\;0.1012,\;0.0998,\;0.0989,\;0.1041,\;0.0965,\;0.0973,\;0.1027,\;0.0992,\;0.0995]}
$$

Highest is class4 (\~0.1041) → prediction would be **4**, but true is **1**.

### Step 5 — Cross‑entropy loss

True class is 1, so:

$$
\mathcal{L}=-\log(p_1)=-\log(0.1012)\approx\boxed{2.29}
$$

---

## 8) Backprop for the same numeric example

### Step 6 — Gradient at logits (Softmax + CE)

$$
\boxed{\delta_2 = \frac{\partial\mathcal{L}}{\partial z_2} = p-y}
$$

So the vector $\delta_2$ is:

- for class1: $\delta_{21}=p_1-1\approx -0.8988$
- for others: $\delta_{2k}=p_k$

Numerically:

$$
\delta_2\approx[0.1006,\;-0.8988,\;0.0998,\;0.0989,\;0.1041,\;0.0965,\;0.0973,\;0.1027,\;0.0992,\;0.0995]
$$

### Step 7 — Gradients for output weights/bias

$$
\boxed{\frac{\partial\mathcal{L}}{\partial W_2}=a_1^T\delta_2}\qquad
\boxed{\frac{\partial\mathcal{L}}{\partial b_2}=\delta_2}
$$

Because $a_1=[0.14,0.20,0]$:

- Row1 of $dW_2$ = $0.14\cdot \delta_2$
- Row2 of $dW_2$ = $0.20\cdot \delta_2$
- Row3 of $dW_2$ = $0\cdot \delta_2 = 0$

Example entries:

- $\frac{\partial\mathcal{L}}{\partial W_{2,(row2,col1)}} = 0.20\cdot(-0.8988)=\boxed{-0.1798}$
- $\frac{\partial\mathcal{L}}{\partial W_{2,(row1,col4)}} = 0.14\cdot(0.1041)=\boxed{0.0146}$

### Step 8 — Backprop to hidden activations

$$
\boxed{\delta_1^{(act)} = \delta_2 W_2^T}
$$

Numerically (from this example):

$$
\delta_1^{(act)}\approx[\;0.01184,\;-0.06891,\;0\;]
$$

### Step 9 — ReLU gate at hidden layer

$z_1=[0.14,0.20,-0.09]$ so ReLU'(z1) = [1, 1, 0].

$$
\boxed{\delta_1 = \delta_1^{(act)}\odot[1,1,0]=[0.01184,\;-0.06891,\;0]}
$$

### Step 10 — Gradients for first layer

$$
\boxed{\frac{\partial\mathcal{L}}{\partial W_1}=x^T\delta_1}\qquad
\boxed{\frac{\partial\mathcal{L}}{\partial b_1}=\delta_1}
$$

Because $x=[0,0.2,0.9,0.4]$, each row i of $dW_1$ is $x_i\cdot \delta_1$.

Example rows:

- Row for x2=0.2 → $[0.2\cdot0.01184,\;0.2\cdot(-0.06891),\;0]=[\boxed{0.00237},\;\boxed{-0.01378},\;0]$
- Row for x3=0.9 → $[\boxed{0.01066},\;\boxed{-0.06202},\;0]$

---

## 9) Parameter update (one step)

Pick learning rate $\eta=0.1$.

Example update for one weight in output layer:

- Suppose the weight from hidden neuron2 → class1 is $W_{2,(row2,col1)}$.
- Gradient computed: $dW= -0.1798$.

Update:

$$
W_{new}=W_{old}-\eta\,dW = W_{old}-0.1(-0.1798)=W_{old}+\boxed{0.01798}
$$

So that weight increases, pushing class1 score upward next time.

> With mini‑batch training, you average gradients over B samples before updating.

---

## 10) Training terms (quick, precise)

- **Epoch**: one full pass over all training images.
- **Batch size (B)**: how many samples used to compute one gradient update.
- **Iteration/Step**: one parameter update (one batch).
- **Learning rate (η)**: update step size.
- **Initialization**: how W is initialized (e.g., He init for ReLU).
- **Optimizer**: variant of gradient descent (SGD, Adam, RMSProp).
- **Overfitting**: train accuracy high, test/val accuracy low.
- **Regularization**: techniques to reduce overfitting (Dropout, L2/weight decay, early stopping).

---

## 11) Complete process summary (what happens every training step)

```text
1) Take a batch of images
2) Flatten + normalize → x
3) Forward pass → logits z → probabilities p
4) Compute loss (cross-entropy)
5) Backprop → gradients dW, db
6) Update parameters with optimizer
7) Repeat for all batches (epoch), repeat for many epochs
```

---

If you want, I can also write the **same full derivation** for a 2‑hidden‑layer MLP (784→128→64→10) with exact matrix shapes at every step (forward + backward), or provide a clean “from‑scratch” pseudo‑code training loop (no libraries).

## 12) Why we chose Dense, why these neuron counts, and why ReLU then Softmax

### Why use a Dense (Fully-Connected) layer?

We use a Dense layer because, after flattening a 28×28 image into a 784-length vector, we want the model to learn combinations of all input pixels together.

- Each neuron can look at every input feature.
- This helps the model learn useful mixtures of pixels such as edges, rough shapes, texture cues, and class-specific patterns.
- In an MLP, Dense layers are the standard building block because the input is a feature vector and the layer learns a weighted transformation of that vector.

### What are the “parameters” in a Dense layer?

A Dense layer has trainable parameters:

- Weights (W)
- Biases (b)

If a layer goes from n\_in inputs to n\_out neurons, then:

Parameters = (n\_in × n\_out) + n\_out

Examples for 784 → 128 → 64 → 10:

- 784 → 128: (784 × 128) + 128 = 100480
- 128 → 64: (128 × 64) + 64 = 8256
- 64 → 10: (64 × 10) + 10 = 650

Total trainable parameters = 100480 + 8256 + 650 = 109386

So when we say Dense(128), it means:

- this layer has 128 neurons,
- each neuron has its own weights and one bias,
- the exact parameter count depends on the previous layer size.

---

### Why choose 128 and 64 neurons?

These are not fixed rules. They are hyperparameters chosen by us. We choose them to balance:

- model capacity,
- speed,
- memory use,
- overfitting risk.

Why not too small?

- If we use very few neurons, the model may be too weak to learn enough patterns from 784 pixels.
- That leads to underfitting.

Why not too large?

- If we use too many neurons, parameter count grows a lot.
- Training becomes slower.
- Memory use increases.
- Overfitting becomes more likely.

Why 128 then 64?

- The first hidden layer is usually larger so it can learn a richer first representation.
- The second hidden layer is often smaller so it compresses those features into more class-useful patterns.
- Then the output layer has 10 neurons because Fashion-MNIST has 10 classes.

That gives a useful funnel structure: 784 → 128 → 64 → 10

So 128 and 64 are practical starting values, not magic values.

---

### Why use ReLU in hidden layers?

ReLU means: ReLU(x) = max(0, x)

We use ReLU in hidden layers because:

1. It adds non-linearity

    - Without an activation function, stacked Dense layers are still just one big linear transformation.
    - Then the network cannot learn complex patterns.
    - ReLU makes the network able to learn non-linear decision boundaries.

2. It is simple and fast

    - It only compares the value with 0.
    - It is computationally cheaper than sigmoid or tanh.

3. It helps gradient flow

    - For positive inputs, its derivative is 1.
    - This reduces the vanishing gradient problem compared with sigmoid.

4. It creates sparse activations

    - Negative values become 0.
    - This often helps the model focus on more useful signals.

Does ReLU have parameters?

- Standard ReLU has no trainable parameters.
- It is only an activation rule.
- If we used Leaky ReLU, then there would be a small negative-side slope parameter.

---

### Why use Softmax in the output layer?

Fashion-MNIST is a multi-class classification problem with 10 mutually exclusive classes. That means one image should belong to exactly one class.

So we want:

- one score for each class,
- then those scores converted into probabilities,
- and all probabilities should add up to 1.

That is exactly what Softmax does. It converts the final logits into a probability distribution over the 10 classes.

Why Softmax is correct here:

- Output values are between 0 and 1.
- All class probabilities sum to 1.
- The largest probability tells us the predicted class.
- It works naturally with cross-entropy loss.

Why not ReLU in the output layer?

- ReLU outputs are not proper probabilities.
- They do not sum to 1.
- Cross-entropy with one-hot labels would not work in the usual classification way.

Does Softmax have parameters?

- No. Softmax has no trainable parameters.
- It only transforms the final logits into normalized probabilities.
- The trainable parameters are in the Dense layer before it.

---

### What does “with parameter” mean for each layer?

This is the key difference:

1. Dense layer

    - Has trainable parameters: weights and biases

2. ReLU

    - Usually has no trainable parameters
    - It is just an activation function

3. Softmax

    - Has no trainable parameters
    - It is just a normalization function over the final logits

So in this network: Dense(128) → ReLU → Dense(64) → ReLU → Dense(10) → Softmax

- only the Dense layers have trainable parameters,
- ReLU and Softmax do not learn weights.

---

### Why this exact order: Dense → ReLU → Dense → ReLU → Dense → Softmax?

Because each part has a different job:

1. Dense = learns feature combinations
2. ReLU = adds non-linearity
3. Dense = builds higher-level features
4. ReLU = keeps non-linearity and stable gradients
5. Dense(10) = gives one raw score for each class
6. Softmax = converts those 10 scores into class probabilities

So:

- Dense layers learn what matters,
- ReLU helps learn complex boundaries,
- Softmax gives a probability distribution for classification.

---

### Practical intuition for Fashion-MNIST

In simple words:

- the first Dense layer learns rough pixel interactions,
- the second Dense layer combines them into more meaningful clothing features,
- the final Dense layer scores each class,
- Softmax turns those scores into “how likely is each class?”.

That is why this is one of the most common baseline architectures for Fashion-MNIST.

## 13) TensorFlow code sample + monitoring weight updates

Below is a practical TensorFlow / Keras example for Fashion-MNIST. It does three useful kinds of monitoring:

1. **TensorBoard Graph** → shows the model graph / layer structure
2. **TensorBoard Histograms** → shows how the full weight distribution of each Dense layer changes over time
3. **Tracked weight lines** → records a few specific node-to-node weights as scalar time-series so you can see exact weight updates across epochs

> Important: for a model like 784→128→64→10, there are more than 109k trainable parameters. Plotting every single weight as a separate line is too dense to read. So the best practice is:
>
> - use **histograms** for all weights,
> - use **line graphs** for a small set of representative weights.

### TensorFlow / Keras example

```python
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# --------------------------------------------------
# 1) Load and preprocess Fashion-MNIST
# --------------------------------------------------
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# --------------------------------------------------
# 2) Build the MLP model
# --------------------------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28), name="flatten"),
    tf.keras.layers.Dense(128, activation="relu", name="dense_1"),
    tf.keras.layers.Dense(64, activation="relu", name="dense_2"),
    tf.keras.layers.Dense(10, activation="softmax", name="output")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# --------------------------------------------------
# 3) TensorBoard logging setup
# --------------------------------------------------
run_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join("logs", "fashion_mlp", run_id)

# Built-in TensorBoard callback:
# - write_graph=True lets TensorBoard display the model graph
# - histogram_freq=1 logs layer histograms every epoch
# - update_freq='epoch' logs once per epoch

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    update_freq="epoch"
)

# --------------------------------------------------
# 4) Custom callback to track exact weight updates
# --------------------------------------------------
class WeightTrackingCallback(tf.keras.callbacks.Callback):
    """
    Logs:
    - histogram of each Dense layer's kernel and bias
    - mean and std of weights
    - a few selected individual weights as scalar lines
    """
    def __init__(self, log_dir):
        super().__init__()
        self.writer = tf.summary.create_file_writer(os.path.join(log_dir, "custom_weights"))

        # Track a few specific connections per layer (row, col)
        # row = input index, col = neuron index
        self.tracked_indices = {
            "dense_1": [(0, 0), (10, 3), (100, 7)],
            "dense_2": [(0, 0), (5, 1), (20, 10)],
            "output":  [(0, 0), (5, 2), (10, 7)],
        }

        # Save lines for matplotlib plotting after training
        self.weight_history = {}

    def on_epoch_end(self, epoch, logs=None):
        with self.writer.as_default():
            for layer in self.model.layers:
                if not isinstance(layer, tf.keras.layers.Dense):
                    continue

                kernel, bias = layer.get_weights()

                # Histogram logs: full distribution of weights / biases
                tf.summary.histogram(f"{layer.name}/kernel", kernel, step=epoch)
                tf.summary.histogram(f"{layer.name}/bias", bias, step=epoch)

                # Scalar summaries: weight statistics
                tf.summary.scalar(f"{layer.name}/kernel_mean", np.mean(kernel), step=epoch)
                tf.summary.scalar(f"{layer.name}/kernel_std", np.std(kernel), step=epoch)
                tf.summary.scalar(f"{layer.name}/bias_mean", np.mean(bias), step=epoch)

                # Track a few exact node-to-node connections
                for (r, c) in self.tracked_indices.get(layer.name, []):
                    if r < kernel.shape[0] and c < kernel.shape[1]:
                        value = float(kernel[r, c])
                        tag = f"tracked/{layer.name}/w_{r}_{c}"
                        tf.summary.scalar(tag, value, step=epoch)

                        self.weight_history.setdefault(tag, []).append(value)

            self.writer.flush()

weight_tracking_cb = WeightTrackingCallback(log_dir)

# --------------------------------------------------
# 5) Train the model
# --------------------------------------------------
history = model.fit(
    x_train,
    y_train,
    epochs=8,
    batch_size=128,
    validation_split=0.1,
    callbacks=[tensorboard_cb, weight_tracking_cb],
    verbose=1
)

# --------------------------------------------------
# 6) Evaluate on test data
# --------------------------------------------------
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# --------------------------------------------------
# 7) Plot training curves in matplotlib
# --------------------------------------------------
plt.figure(figsize=(8, 4))
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(history.history["accuracy"], label="train_accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy vs Epoch")
plt.legend()
plt.grid(True)
plt.show()

# --------------------------------------------------
# 8) Plot selected weight updates (exact connections)
# --------------------------------------------------
plt.figure(figsize=(10, 5))
for tag, values in weight_tracking_cb.weight_history.items():
    plt.plot(values, label=tag)

plt.xlabel("Epoch")
plt.ylabel("Weight Value")
plt.title("Selected Node-to-Node Weight Updates Across Epochs")
plt.legend(fontsize=8)
plt.grid(True)
plt.show()

# --------------------------------------------------
# 9) How to open TensorBoard
# --------------------------------------------------
print("
To view TensorBoard, run this in terminal:")
print(f"tensorboard --logdir {os.path.abspath('logs')}")
print("Then open the local URL shown in the terminal (usually http://localhost:6006)")
```

### What you will see in TensorBoard

- **Scalars tab**:

  - loss
  - accuracy
  - validation loss
  - validation accuracy
  - selected tracked weights like `tracked/dense_1/w_0_0`

- **Histograms tab**:

  - full kernel distribution for `dense_1`, `dense_2`, and `output`
  - full bias distribution for those layers
  - this is the best way to see whether weights are spreading, shrinking, or saturating

- **Graphs tab**:

  - model structure (Flatten → Dense(128) → Dense(64) → Dense(10))

