# Caries Detection & Classification — Jupyter Lab Implementation Guide

Framework: **TensorFlow / Keras** (best fit for a 6GB RTX 3050 — light on setup, native transfer learning, easy Grad-CAM).

---

## 0. GPU notes for a 6GB card (read this first)

Your RTX 3050 6GB will handle this fine as long as you respect a few limits:

- **Image size:** stick to 224×224. Don't go to 380/456 (EfficientNetB3+) — you'll OOM.
- **Batch size:** start at 16. If no OOM, try 32. If OOM at 16, drop to 8.
- **Mixed precision:** turn it on — roughly halves VRAM use and speeds up training on RTX cards.
- **Close other GPU apps** (browser hardware acceleration, other notebooks) before training.
- Check usage live with `nvidia-smi -l 1` in a terminal while training runs.

---

## 1. Environment setup

Run in a terminal (not in the notebook):

```bash
conda create -n caries python=3.10 -y
conda activate caries
pip install tensorflow[and-cuda]==2.16.* jupyterlab ipywidgets
pip install numpy pandas matplotlib seaborn scikit-learn opencv-python pillow
pip install grad-cam gradio
python -m ipykernel install --user --name caries --display-name "Python (caries)"
```

Verify GPU is visible — run this as your **first notebook cell**:

```python
# Cell 1 — GPU check
import tensorflow as tf
print("TF version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')   # saves VRAM on RTX cards
```

If `GPUs` prints empty, stop here and fix the CUDA/driver install before continuing — everything below assumes GPU is working.

---

## 2. Organize your data

Arrange your images in an `ImageFolder`-style structure — Keras can load this directly:

```
data/
  train/
    non_caries/
    early_caries/
    advanced_caries/
  val/
    non_caries/
    early_caries/
    advanced_caries/
  test/
    non_caries/
    early_caries/
    advanced_caries/
```

If you only have one big folder per class right now, split it with:

```python
# Cell 2 — train/val/test split (run once)
import splitfolders  # pip install split-folders
splitfolders.ratio("data/raw", output="data",
                    seed=42, ratio=(0.7, 0.15, 0.15))
```

(`data/raw/non_caries`, `data/raw/early_caries`, `data/raw/advanced_caries` → auto-split into `train/val/test`.)

---

## 3. Load the data & check class balance

```python
# Cell 3 — load datasets
IMG_SIZE = (224, 224)
BATCH_SIZE = 16

train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical", shuffle=True)
val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/val", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")
test_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode="categorical")

class_names = train_ds.class_names
print(class_names)
```

```python
# Cell 4 — class balance check (important — you likely have imbalance)
import collections
counts = collections.Counter()
for _, labels in train_ds.unbatch():
    counts[class_names[int(tf.argmax(labels))]] += 1
print(counts)
```

If one class is much smaller, compute class weights and pass them into `model.fit(..., class_weight=...)` later:

```python
# Cell 5 — class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

y_train = np.concatenate([np.argmax(l, axis=1) for _, l in train_ds])
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(weights))
print(class_weight)
```

---

## 4. Preprocessing & augmentation

Keep augmentation realistic for intraoral photos — don't distort color too aggressively, since color/discoloration *is* your signal.

```python
# Cell 6 — augmentation + preprocessing pipeline
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomBrightness(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

AUTOTUNE = tf.data.AUTOTUNE

def prepare(ds, training=False):
    if training:
        ds = ds.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x, y: (tf.keras.applications.resnet50.preprocess_input(x), y),
                num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds = prepare(train_ds, training=True)
val_ds = prepare(val_ds)
test_ds = prepare(test_ds)
```

---

## 5. Build the model (transfer learning, ResNet50)

```python
# Cell 7 — model
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False   # freeze for stage 1

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax", dtype="float32")(x)  # float32 out for mixed precision

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
model.summary()
```

---

## 6. Train — stage 1 (frozen base)

```python
# Cell 8 — callbacks
callbacks = [
    tf.keras.callbacks.ModelCheckpoint("best_model.keras", save_best_only=True, monitor="val_accuracy"),
    tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    train_ds, validation_data=val_ds, epochs=20,
    class_weight=class_weight, callbacks=callbacks
)
```

## 7. Fine-tune — stage 2 (unfreeze top layers)

```python
# Cell 9 — unfreeze last ~30 layers, train with a low LR
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy",
              metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])

history_fine = model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    class_weight=class_weight, callbacks=callbacks
)
```

---

## 8. Evaluate properly (not just accuracy)

```python
# Cell 10 — confusion matrix + classification report
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

y_true, y_pred = [], []
for x, y in test_ds:
    preds = model.predict(x, verbose=0)
    y_true.extend(np.argmax(y.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.show()
```

Check **per-class recall** carefully — for a diagnostic tool, missing early-stage caries (false negatives) is worse than a false positive. If recall on `early_caries` is weak, that's your main thing to improve (more augmentation, more early-stage samples, or a lower decision threshold for that class).

---

## 9. Explainability — Grad-CAM

This is what replaces the old "grayscale black-pixel" idea with a real, learned visual explanation, and it's your novelty point for the synopsis.

```python
# Cell 11 — Grad-CAM
import cv2

def grad_cam(model, img_array, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), class_names[class_idx.numpy()]

def overlay_heatmap(img_path, heatmap, alpha=0.4):
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
```

(Note: `last_conv_layer_name` is `"conv5_block3_out"` for ResNet50 — confirm with `model.summary()` if you swap architectures.)

---

## 10. ICDAS / G.V. Black mapping layer

Simple, transparent — just a lookup, no need to train this:

```python
# Cell 12 — clinical mapping
ICDAS_MAP = {
    "non_caries":       {"icdas": "0", "gv_black": "N/A"},
    "early_caries":     {"icdas": "1-2", "gv_black": "Not typically classified (pre-cavitation)"},
    "advanced_caries":  {"icdas": "3-6", "gv_black": "Class I-V (site-dependent — needs visual site confirmation)"},
}

def predict_with_clinical_mapping(model, img_path):
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    arr = tf.keras.utils.img_to_array(img)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    return {
        "predicted_class": pred_class,
        "confidence": round(confidence, 3),
        "icdas_code": ICDAS_MAP[pred_class]["icdas"],
        "gv_black": ICDAS_MAP[pred_class]["gv_black"],
    }
```

Be upfront in your report: **G.V. Black is site-based** (which tooth surface — occlusal, proximal, cervical, etc.), which a plain classifier can't determine from stage alone. Either add a second lightweight classifier for tooth surface/site, or state this as a scoping limitation with future work.

---

## 11. Save the trained model

```python
# Cell 13 — save
model.save("caries_model.keras")
```

---

## 12. Working prototype — quick demo UI (Gradio)

This gives you an actual runnable prototype with almost no extra code — works right inside Jupyter or as a standalone script.

```python
# Cell 14 — prototype demo
import gradio as gr

def predict_fn(image):
    image = image.resize((224, 224))
    arr = tf.keras.utils.img_to_array(image)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)[0]
    pred_class = class_names[np.argmax(preds)]
    info = ICDAS_MAP[pred_class]
    return {c: float(p) for c, p in zip(class_names, preds)}, f"ICDAS: {info['icdas']} | G.V. Black: {info['gv_black']}"

demo = gr.Interface(
    fn=predict_fn,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Label(num_top_classes=3), gr.Textbox(label="Clinical Mapping")],
    title="Dental Caries Detection & Classification Prototype"
)
demo.launch(share=False)  # inline in Jupyter, or opens in browser
```

This is enough to demo live to your guide: upload a photo, get class + confidence + ICDAS/G.V. Black mapping. Add the Grad-CAM overlay as a second output image once the base pipeline works, if you want the visual explanation in the demo too.

---

## Suggested order of work

1. Cells 1–5 (setup, load data, check balance) — get this right first, it decides everything downstream.
2. Cells 6–9 (train stage 1, then fine-tune) — don't fine-tune until stage-1 val accuracy plateaus.
3. Cell 10 (evaluate) — look at per-class recall, not just overall accuracy.
4. Cell 11 (Grad-CAM) — sanity-check that heatmaps actually land on visible lesions, not random background. If they don't, that's a sign of a labeling or preprocessing issue, not just a "model needs more training" issue.
5. Cells 12–14 (mapping + prototype) — wire up once accuracy is acceptable.
