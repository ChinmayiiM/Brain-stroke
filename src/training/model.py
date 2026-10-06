"""EfficientNetB0 transfer-learning model: construction, freezing policy, compilation.

Architecture::

    Input (H, W, 3) float32 in [0, 255]
      └─ EfficientNetB0 backbone (ImageNet, include_top=False,
         built-in Rescaling + Normalization, called with training=False)
           └─ GlobalAveragePooling2D
                └─ Dropout
                     └─ Dense(output) — softmax (N>2 classes) / sigmoid (2 classes)
"""

from __future__ import annotations

import keras

from src.training.settings import TrainingSettings

BACKBONE_NAME = "efficientnetb0"
HEAD_LAYER_NAMES = ("global_avg_pool", "head_dropout", "classifier")


def build_model(settings: TrainingSettings) -> keras.Model:
    """Build the stage-1 model (backbone frozen)."""
    inputs = keras.Input(shape=settings.input_shape, name="mri_image")
    backbone = keras.applications.EfficientNetB0(
        include_top=False,
        weights=settings.backbone_weights,
        input_shape=settings.input_shape,
        name=BACKBONE_NAME,
    )
    backbone.trainable = False

    # training=False keeps BatchNorm in inference mode even after layers are
    # unfrozen for fine-tuning, so small-batch statistics never overwrite the
    # ImageNet running statistics.
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name=HEAD_LAYER_NAMES[0])(x)
    x = keras.layers.Dropout(settings.dropout, seed=settings.seed, name=HEAD_LAYER_NAMES[1])(x)
    if settings.is_binary:
        outputs = keras.layers.Dense(1, activation="sigmoid", name=HEAD_LAYER_NAMES[2])(x)
    else:
        outputs = keras.layers.Dense(settings.num_classes, activation="softmax", name=HEAD_LAYER_NAMES[2])(x)
    return keras.Model(inputs, outputs, name=settings.model_name)


def get_backbone(model: keras.Model) -> keras.Model:
    return model.get_layer(BACKBONE_NAME)


def freeze_backbone(model: keras.Model) -> None:
    """Stage 1: the entire backbone is non-trainable."""
    get_backbone(model).trainable = False


def unfreeze_top_layers(model: keras.Model, n_layers: int) -> list[str]:
    """Stage 2: make the top ``n_layers`` backbone layers trainable, except BatchNorm.

    Returns the names of layers that now hold trainable weights.
    """
    backbone = get_backbone(model)
    if not 0 < n_layers <= len(backbone.layers):
        raise ValueError(f"unfreeze_top_n_layers must be in 1..{len(backbone.layers)}, got {n_layers}")

    backbone.trainable = True
    boundary = len(backbone.layers) - n_layers
    for index, layer in enumerate(backbone.layers):
        layer.trainable = index >= boundary and not isinstance(layer, keras.layers.BatchNormalization)
    return [layer.name for layer in backbone.layers[boundary:] if layer.trainable and layer.trainable_weights]


def expected_unfrozen_layers(model: keras.Model, n_layers: int) -> set[str]:
    """Layers that *should* hold trainable weights after :func:`unfreeze_top_layers`."""
    backbone = get_backbone(model)
    return {
        layer.name
        for layer in backbone.layers[len(backbone.layers) - n_layers:]
        if not isinstance(layer, keras.layers.BatchNormalization) and layer.weights
    }


def build_metrics(settings: TrainingSettings) -> list[keras.metrics.Metric]:
    """Accuracy, macro-F1 and per-class precision/recall.

    Per-class recall exposes minority-class behaviour that overall accuracy hides.
    These are training-monitoring metrics on validation slices, not clinical performance.
    """
    if settings.is_binary:
        positive = settings.class_names[1].lower()
        return [
            keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name=f"precision_{positive}"),
            keras.metrics.Recall(name=f"recall_{positive}"),
            keras.metrics.F1Score(average="macro", threshold=0.5, name="f1_macro"),
            keras.metrics.AUC(name="auc"),
        ]
    metrics: list[keras.metrics.Metric] = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.F1Score(average="macro", threshold=None, name="f1_macro"),  # argmax-based
    ]
    for index, name in enumerate(settings.class_names):
        slug = name.lower()
        metrics.append(keras.metrics.Precision(top_k=1, class_id=index, name=f"precision_{slug}"))
        metrics.append(keras.metrics.Recall(top_k=1, class_id=index, name=f"recall_{slug}"))
    return metrics


def compile_model(model: keras.Model, settings: TrainingSettings, learning_rate: float) -> None:
    """Compile (or re-compile after changing trainability) with the matching loss."""
    loss = (
        keras.losses.BinaryCrossentropy(name="loss")
        if settings.is_binary
        else keras.losses.CategoricalCrossentropy(name="loss")
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=build_metrics(settings),
    )
