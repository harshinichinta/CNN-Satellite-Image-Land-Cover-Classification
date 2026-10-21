from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tensorflow as tf

from backend.app.config import BASE_DIR, EUROSAT_CLASSES
from backend.app.services.probabilities import fit_temperature


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _decode_image(path: str, label: int, image_size: int, use_preprocess: bool = False):
    image = tf.io.read_file(path)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(
        image,
        [image_size, image_size],
        method=tf.image.ResizeMethod.BILINEAR,
    )
    image = tf.cast(image, tf.float32)
    if use_preprocess:
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    else:
        image = image / 255.0
    return image, label


def load_split_dataset(
    manifest_path: Path,
    split_name: str,
    image_size: int,
    batch_size: int,
    use_preprocess: bool = False,
    shuffle: bool = False,
    max_samples_per_class: int | None = None,
    random_seed: int = 42,
):
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    class_to_index = manifest["class_to_index"]
    file_paths = manifest[f"{split_name}_files"]
    if max_samples_per_class is not None:
        if max_samples_per_class < 1:
            raise ValueError("max_samples_per_class must be positive.")
        random = np.random.default_rng(random_seed)
        selected_paths = []
        for class_name in manifest.get("classes", class_to_index):
            class_paths = [
                file_path
                for file_path in file_paths
                if Path(file_path).parent.name == class_name
            ]
            random.shuffle(class_paths)
            selected_paths.extend(class_paths[:max_samples_per_class])
        file_paths = selected_paths

    labels = [class_to_index[Path(path).parent.name] for path in file_paths]
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths),
            seed=random_seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda path, label: _decode_image(
            path,
            label,
            image_size,
            use_preprocess,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def build_baseline_model(input_shape=(224, 224, 3), num_classes=len(EUROSAT_CLASSES)):
    inputs = tf.keras.layers.Input(shape=input_shape)
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        ],
        name="satellite_augmentation",
    )
    values = augmentation(inputs)
    for filters in (32, 64, 128):
        values = tf.keras.layers.Conv2D(
            filters, 3, padding="same", activation="relu"
        )(values)
        values = tf.keras.layers.MaxPooling2D(pool_size=2)(values)
    values = tf.keras.layers.GlobalAveragePooling2D()(values)
    values = tf.keras.layers.Dense(128, activation="relu")(values)
    values = tf.keras.layers.Dropout(0.35)(values)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(values)
    return tf.keras.Model(inputs, outputs, name="eurosat_compact_cnn")


def build_transfer_model(input_shape=(224, 224, 3), num_classes=len(EUROSAT_CLASSES)):
    mobile_net = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    mobile_net.trainable = False

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="satellite_augmentation",
    )
    inputs = tf.keras.layers.Input(shape=input_shape)
    augmented = augmentation(inputs)
    outputs = mobile_net(augmented, training=False)
    outputs = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling")(outputs)
    outputs = tf.keras.layers.Dense(
        256,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(1e-4),
        name="classifier_hidden",
    )(outputs)
    outputs = tf.keras.layers.Dropout(0.3)(outputs)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="output_layer")(outputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def fit_model(model, train_ds, val_ds, epochs: int, callbacks: List[tf.keras.callbacks.Callback]):
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
        shuffle=False,
    )


def save_model_artifacts(model, model_path: Path, metadata: Dict[str, Any], history: tf.keras.callbacks.History | None = None):
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    with model_path.with_name(f"{model_path.stem}_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    if history is not None:
        history_path = model_path.with_name(f"{model_path.stem}_history.json")
        with history_path.open("w", encoding="utf-8") as handle:
            json.dump(history.history, handle, indent=2)
    print(f"Saved model to {model_path}")


def choose_best_model(model_a_path: Path, model_b_path: Path):
    metric_a = json.loads((model_a_path.with_name(f"{model_a_path.stem}_metadata.json")).read_text(encoding="utf-8"))
    metric_b = json.loads((model_b_path.with_name(f"{model_b_path.stem}_metadata.json")).read_text(encoding="utf-8"))
    if metric_a.get("best_val_accuracy", 0.0) >= metric_b.get("best_val_accuracy", 0.0):
        return model_a_path
    return model_b_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a baseline CNN or MobileNetV2 transfer model.")
    parser.add_argument("--model-type", choices=["baseline", "transfer", "both"], default="baseline")
    parser.add_argument("--manifest", default="data/processed/dataset_manifest.json", help="Prepared dataset manifest")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--quick", action="store_true", help="Use a reduced training schedule for smoke testing.")
    args = parser.parse_args()

    set_seed(args.seed)
    model_dir = BASE_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    manifest_path = (BASE_DIR / args.manifest).resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Dataset manifest not found at {manifest_path}. Run scripts/prepare_dataset.py first.")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    class_names = manifest.get("classes", EUROSAT_CLASSES)
    if class_names != EUROSAT_CLASSES:
        raise ValueError(
            "Dataset class order does not match the supported EuroSAT classes. "
            "Rebuild the dataset manifest from the EuroSAT_RGB class folders."
        )
    if manifest.get("class_to_index") != {
        class_name: index for index, class_name in enumerate(class_names)
    }:
        raise ValueError("Dataset manifest class_to_index does not match its classes list.")
    if args.epochs < 1 or args.batch_size < 1 or args.fine_tune_epochs < 0:
        raise ValueError("Epoch counts and batch size must be positive (fine-tune epochs may be zero).")
    if args.image_size < 32:
        raise ValueError("Model input image size must be at least 32 pixels.")

    if args.quick:
        args.epochs = min(args.epochs, 1)
        args.batch_size = min(args.batch_size, 32)
        args.fine_tune_epochs = 0

    max_samples_per_class = 100 if args.quick else None

    model_types = [args.model_type] if args.model_type != "both" else ["baseline", "transfer"]
    trained_paths = []

    for model_type in model_types:
        if model_type == "baseline":
            model = build_baseline_model(input_shape=(args.image_size, args.image_size, 3), num_classes=len(class_names))
            output_name = "baseline_cnn.keras"
            preprocess = False
        else:
            model = build_transfer_model(input_shape=(args.image_size, args.image_size, 3), num_classes=len(class_names))
            output_name = "transfer_mobilenet.keras"
            preprocess = True

        initial_learning_rate = 5e-4 if model_type == "baseline" else 1e-4
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        train_ds = load_split_dataset(
            manifest_path,
            "train",
            args.image_size,
            args.batch_size,
            use_preprocess=preprocess,
            shuffle=True,
            max_samples_per_class=max_samples_per_class,
            random_seed=args.seed,
        )
        val_ds = load_split_dataset(
            manifest_path,
            "val",
            args.image_size,
            args.batch_size,
            use_preprocess=preprocess,
            max_samples_per_class=max_samples_per_class,
        )

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=model_dir / f"{output_name.replace('.keras', '')}_checkpoint.weights.h5",
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        )
        history = fit_model(
            model, train_ds, val_ds, args.epochs, [early_stop, reduce_lr, checkpoint]
        )
        best_val_accuracy = max(history.history.get("val_accuracy", [0.0]))

        fine_tune_val_accuracy = None
        if model_type == "transfer" and args.fine_tune_epochs:
            head_weights = model.get_weights()
            backbone = next(
                layer for layer in model.layers if isinstance(layer, tf.keras.Model)
            )
            backbone.trainable = True
            fine_tune_layers = 30
            for layer in backbone.layers[:-fine_tune_layers]:
                layer.trainable = False
            for layer in backbone.layers:
                if isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = False

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            fine_tune_history = fit_model(
                model,
                train_ds,
                val_ds,
                args.fine_tune_epochs,
                [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_accuracy",
                        mode="max",
                        patience=2,
                        restore_best_weights=True,
                    )
                ],
            )
            fine_tune_val_accuracy = max(
                fine_tune_history.history.get("val_accuracy", [0.0])
            )
            if fine_tune_val_accuracy >= best_val_accuracy:
                history.history["loss"].extend(fine_tune_history.history.get("loss", []))
                history.history["accuracy"].extend(
                    fine_tune_history.history.get("accuracy", [])
                )
                history.history["val_loss"].extend(
                    fine_tune_history.history.get("val_loss", [])
                )
                history.history["val_accuracy"].extend(
                    fine_tune_history.history.get("val_accuracy", [])
                )
                best_val_accuracy = fine_tune_val_accuracy
            else:
                model.set_weights(head_weights)
                backbone.trainable = False
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
                    loss="sparse_categorical_crossentropy",
                    metrics=["accuracy"],
                )

        metadata = {
            "model_type": model_type,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "random_seed": args.seed,
            "classes": class_names,
            "preprocessing": "mobilenet_v2" if preprocess else "rescale_0_1",
            "resize_method": "tensorflow_bilinear_stretch",
            "best_val_accuracy": float(best_val_accuracy),
            "calibration_temperature": 1.0,
            "history_file": f"{output_name.replace('.keras', '')}_history.json",
            "manifest": str(manifest_path),
            "quick_training": args.quick,
            "max_samples_per_class": max_samples_per_class,
            "fine_tune_epochs": args.fine_tune_epochs,
            "fine_tune_best_val_accuracy": fine_tune_val_accuracy,
            "train_samples": sum(
                min(
                    max_samples_per_class or len(manifest["train_files"]),
                    sum(Path(path).parent.name == name for path in manifest["train_files"]),
                )
                for name in class_names
            ),
            "augmentation": (
                ["horizontal_vertical_flip"]
                if model_type in ("transfer", "baseline")
                else []
            ),
        }
        val_scores = model.predict(val_ds, verbose=0)
        val_labels = np.concatenate([labels.numpy() for _, labels in val_ds])
        metadata["calibration_temperature"] = fit_temperature(val_scores, val_labels)
        metadata["calibration_method"] = "temperature_scaling_on_validation_split"

        model_path = model_dir / output_name
        save_model_artifacts(model, model_path, metadata, history)
        trained_paths.append(model_path)

    if len(trained_paths) > 1:
        selected_model = choose_best_model(trained_paths[0], trained_paths[1])
    else:
        selected_model = trained_paths[0]

    if selected_model:
        best_model_target = model_dir / "best_model.keras"
        if selected_model != best_model_target:
            tf.keras.models.load_model(selected_model, compile=False).save(best_model_target)
        best_metadata = json.loads(
            selected_model.with_name(f"{selected_model.stem}_metadata.json").read_text(encoding="utf-8")
        )
        with best_model_target.with_name("best_model_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(best_metadata, handle, indent=2)
        print(f"Selected model: {best_model_target}")

    print("Training complete. Use scripts/evaluate_model.py to score the saved model on the test split.")


if __name__ == "__main__":
    main()
