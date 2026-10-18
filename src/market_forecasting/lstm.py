"""Small residual LSTMs with deterministic, auditable training and selection."""
import json
import os
import time
from zipfile import ZipFile
from pathlib import Path
import numpy as np
from .preprocessing import reconstruct_prices

# CPU execution avoids switching numerical backends between paired experiments.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import tensorflow as tf


def configure_runtime():
    """Use deterministic CPU operations with bounded thread counts."""
    tf.config.set_visible_devices([], "GPU")
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()


def build_model(settings, seed, lookback=30, features=1):
    """Initialise an independent LSTM that predicts a standardised correction."""
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, features)),
        tf.keras.layers.LSTM(settings["units"], stateful=False),
        tf.keras.layers.Dropout(settings["dropout"], seed=seed),
        tf.keras.layers.Dense(1),
    ])
    # MSE is in correction-standardised units; evaluation reconstructs USD prices.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
                  loss="mean_squared_error")
    return model


def predict_standardised(model, X, batch_size=512):
    """Predict corrections in inference mode without changing model state."""
    return np.concatenate([np.asarray(model(X[i:i + batch_size], training=False)).reshape(-1)
                           for i in range(0, len(X), batch_size)]).astype(float)



def load_frozen_model(path, feature_names, settings, lookback=30):
    """Load an inference-only checkpoint after verifying its frozen specification."""
    names = list(feature_names)
    if not names or len(set(names)) != len(names) or not all(isinstance(v, str) and v.strip() for v in names):
        raise ValueError("Feature names must be nonempty, unique strings in saved order")
    # compile=False avoids restoring a training workflow; inspect the saved compile
    # metadata first so an incompatible optimiser or loss cannot pass unnoticed.
    with ZipFile(path) as archive:
        saved = json.loads(archive.read("config.json"))
    compiled = saved.get("compile_config", {})
    optimizer = compiled.get("optimizer") or {}
    optimizer_config = optimizer.get("config", {})
    if (optimizer.get("class_name") != "Adam"
            or compiled.get("loss") != "mean_squared_error"
            or not np.isclose(optimizer_config.get("learning_rate", np.nan),
                              settings["learning_rate"], rtol=1e-6, atol=0)):
        raise ValueError("Saved optimiser, learning rate or loss differs from frozen settings")
    expected_optimizer = tf.keras.optimizers.Adam(
        learning_rate=settings["learning_rate"]).get_config()
    for key, expected in expected_optimizer.items():
        if key not in {"name", "learning_rate"} and optimizer_config.get(key) != expected:
            raise ValueError(f"Saved Adam setting differs from the implementation: {key}")
    model = tf.keras.models.load_model(path, compile=False, safe_mode=True)
    if tuple(model.input_shape) != (None, lookback, len(names)) or tuple(model.output_shape) != (None, 1):
        raise ValueError("Checkpoint input/output dimensions differ from frozen features")
    layers = model.layers
    if (len(layers) != 3 or not isinstance(layers[0], tf.keras.layers.LSTM)
            or not isinstance(layers[1], tf.keras.layers.Dropout)
            or not isinstance(layers[2], tf.keras.layers.Dense)):
        raise ValueError("Checkpoint must contain LSTM, Dropout and Dense layers")
    recurrent, dropout, output = (layer.get_config() for layer in layers)
    if (recurrent["units"] != settings["units"] or recurrent["stateful"]
            or recurrent["return_sequences"] or recurrent["return_state"]
            or recurrent["go_backwards"] or recurrent["activation"] != "tanh"
            or recurrent["recurrent_activation"] != "sigmoid"
            or recurrent["dropout"] != 0 or recurrent["recurrent_dropout"] != 0
            or not recurrent["use_bias"] or not recurrent["unit_forget_bias"]
            or dropout["rate"] != settings["dropout"]
            or output["units"] != 1 or output["activation"] != "linear"
            or not output["use_bias"]):
        raise ValueError("Checkpoint layer settings differ from the frozen architecture")
    units = settings["units"]
    expected_parameters = 4 * units * (len(names) + units + 1) + units + 1
    if (model.count_params() != expected_parameters
            or not all(np.isfinite(value).all() for value in model.get_weights())):
        raise ValueError("Checkpoint parameters are nonfinite or have an invalid count")
    return model


def frozen_forecast(model, X, origins, correction_scaler):
    """Reconstruct USD forecasts while checking that inference preserves fitted state."""
    X, origins = np.asarray(X), np.asarray(origins, dtype=float)
    if (X.ndim != 3 or not len(X) or tuple(X.shape[1:]) != tuple(model.input_shape[1:])
            or origins.shape != (len(X),) or not np.isfinite(X).all()
            or not np.isfinite(origins).all() or not (origins > 0).all()):
        raise ValueError("Finite aligned input windows and positive origin prices are required")
    scaler_before = (correction_scaler.mean, correction_scaler.scale)
    if not np.isfinite(scaler_before).all() or correction_scaler.scale <= 0:
        raise ValueError("Correction scaler must have finite mean and positive scale")
    weights_before = [value.copy() for value in model.get_weights()]
    standardised = predict_standardised(model, X)
    # The target Close is deliberately absent; only the known origin is added.
    correction = correction_scaler.inverse_transform(standardised)
    prediction = reconstruct_prices(origins, standardised, correction_scaler)
    weights_after = model.get_weights()
    if (len(weights_before) != len(weights_after)
            or any(not np.array_equal(a, b) for a, b in zip(weights_before, weights_after))):
        raise RuntimeError("Frozen LSTM weights changed during inference")
    if scaler_before != (correction_scaler.mean, correction_scaler.scale):
        raise RuntimeError("Frozen correction scaler changed during inference")
    if (standardised.shape != origins.shape or not np.isfinite(standardised).all()
            or not np.isfinite(correction).all() or not np.isfinite(prediction).all()):
        raise RuntimeError("Nonfinite or misaligned LSTM forecast")
    return {"standardised_correction": standardised,
            "correction_usd": correction, "prediction": prediction}


class EarlyStopState:
    """Track strict validation improvements and restore the best model weights."""
    def __init__(self, patience):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = None
        self.wait = 0
        self.weights = None

    def observe(self, model, loss, epoch):
        """Save an improved checkpoint or count an epoch without improvement."""
        if not np.isfinite(loss):
            raise ValueError("Nonfinite validation loss")
        if loss < self.best_loss:
            self.best_loss, self.best_epoch, self.wait = float(loss), epoch, 0
            self.weights = [w.copy() for w in model.get_weights()]
        else:
            self.wait += 1
        return self.wait >= self.patience

    def restore(self, model):
        """Restore weights selected solely by internal-validation MSE."""
        if self.weights is None:
            raise ValueError("No finite checkpoint available")
        model.set_weights(self.weights)


def train_model(model, X, y, settings, output, validation=None):
    """Train chronological minibatches and persist history after every epoch."""
    output = Path(output)
    history = []
    tracker = EarlyStopState(settings["patience"]) if validation is not None else None
    batch_size, epochs = settings["batch_size"], settings["epochs"]
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # No shuffling: each epoch traverses the same target-date order.
        for start in range(0, len(X), batch_size):
            stop = min(start + batch_size, len(X))
            model.reset_metrics()
            loss = float(model.train_on_batch(X[start:stop], y[start:stop, None]))
            if not np.isfinite(loss):
                raise ValueError(f"Nonfinite training loss at epoch {epoch}")
            total_loss += loss * (stop - start)
        # Compare both sets using end-of-epoch weights and dropout disabled.
        train_predictions = predict_standardised(model, X)
        row = {"epoch": epoch, "training_loss": total_loss / len(X),
               "training_inference_mse": float(np.mean((train_predictions - y) ** 2)),
               "optimizer_steps": int(model.optimizer.iterations.numpy()),
               "elapsed_seconds": time.perf_counter() - started}
        should_stop = False
        if validation is not None:
            vx, vy = validation
            row["validation_mse"] = float(np.mean((predict_standardised(model, vx) - vy) ** 2))
            should_stop = tracker.observe(model, row["validation_mse"], epoch)
        if not np.isfinite(row["training_inference_mse"]):
            raise ValueError("Nonfinite inference loss")
        history.append(row)
        # Persist progress so a failed run retains its completed epochs.
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        if should_stop:
            break
    terminal_validation_mse = history[-1].get("validation_mse")
    if tracker is not None:
        tracker.restore(model)
    # The restored selection checkpoint is for inference; final fits start afresh.
    model.save(output / "model.keras")
    return {"epochs_completed": len(history), "best_epoch": tracker.best_epoch if tracker else None,
            "best_validation_mse": tracker.best_loss if tracker else None,
            "terminal_validation_mse": terminal_validation_mse,
            "stopped_early": bool(tracker and len(history) < epochs),
            "reached_epoch_cap": len(history) == epochs,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_steps": int(model.optimizer.iterations.numpy()),
            "trainable_parameters": model.count_params()}


def choose_configuration(runs, configurations, seeds):
    """Apply the predefined mean-MAE band, complexity and stability rule."""
    groups = []
    for cfg in configurations:
        entries = [run for run in runs if run["configuration_id"] == cfg["id"]]
        valid = (len(entries) == len(seeds)
                 and sorted(run["seed"] for run in entries) == sorted(seeds)
                 and all(run["status"] == "complete" for run in entries))
        row = {"configuration_id": cfg["id"], "eligible": False, "units": cfg["units"]}
        if valid:
            mae = np.array([run["evaluation_metrics"]["mae_usd"] for run in entries])
            rmse = np.array([run["evaluation_metrics"]["rmse_usd"] for run in entries])
            if np.isfinite(mae).all() and np.isfinite(rmse).all():
                row.update(eligible=True, mean_mae=float(mae.mean()), std_mae=float(mae.std(ddof=1)),
                           mean_rmse=float(rmse.mean()), std_rmse=float(rmse.std(ddof=1)))
        groups.append(row)
    eligible = [row for row in groups if row["eligible"]]
    if not eligible:
        raise ValueError("No configuration has three valid runs")
    threshold = min(row["mean_mae"] for row in eligible) * 1.01
    shortlist = [row for row in eligible if row["mean_mae"] <= threshold]
    selected = min(shortlist, key=lambda row: (row["units"], row["std_mae"],
                                             row["mean_mae"], row["configuration_id"]))
    epochs = [run["best_epoch"] for run in runs if run["configuration_id"] == selected["configuration_id"]]
    return {"selected_id": selected["configuration_id"], "epochs": int(np.median(epochs)),
            "mae_band_upper": threshold, "groups": groups}
