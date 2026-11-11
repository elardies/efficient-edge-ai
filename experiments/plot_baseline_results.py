from pathlib import Path
import json
import matplotlib.pyplot as plt

# PROJECT FOLDERS
project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"
figures_dir = project_dir / "figures"
figures_dir.mkdir(exist_ok=True)


# LOAD RESULTS
def load_results(filename):
    with open(results_dir / filename, "r") as f:
        return json.load(f)


baseline = load_results("baseline_cnn_results.json")
original = load_results("mobilenet_v3_lr001_results.json")
lower_lr = load_results("mobilenet_v3_lr0001_results.json")
scheduler = load_results("mobilenet_v3_scheduler_results.json")


# PLOT FUNCTION
def plot_metric(experiments, metric, title, ylabel, filename):
    plt.figure(figsize=(9, 5))

    for name, results in experiments.items():
        values = results[metric]
        epochs = range(1, len(values) + 1)

        plt.plot(epochs, values, label=name, linewidth=2)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=150)
    plt.close()


# 1. BASELINE VS SELECTED MOBILENETV3
main_experiments = {
    "SimpleCNN": baseline,
    "MobileNetV3 (LR 0.001)": original
}

plot_metric(
    main_experiments,
    "train_losses",
    "SimpleCNN vs MobileNetV3 - Training Loss",
    "Cross-entropy loss",
    "model_training_loss.png"
)

plot_metric(
    main_experiments,
    "val_losses",
    "SimpleCNN vs MobileNetV3 - Validation Loss",
    "Cross-entropy loss",
    "model_validation_loss.png"
)

plot_metric(
    main_experiments,
    "val_accuracies",
    "SimpleCNN vs MobileNetV3 - Validation Accuracy",
    "Validation accuracy (%)",
    "model_validation_accuracy.png"
)


# 2. MOBILENETV3 HYPERPARAMETER EXPERIMENTS
mobilenet_experiments = {
    "LR 0.001": original,
    "LR 0.0001": lower_lr,
    "LR scheduler": scheduler
}

plot_metric(
    mobilenet_experiments,
    "train_losses",
    "MobileNetV3 - Training Loss Comparison",
    "Cross-entropy loss",
    "mobilenet_training_loss.png"
)

plot_metric(
    mobilenet_experiments,
    "val_losses",
    "MobileNetV3 - Validation Loss Comparison",
    "Cross-entropy loss",
    "mobilenet_validation_loss.png"
)

plot_metric(
    mobilenet_experiments,
    "val_accuracies",
    "MobileNetV3 - Validation Accuracy Comparison",
    "Validation accuracy (%)",
    "mobilenet_validation_accuracy.png"
)


# 3. LEARNING RATE SCHEDULE
plt.figure(figsize=(9, 5))

epochs = range(1, len(scheduler["learning_rates"]) + 1)

plt.step(
    epochs,
    scheduler["learning_rates"],
    where="mid",
    linewidth=2
)

plt.xlabel("Epoch")
plt.ylabel("Learning rate")
plt.title("MobileNetV3 - Learning Rate Schedule")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "mobilenet_learning_rate.png", dpi=150)
plt.close()


# SUMMARY
print("All 7 figures saved in:", figures_dir)
print()
print("Saved checkpoint validation accuracies:")

for name, results in {
    "SimpleCNN": baseline,
    "MobileNetV3 original": original,
    "MobileNetV3 lower LR": lower_lr,
    "MobileNetV3 scheduler": scheduler
}.items():
    print(f"{name}: {results['saved_val_accuracy']:.2f}%")

