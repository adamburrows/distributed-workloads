import re
import matplotlib.pyplot as plt

log_file = "/opt/data/log.txt"

epochs = []
losses = []
accuracies = []

# Keep track of already seen epoch metrics to ignore duplicates
seen_metrics = set()

# Regex patterns
epoch_metric_pattern = re.compile(
    r"\{metricName: accuracy, metricValue: ([0-9.]+)\};\{metricName: loss, metricValue: ([0-9.]+)\}"
)

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        match = epoch_metric_pattern.search(line)
        if match:
            acc, loss = float(match.group(1)), float(match.group(2))
            metric_tuple = (acc, loss)
            # Only add if we haven't seen this metric before
            if metric_tuple not in seen_metrics:
                seen_metrics.add(metric_tuple)
                epoch = len(epochs) + 1
                epochs.append(epoch)
                losses.append(loss)
                accuracies.append(acc)

# Plot
plt.figure(figsize=(8,4))
plt.plot(epochs, losses, marker='o', label="Loss")
plt.plot(epochs, accuracies, marker='x', label="Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.title("Training Metrics")
plt.xticks(range(1, max(epochs)+1))
plt.legend()
plt.grid(True)
plt.show()