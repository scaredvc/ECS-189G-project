from typing import Dict, List, Tuple

def show_menu(title: str, options: List[Tuple[str, str]]) -> str:
    while True:
        print(f"\n{'='*60}")
        print(f"{title}")
        print(f"{'='*60}")
        
        for i, (option, description) in enumerate(options, 1):
            print(f"{i}. {option}")
        
        try:
            choice = int(input("\nEnter your choice (number): "))
            if 1 <= choice <= len(options):
                selected_option = options[choice-1]
                print(f"\n💡 Description: {selected_option[1]}")
                return selected_option[0]
            else:
                print("\n❌ Invalid choice. Please try again.")
        except ValueError:
            print("\n❌ Please enter a valid number.")

dataset_options = [
    ("ORL", "Human face images - 40 people, 10 images each (9 train, 1 test), 112x92x3 grayscale"),
    ("MNIST", "Hand-written digit images - 60,000 training, 10,000 testing, 28x28 grayscale"),
    ("CIFAR", "Colored object images - 50,000 training, 10,000 testing, 32x32x3 color")
]

metric_options = [
    ("accuracy_score", "Standard accuracy - ratio of correct predictions"),
    ("f1_score", "Harmonic mean of precision and recall"),
    ("precision_score", "Ratio of true positives to all positive predictions"),
    ("recall_score", "Ratio of true positives to all actual positives")
]

epoch_options = [
    ("100", "Quick training, good for testing"),
    ("200", "Balanced training time"),
    ("500", "Thorough training"),
    ("1000", "Extended training for complex problems")
]

lr_options = [
    ("0.1", "Aggressive learning - faster but may overshoot"),
    ("0.01", "Standard learning rate - good balance"),
    ("0.001", "Conservative learning - more stable"),
    ("0.0001", "Very careful learning - for sensitive problems")
]

def get_training_params() -> Dict[str, str]:
    print("\n🔧 CNN Model Configuration")

    params = {
        'dataset': show_menu("📊 Select Dataset", dataset_options),
        'metric': show_menu("📊 Select Evaluation Metric", metric_options),
        'epochs': show_menu("🔄 Select Number of Epochs", epoch_options),
        'learning_rate': show_menu("📈 Select Learning Rate", lr_options),
    }

    return params

def print_training_status(message: str, is_complete: bool = False) -> None:
    emoji = "✅" if is_complete else "✨"
    print(f"\n{emoji} {message}")

def print_training_summary(metric: str, mean_score: float, std_score: float):
    print("\n============================================================")
    print("\U0001F3AF Final Performance Results")
    print("============================================================")
    if std_score is not None:
        print(f"\U0001F4C8 {metric}: {mean_score:.4f} ± {std_score:.4f}")
    else:
        print(f"\U0001F4C8 {metric}: {mean_score:.4f}")
    print("============================================================")