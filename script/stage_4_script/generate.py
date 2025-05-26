from local_code.stage_4_code.Dataset_Loader import Dataset_Loader
from local_code.stage_4_code.Method_RNN_Generator import Method_RNN_Generator
from local_code.stage_4_code.Result_Saver import Result_Saver
import torch
import argparse

def get_generator_params():
    """Get parameters for the joke generator"""
    print("\n🔧 RNN Joke Generator Configuration")
    print("="*60)
    
    # Epochs
    print("\n🔄 Select Number of Epochs")
    print("="*60)
    epochs_options = [
        ("20", "Quick training for testing"),
        ("50", "Medium-length training"),
        ("100", "Full training for better results"),
        ("150", "Extended training for best results")
    ]
    
    for i, (value, desc) in enumerate(epochs_options, 1):
        print(f"{i}. {value}")
    
    epochs_choice = input("\nEnter your choice (number): ")
    try:
        epochs = epochs_options[int(epochs_choice) - 1][0]
        print(f"\n💡 Description: {epochs_options[int(epochs_choice) - 1][1]}")
    except (ValueError, IndexError):
        print("Invalid choice, using default (50)")
        epochs = "50"
    
    # Learning Rate
    print("\n📈 Select Learning Rate")
    print("="*60)
    lr_options = [
        ("0.01", "Faster learning but may be unstable"),
        ("0.005", "Balanced learning rate"),
        ("0.001", "Conservative learning rate (recommended)"),
        ("0.0005", "Slow but stable learning")
    ]
    
    for i, (value, desc) in enumerate(lr_options, 1):
        print(f"{i}. {value}")
    
    lr_choice = input("\nEnter your choice (number): ")
    try:
        learning_rate = lr_options[int(lr_choice) - 1][0]
        print(f"\n💡 Description: {lr_options[int(lr_choice) - 1][1]}")
    except (ValueError, IndexError):
        print("Invalid choice, using default (0.001)")
        learning_rate = "0.001"
    
    # Temperature for generation
    print("\n🌡️ Select Temperature for Generation")
    print("="*60)
    temp_options = [
        ("0.5", "More predictable, less creative"),
        ("0.7", "Balanced temperature"),
        ("1.0", "Standard sampling"),
        ("1.2", "More random and creative")
    ]
    
    for i, (value, desc) in enumerate(temp_options, 1):
        print(f"{i}. {value}")
    
    temp_choice = input("\nEnter your choice (number): ")
    try:
        temperature = temp_options[int(temp_choice) - 1][0]
        print(f"\n💡 Description: {temp_options[int(temp_choice) - 1][1]}")
    except (ValueError, IndexError):
        print("Invalid choice, using default (0.7)")
        temperature = "0.7"
    
    return {
        'epochs': epochs,
        'learning_rate': learning_rate,
        'temperature': temperature
    }

def main():
    # Load the joke dataset
    print("\nLoading joke dataset...")
    dataset = Dataset_Loader(dName="stage 4 generator", dDescription="joke generator", data_type="generation")
    dataset.dataset_source_folder_path = './data/stage_4_data/'
    dataset.load()
    
    # Get parameters
    params = get_generator_params()
    
    # Create the generator model
    print("\n✨ Creating RNN Generator with selected parameters...")
    model = Method_RNN_Generator(
        "stage 4 generator", 
        "RNN for joke generation",
        max_epoch=int(params['epochs']),
        learning_rate=float(params['learning_rate'])
    )
    model.temperature = float(params['temperature'])
    print(f"Model will be trained for {params['epochs']} epochs with learning rate {params['learning_rate']}")
    print(f"Generation temperature: {params['temperature']}")
    
    # Set up result saver
    result_saver = Result_Saver('saver', '')
    result_saver.result_destination_folder_path = './result/stage_4_result/RNN_Generator_'
    result_saver.result_destination_file_name = 'generation_result'
    
    # Assign data to the model
    model.data = dataset.data
    
    # Run the model
    print("\n🚀 Starting the training and generation process...")
    results = model.run()
    
    # Save results
    print("\n💾 Saving generation results...")
    result_saver.data = results
    result_saver.save()
    
    print("\n✅ Generation process completed! Results saved successfully.")
    print("\n📝 Generated Jokes Summary:")
    print("="*60)
    
    # Show sample generations from common joke starters
    for start, joke in results['generated_jokes'].items():
        if start in ["what did the", "why did the", "two men walk"]:
            print(f"Input: \"{start}\"")
            print(f"Generated: \"{joke}\"\n")
    
    print("\n💡 You can generate more jokes by running this script again or by loading the saved model.")

if __name__ == "__main__":
    main()
