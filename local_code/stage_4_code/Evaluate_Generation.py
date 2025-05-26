from local_code.base_class.evaluate import evaluate
import numpy as np

class Evaluate_Generation(evaluate):
    def __init__(self, eName, eDescription):
        super().__init__(eName, eDescription)
    
    def evaluate(self):
        print('evaluating generated text...')
        
        # Get the generated text and metrics
        generated_text = self.data['generated_text']
        metrics = self.data['plotting_data']
        
        # Print evaluation results
        print('\nGenerated Text:')
        print(generated_text)
        
        print('\nTraining Metrics:')
        print(f"Final Loss: {metrics['loss'][-1]:.4f}")
        print(f"Training Time: {metrics['training_time']:.2f}s")
        
        if 'perplexity' in metrics:
            print(f"Final Perplexity: {metrics['perplexity'][-1]:.2f}")
        
        # Return evaluation results
        return {
            'generated_text': generated_text,
            'final_loss': metrics['loss'][-1],
            'training_time': metrics['training_time']
        } 