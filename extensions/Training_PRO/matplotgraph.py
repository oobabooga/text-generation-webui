import os
import json

def create_graph(lora_path, lora_name):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
        
        peft_model_path = f'{lora_path}/training_graph.json'
        image_model_path = f'{lora_path}/training_graph.png'
        # Check if the JSON file exists
        if os.path.exists(peft_model_path):
            # Load data from JSON file
            with open(peft_model_path, 'r') as file:
                data = json.load(file)
            # Extract x, y1, and y2 values
            x = [item['epoch'] for item in data]
            y1 = [item['learning_rate'] for item in data]
            y2 = [item['loss'] for item in data]

            # Create the line chart
            fig, ax1 = plt.subplots(figsize=(10, 6))
        

            # Plot y1 (learning rate) on the first y-axis
            ax1.plot(x, y1, 'b-', label='Learning Rate')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Learning Rate', color='b')
            ax1.tick_params('y', colors='b')

            # Create a second y-axis
            ax2 = ax1.twinx()

            # Plot y2 (loss) on the second y-axis
            ax2.plot(x, y2, 'r-', label='Loss')
            ax2.set_ylabel('Loss', color='r')
            ax2.tick_params('y', colors='r')

            # Set the y-axis formatter to display numbers in scientific notation
            ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

            # Add grid
            ax1.grid(True)

            # Combine the legends for both plots
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax2.legend(lines + lines2, labels + labels2, loc='best')

            # Set the title
            plt.title(f'{lora_name} LR and Loss vs Epoch')

            # Save the chart as an image
            plt.savefig(image_model_path)

            print(f"Graph saved in {image_model_path}")
        else:
            print(f"File 'training_graph.json' does not exist in the {lora_path}")
      
    except ImportError:
        print("matplotlib is not installed. Please install matplotlib to create PNG graphs")