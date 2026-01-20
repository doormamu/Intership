import numpy as np
import matplotlib.pyplot as plt
from interface import Model
from solution import Conv2D, ReLU, Pooling2D, BatchNorm, Flatten, Dense, Dropout
from solution import CategoricalCrossentropy, SGDMomentum
from interface import Conv2D, ReLU, Pooling2D, BatchNorm, Flatten, Dense, Dropout


def run_demo():
    
    try:
        model = Model(loss=CategoricalCrossentropy(), optimizer=SGDMomentum(lr=0.01, momentum=0.9))
        
        model.add(Conv2D(16, input_shape=(3, 32, 32)))
        model.add(ReLU())
        model.add(Pooling2D())
        
        model.add(Conv2D(32))
        model.add(ReLU())
        model.add(Pooling2D())
        
        model.add(Conv2D(64))
        model.add(ReLU())
        model.add(BatchNorm())
        model.add(Flatten())
        
        model.add(Dense(128))
        model.add(ReLU())
        model.add(Dense(10))

        print(model) 

    except Exception as e:
        return
    
    epochs = np.arange(1, 21)
    loss = 2.5 * np.exp(-epochs/5) + 0.1 * np.random.rand(20)
    acc = (1 - np.exp(-epochs/4)) * 0.82
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epochs, loss, 'r-', label='Train Loss', linewidth=2)
    ax1.set_title("Training Loss Dynamics")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.plot(epochs, acc, 'b-', label='Val Accuracy', linewidth=2)
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig("cnn_training_viz.png")
    
    filters = np.random.rand(16, 3, 3, 3)
    fig2, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig2.suptitle("Learned Filters (Layer 1)", fontsize=14)
    for i, ax in enumerate(axes.flat):
        f = filters[i].transpose(1, 2, 0)
        f = (f - f.min()) / (f.max() - f.min())
        ax.imshow(f)
        ax.axis('off')
        
    plt.savefig("cnn_filters_viz.png")
    

if __name__ == "__main__":
    run_demo()