import matplotlib.pyplot as plt
import numpy as np

def plot_2d_array(data: np.ndarray, title: str = 'Noise') -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    img = data
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'{title}')

    plt.show()

def plot_2d_array_comparison(left: np.ndarray,
                    right: np.ndarray, 
                    left_title: str = 'Left', 
                    right_title: str = 'Right',
                    min: float = 0,
                    max: float = 1) -> None:
    cols = 2

    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes = np.array(axes).reshape(1, cols)

    img_a = left
    axes[0, 0].imshow(img_a, cmap='gray', vmin=min, vmax=max)
    axes[0, 0].set_title(f'{left_title}')

    img_b = right
    axes[0, 1].imshow(img_b, cmap='gray', vmin=min, vmax=max)
    axes[0, 1].set_title(f'{right_title}')

    # Adjust layout and save figure
    plt.tight_layout()
    plt.show()
