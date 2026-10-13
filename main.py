import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button

# ============================================================
# CONFIGURATION & PARAMETERS
# ============================================================
# Choose your plotting method here: 
#   'static'      -> Displays the final state image
#   'animation'   -> Plays an automatic animation through time
#   'interactive' -> Provides a slider and buttons to scrub through frames
#   'html'        -> Good for using in Jupiter and Google Collab aka online running of code

PLOT_METHOD = 'static'  

N = 100       # Grid size
T = 1000     # Timesteps
res = 10      # Record spacetime every 'res' steps
rn_val = 983  # Random seed value for mapping
S = 20       # Central square size


# Found interesting seed values
'''
983, 871, 969, 858, 330, 415, 292, 1047
'''


# ============================================================
# TRUTH TABLE MATRIX & INITIALIZATION
# ============================================================
truth_table_matrix = np.array([
    [0, 0, 2, 3, 0, 0, 6, 13, 7, 6, 12, 15, 3, 15, 15, 15],
    [0, 0, 2, 2, 1, 0, 7, 6, 6, 6, 14, 14, 3, 2, 15, 14],
    [2, 2, 0, 3, 0, 0, 4, 7, 7, 6, 13, 15, 1, 8, 13, 15],
    [3, 2, 3, 9, 0, 0, 6, 4, 7, 6, 15, 12, 0, 0, 14, 12],
    [0, 1, 0, 0, 0, 5, 0, 4, 11, 11, 1, 0, 11, 15, 1, 4],
    [0, 0, 0, 0, 5, 0, 5, 0, 10, 10, 0, 0, 15, 10, 5, 0],
    [6, 7, 4, 6, 0, 5, 0, 6, 15, 15, 5, 6, 9, 13, 1, 6],
    [13, 6, 7, 4, 4, 0, 6, 0, 15, 14, 7, 4, 12, 8, 6, 0],
    [7, 6, 7, 7, 11, 10, 15, 15, 0, 8, 2, 11, 0, 8, 9, 15],
    [6, 6, 6, 6, 11, 10, 15, 14, 8, 3, 10, 2, 9, 0, 15, 6],
    [12, 14, 13, 15, 1, 0, 5, 7, 2, 10, 0, 11, 0, 8, 9, 7],
    [15, 14, 15, 12, 0, 0, 6, 4, 11, 2, 11, 0, 8, 12, 14, 4],
    [3, 3, 1, 0, 11, 15, 9, 12, 0, 9, 0, 8, 0, 13, 8, 12],
    [15, 2, 8, 0, 15, 10, 13, 8, 8, 0, 8, 12, 13, 0, 13, 0],
    [15, 15, 13, 14, 1, 5, 1, 6, 9, 15, 9, 14, 8, 13, 0, 4],
    [15, 14, 15, 12, 4, 0, 6, 0, 15, 6, 7, 4, 12, 0, 4, 0]
])

print(f"Running simulation with seed: {rn_val}")

# Set up grid and initial conditions
W = np.zeros((N, N), dtype=int) + 5
start = (N - S) // 2
end = start + S

# Fill the central square
W[start:end, start:end] = np.random.randint(1, 15, (S, S))

Spacetime = [W.copy()]

# ============================================================
# NEIGHBOURHOOD & MAPPINGS
# ============================================================
offsets = np.array([[-1,-1], [-1,0], [-1,1], [ 0,1], [ 1,1], [ 1,0], [ 1,-1], [ 0,-1]])

def mapping_Brn(rn):
    rng = np.random.default_rng(rn)
    m = {}
    for v in range(1, 15):
        k = rng.integers(1, 5)  # 1–4 random pairs
        pairs = rng.integers(0, 8, size=(k, 2))
        m[v] = [tuple(p) for p in pairs]
    return m

value_to_pairs = mapping_Brn(rn_val)

# Build lookup tables for fast vectorised access
max_val = 14
pair_table = np.zeros((max_val + 1, 8, 2), dtype=int)
pair_counts = np.zeros(max_val + 1, dtype=int)

for val, pairs in value_to_pairs.items():
    pair_counts[val] = len(pairs)
    pair_table[val, :len(pairs)] = np.array(pairs)

# Precompute base coordinates grid
coords = np.stack(np.meshgrid(np.arange(N), np.arange(N), indexing='ij'), axis=-1)

# ============================================================
# MAIN SIMULATION LOOP
# ============================================================
for t in range(T):
    vals = W  
    counts = pair_counts[vals]  
    pair_idx = (counts + t) % counts

    neigh_pairs = pair_table[vals, pair_idx]
    neigh_coords = (coords[..., None, :] + offsets[neigh_pairs]) % N

    v1 = W[neigh_coords[..., 0, 0], neigh_coords[..., 0, 1]]
    v2 = W[neigh_coords[..., 1, 0], neigh_coords[..., 1, 1]]

    new_vals = truth_table_matrix[v1, v2]
    mask = (new_vals != 0) & (new_vals != 15)
    W = np.where(mask, new_vals, W)

    if t % res == 0:
        Spacetime.append(W.copy())

# ============================================================
# PLOTTING & VISUALIZATION SELECTION
# ============================================================
if PLOT_METHOD == 'static':
    fig = plt.figure(figsize=(6, 6), facecolor='none')
    ax = plt.axes([0, 0, 1, 1])
    ax.imshow(W)
    ax.axis('off')
    plt.show()

elif PLOT_METHOD == 'animation':
    fig, ax = plt.subplots()
    im = ax.imshow(Spacetime[0], interpolation='nearest', vmin=0, vmax=np.max(Spacetime), cmap='viridis')
    title = ax.set_title("Frame: 0")

    def update_anim(frame):
        im.set_data(Spacetime[frame])
        title.set_text(f"Frame: {frame}")
        return [im, title]

    ani = FuncAnimation(fig, update_anim, frames=len(Spacetime), interval=10, blit=False, repeat=False)
    plt.show()

elif PLOT_METHOD == 'interactive':
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)  # Make space for slider and buttons

    frame = 0
    im = ax.imshow(Spacetime[frame], vmin=0, vmax=np.max(Spacetime), cmap='viridis')
    title = ax.set_title(f"Frame: {frame}")

    # Slider widget
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.05])
    slider = Slider(ax_slider, 'Frame', 0, len(Spacetime) - 1, valinit=0, valstep=1)

    def update_slider(val):
        f = int(slider.val)
        im.set_data(Spacetime[f])
        title.set_text(f"Frame: {f}")
        fig.canvas.draw_idle()

    slider.on_changed(update_slider)

    # Buttons widgets
    ax_prev = plt.axes([0.15, 0.02, 0.1, 0.05])
    ax_next = plt.axes([0.75, 0.02, 0.1, 0.05])
    btn_prev = Button(ax_prev, 'Prev')
    btn_next = Button(ax_next, 'Next')

    def prev_frame(event):
        slider.set_val(max(0, slider.val - 1))

    def next_frame(event):
        slider.set_val(min(len(Spacetime) - 1, slider.val + 1))

    btn_prev.on_clicked(prev_frame)
    btn_next.on_clicked(next_frame)

    plt.show()


elif PLOT_METHOD == 'html':

    print('plot html')
    from IPython.display import HTML

    # Create your figure and initial frame
    fig, ax = plt.subplots()

    im = ax.imshow(Spacetime[0], interpolation='nearest', vmin=0, vmax=np.max(Spacetime), cmap='viridis')
    title = ax.set_title("Frame: 0")

    def update(frame):
        """This function is called for every frame of the animation"""
        im.set_data(Spacetime[frame])
        title.set_text(f"Frame: {frame}")
        return [im, title]

    # Prevent matplotlib from displaying a duplicate static plot before the video
    plt.close(fig)

    ani = FuncAnimation(fig, update, frames=len(Spacetime), interval=100, blit=False, repeat=False)

    # Render interactive HTML5 animation controls directly in Colab output
    display(HTML(ani.to_jshtml()))
