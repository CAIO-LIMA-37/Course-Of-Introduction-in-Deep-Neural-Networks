import matplotlib.pyplot as plt
import tensorflow as tf

def plot_neural_graph(model, max_neurons_per_layer=15):
    """
    Desenha o grafo (nós e arestas) de um modelo Keras Sequential.
    Limita visualmente o número de neurônios para o gráfico não virar um borrão.
    """
    layer_sizes = []
    
    # Extrai o tamanho da entrada (Input)
    if hasattr(model, 'layers') and len(model.layers) > 0:
        if hasattr(model.layers[0], 'input_shape') and model.layers[0].input_shape is not None:
             in_dim = model.layers[0].input_shape[-1]
             layer_sizes.append(in_dim)
    
    # Extrai o tamanho das camadas Densas (Ocultas e Saída)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            layer_sizes.append(layer.units)
            
    if not layer_sizes:
        print("Modelo sem camadas Densas identificadas.")
        return

    # Limita o visual para redes gigantes não quebrarem o gráfico
    visual_sizes = [min(size, max_neurons_per_layer) for size in layer_sizes]
    
    fig = plt.figure(figsize=(10, 6))
    ax = fig.gca()
    ax.axis('off') # Esconde os eixos do gráfico
    
    # Margens e espaçamentos
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    n_layers = len(visual_sizes)
    h_spacing = (right - left) / float(max(n_layers - 1, 1))
    max_nodes = max(visual_sizes)
    v_spacing = (top - bottom) / float(max(max_nodes, 2))
    
    # 1. Desenhar Arestas (As conexões/pesos)
    for n, (size_a, size_b) in enumerate(zip(visual_sizes[:-1], visual_sizes[1:])):
        layer_top_a = v_spacing * (size_a - 1) / 2. + (top + bottom) / 2.
        layer_top_b = v_spacing * (size_b - 1) / 2. + (top + bottom) / 2.
        for m in range(size_a):
            for o in range(size_b):
                line = plt.Line2D([n * h_spacing + left, (n + 1) * h_spacing + left],
                                  [layer_top_a - m * v_spacing, layer_top_b - o * v_spacing], 
                                  c='gray', alpha=0.3, zorder=1)
                ax.add_artist(line)
                
    # 2. Desenhar Nós (Os neurônios)
    for n, size in enumerate(visual_sizes):
        layer_top = v_spacing * (size - 1) / 2. + (top + bottom) / 2.
        for m in range(size):
            # Define a cor dependendo se é entrada, oculta ou saída
            color = '#FF9999' if n == 0 else ('#99FF99' if n == n_layers - 1 else '#99CCFF')
            
            circle = plt.Circle((n * h_spacing + left, layer_top - m * v_spacing), 
                                v_spacing / 4., color=color, ec='black', zorder=2)
            ax.add_artist(circle)
            
            # Textos dentro das bolinhas
            if n == 0:
                label = f"In {m+1}"
            elif n == n_layers - 1:
                label = f"Out {m+1}"
            else:
                label = f"H {m+1}"
            
            # Se atingiu o limite, coloca '...' no último nó
            if m == size - 1 and layer_sizes[n] > max_neurons_per_layer:
                label = "..."
                
            ax.text(n * h_spacing + left, layer_top - m * v_spacing, label, 
                    ha='center', va='center', fontsize=8, zorder=3)
                    
    # Títulos da base
    plt.title(f"Arquitetura: {' -> '.join(map(str, layer_sizes))} Neurônios", fontsize=14, fontweight='bold')
    plt.text(left, bottom - 0.05, "Entrada (Features)", ha='center', fontsize=10)
    plt.text(right, bottom - 0.05, "Saída (Predição)", ha='center', fontsize=10)
    
    plt.show()