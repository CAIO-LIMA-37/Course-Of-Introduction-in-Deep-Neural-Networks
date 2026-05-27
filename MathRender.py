import numpy as np
import tensorflow as tf
from IPython.display import display, Markdown

def format_matrix_latex(matrix, precision=2):
    """Converte um array numpy para uma matriz LaTeX (bmatrix)."""
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)  
        
    lines = []
    for row in matrix:
        lines.append(" & ".join([f"{val:.{precision}f}" for val in row]))
        
    return r"\begin{bmatrix} " + r" \\ ".join(lines) + r" \end{bmatrix}"

def model_to_math_markdown(model, precision=2):
    """
    Constrói a string LaTeX aninhada de toda a rede neural Sequential do Keras.
    Gera uma função composta de fora para dentro com as equações matemáticas completas.
    """
    dense_layers = [layer for layer in model.layers if isinstance(layer, tf.keras.layers.Dense)]
    
    if not dense_layers:
        return "O modelo não possui camadas Densas para visualização."
        
    equation = r"\mathbf{x}"
    
    for layer in dense_layers:
        weights = layer.get_weights()
        if not weights:
            continue
            
        W, b = weights
        W_T = W.T
        
        W_str = format_matrix_latex(W_T, precision)
        b_str = format_matrix_latex(b, precision)
        
        # O núcleo da camada antes da ativação (W^T * x + b)
        inner = rf"{W_str} \cdot {equation} + {b_str}"
        
        activation_name = layer.activation.__name__
        
        # Injeção das fórmulas matemáticas completas (usando \exp para evitar matrizes microscópicas no expoente)
        if activation_name == 'relu':
            equation = rf"\max\left(0, {inner} \right)"
        elif activation_name == 'sigmoid':
            equation = rf"\frac{{1}}{{1 + \exp\left(-\left( {inner} \right)\right)}}"
        elif activation_name == 'tanh':
            equation = rf"\frac{{\exp\left({inner}\right) - \exp\left(-\left({inner}\right)\right)}}{{\exp\left({inner}\right) + \exp\left(-\left({inner}\right)\right)}}"
        elif activation_name == 'softmax':
            equation = rf"\frac{{\exp\left({inner}\right)}}{{\sum \exp\left({inner}\right)}}"
        elif activation_name == 'linear':
            equation = rf"\left( {inner} \right)"
        else:
            equation = rf"\text{{{activation_name}}}\left( {inner} \right)"
            
    # O uso do ambiente 'aligned' trava o alinhamento vertical e impede a quebra de linha do f(x)
    return f"$$ \\begin{{aligned}} f(\\mathbf{{x}}) &= {equation} \\end{{aligned}} $$"

class MathRenderCallback(tf.keras.callbacks.Callback):
    """Callback customizado para renderizar a equação e as métricas no notebook."""
    def __init__(self, print_every_n_epochs=10, precision=2):
        super().__init__()
        self.print_every_n_epochs = print_every_n_epochs
        self.precision = precision
        
    def on_epoch_end(self, epoch, logs=None):
        # O 'logs' é um dicionário que contém o 'loss' e outras métricas (como 'val_loss', 'accuracy')
        if (epoch + 1) % self.print_every_n_epochs == 0:
            logs = logs or {}
            
            # Formata as métricas para ficarem bonitas no texto
            metricas_texto = " | ".join([f"**{chave}**: {valor:.4f}" for chave, valor in logs.items()])
            
            # Renderiza o Título e as Métricas
            display(Markdown(f"### 🔄 Atualização - Época {epoch + 1}"))
            display(Markdown(f"📊 {metricas_texto}"))
            
            # Renderiza a Matemática
            math_md = model_to_math_markdown(self.model, self.precision)
            display(Markdown(math_md))
            print("-" * 60) # Separador visual