import numpy as np  
import matplotlib.pyplot as plt  

class HopfieldNetwork:  
    def __init__(self, n_units):  
        self.n_units = n_units  
        self.weights = np.zeros((n_units, n_units))  

    def train(self, patterns):  
        for pattern in patterns:  
            pattern = np.reshape(pattern, (self.n_units,))  
            self.weights += np.outer(pattern, pattern)  
        # Establecer los pesos en 0 en la diagonal  
        np.fill_diagonal(self.weights, 0)  

    def run(self, initial_state, max_steps=100):  
        state = initial_state.copy()  
        for _ in range(max_steps):  
            new_state = np.sign(np.dot(self.weights, state))  
            new_state[new_state == 0] = 1  # Transformar ceros en unos  
            if np.array_equal(state, new_state):  
                break  
            state = new_state  
        return state  

def create_pattern_matrix(n, m):  
    """ Crea un patrón de 1s y -1s para la red """  
    return np.random.choice([-1, 1], size=(n, m))  

# Definir patrones de trabajo (ejemplo: imágenes de piezas)  
patterns = [  
    create_pattern_matrix(1, 16).flatten(),  # Patrón 1  
    create_pattern_matrix(1, 16).flatten(),  # Patrón 2  
    create_pattern_matrix(1, 16).flatten()   # Patrón 3  
]  

# Instancia de la red de Hopfield  
n_units = patterns[0].size  # Número de unidades es igual al tamaño de un patrón  
hopfield_net = HopfieldNetwork(n_units)  

# Entrenar la red  
hopfield_net.train(patterns)  

# Crear una entrada ruidosa (patrón 1 con ruido)  
noisy_input = patterns[0].copy()  
noisy_input[2] = -noisy_input[2]  # Cambiar un elemento para simular ruido  

# Ejecutar la red para recuperar el patrón  
retrieved_pattern = hopfield_net.run(noisy_input)  

# Función para visualizar patrones  
def plot_pattern(pattern, title):  
    plt.imshow(pattern.reshape(4, 4), cmap='gray')  
    plt.title(title)  
    plt.axis('off')  

# Visualizar patrones  
plt.figure(figsize=(8, 6))  

plt.subplot(1, 3, 1)  
plot_pattern(noisy_input, 'Entrada Ruidosa')  

plt.subplot(1, 3, 2)  
plot_pattern(patterns[0], 'Patrón Original')  

plt.subplot(1, 3, 3)  
plot_pattern(retrieved_pattern, 'Patrón Recuperado')  

plt.tight_layout()  
plt.show()  
