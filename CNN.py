# Librerias necesarias
import tensorflow as tf
from tensorflow import keras
from keras import layers

# Librerias complementarias
import numpy as np
import matplotlib.pyplot as plt


# PLAN: Estableciendo los data-sets y las etiquetas de cada posible valor del atributo target.
# Data
datos = keras.datasets.fashion_mnist

# Spliting automatico
(train_inputs, train_outputs), (test_inputs, test_outputs) = datos.load_data()

# Etiquetando cada elemento del tensor OutPut
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# CHECK
print("Shape de train_inputs:", train_inputs.shape)
print("Shape de train_outputs:", train_outputs.shape)
print("Shape de test_inputs:", test_inputs.shape)
print("Shape de test_outputs:", test_outputs.shape)

# Normalizamos los tensores de entrenamiento
    # Para mejorar la convergencia y las diferencias de escalas
train_inputs = train_inputs / 255.0
test_inputs = test_inputs / 255.0

# DO
# ++ Capa de entrada
input_layer = keras.Input(shape=(28,28,1)) # -> 28 px * 28 px, con 1 canal de color

# + Capa de convolucion
    # Hiperparametros:
        # n_filtros = 64
        # filtro_size = (3,3)
        # padding='same' para que se conserve la dimension del tensor de entrada a la salida de la capa
x = layers.Conv2D(64,(3,3), activation='relu', padding='same')(input_layer)

# + Capa de MaxPooling
    # Size del Filtro_Convolucion = 2x2
x = layers.MaxPooling2D((2,2))(x)

# + Capa de Normalizacion para estabilizar
x = layers.BatchNormalization()(x)

# + Capa de aplanamiento para conectar con Capa_Densa
x = layers.Flatten()(x)

# + Capa 'full-conected' de 100 neuronas
    # Nota: El n de neuronas se puede optimizar con Meta-Learning
x = layers.Dense(100, activation='relu')(x)

# ++ Capa de Salida
n_clases = len(class_names) # Cantidad de clases del atributo target
x = layers.Dense(n_clases, activation='sigmoid')(x)

# Sellamos la arquitectura de NN para Clasificacion
output_layer = x


# Compilando
# Instanciamos el modelo en funcion a las arquitecturas de las Capas de Entrada & Capa de Salida
clasificador = keras.Model(input_layer, output_layer)

# Setteamos la 'Tasa de aprendizaje':
    # Learning Rate: Hiperparametro que determina cuanto deben cambiar los pesos de las conexiones entre las neuronas en cada iteracion
lr = tf.keras.optimizers.Adam(learning_rate=0.001) # Valor moderado
lr_alto = tf.keras.optimizers.Adam(learning_rate=0.01) # Valor alto -> Compila mas rapido pero aumenta el riesgo de OverShooting (no seleccionar el optimo)
lr_bajo = tf.keras.optimizers.Adam(learning_rate=0.0001) # Valor bajo -> Compila menos rapido pero disminuye el riesgo de OverShooting (no seleccionar el optimo)

# Compilando ...
clasificador.compile(optimizer = lr, loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# CHECK: Resumen de las Capas
clasificador.summary()

# DO: Trainning
# Entranamos al modelo
registro = clasificador.fit(train_inputs, train_outputs,
                epochs=5, # Cuantas veces la maquina recorrera los datos
                batch_size=100, # Cuantas muestras tomara para ajustar los pesos en el siguiente ...
                shuffle=True,
                validation_data=(test_inputs, test_outputs))

# Check -> Indicadores de Machine
# ACCURACY
plt.figure()
plt.title("Accuracy = f(epocas)")
plt.plot( registro.history['accuracy'])
plt.plot( registro.history['val_accuracy'] )
plt.legend(['Training Accuracy', 'Testing Accuracy'])
# LOSS
plt.figure()
plt.title("Loss = f(epocas)")
plt.plot( registro.history['loss'] )
plt.plot( registro.history['val_loss'] )
plt.legend(['Training Loss', 'Trainning Loss'])


# CHECK: Matriz de Confusion
predictions = clasificador.predict(test_inputs)

# Importando librerias y metodos necesarios
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

cm=confusion_matrix(test_outputs, np.argmax(predictions,axis=1))
cm_df=pd.DataFrame(cm)
plt.figure(figsize=(10,10))
sns.heatmap(cm_df, annot=True, cmap= "YlGnBu", annot_kws={"size": 20}, cbar=False, fmt='g')
plt.xlabel('Predichos como')
plt.ylabel('Valor real del Target')



# Recordando ...

# Etiquetando cada elemento del tensor OutPut
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

class_df = pd.DataFrame(class_names)
print (class_df)


plt.figure(figsize=(10,10))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_inputs[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_outputs[i]])
plt.show()
