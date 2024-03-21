<h1> Seguimiento de Trayectorias - Manipulador Robotico - DRL </h1>

![Zona de trabajo](https://github.com/)
<h2> Description </h2>

- Seguimiento de Trayectorias usando un manipulador robotico (Franka Emika Panda - 7 DoF) con un sistema de control moderno basado en DRL en la plataforma de simulacion CoppeliaSim.
- La entrada al modelo de Red Neuronal Convolucional es un ROI de una imagen preprocesada y la salida es un vector con tres parametros de posicion (X, Y) y orientacion (Z) del efector final. el robot esta configurado en modo IK desde el simulador y solo se toma el control de movimeinto del efector final. Las trayectorias son generadas en la escena de simulacion. 
<h2> Software: </h2>

- CoppeliaSim 6.1
- Python 3.9.16
- keras 2.11
- Numpy 1.21.5
- OpenCV 4.6
- Matplotlib 0.1.6
- Tensorflow 2.11
<h2> Contenido </h2>

- Escena CoppeliaSim para Inferencia y Entrenamiento.
- Archivos .ipynb para Inferencia y Entrenamiento.
- Archivo .py del entorno
- Modelo entrenado con 10k pasos de tiempo.
<h2> Procedimiento </h2>

- Entrenamiento: Abrir y ejecutar escena de Entrenamiento y archivo.ipynb de Entrenamiento
- Evaluación: Abrir y ejecutar escena de Evaluación y archivo.ipynb de Evaluacion.
<h2> Instalacion de Gymnasium </h2>

![Zona de trabajo](https://github.com/Roger)
<h2> Media </h2>

Demo video: Robot training - https://youtu.be/4M1eHd9K1vw
