========================================================================
CARPETA DE RECOPILACIÓN: OJOS DE ANIMALES
========================================================================

Instrucciones para entrenar con ojos de animales:

1. Guarda aquí las imágenes completas de tus mascotas o animales (perros, gatos, etc.).
2. Para entrenar el modelo:
   - Recorta manualmente en forma cuadrada el área de los ojos de los animales.
   - Guarda esos recortes cuadrados directamente en la carpeta principal de entrenamiento:
     ./Ojos/img_transformada_animal_01.jpg
     ./Ojos/img_transformada_animal_02.jpg
     (y así sucesivamente...)
3. Una vez colocados, vuelve a ejecutar el entrenamiento con:
   python train_model.py

Esto reentrenará la Red Neuronal Convolucional (CNN) real para que aprenda a reconocer tanto ojos humanos como ojos de animales con la misma alta precisión.
========================================================================
