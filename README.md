Лабораторная работа 5
===
Решение задачи классификации изображений из набора данных Oregon Wildlife с использованием нейронных сетей глубокого обучения и техники обучения Fine Tuning
---
### 1  С использованием техники обучения Transfer Learning, оптимальной политики изменения темпа обучения, аугментации данных с оптимальными настройками обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife.
Для решения поставленной задачи были использованы техники аугментации данных с оптимальными параметрами, определенными в предыдущих лабораторных работах:

* Для яркости/контраста оптимальными параметрами были определены contrast_factor = 1, delta = 1, где delta - скалярная величина, добавляемая к значениям пикселей, а contrast_factor - множитель типа float для регулировки контраста.
* Для случайного вращения оптимальными параметрами были определены factor = (0, 0.025), где factor - величина, представленная как часть 2pi, представляющая нижнюю и верхнюю границу для вращения по и против часовой стрелки.
* Для добавления случайного шума оптимальным параметром был определен stddev = 0.01, где stddev - значение среднеквадратичного отклонения добавляемого шума.
* Для использования случайной части изображения оптимальными параметрами были определены resizing(235,235) RandomCrop(224,224), где height и width - высота и ширина соответственно.

В качестве оптимальной политики изменения темпа обучения была принята политика step_decay с параметрами:

initial_lrate = 0.01 - начальный темп обучения,
drop = 0.5,
epochs_drop = 5.0 

```
def build_model():
  inputs = tf.keras.Input(shape=(235, 235, 3))
  x = tf.keras.layers.GaussianNoise(stddev = 0.01)(inputs)
  x = tf.keras.layers.experimental.preprocessing.RandomCrop(224,224)(x)
  x = tf.keras.layers.experimental.preprocessing.RandomRotation(factor = (0, 0.025))(x)
  model = EfficientNetB0(input_tensor = x, include_top=False, pooling = 'avg', weights='imagenet')
  model.trainable = False
  x = tf.keras.layers.Flatten()(model.output)
  outputs = tf.keras.layers.Dense(NUM_CLASSES, activation = tf.keras.activations.softmax)(x)
  return tf.keras.Model(inputs=inputs, outputs=outputs)
```
**График метрики точности:**

![1](https://user-images.githubusercontent.com/59210216/113638884-8ff35000-9680-11eb-86db-5228bb93380c.jpg)

![2](https://user-images.githubusercontent.com/59210216/113638895-941f6d80-9680-11eb-9191-15417db6bbe5.jpg)

**График функции потерь:**

![3](https://user-images.githubusercontent.com/59210216/113638904-9bdf1200-9680-11eb-8814-3308a2fff2b0.jpg)

![4](https://user-images.githubusercontent.com/59210216/113638908-a0a3c600-9680-11eb-9903-1a563f46f96f.jpg)

#### Анализ полученных результатов:
