# Intel Image Classification: Computer Vision Project

**Цель**: Разработать и обучить CNN для классификации 6 категорий объектов (гора, ледник, лес, здание, улица, море).

### Этапы

1. **Preprocessing**: Ресайз изображений, нормальзация, data augmentation (отражения, сдвиги) для предотвращения переобучения.
2. **Custom Model**: Построение собственной CNN с использованием архитектуры DenseNet (Свертки, MaxPool, Конкатенация, Dropout).
3. **Transfer Learning**: Fine-tuning ResNet18 c использование различных lr для финального и сверточного слоя.

## Результаты

### 1. Custom net

| Accuracy | F1 Score | Precision | Recall |
| :------- | :------- | :-------- | :----- |
| 0.8800   | 0.8813   | 0.8812    | 0.8818 |

| True \ Predicted | buildings | forest | glacier | mountain |  sea  | street |
| :--------------- | :-------: | :----: | :-----: | :------: | :---: | :----: |
| buildings        |    182    |   2    |    0    |    5     |   3   |   18   |
| forest           |     1     |  228   |    1    |    3     |   0   |   1    |
| glacier          |     2     |   1    |   236   |    30    |  16   |   0    |
| mountain         |     0     |   2    |   28    |   215    |   9   |   0    |
| sea              |     6     |   0    |   13    |    4     |  236  |   1    |
| street           |    25     |   2    |    2    |    0     |   5   |  223   |

**Custom net**: <br>
**Total params** = 375,622 <br>
**Trainable params** = 375,622

### 2. Resnet

| Accuracy | F1 Score | Precision | Recall |
| :------- | :------- | :-------- | :----- |
| 0.9380   | 0.9386   | 0.9382    | 0.9397 |

| True \ Predicted | buildings | forest | glacier | mountain |  sea  | street |
| :--------------- | :-------: | :----: | :-----: | :------: | :---: | :----: |
| buildings        |    200    |   0    |    1    |    0     |   0   |   9    |
| forest           |     0     |  230   |    1    |    1     |   1   |   1    |
| glacier          |     2     |   0    |   256   |    19    |   8   |   0    |
| mountain         |     2     |   2    |   18    |   229    |   3   |   0    |
| sea              |     0     |   0    |    3    |    1     |  256  |   0    |
| street           |    19     |   0    |    0    |    0     |   2   |  236   |

**Resnet** <br>
**Total params** = 11,179,590 <br>
**Trainable params** = 8,396,806

---

Веса должны лежать в папке `checkpoints` (веса доступны в Releases)
[Датасет](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) должен лежать в папке `data`