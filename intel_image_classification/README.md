# Intel Image Classification: Computer Vision Project

**Цель**: Разработать и обучить CNN для классификации 6 категорий объектов (гора, ледник, лес, здание, улица, море).

## Этапы

1. **Preprocessing**: Ресайз изображений, нормальзация, data augmentation (отражения, сдвиги) для предотвращения переобучения.
2. **Custom Model**: Построение собственной CNN с использованием архитектуры DenseNet (Свертки, MaxPool, Конкатенация, Dropout).
3. **Transfer Learning**: Fine-tuning ResNet18 c использование различных lr для финального и сверточного слоя.

## Результаты

### Custom net

<div style="display: flex; margin-top: -20px"><div style="margin-right: 20px"><h4>Confusion matrix</h4><style type="text/css">
#T_04d07_row0_col0, #T_04d07_row1_col1, #T_04d07_row2_col2, #T_04d07_row3_col3, #T_04d07_row4_col4, #T_04d07_row5_col5 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_04d07_row0_col1, #T_04d07_row2_col0, #T_04d07_row3_col1, #T_04d07_row5_col1, #T_04d07_row5_col2 {
  background-color: #f5fafe;
  color: #000000;
}
#T_04d07_row0_col2, #T_04d07_row1_col4, #T_04d07_row2_col5, #T_04d07_row3_col0, #T_04d07_row3_col5, #T_04d07_row4_col1, #T_04d07_row5_col3 {
  background-color: #f7fbff;
  color: #000000;
}
#T_04d07_row0_col3, #T_04d07_row5_col4 {
  background-color: #f3f8fe;
  color: #000000;
}
#T_04d07_row0_col4, #T_04d07_row1_col3 {
  background-color: #f5f9fe;
  color: #000000;
}
#T_04d07_row0_col5 {
  background-color: #e7f1fa;
  color: #000000;
}
#T_04d07_row1_col0, #T_04d07_row1_col2, #T_04d07_row1_col5, #T_04d07_row2_col1, #T_04d07_row4_col5 {
  background-color: #f6faff;
  color: #000000;
}
#T_04d07_row2_col3, #T_04d07_row5_col0 {
  background-color: #dce9f6;
  color: #000000;
}
#T_04d07_row2_col4 {
  background-color: #eaf2fb;
  color: #000000;
}
#T_04d07_row3_col2 {
  background-color: #dfecf7;
  color: #000000;
}
#T_04d07_row3_col4 {
  background-color: #f0f6fd;
  color: #000000;
}
#T_04d07_row4_col0 {
  background-color: #f1f7fd;
  color: #000000;
}
#T_04d07_row4_col2 {
  background-color: #ecf4fb;
  color: #000000;
}
#T_04d07_row4_col3 {
  background-color: #f4f9fe;
  color: #000000;
}
</style>
<table id="T_04d07">
  <thead>
    <tr>
      <th class="index_name level0" >Predicted</th>
      <th id="T_04d07_level0_col0" class="col_heading level0 col0" >buildings</th>
      <th id="T_04d07_level0_col1" class="col_heading level0 col1" >forest</th>
      <th id="T_04d07_level0_col2" class="col_heading level0 col2" >glacier</th>
      <th id="T_04d07_level0_col3" class="col_heading level0 col3" >mountain</th>
      <th id="T_04d07_level0_col4" class="col_heading level0 col4" >sea</th>
      <th id="T_04d07_level0_col5" class="col_heading level0 col5" >street</th>
    </tr>
    <tr>
      <th class="index_name level0" >True</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_04d07_level0_row0" class="row_heading level0 row0" >buildings</th>
      <td id="T_04d07_row0_col0" class="data row0 col0" >182</td>
      <td id="T_04d07_row0_col1" class="data row0 col1" >2</td>
      <td id="T_04d07_row0_col2" class="data row0 col2" >0</td>
      <td id="T_04d07_row0_col3" class="data row0 col3" >5</td>
      <td id="T_04d07_row0_col4" class="data row0 col4" >3</td>
      <td id="T_04d07_row0_col5" class="data row0 col5" >18</td>
    </tr>
    <tr>
      <th id="T_04d07_level0_row1" class="row_heading level0 row1" >forest</th>
      <td id="T_04d07_row1_col0" class="data row1 col0" >1</td>
      <td id="T_04d07_row1_col1" class="data row1 col1" >228</td>
      <td id="T_04d07_row1_col2" class="data row1 col2" >1</td>
      <td id="T_04d07_row1_col3" class="data row1 col3" >3</td>
      <td id="T_04d07_row1_col4" class="data row1 col4" >0</td>
      <td id="T_04d07_row1_col5" class="data row1 col5" >1</td>
    </tr>
    <tr>
      <th id="T_04d07_level0_row2" class="row_heading level0 row2" >glacier</th>
      <td id="T_04d07_row2_col0" class="data row2 col0" >2</td>
      <td id="T_04d07_row2_col1" class="data row2 col1" >1</td>
      <td id="T_04d07_row2_col2" class="data row2 col2" >236</td>
      <td id="T_04d07_row2_col3" class="data row2 col3" >30</td>
      <td id="T_04d07_row2_col4" class="data row2 col4" >16</td>
      <td id="T_04d07_row2_col5" class="data row2 col5" >0</td>
    </tr>
    <tr>
      <th id="T_04d07_level0_row3" class="row_heading level0 row3" >mountain</th>
      <td id="T_04d07_row3_col0" class="data row3 col0" >0</td>
      <td id="T_04d07_row3_col1" class="data row3 col1" >2</td>
      <td id="T_04d07_row3_col2" class="data row3 col2" >28</td>
      <td id="T_04d07_row3_col3" class="data row3 col3" >215</td>
      <td id="T_04d07_row3_col4" class="data row3 col4" >9</td>
      <td id="T_04d07_row3_col5" class="data row3 col5" >0</td>
    </tr>
    <tr>
      <th id="T_04d07_level0_row4" class="row_heading level0 row4" >sea</th>
      <td id="T_04d07_row4_col0" class="data row4 col0" >6</td>
      <td id="T_04d07_row4_col1" class="data row4 col1" >0</td>
      <td id="T_04d07_row4_col2" class="data row4 col2" >13</td>
      <td id="T_04d07_row4_col3" class="data row4 col3" >4</td>
      <td id="T_04d07_row4_col4" class="data row4 col4" >236</td>
      <td id="T_04d07_row4_col5" class="data row4 col5" >1</td>
    </tr>
    <tr>
      <th id="T_04d07_level0_row5" class="row_heading level0 row5" >street</th>
      <td id="T_04d07_row5_col0" class="data row5 col0" >25</td>
      <td id="T_04d07_row5_col1" class="data row5 col1" >2</td>
      <td id="T_04d07_row5_col2" class="data row5 col2" >2</td>
      <td id="T_04d07_row5_col3" class="data row5 col3" >0</td>
      <td id="T_04d07_row5_col4" class="data row5 col4" >5</td>
      <td id="T_04d07_row5_col5" class="data row5 col5" >223</td>
    </tr>
  </tbody>
</table>
</div><div style="margin-right: 20px"><h4>Metrics</h4><style type="text/css">
</style>
<table id="T_eed66">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_eed66_level0_col0" class="col_heading level0 col0" >Metric</th>
      <th id="T_eed66_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_eed66_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_eed66_row0_col0" class="data row0 col0" >Accuracy</td>
      <td id="T_eed66_row0_col1" class="data row0 col1" >0.8800</td>
    </tr>
    <tr>
      <th id="T_eed66_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_eed66_row1_col0" class="data row1 col0" >F1</td>
      <td id="T_eed66_row1_col1" class="data row1 col1" >0.8813</td>
    </tr>
    <tr>
      <th id="T_eed66_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_eed66_row2_col0" class="data row2 col0" >Precision</td>
      <td id="T_eed66_row2_col1" class="data row2 col1" >0.8812</td>
    </tr>
    <tr>
      <th id="T_eed66_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_eed66_row3_col0" class="data row3 col0" >Recall</td>
      <td id="T_eed66_row3_col1" class="data row3 col1" >0.8818</td>
    </tr>
  </tbody>
</table>
</div></div>

**Custom net**: <br>
**Total params** = 375,622 <br>
**Trainable params** = 375,622

### Resnet

<div style="display: flex; margin-top: -20px"><div style="margin-right: 20px"><h4>Confusion matrix</h4><style type="text/css">
#T_b834c_row0_col0, #T_b834c_row1_col1, #T_b834c_row2_col2, #T_b834c_row3_col3, #T_b834c_row4_col4, #T_b834c_row5_col5 {
  background-color: #08306b;
  color: #f1f1f1;
}
#T_b834c_row0_col1, #T_b834c_row0_col3, #T_b834c_row0_col4, #T_b834c_row1_col0, #T_b834c_row2_col1, #T_b834c_row2_col5, #T_b834c_row3_col5, #T_b834c_row4_col0, #T_b834c_row4_col1, #T_b834c_row4_col5, #T_b834c_row5_col1, #T_b834c_row5_col2, #T_b834c_row5_col3 {
  background-color: #f7fbff;
  color: #000000;
}
#T_b834c_row0_col2, #T_b834c_row1_col2, #T_b834c_row1_col3, #T_b834c_row1_col4, #T_b834c_row1_col5, #T_b834c_row4_col3 {
  background-color: #f6faff;
  color: #000000;
}
#T_b834c_row0_col5 {
  background-color: #f0f6fd;
  color: #000000;
}
#T_b834c_row2_col0, #T_b834c_row3_col0, #T_b834c_row3_col1, #T_b834c_row5_col4 {
  background-color: #f5fafe;
  color: #000000;
}
#T_b834c_row2_col3 {
  background-color: #e7f0fa;
  color: #000000;
}
#T_b834c_row2_col4 {
  background-color: #f1f7fd;
  color: #000000;
}
#T_b834c_row3_col2 {
  background-color: #e9f2fa;
  color: #000000;
}
#T_b834c_row3_col4, #T_b834c_row4_col2 {
  background-color: #f5f9fe;
  color: #000000;
}
#T_b834c_row5_col0 {
  background-color: #e4eff9;
  color: #000000;
}
</style>
<table id="T_b834c">
  <thead>
    <tr>
      <th class="index_name level0" >Predicted</th>
      <th id="T_b834c_level0_col0" class="col_heading level0 col0" >buildings</th>
      <th id="T_b834c_level0_col1" class="col_heading level0 col1" >forest</th>
      <th id="T_b834c_level0_col2" class="col_heading level0 col2" >glacier</th>
      <th id="T_b834c_level0_col3" class="col_heading level0 col3" >mountain</th>
      <th id="T_b834c_level0_col4" class="col_heading level0 col4" >sea</th>
      <th id="T_b834c_level0_col5" class="col_heading level0 col5" >street</th>
    </tr>
    <tr>
      <th class="index_name level0" >True</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
      <th class="blank col4" >&nbsp;</th>
      <th class="blank col5" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_b834c_level0_row0" class="row_heading level0 row0" >buildings</th>
      <td id="T_b834c_row0_col0" class="data row0 col0" >200</td>
      <td id="T_b834c_row0_col1" class="data row0 col1" >0</td>
      <td id="T_b834c_row0_col2" class="data row0 col2" >1</td>
      <td id="T_b834c_row0_col3" class="data row0 col3" >0</td>
      <td id="T_b834c_row0_col4" class="data row0 col4" >0</td>
      <td id="T_b834c_row0_col5" class="data row0 col5" >9</td>
    </tr>
    <tr>
      <th id="T_b834c_level0_row1" class="row_heading level0 row1" >forest</th>
      <td id="T_b834c_row1_col0" class="data row1 col0" >0</td>
      <td id="T_b834c_row1_col1" class="data row1 col1" >230</td>
      <td id="T_b834c_row1_col2" class="data row1 col2" >1</td>
      <td id="T_b834c_row1_col3" class="data row1 col3" >1</td>
      <td id="T_b834c_row1_col4" class="data row1 col4" >1</td>
      <td id="T_b834c_row1_col5" class="data row1 col5" >1</td>
    </tr>
    <tr>
      <th id="T_b834c_level0_row2" class="row_heading level0 row2" >glacier</th>
      <td id="T_b834c_row2_col0" class="data row2 col0" >2</td>
      <td id="T_b834c_row2_col1" class="data row2 col1" >0</td>
      <td id="T_b834c_row2_col2" class="data row2 col2" >256</td>
      <td id="T_b834c_row2_col3" class="data row2 col3" >19</td>
      <td id="T_b834c_row2_col4" class="data row2 col4" >8</td>
      <td id="T_b834c_row2_col5" class="data row2 col5" >0</td>
    </tr>
    <tr>
      <th id="T_b834c_level0_row3" class="row_heading level0 row3" >mountain</th>
      <td id="T_b834c_row3_col0" class="data row3 col0" >2</td>
      <td id="T_b834c_row3_col1" class="data row3 col1" >2</td>
      <td id="T_b834c_row3_col2" class="data row3 col2" >18</td>
      <td id="T_b834c_row3_col3" class="data row3 col3" >229</td>
      <td id="T_b834c_row3_col4" class="data row3 col4" >3</td>
      <td id="T_b834c_row3_col5" class="data row3 col5" >0</td>
    </tr>
    <tr>
      <th id="T_b834c_level0_row4" class="row_heading level0 row4" >sea</th>
      <td id="T_b834c_row4_col0" class="data row4 col0" >0</td>
      <td id="T_b834c_row4_col1" class="data row4 col1" >0</td>
      <td id="T_b834c_row4_col2" class="data row4 col2" >3</td>
      <td id="T_b834c_row4_col3" class="data row4 col3" >1</td>
      <td id="T_b834c_row4_col4" class="data row4 col4" >256</td>
      <td id="T_b834c_row4_col5" class="data row4 col5" >0</td>
    </tr>
    <tr>
      <th id="T_b834c_level0_row5" class="row_heading level0 row5" >street</th>
      <td id="T_b834c_row5_col0" class="data row5 col0" >19</td>
      <td id="T_b834c_row5_col1" class="data row5 col1" >0</td>
      <td id="T_b834c_row5_col2" class="data row5 col2" >0</td>
      <td id="T_b834c_row5_col3" class="data row5 col3" >0</td>
      <td id="T_b834c_row5_col4" class="data row5 col4" >2</td>
      <td id="T_b834c_row5_col5" class="data row5 col5" >236</td>
    </tr>
  </tbody>
</table>
</div><div style="margin-right: 20px"><h4>Metrics</h4><style type="text/css">
</style>
<table id="T_973b0">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_973b0_level0_col0" class="col_heading level0 col0" >Metric</th>
      <th id="T_973b0_level0_col1" class="col_heading level0 col1" >Value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_973b0_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_973b0_row0_col0" class="data row0 col0" >Accuracy</td>
      <td id="T_973b0_row0_col1" class="data row0 col1" >0.9380</td>
    </tr>
    <tr>
      <th id="T_973b0_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_973b0_row1_col0" class="data row1 col0" >F1</td>
      <td id="T_973b0_row1_col1" class="data row1 col1" >0.9386</td>
    </tr>
    <tr>
      <th id="T_973b0_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_973b0_row2_col0" class="data row2 col0" >Precision</td>
      <td id="T_973b0_row2_col1" class="data row2 col1" >0.9382</td>
    </tr>
    <tr>
      <th id="T_973b0_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_973b0_row3_col0" class="data row3 col0" >Recall</td>
      <td id="T_973b0_row3_col1" class="data row3 col1" >0.9397</td>
    </tr>
  </tbody>
</table>
</div></div>

**Resnet** <br>
**Total params** = 11,179,590 <br>
**Trainable params** = 8,396,806

---

Веса должны лежать в папке `checkpoints` (веса доступны в Releases)
[Датасет](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) должен лежать в папке `data`

