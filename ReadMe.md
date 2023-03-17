<!--
 * @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @Date: 2023-02-28 15:29:44
 * @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @LastEditTime: 2023-03-18 00:07:42
 * @FilePath: /mru/Knowledge-Distillation/ReadMe.md
 * @Description: 
 * 
-->
# Knowledge Distillation
**2023 Spring, Data Science HW2**

[Spec](./assests/HW2-Model%20Compression-v1.pdf)

<br/>

## Folder structure
```
.
├── assets/
├── data (after running the code below)/
├── Readme.md
├── resnet-50.pth
├── requirements.txt
├── kd_loss.py
├── main.py
├── model.py
├── sample_predict.py
├── train.py
└── utils.py
```
Note: Make sure that the pretrained model `resnet-50.pth` is at the path `./`

<br/>

## Environment setup
- Python 3.8.10
    ```sh
    pip3 install -r requirements.txt
    ```
    
<br/>

## Run code
- KD & pruning:
    ```sh
    python3 main.py --fname test
    ```

- Evaluation:
    ```sh
    python3 predict.py ./logs/test/pruned_model.pth
    ```
    Note: running this code will generate `pred.csv`

- Directly test the well-trained pruned model
    ```sh
    python3 predict.py TBD
    ```
    Note: running this code will generate `pred.csv`

## Referenced
- https://github.com/DefangChen/SimKD
- https://github.com/VainF/Torch-Pruning