<!--
 * @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @Date: 2023-02-28 15:29:44
 * @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 * @LastEditTime: 2023-03-18 11:31:36
 * @FilePath: /mru/Knowledge-Distillation/ReadMe.md
 * @Description: 
 * 
-->
# Knowledge Distillation
**2023 Spring, Data Science HW2**

[\[Spec\]](./assests/HW2-Model%20Compression-v1.pdf), [\[Lecture\]](./assests/Lecture%202%20-%20Model%20Compression.pdf)

<br/>

## Folder structure
```
.
├── assets/
├── data (after running the code below)/
├── Readme.md
├── run_seed.sh
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

Note: The well-trained pruned model can be found at `./assests`

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
    python3 main.py --fname testing
    ```

- Evaluation:
    ```sh
    python3 predict.py ./logs/testing/pruned_model.pth
    ```
    Note: running this code will generate `pred.csv`

- Directly test the well-trained pruned model
    ```sh
    python3 predict.py ./assests/best_pruned_model.pth
    ```
    Note: running this code will generate `pred.csv`

<br/>

## Hyperparmeter tuning
- seed tuning
    ```sh
    chmod +x ./run_seed.sh
    ./run_seed.sh
    ```
    Note: one can modify this file to tune any parameter

<br/>

## Referenced
- https://github.com/DefangChen/SimKD
- https://github.com/VainF/Torch-Pruning