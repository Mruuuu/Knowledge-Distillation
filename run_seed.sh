###
 # @Author: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @Date: 2023-03-14 10:52:31
 # @LastEditors: Yen-Ju Chen  mru.11@nycu.edu.tw
 # @LastEditTime: 2023-03-18 12:16:01
 # @FilePath: /mru/Knowledge-Distillation/run_seed.sh
 # @Description: 
 # 
### 
source myenv/bin/activate
tmp=3
log_root="./logs/seed"
for seed in 777 77 7 666 66 6 555 55 5
do
    python3 main.py --log_root $log_root --kd_T 2.0 --kd_weight 0.4 --cls_weight 0.8 --optimizer adam --seed $seed --device cuda:3 --fname $((1 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 4.0 --kd_weight 0.6 --cls_weight 0.6 --optimizer adam --seed $seed --device cuda:3 --fname $((2 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 4.0 --kd_weight 0.2 --cls_weight 0.6 --optimizer adam --seed $seed --device cuda:3 --fname $((3 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 8.0 --kd_weight 0.2 --cls_weight 0.2 --optimizer adam --seed $seed --device cuda:3 --fname $((4 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 2.0 --kd_weight 0.4 --cls_weight 0.8 --optimizer adam --seed $seed --device cuda:2 --fname $((5 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 4.0 --kd_weight 0.2 --cls_weight 0.6 --optimizer adam --seed $seed --device cuda:2 --fname $((6 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 8.0 --kd_weight 0.2 --cls_weight 0.2 --optimizer adam --seed $seed --device cuda:2 --fname $((7 + tmp * 8))&
    python3 main.py --log_root $log_root --kd_T 6.0 --kd_weight 0.4 --cls_weight 0.4 --optimizer adam --seed $seed --device cuda:2 --fname $((8 + tmp * 8))&
    wait
    tmp=$((tmp+1))
done