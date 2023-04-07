# Evasion Attacks to Graph Neural Networks via Influence Function
An official implementation of the paper **["Evasion Attacks to Graph Neural Networks via Influence Function"]**.
## Running the code
For example, to check the performance of our one-time white box attack against GCN in cora dataset, run the following code:
```
cd OTA
```

```
python main.py --attack_object white --model GCN --dataset cora --attack_algorithm one-time
```
