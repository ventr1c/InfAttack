# Efficient, Direct, and Restricted Black-Box Graph Evasion Attacks to Any-Layer Graph Neural Networks via Influence Function
An official PyTorch implementation of "Efficient, Direct, and Restricted Black-Box Graph Evasion Attacks to Any-Layer Graph Neural Networks via Influence Function" (WSDM 2024). [[paper]]()
## Note
The code has not been organized yet. We will organize it as soon as possible. Thanks.

## Full Version of Our Paper
You can check the full version of our paper at `.WSDM2024_InfluenceAttack_full.pdf` or this [link](https://github.com/ventr1c/InfAttack/blob/main/WSDM2024_InfluenceAttack_full.pdf).

## Running the code
For example, to check the performance of our one-time white box attack against GCN in cora dataset, run the following code:
```
cd OTA
```

```
python main.py --attack_object white --model GCN --dataset cora --attack_algorithm one-time
```

