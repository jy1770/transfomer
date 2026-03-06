# transfomer
数据来源 : https://www.statmt.org/wmt14/translation-task.html  
运行环境 : python3.10.1 , PyTorch: 2.9.1+cu128  
有一个需求 : cuda的序号必须从0开始并且连续。  
Data的格式为,这个文件夹需要按照这个格式创建，下载的训练数据放在RaWTrainData中，下载的评估数据放在RawTestData中  
Data/  
├── .pt/  
├── RawData/  
│   ├── RawTestData/  
│   └── RawTrainData/  
├── Data/  
│   ├── TestData/  
│   └── TrainData/  
最终实现的效果(d_model=512,en-fr)  
#10000 steps  
BLEU = 32.66 62.5/38.8/26.1/17.9 (BP = 1.000 ratio = 1.000 hyp_len = 77319 ref_len = 77306)  
BLEU = 32.656831315117756  
#20000 steps  
BLEU = 35.09 64.0/41.3/28.6/20.1 (BP = 1.000 ratio = 1.014 hyp_len = 78388 ref_len = 77306)  
BLEU = 35.09044594329327  
#30000 steps  
BLEU = 36.27 65.2/42.5/29.7/21.0 (BP = 1.000 ratio = 1.004 hyp_len = 77610 ref_len = 77306)  
BLEU = 36.27171626809166  
#40000 steps  
BLEU = 36.95 65.7/43.2/30.3/21.6 (BP = 1.000 ratio = 1.001 hyp_len = 77345 ref_len = 77306)  
BLEU = 36.95083404306642  
#50000 steps  
BLEU = 37.13 66.4/43.9/31.0/22.1 (BP = 0.988 ratio = 0.988 hyp_len = 76386 ref_len = 77306)  
BLEU = 37.12782297081585  
#60000 steps  
BLEU = 37.92 66.6/44.4/31.5/22.8 (BP = 0.993 ratio = 0.993 hyp_len = 76775 ref_len = 77306)  
BLEU = 37.91704972455014  
#70000 steps  
BLEU = 38.36 66.6/44.6/31.8/23.0 (BP = 1.000 ratio = 1.000 hyp_len = 77289 ref_len = 77306)  
BLEU = 38.36171372677844  
#80000 steps  
BLEU = 38.51 66.9/44.9/32.1/23.3 (BP = 0.995 ratio = 0.995 hyp_len = 76886 ref_len = 77306)  
BLEU = 38.51351751735988  
#90000 steps  
BLEU = 38.47 66.8/44.8/32.0/23.2 (BP = 0.996 ratio = 0.996 hyp_len = 77000 ref_len = 77306)  
BLEU = 38.46710388870065  
#100000 steps  
BLEU = 38.55 66.8/44.9/32.1/23.3 (BP = 0.996 ratio = 0.996 hyp_len = 76993 ref_len = 77306)  
BLEU = 38.55037097078598  
#100000 steps (Beam)  
BLEU = 39.10 67.3/45.7/32.8/24.0 (BP = 0.992 ratio = 0.992 hyp_len = 76652 ref_len = 77306)  
BLEU = 39.102823673040824  
