import torch 
import os 
import time
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from functools import partial
from training.DataSet import *
from training.Config  import *
from Function.Function import *
from transformers.optimization import get_cosine_schedule_with_warmup
from torch.amp import autocast
from tqdm import tqdm

def split_decay(named_params):
    """把参数分成 decay / no_decay(LN / bias 不做 weight decay)"""
    decay, no_decay = [], []
    for n, p in named_params:
        if (not p.requires_grad) or (p is None):
            continue
        if n.endswith(".bias") or ("LayerNorm" in n) or ("layer_norm" in n) or ("layernorm" in n):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay

def make_param_groups(args,trans_named):
    param_groups = []
    # —————————————— # shared
    trans_decay, trans_no_decay = split_decay(trans_named)
    if trans_decay:
        param_groups.append({"params": trans_decay, "lr": args.lr_trans, "weight_decay": args.wd_trans})
    if trans_no_decay:
        param_groups.append({"params": trans_no_decay, "lr": args.lr_trans, "weight_decay": 0.0})
    return param_groups

def make_named(transfomer):
    return list(transfomer.named_parameters())

def save_model(args,transfomer,ProcessId,steps):
    if ProcessId == 0:
        torch.save(transfomer.module.state_dict(),f'{args.DataPath}/.pt/transfomer_{args.d_model}_{args.SrcName}_{args.TgtName}_{str(steps)}.pt')

def get_PadId(args):
    BPEPath = f'{args.DataPath}Data/TrainData/BPE-{args.SrcName}-{args.TgtName}.model'
    BPE = spm.SentencePieceProcessor(BPEPath) # 导入BPE
    return BPE.pad_id()

def function(args: argparse.Namespace):
    # —————————————— # 打开f32加速
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    # —————————————— # 初始化参数和文件
    ProcessId = GetProcessId()
    PadId = get_PadId(args)
    if ProcessId == 0:
        TrainLossPath = f'{args.DataPath}/TrainLoss/TrainLoss_1.txt'
        with open(TrainLossPath , 'w' ,encoding="utf-8") as f:
            f.write(f'LossSteps = {args.LossSteps}\n')
        LossSum = 0
    # —————————————— # 实例并应用数据生成器(DataLoader)
    modeldataset = ModelDataSet(args)
    batch_sampler = ModelSampler(modeldataset,args,ProcessId)
    loader = DataLoader(modeldataset,batch_sampler=batch_sampler,collate_fn=partial(collate_fn, PadId=PadId),num_workers=3,pin_memory=True)
    # —————————————— # 实例并应用模型参数配置器(Config)
    transfomerconfig = Config(args,PadId)
    transfomer = transfomerconfig.make_model()
    # —————————————— # 配置参数优化器
    # —————— # 【配置一 : transfomer原论文配置】
    '''
    optimizer = optim.Adam(transfomer.parameters(),lr=1.0,betas=(0.9, 0.98),eps=1e-9)
    scheduler = NoamLR(optimizer, d_model=args.d_model, warmup_steps=4000, factor=1.0)
    criterion = nn.CrossEntropyLoss(ignore_index=PadId, label_smoothing=0.1)
    '''
    # —————— # 【配置二 : 更灵活】
    trans_named = make_named(transfomer)
    param_groups = make_param_groups(args,trans_named)
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.98), eps=1e-8)
    scheduler = get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=args.warmup_steps,num_training_steps=args.max_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=PadId, label_smoothing=0.1)
    # —————————————— # 预先缓存参数列表
    trans_params = [p for _,p in trans_named]
    # —————————————— # 开始训练
    transfomer.train()
    steps = 0
    pbar = tqdm(total=args.max_steps, desc="Steps", disable = (ProcessId!=0), mininterval=1)
    try:
        while True:
            for batch in loader:
                # —————————————— # 初始化参数
                src,tgt = batch.src.cuda(),batch.tgt.cuda()
                optimizer.zero_grad(set_to_none=True)
                # —————————————— # 前向传播
                with autocast(device_type='cuda',dtype=torch.bfloat16):
                    out = transfomer(src, tgt[:, :-1])
                    out = out.contiguous().view(-1, args.vocab_size)
                    tgt_y = tgt[:, 1:].contiguous().view(-1)
                    loss = criterion(out, tgt_y)
                loss.backward()  # 反向传播放在 autocast 外面更稳一些
                del out,tgt_y
                # —————————————— # 梯度裁剪
                # —————— #【配置一：原论文配置】
                '''
                # 只裁剪“共享/同步”的参数(DDP里面的Transformer)
                shared_params = [p for p in transfomer.Transfomer.parameters() if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(shared_params, args.clip)
                # 专家参数单独裁剪（可选，但建议给一个更大的阈值或单独开关）
                expert_params = [p for p in transfomer.ExpertsSet.parameters() if p.grad is not None]
                torch.nn.utils.clip_grad_norm_(expert_params, args.clip)  # 或者 args.clip_expert
                '''
                # —————— # 【配置二：改进版】
                torch.nn.utils.clip_grad_norm_([p for p in trans_params  if p.grad is not None], args.clip_trans  )

                # —————————————— # 参数更新
                optimizer.step() ; scheduler.step() ; steps+=1 ; pbar.update(1)

                # —————————————— # 统计损失值
                if ProcessId == 0:
                    LossSum+=loss.item()
                    if steps%args.LossSteps==0:
                        with open(TrainLossPath , 'a' ,encoding="utf-8") as f:
                            f.write(f'{LossSum/args.LossSteps}\n')
                        LossSum = 0

                # —————————————— # 保存模型
                if steps % 10000==0:
                    save_model(args,transfomer,ProcessId,steps)
                    torch.cuda.empty_cache() # 清理显存块
                    if steps >= args.max_steps: 
                        if ProcessId == 0:
                            with open(TrainLossPath , 'a' ,encoding="utf-8") as f:
                                f.write(str(pbar))
                        time.sleep(60)
                        os.system("su -c 'shutdown -h now'")
                del loss
    except:
        time.sleep(3)
        os.system("su -c 'shutdown -h now'") # 笑死，出现错误直接关机，省钱         

def add_subparser(subparsers: argparse._SubParsersAction, parents=None):
    if parents is None:
        parents = []
    parser = subparsers.add_parser('train', help='数据处理',parents=parents)
    group = parser.add_argument_group('训练参数')
    # —————————————— # 训练基础参数
    group.add_argument("--LossSteps",default=1000   , type=str2int , help="记录loss的步数")
    group.add_argument("--S"        , default=18000 , type=str2int , help="单次训练token数")
    group.add_argument("--tf32"     , default=True  , type=str2bool, help="是否打开tf32")
    group.add_argument("--sort"     , default=True  , type=str2bool, help="是否桶内排序")
    # —————————————— # 配置器参数
    group.add_argument("--max_steps"   , default=100000, type=str2int, help="")
    group.add_argument("--warmup_steps", default=8000  , type=str2int, help="")
    # —————— # 模型参数
    group.add_argument("--lr_trans"  , default=5e-4, type=str2float, help="")
    group.add_argument("--wd_trans"  , default=1e-2, type=str2float, help="")
    group.add_argument("--clip_trans", default=1.0 , type=str2float, help="")

    parser.set_defaults(func = function)