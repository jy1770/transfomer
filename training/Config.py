from torch.nn.parallel import DistributedDataParallel as DDP
from training.Encoder import *
from training.Decoder import *
from training.Transfomer  import *
from training.Transfomer  import *

class Config():
    def __init__ (self,args,PadId,device,ProcessId):
        self.d_model=args.d_model
        self.h = args.HeadNum
        self.d_ff = args.d_ff
        self.dropout = args.dropout
        self.N = args.N
        self.vocab_size = args.vocab_size
        self.PadId = PadId
        self.device = device
        self.ProcessId = ProcessId
    def make_model(self):
        # —————————————— # 实例化模型
        encoder = Encoder(self.d_model,self.d_ff,self.h,self.N,self.vocab_size,self.dropout,self.ProcessId)
        decoder = Decoder(self.d_model,self.d_ff,self.h,self.N,self.vocab_size,self.dropout,self.ProcessId)
        transfomer = Transfomer(encoder,decoder,self.PadId,self.h,self.device).to(self.device)
        # —————————————— # 初始化参数
        transfomer.apply(init_weights)
        # —————————————— # DDP包装 
        transfomer = DDP(transfomer)
        return transfomer