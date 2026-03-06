import torch
from testing.Encoder_ import *
from testing.Decoder_ import *
from testing.Transfomer_  import *
from testing.Transfomer_  import *

class Config():
    def __init__ (self,args,PadId,device):
        self.DataPath = args.DataPath
        self.num = args.num
        self.SrcName = args.SrcName
        self.TgtName = args.TgtName

        self.d_model=args.d_model
        self.h = args.HeadNum
        self.d_ff = args.d_ff
        self.N = args.N
        self.vocab_size = args.vocab_size
        self.PadId = PadId
        self.device = device

    def load_model(self):
        # —————————————— # 实例化模型
        encoder = Encoder(self.d_model,self.d_ff,self.h,self.N,self.vocab_size)
        decoder = Decoder(self.d_model,self.d_ff,self.h,self.N,self.vocab_size)
        transfomer = Transfomer(encoder,decoder,self.PadId,self.h,self.device)
        # —————————————— # 导入共享参数
        checkpoint = torch.load(f'{self.DataPath}/.pt1/transfomer_{self.d_model}_{self.SrcName}_{self.TgtName}_{self.num}.pt', map_location="cpu")
        transfomer.load_state_dict(checkpoint)
        return transfomer.to(self.device)