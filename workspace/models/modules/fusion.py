import torch
import torch.nn as nn
import torch.nn.functional as F

# class Cfgs(BaseCfgs):
#     def __init__(self):
#         super(Cfgs, self).__init__()

#         self.HIGH_ORDER = False
#         self.HIDDEN_SIZE = 2048
#         self.MFB_K = 5
#         self.MFB_O = 1000
#         self.LSTM_OUT_SIZE = 1024
#         self.DROPOUT_R = 0.1
#         self.I_GLIMPSES = 2
#         self.Q_GLIMPSES = 2


class MFB(nn.Module):
    def __init__(self, img_feat_size, ques_feat_size, is_first=True, MFB_K=5, MFB_O=2048, DROPOUT_R=0.1):
        super(MFB, self).__init__()
        self.is_first = is_first
        self.MFB_K = MFB_K
        self.MFB_O = MFB_O
        self.DROPOUT_R = DROPOUT_R
        self.proj_i = nn.Linear(img_feat_size, self.MFB_K * self.MFB_O)
        self.proj_q = nn.Linear(ques_feat_size, self.MFB_K * self.MFB_O)
        self.dropout = nn.Dropout(self.DROPOUT_R)
        self.pool = nn.AvgPool1d(self.MFB_K, stride=self.MFB_K)

    def forward(self, img_feat, ques_feat, exp_in=1):
        '''
            img_feat.size() -> (N, C, img_feat_size)    C = 1 or 100
            ques_feat.size() -> (N, 1, ques_feat_size)
            z.size() -> (N, C, MFB_O)
            exp_out.size() -> (N, C, K*O)
        '''
        batch_size = img_feat.shape[0]
        img_feat = self.proj_i(img_feat)                # (N, C, K*O)
        ques_feat = self.proj_q(ques_feat)              # (N, 1, K*O)

        exp_out = img_feat * ques_feat                  # (N, C, K*O)
        exp_out = self.dropout(exp_out) if self.is_first else self.dropout(exp_out * exp_in)     # (N, C, K*O)
        z = self.pool(exp_out) * self.MFB_K         # (N, C, O)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z.view(batch_size, -1))         # (N, C*O)
        z = z.view(batch_size, -1, self.MFB_O)      # (N, C, O)
        return z, exp_out
    

class CoAtt(nn.Module):
    def __init__(self, __C):
        super(CoAtt, self).__init__()
        self.__C = __C

        img_feat_size = __C.FEAT_SIZE[__C.DATASET]['FRCN_FEAT_SIZE'][1]
        img_att_feat_size = img_feat_size * __C.I_GLIMPSES
        ques_att_feat_size = __C.LSTM_OUT_SIZE * __C.Q_GLIMPSES

        self.q_att = QAtt(__C)
        self.i_att = IAtt(__C, img_feat_size, ques_att_feat_size)

        if self.__C.HIGH_ORDER:  # MFH
            self.mfh1 = MFB(__C, img_att_feat_size, ques_att_feat_size, True)
            self.mfh2 = MFB(__C, img_att_feat_size, ques_att_feat_size, False)
        else:  # MFB
            self.mfb = MFB(__C, img_att_feat_size, ques_att_feat_size, True)

    def forward(self, img_feat, ques_feat):
        '''
            img_feat.size() -> (N, C, FRCN_FEAT_SIZE)
            ques_feat.size() -> (N, T, LSTM_OUT_SIZE)
            z.size() -> MFH:(N, 2*O) / MFB:(N, O)
        '''
        ques_feat = self.q_att(ques_feat)               # (N, LSTM_OUT_SIZE*Q_GLIMPSES)
        fuse_feat = self.i_att(img_feat, ques_feat)     # (N, FRCN_FEAT_SIZE*I_GLIMPSES)

        if self.__C.HIGH_ORDER:  # MFH
            z1, exp1 = self.mfh1(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))        # z1:(N, 1, O)  exp1:(N, C, K*O)
            z2, _ = self.mfh2(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1), exp1)     # z2:(N, 1, O)  _:(N, C, K*O)
            z = torch.cat((z1.squeeze(1), z2.squeeze(1)), 1)                            # (N, 2*O)
        else:  # MFB
            z, _ = self.mfb(fuse_feat.unsqueeze(1), ques_feat.unsqueeze(1))             # z:(N, 1, O)  _:(N, C, K*O)
            z = z.squeeze(1)                                                            # (N, O)

        return z