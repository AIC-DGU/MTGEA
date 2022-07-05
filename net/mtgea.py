from pyrsistent import v
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlight import import_class
import math



class MTGEA(nn.Module):

    def __init__(self, base_model, in_channels, num_class, output_class, graph_args,
                edge_importance_weighting, **kwargs):

        super().__init__()
        base_model = import_class(base_model)

        self.rad_stgcn = base_model(in_channels=in_channels, num_class=num_class,
                                        graph_args=graph_args,edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)
        self.kin_stgcn = base_model(in_channels=in_channels, num_class=num_class,
                                        graph_args=graph_args,edge_importance_weighting=edge_importance_weighting,
                                        **kwargs)


        self.fc_layer_1  = nn.Linear(num_class*2, output_class)



        self.rad_k = nn.Linear(num_class, num_class)
        self.rad_v = nn.Linear(num_class, num_class)

        self.kin_q = nn.Linear(num_class, num_class)
        self.softmax = nn.Softmax(-1)



    def att(self, query, key, value):
        d_k = query.size(-1)

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(d_k)
        att_map = F.softmax(scores, dim=-1)
        return torch.matmul(att_map, value)                                


    def forward(self, first, second):

        rad_output = self.rad_stgcn(first) 
        kin_output = self.kin_stgcn(second) 


        rad_k = self.rad_k(rad_output)
        rad_v = self.rad_v(rad_output)
        kin_q = self.kin_q(kin_output)


        att_output_rad = torch.sigmoid(self.att(kin_q, rad_k, rad_v))

        cat_rad = torch.cat([rad_output,att_output_rad], dim=1)
        output = self.fc_layer_1(cat_rad)
        output = self.softmax(output)



        return output    