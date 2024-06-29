import torch
from torch import nn
from typing import Optional
from config import *
from load_data import load_data,get_contexts,CustomDataset
from torch.utils.data import TensorDataset, DataLoader


def softmask_with_mask(inputs,masks):
    
    inputs = torch.exp(inputs)
    inputs = inputs* masks
    
    sum_inputs = torch.sum(inputs,dim=1,keepdim=True) + 1e-9
    return inputs / sum_inputs

def mean_with_mask(inputs,masks):
    
    inputs = torch.sum(inputs * masks.unsqueeze(-1),dim=1)
    
    sum_masks = torch.sum(masks,dim=1,keepdim=True) 
    
    return inputs / sum_masks

class BilinearAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        
        self.bilinear = nn.Bilinear(
            in1_features=input_size,
            in2_features=input_size,
            out_features=1,
            bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.uniform_(self.bilinear.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
        nn.init.constant_(self.bilinear.bias, 0)

    def forward(self, hidden_states: torch.Tensor, representation: torch.Tensor,att_masks: torch.Tensor):
        """
        :param hidden_states: [n x input_size] where n is the number of tokens
        :param representation: [input_size]
        :return: [input_size] the new representation
        """
        
        
        # [n_hidden_states x 1]
        #representation.unsqueeze(1).expand(representation.size(0),hidden_states.size(1),representation.size(1))
        att_scores = torch.tanh(self.bilinear(hidden_states, representation.unsqueeze(1).repeat(1,hidden_states.size(1),1)))
        att_scores = softmask_with_mask(att_scores.squeeze(-1),att_masks)
        #att_scores = att_scores.squeeze(-1) * att_masks
        
        #att_scores = self.softmax(att_scores)
        updated_representation = att_scores.unsqueeze(-1) * hidden_states
        
        updated_representation = torch.sum(updated_representation, dim=1)
        
        return updated_representation
    

class HierarchicalAttention(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=1, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)
        self.init_weights()


    def init_weights(self):
        nn.init.uniform_(self.linear.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
        nn.init.constant_(self.linear.bias, 0)

    def forward(self, representation1: torch.Tensor, representation2: torch.Tensor):
        """
        :param representation1: [input_size]
        :param representation2: [input_size]
        :return: representation1, representation2: the representations scaled by their corresponding attention score
        """
        representations = torch.cat((
            self.tanh(self.linear(representation1)),
            self.tanh(self.linear(representation2))
        ),dim=-1)
        
        attention_scores = self.softmax(representations)
        
        representation1 = attention_scores[:,0].unsqueeze(1) * representation1
        representation2 = attention_scores[:,1].unsqueeze(1) * representation2
        
        return representation1, representation2

class LCRRotHopPlusPlus(nn.Module):
    def __init__(self,sentiment_prediction = True,dropout_prob1 = dropout_prob1,dropout_prob2 = dropout_prob2):
        super(LCRRotHopPlusPlus,self).__init__()
        self.hops = hops
        self.sentiment_prediction = sentiment_prediction
        if self.hops < 1:
            raise ValueError("Invalid number of hops")
    	
        self.left_bilstm = nn.LSTM(input_size=hidden_dim,hidden_size=hidden_lstm,bidirectional=True,batch_first=True)
        self.target_bilstm = nn.LSTM(input_size=hidden_dim,hidden_size=hidden_lstm,bidirectional=True,batch_first=True)
        self.right_bilstm = nn.LSTM(input_size=hidden_dim,hidden_size=hidden_lstm,bidirectional=True,batch_first=True)  

        self.init_lstm(self.left_bilstm)
        self.init_lstm(self.right_bilstm)
        self.init_lstm(self.target_bilstm)
        
        self.representation_size = 2 * hidden_lstm
        
        self.left_bilinear = BilinearAttention(self.representation_size)
        self.target_left_bilinear = BilinearAttention(self.representation_size)
        self.target_right_bilinear = BilinearAttention(self.representation_size)
        self.right_bilinear = BilinearAttention(self.representation_size)
        
        self.context_hierarchical = HierarchicalAttention(self.representation_size)
        self.target_hierarchical = HierarchicalAttention(self.representation_size)
        
        self.dropout1 = nn.Dropout(p=dropout_prob1)
        self.dropout2 = nn.Dropout(p=dropout_prob2)
        
        
        self.output_linear = nn.Linear(in_features=4 * self.representation_size, out_features=num_polarities)
        self.init_weights()

        #self.softmax = nn.Softmax(dim=1)
    
    def init_weights(self):
        nn.init.uniform_(self.output_linear.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
        nn.init.constant_(self.output_linear.bias, 0)

    def init_lstm(self,lstm):
        
        for name, par in lstm.named_parameters():
            
            if 'weight' in name:
                nn.init.uniform_(par,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
            elif 'bias' in name:
    
                nn.init.constant_(par, 0)

    def forward(self,left,target,right,att_left,att_target,att_right):
        
        
        left_hidden_states, _ = self.left_bilstm(self.dropout1(left))
        target_hidden_states, _ = self.target_bilstm(self.dropout1(target))
        right_hidden_states, _ = self.right_bilstm(self.dropout1(right))
        
    
        # initial representations using pooling
        representation_target_left = mean_with_mask(target_hidden_states,att_target)#torch.mean(target_hidden_states, dim=1)
        
        representation_target_right = representation_target_left

        # rotatory attention
        for i in range(self.hops):
            # target-to-context

            representation_left = self.left_bilinear(left_hidden_states, representation_target_left,att_left)
            representation_right = self.right_bilinear(right_hidden_states, representation_target_right,att_right)

            representation_target_left = self.target_left_bilinear(target_hidden_states, representation_left,att_target)
            representation_target_right = self.target_right_bilinear(target_hidden_states, representation_right,att_target)

            representation_left, representation_right = self.context_hierarchical(
                    representation_left,
                    representation_right
                )

            # context-to-target
        
            representation_target_left, representation_target_right = self.target_hierarchical(
                    representation_target_left,representation_target_right
                )
                  
        # determine output probabilities
        output = torch.concat([
            representation_left,
            representation_target_left,
            representation_target_right,
            representation_right,
        ],dim=1)
        
        if self.sentiment_prediction:
            output = self.dropout2(output)
            output = self.output_linear(output) 
            
            #output = self.softmax(output)
            return output
        
        return output
    