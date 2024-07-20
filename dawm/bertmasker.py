import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from config import *
from pytorch_revgrad import RevGrad


    
def gumbel_softmax(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Applies the Gumbel-Softmax trick to a given logits tensor.

    Args:
        logits (torch.Tensor): The input logits tensor of shape (batch_size, MAX_LENGTH, 2).
        tau (float): The temperature parameter for the Gumbel-Softmax.

    Returns:
        torch.Tensor: A tensor of the same shape as logits after applying Gumbel-Softmax.
    """
    # A small constant to avoid numerical issues with log function
    eps = 1e-20
    
    # Draw samples from the Gumbel distribution
    # This creates a tensor of the same shape as logits, with samples from the Gumbel(0, 1) distribution
    samples = -torch.log(-torch.log(torch.rand((logits.size(0), MAX_LENGTH, 2)) + eps) + eps).to(device)
    
    # Apply the Gumbel-Softmax trick
    # Adjust the logits by adding the Gumbel samples and scaling by the temperature parameter tau
    p = (torch.log(logits) + samples) / tau
    
    # Apply the softmax function to get the probabilities
    p = nn.functional.softmax(p, dim=-1)
    
    # Clean up the samples tensor to free memory
    del samples
    
    # Return the Gumbel-Softmax probabilities
    return p

def softmask_with_mask(inputs: torch.Tensor, masks: torch.Tensor, zeros: torch.Tensor) -> torch.Tensor:
    """
    Applies a masked softmax operation on the input tensor.

    Args:
        inputs (torch.Tensor): The input tensor
        masks (torch.Tensor): masking tensor for tokens and PAD tokens
        zeros (torch.Tensor): masking tensor for special tokens (CLS and SEP)

    Returns:
        torch.Tensor: A tensor of the same shape as inputs after applying the masked softmax.
    """
    # Apply the exponential function to the inputs
    inputs = torch.exp(inputs)
    
    # Apply the masks and zeros to the inputs
    # The masks are unsqueezed to match the dimensions of the inputs
    inputs = inputs * masks.unsqueeze(-1) * (1 - zeros).unsqueeze(-1)
    
    # Compute the sum of the masked inputs along the sequence length dimension
    sum_inputs = torch.sum(inputs, dim=1, keepdim=True) + 1e-9
    
    # Return the masked softmax output
    return inputs / sum_inputs

def concatenate_tensors(representation: torch.Tensor, descriptor: torch.Tensor, domain_list: torch.Tensor,att_masks:torch.Tensor) -> torch.Tensor:
    """
    Function to concatenate two tensors

    Args:
        representation (torch.Tensor): The hidden representation of shape (batch_size, MAX_LENGTH, d)
        descriptor (torch.Tensor): the domain descriptors of shape (num_domains, L)
        domain_list (torch.Tensor): tensor with domains (0,1,2,...)  for each sample
        att_masks (torch.Tensor): masking tensor for tokens and PAD tokens

    Returns:
        torch.Tensor: A tensor of the same shape as inputs after applying the masked softmax.
    """
    
    # Get mean of domain descriptors
    ave_descr = torch.mean(descriptor,dim=0)
    
    # match size
    ave_descr2 = ave_descr.unsqueeze(0).unsqueeze(0).repeat(representation.size(0),MAX_LENGTH,1)

    domain_descr = torch.index_select(descriptor,0,domain_list)
    
    repeated_descriptor = domain_descr.unsqueeze(1).repeat(1, MAX_LENGTH, 1)

    # Concatenate representation and corresponding domain descriptor of each sample and the average domain descriptor
    concatenated_tensors = torch.cat((representation, repeated_descriptor), dim=-1)
    concatenated_tensors = concatenated_tensors * att_masks.unsqueeze(-1)
    concatenated_tensors2 = torch.cat((representation, ave_descr2), dim=-1)
    concatenated_tensors2 = concatenated_tensors2 * att_masks.unsqueeze(-1)

    # stack both concatenations
    concatenated_tensor3 = torch.stack((concatenated_tensors,concatenated_tensors2),dim=-2)

    return concatenated_tensor3

class SentimentClassifier(nn.Module):
    """
    Simple sentiment classifier, takes as input as a representation (maybe cls) and outputs probability scores
    """
    def __init__(self, hidden_size = hidden_size):
        super(SentimentClassifier,self).__init__()
        
        self.classifier = nn.Sequential(
            
            nn.Linear(int(2*hidden_dim),hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_polarities),
            #nn.Softmax(dim=-1)
        )
        self.init_weight(self.classifier)

    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]                
                nn.init.constant_(layer.bias, 0)
   
    def forward(self, representation: torch.Tensor):
        """
        forward function

        Args:
            representation (torch.Tensor): hidden representation which is fed into the classifier
        
        Returns:
            torch.Tensor: unnormalized probabilities
        """
        predicted_polarity = self.classifier(representation)
        return predicted_polarity


    
class SharedPart(nn.Module):
    """
    Token mask lyer of BERTMasker, and outputs predicted probabilities for domain classification
    """
    def __init__(self, hidden_size = hidden_size, temp = temperature,alpha=alpha,masking=0.1):
        super(SharedPart, self).__init__()

        self.mlps = nn.Sequential(
            nn.Linear(hidden_dim + descriptor_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.grl = RevGrad(alpha=alpha)
        self.dcs = nn.Sequential(
            nn.Linear(hidden_dim,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_domains),
            #nn.Softmax(dim=1),
        )

        self.temp = temp
        self.init_weight(self.mlps)
        self.init_weight(self.dcs)

        self.dcp = nn.Sequential(
            
            nn.Linear(hidden_dim,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_domains),
            #nn.Softmax(dim=1),
        )

        self.masking = masking
        self.sigmoid = nn.Sigmoid()

        self.init_weight(self.dcp)

        
        self.bert = model_bert

    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                nn.init.constant_(layer.bias, 0)

    def forward(self, hidden_embeddings: torch.Tensor, input_embedding: torch.Tensor, mask_embedding: torch.Tensor, segments_tensor: torch.Tensor, domain_list: torch.Tensor,z:torch.Tensor):
        """
        Forward pass of the model.

        Args:
            hidden_embeddings (torch.Tensor): The hidden embeddings tensor.
            input_embedding (torch.Tensor): The input embeddings tensor used by BERT to create hidden embeddings (same as using word tokens).
            mask_embedding (torch.Tensor): The mask input embedding used by BERT.
            segments_tensor (torch.Tensor): The segments tensor indicating PAD tokens.
            domain_list (torch.Tensor): The domain list.
            z (torch.Tensor): concatenated hidden representations and domain descriptors.

        Returns:
            y_pred (torch.Tensor): predicted domain classification for shared part
            summed_hs (torch.Tensor): shared hidden representations
            y_pred2 (torch.Tensor): predicted domain classification for private part
            private_rep (torch.Tensor): provate hidden representations
            mask_perc (torch.Tensor): masking percentage 
        """
        
        # calculate similarity 
        pi = self.mlps(z).squeeze(-1)
        
        # get discrete values 
        Ps = gumbel_softmax(logits=pi,tau=self.temp,t=0) - self.masking
        Ph = torch.round(Ps)
        P = Ph.detach() - Ps.detach() + Ps

        # get masking tensor for CLS and SEP tokens
        s = torch.sum(segments_tensor,dim=1).to(device) - 1
        zeros = torch.zeros(input_embedding.size(0),MAX_LENGTH).to(device)
        zeros.scatter_(1,s.unsqueeze(1),1).scatter_(1,torch.zeros_like(zeros[:, :1]).to(device).long(), 1)

        # replace input embeddings of masked tokens with [MASK] input embeddings 
        embedded_inputs =  (1-P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * input_embedding + (P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * mask_embedding
        embedded_inputs = embedded_inputs  * (1-zeros).unsqueeze(-1) + input_embedding * zeros.unsqueeze(-1)
        outputs = self.bert(inputs_embeds = embedded_inputs, attention_mask = segments_tensor)

        

        # sum over last four layers and get cls token [batch_size x hidden_dim]
        hidden_states = torch.stack(outputs.hidden_states[-4:],dim=0)
        shared_rep = torch.sum(hidden_states[:,:,0,:], dim=0)
       
        # apply GRL
        shared_rep = self.grl(shared_rep)
        
        # get shared domain classification probabilities
        y_pred = self.dcs(shared_rep)
        
        # get domain informative clue
        embedded_inputs = (P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * hidden_embeddings 
        sum = torch.sum((P[torch.arange(P.size(0)),:, 0]) * segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        condition = sum == 0
        private_rep2 = torch.sum(embedded_inputs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / (sum + condition.float())

        # get attention scores
        pr = private_rep2.unsqueeze(1).repeat(1,MAX_LENGTH,1)
        h_private = hidden_embeddings * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1) * pr
        a = softmask_with_mask(self.sigmoid(h_private),segments_tensor,zeros)
        
        # domain enriched hidden embedding for sentiment classification
        private_rep = torch.sum(hidden_embeddings *a* segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) 

        # use domain informative clue for domain classification
        y_pred2 = self.dcp(private_rep2)

        mask_perc = torch.sum((P[torch.arange(P.size(0)),:, 0]) * segments_tensor * (1-zeros),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1)
        
        
        return y_pred,shared_rep,y_pred2,private_rep,mask_perc

class BERTMasker(nn.Module):
    " The BERTMasker framework"
    
    def __init__(self,shared_domain_classifier,private_domain_classifier,sentiment_classifier):
        super(BERTMasker,self).__init__()
        self.shared_domain_classifier = shared_domain_classifier
        self.sentiment_classifier = sentiment_classifier
        self.descriptors = nn.Parameter(torch.rand(num_domains,descriptor_dimension) * 0.2-0.1,requires_grad=True)
  
    def init_layer(self,layer):
        nn.init.uniform_(layer.weight,a=-0.1,b=0.1)

    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                #nn.init.normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, hidden_embeddings: torch.Tensor, input_embedding: torch.Tensor, mask_embedding: torch.Tensor, segments_tensor: torch.Tensor, domain_list: torch.Tensor):
        """
        Forward pass of the model.

        Args:
            hidden_embeddings (torch.Tensor): The hidden embeddings tensor.
            input_embedding (torch.Tensor): The input embeddings tensor used by BERT to create hidden embeddings (same as using word tokens).
            mask_embedding (torch.Tensor): The mask input embedding used by BERT.
            pad_embedding (torch.Tensor): The PAD input embedding used by BERT.
            segments_tensor (torch.Tensor): The segments tensor indicating PAD tokens.
            domain_list (torch.Tensor): The domain list.


        Returns:
            shared_output (torch.Tensor): unnormalized shared domain classification probabilities
            private_output (torch.Tensor): unnormalized private domain classification probabilities
            sentiment_pred (torch.Tensor): unnormalized sentiment probabilties
            mask_perc (torch.Tensor): masking percentage 
        """
        # concate embeddings and domain descriptors
        z = concatenate_tensors(representation=hidden_embeddings,descriptor=self.descriptors,domain_list=domain_list,att_masks = segments_tensor)
        
        # token mask layer 
        shared_output, shared_cls, private_output,private_cls,mask_perc = self.shared_domain_classifier(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list,z=z)
        
        
        # sentiment classification
        total_cls = torch.cat((shared_cls,private_cls),dim=-1)
        sentiment_pred = self.sentiment_classifier(total_cls)
    
        return shared_output,private_output,sentiment_pred,mask_perc
        
