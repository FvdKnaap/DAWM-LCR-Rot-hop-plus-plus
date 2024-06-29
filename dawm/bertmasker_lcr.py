import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from load_data import get_contexts
from grad_reverse import revgrad
from pytorch_revgrad import RevGrad
def gumbel_softmax(logits,tau,t):
    #gumbel_dist = torch.distributions.gumbel.Gumbel(0, 1)
    eps = 1e-20
    # Draw samples from the Gumbel distribution
    samples = -torch.log(-torch.log(torch.rand((logits.size(0),MAX_LENGTH,2)) + eps) + eps).to(device)
    #samples = gumbel_dist.sample((logits.size(0),MAX_LENGTH,2)).to(device)
    
    #p = torch.exp(samples/tau) * logits**(1/tau)
    p = (torch.log(logits) + samples) / tau
    
    p = nn.functional.softmax(p,dim=-1)
    #p = torch.exp(samples/tau) * logits**(1/tau) / torch.sum(torch.exp(samples/tau) * logits**(1/tau),dim=-1).unsqueeze(-1)
    #p = torch.exp(samples + torch.log(logits) / tau) / torch.sum(samples + torch.log(logits) / tau,dim=-1).unsqueeze(-1)
    del samples
    return p

def softmask_with_mask(inputs,masks,zeros):
    
    inputs = torch.exp(inputs)
    
    inputs = inputs* masks.unsqueeze(-1) * (1-zeros).unsqueeze(-1)
    
    sum_inputs = torch.sum(inputs,dim=1,keepdim=True) + 1e-9
    

    return inputs / sum_inputs

def concatenate_tensors(representation: torch.Tensor, descriptor: torch.Tensor, domain_list: torch.Tensor,att_masks):
    """
    Function to concatenate two tensors
    
    :param representation: [batch_size x MAX_LENGH x hidden_dim] is the generated embeddings from BERT in batches
    :param descriptor: [num_domains x descriptor_dimension] is a tensor containing the domain descriptors for all domains, which are learned during training
    :param list: [num_domains] list containing integers to which domain each sample belongs to
    
    :return concatenated_tensors: [batch_size x MAX_LENGH x (hidden_dim + descriptor_dimension)] tensor which concatenates each token embedding with the correct domain_descriptor
    """
    
    
    
    # Loop over value sin the list
    ave_descr = torch.mean(descriptor,dim=0)
    
    ave_descr2 = ave_descr.unsqueeze(0).unsqueeze(0).repeat(representation.size(0),MAX_LENGTH,1)

    domain_descr = torch.index_select(descriptor,0,domain_list)
    
    repeated_descriptor = domain_descr.unsqueeze(1).repeat(1, MAX_LENGTH, 1)
    # Concatenate representation and repeated descriptor along the last dimension
    concatenated_tensors = torch.cat((representation, repeated_descriptor), dim=-1)
    concatenated_tensors = concatenated_tensors * att_masks.unsqueeze(-1)
    concatenated_tensors2 = torch.cat((representation, ave_descr2), dim=-1)
    concatenated_tensors2 = concatenated_tensors2 * att_masks.unsqueeze(-1)
    concatenated_tensor3 = torch.stack((concatenated_tensors,concatenated_tensors2),dim=-2)
    
    
    return concatenated_tensor3

def mix_domain_descriptors(representation: torch.Tensor, descriptor: torch.Tensor):
    """
    Function to concatenate all domain descriptors with all tokens in a sentence
    
    :param representation: [batch_size x hidden_dim] is the generated embeddings from BERT in batches -> CLS representation
    :param descriptor: [num_domains x descriptor_dimension] is a tensor containing the domain descriptors for all domains, which are learned during training
    
    :return concatenated_tensors: [batch_size x num_domains x (hidden_dim + descriptor_dimension)] tensor which concatenates each token embedding with all domain_descriptors
    """
    
    # add dimension to the cls representation to match dimensions of domain descriptor [batch_size x num_domains x hidden_dim]
    representation = representation.unsqueeze(1).expand(representation.size(0),num_domains,hidden_dim).clone()
    
    # add dimension to the domain descriptor to match cls representation [batch_size x num_domains x descriptor_dimension]
    descriptor = descriptor.unsqueeze(0).expand(representation.size(0),num_domains,descriptor_dimension).clone()

    # Concatenate along the last dimension [batch_size x num_domains x (hidden_dim + descriptor_dimension)]
    concatenated_tensor = torch.cat((representation, descriptor), dim=-1)
    return concatenated_tensor

class GradientReversal(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        return revgrad(x, self.alpha)
    
class SentimentClassifier(nn.Module):
    """
    Simple sentiment classifier, takes as input as a representation (maybe cls) and outputs probability scores
    """
    def __init__(self):
        super(SentimentClassifier,self).__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(int(16*hidden_lstm),num_polarities),
            #nn.Softmax(dim=-1)
        )
        self.init_weight(self.classifier)

    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                #nn.init.normal_(layer.weight)
                
                nn.init.constant_(layer.bias, 0)

    def forward(self, representation: torch.Tensor):
        """
        :param representation: [batch_size x 2*hidden_dim]
        
        :return predicted_polarity: [batch_size x num_polarities]
        """
        predicted_polarity = self.classifier(representation)
        return predicted_polarity


class PrivatePart(nn.Module):
    """
    Private part of BERTMasker, adapted for LCR-Rot-hop++, and outputs predicted probabilities for domain classification
    """
    def __init__(self, hidden_size = hidden_size, temp = temperature ):
        super(PrivatePart,self).__init__()
        
        self.mlp1 = nn.Sequential(
            nn.Linear(hidden_dim + descriptor_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        #self.descriptors = nn.Parameter(descriptors,requires_grad=True)
        
        #self.similarity_layer = nn.Sequential(
        #    nn.Linear(hidden_dim + descriptor_dimension,hidden_size),
        #    nn.Tanh(),
        #    nn.Linear(hidden_size,1)
        #)
        
        self.softmax = nn.Softmax(dim=1)
        
        self.dcs = nn.Sequential(
            nn.Linear(hidden_dim,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_domains),
            #nn.Softmax(dim=1),
        )

       
        self.temp = temp
        self.init_weight(self.mlp1)
        self.init_weight(self.dcs)

        #self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        
        #for param in self.bert.parameters():
        #        param.requires_grad = False
        self.i = 1
    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                #nn.init.normal_(layer.weight)
               
                nn.init.constant_(layer.bias, 0)
            
    def forward(self, hidden_embeddings: torch.Tensor, input_embedding: torch.Tensor, mask_embedding: torch.Tensor, segments_tensor: torch.Tensor,domain_list,z):
        """
        :param hidden_embeddings: [batch_size x MAX_LENGTH x hidden_dim] containing all token embeddings
        :param input_embedding: [batch_size x MAX_LENGTH x hidden_dim] containing the input embeddings used by BERT to egenrate contextualized embeddings 
        :param mask_embedding: [hidden_dim] containing the input embedding of the [MASK] token
        :param segments_tensor: [batch_size x MAX_LENGTH] containing the attention masks used by BERT
        
        :return y_pred: [batch_size x num_domains] predicted domain probabilities, private_rep: [batch_size x hidden_dim] cls representation
        """
        
        # concatenate every domain descriptor with cls token [batch_size x num_domains x (hidden_dim + descriptor_dimension)]
        #concat_embed = mix_domain_descriptors(representation= hidden_embeddings[:,0,:], descriptor= self.descriptors)
        #print(concat_embed.size())
        # Get similarity with every domain for every sample [batch_size x num_domains]
        #similarity = self.similarity_layer(z[:,0,:,:]).squeeze(-1)
       
        #similarity_sm = self.softmax(similarity)
        
        # Calculate the mixture of domain decriptors bu multiplying the weights with the actual descriptors [batch_size x MAX_LENGTH x descriptor_dimension]
        #domain_descr_mix = torch.matmul(similarity_sm, self.descriptors).unsqueeze(1).expand(hidden_embeddings.size(0),MAX_LENGTH,descriptor_dimension).clone()
        
        # Concatenate mixture of domains and actual token embeddings [batch_size x MAX_LENGTH x (hidden_dim + descriptor_dimension)]
        #z = torch.cat((hidden_embeddings,domain_descr_mix),dim=-1)
        
        # Measure similarity [batch_size x MAX_LENGTH x output_size]
        #pi = self.mlp1(z)
        
        # Convert to discrete decisions [batch_size x num_domains x output_size]
        #P = F.gumbel_softmax(logits= pi, tau=temperature, hard=True)
        
        # Replace the input embeddings of domain-invariant tokens with [MASK] input embeddings, and generate BERT embeddings [batch_size x MAX_LENGTH x hidden_dim]
        #embedded_inputs = P[:,:,0].unsqueeze(-1) * input_embedding + P[:,:,1].unsqueeze(-1) * mask_embedding

        pi = self.mlp1(z).squeeze(-1)
        
        # get discrete values [batch_size x MAX_LENGTH x output_size]
        #P = F.gumbel_softmax(logits=pi, tau=1, hard=False)
        #P = F.gumbel_softmax(torch.log(pi),tau=self.temp,hard=True)
        Ps = gumbel_softmax(logits=pi,tau=self.temp,t=0) - 0.1
        
        P = torch.round(Ps)
        #P = Ph - Ps.detach() + Ps
        
        #Ps = torch.round(P)
        #P = Ps - P.detach() + P
        #P = torch.round(P)
        #print(P)
        s = torch.sum(segments_tensor,dim=1).to(device) - 1
        zeros = torch.zeros(input_embedding.size(0),MAX_LENGTH).to(device)
        zeros.scatter_(1,s.unsqueeze(1),1).scatter_(1,torch.zeros_like(zeros[:, :1]).to(device).long(), 1)
        
        # replace input embeddings of masked tokens with [MASK] input embeddings [batch_size x MAX_LENGTH x hidden_dim]
        # 1 - P[...,domain_list]
        embedded_inputs = (P[torch.arange(P.size(0)),:, domain_list]).unsqueeze(-1) * hidden_embeddings #+ (1-P[torch.arange(P.size(0)),:, domain_list]).unsqueeze(-1) * mask_embedding
        #embedded_inputs = embedded_inputs  * (1-zeros).unsqueeze(-1) + input_embedding * zeros.unsqueeze(-1)

        sum = torch.sum((P[torch.arange(P.size(0)),:, domain_list]) * segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        condition = sum == 0
        private_rep2 = torch.sum(embedded_inputs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / (sum + condition.float())

        #private_rep = torch.sum(embedded_inputs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)


        pr = private_rep2.unsqueeze(1).repeat(1,MAX_LENGTH,1)
        h_private = hidden_embeddings * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1) * pr
        a = self.softmax(h_private)
        
        summed_hs = hidden_embeddings *a* segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1)#/ torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)

        #outputs = self.bert(inputs_embeds = embedded_inputs, attention_mask = segments_tensor)

        # Sum over last four layers and get cls token embedding [batch_size x hidden_dim]
        #hidden_states = torch.stack(outputs.hidden_states[-4:],dim=0)
        #private_rep = torch.sum(hidden_states[:,:,0,:], dim=0)
        #summed_hs = torch.sum(hidden_states[:,:,:,:], dim=0)
        #sprivate_rep = torch.sum(summed_hs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        #sum = torch.sum((P[torch.arange(P.size(0)),:, domain_list]) * segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        #condition = sum == 0
        #private_rep_only = torch.sum(private_rep * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1) * (P[torch.arange(P.size(0)),:, domain_list]).unsqueeze(-1),dim=1) / (sum + condition.float())

        #private_rep = torch.sum(hidden_embeddings *a* segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)

        #private_rep = P[torch.arange(P.size(0)),:, 0].unsqueeze(-1) * private_rep * (1-zeros).unsqueeze(-1)
        
        #if self.i == 500:
        #    print(torch.sum(P[torch.arange(P.size(0)),:, 0],dim=1))
        #    
        #private_rep = torch.sum(private_rep,dim=1) / torch.sum(P[torch.arange(P.size(0)),:, 0],dim=1).unsqueeze(-1)
        # Get predictions for domain classification [batch_size x num_domains]
        y_pred = self.dcs(private_rep2)

    
        return y_pred,summed_hs

    
class SharedPart(nn.Module):
    """
    Shared part of BERTMasker, and outputs predicted probabilities for domain classification
    """
    def __init__(self, hidden_size = hidden_size, temp = temperature,alpha=alpha,masking=0.49):
        super(SharedPart, self).__init__()
        
        self.mlps = nn.Sequential(
            nn.Linear(hidden_dim + descriptor_dimension, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        self.grl = RevGrad(alpha=alpha)
        self.dcs = nn.Sequential(
        
            #RevGrad(),
            #GradientReversal(alpha=alpha),
            nn.Linear(hidden_dim,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_domains),
            #nn.Softmax(dim=1),
        )
        self.masking = masking
        self.temp = temp
        self.init_weight(self.mlps)
        self.init_weight(self.dcs)
        
        self.dcp = nn.Sequential(
            
            nn.Linear(hidden_dim,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size,num_domains),
            #nn.Softmax(dim=1),
        )

        self.sigmoid = nn.Sigmoid()

        self.init_weight(self.dcp)

        #self.descriptors = nn.Parameter(descriptors,requires_grad=True) 
        #self.bert = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)
        self.bert = model_bert
        #for param in self.bert.parameters():
        #        param.requires_grad = False
        self.i = 1
    def init_weight(self,seq):
        for layer in seq:
            if isinstance(layer, nn.Linear):
                
                nn.init.uniform_(layer.weight,a=-0.1,b=0.1)  # Uniform initialization for weights in range [-0.1, 0.1]
                #nn.init.normal_(layer.weight)
                nn.init.constant_(layer.bias, 0)

    def forward(self, hidden_embeddings: torch.Tensor, input_embedding: torch.Tensor, mask_embedding: torch.Tensor, segments_tensor: torch.Tensor, domain_list,z):
        """
        :param hidden_embeddings: [batch_size x MAX_LENGTH x hidden_dim] containing all token embeddings
        :param input_embedding: [batch_size x MAX_LENGTH x hidden_dim] containing the input embeddings used by BERT to egenrate contextualized embeddings 
        :param mask_embedding: [hidden_dim] containing the input embedding of the [MASK] token
        :param segments_tensor: [batch_size x MAX_LENGTH] containing the attention masks used by BERT
        :param domain_list: [batch_size] list containing the domains of the respective samples
        
        :return y_pred: [batch_size x num_domains] predicted domain probabilities, shared_rep: [batch_size x hidden_dim] cls representation
        """
        
        # concatenate token embeddings and descriptors [batch_size x MAX_LENGTH x (hidden_dim + descriptor_dimension)]
        #z = concatenate_tensors(representation=hidden_embeddings,descriptor=self.descriptors,domain_list=domain_list)
        #descriptor2 = self.descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension).clone()
        #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
        #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
        
        #b = self.descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
        
        #hidden_embeddings = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
        #z = torch.cat((hidden_embeddings,b),dim=-1)
        
        # calculate similarity [batch_size x MAX_LENGTH x output_size]
        pi = self.mlps(z).squeeze(-1)
        
        # get discrete values [batch_size x MAX_LENGTH x output_size]
        #P = F.gumbel_softmax(logits=pi, tau=1, hard=False)
       # P = gumbel_softmax(logits=pi,tau=self.temp,t=0)
        
        #Ps = torch.round(P)
        #P = Ps - P.detach() + P
        Ps = gumbel_softmax(logits=pi,tau=self.temp,t=0) - self.masking
        #P = F.gumbel_softmax(torch.log(pi),tau=self.temp,hard=True)
     
        Ph = torch.round(Ps)
        P = Ph.detach() - Ps.detach() + Ps
        s = torch.sum(segments_tensor,dim=1).to(device) - 1
        zeros = torch.zeros(input_embedding.size(0),MAX_LENGTH).to(device)
        zeros.scatter_(1,s.unsqueeze(1),1).scatter_(1,torch.zeros_like(zeros[:, :1]).to(device).long(), 1)

        # replace input embeddings of masked tokens with [MASK] input embeddings [batch_size x MAX_LENGTH x hidden_dim]
        embedded_inputs =  (1-P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * input_embedding + (P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * mask_embedding
        embedded_inputs = embedded_inputs  * (1-zeros).unsqueeze(-1) + input_embedding * zeros.unsqueeze(-1)
        outputs = self.bert(inputs_embeds = embedded_inputs, attention_mask = segments_tensor)

        input_e = embedded_inputs
        #shared_rep = torch.sum(embedded_inputs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)

        # sum over last four layers and get cls token [batch_size x hidden_dim]
        hidden_states = torch.stack(outputs.hidden_states[-4:],dim=0)
        summed_hs = torch.sum(hidden_states[:,:,:,:], dim=0)
        shared_rep = summed_hs[:,0,:]
        #shared_rep = torch.sum(shared_rep * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
       # shared_rep = torch.sum(hidden_states[:,:,:,:], dim=0)
        #shared_rep = torch.sum(shared_rep * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        #hook_handle = shared_rep.register_hook(zeros grad: print("Gradient:", grad))
        # get predictions [batch_size x num_domains]
        #if self.training:
        #    h = shared_rep.register_hook(lambda grad: grad * -1)
        
        shared_rep = self.grl(shared_rep)
        
        y_pred = self.dcs(shared_rep)
        
        embedded_inputs = (P[torch.arange(P.size(0)),:, 0]).unsqueeze(-1) * hidden_embeddings #+ (1-P[torch.arange(P.size(0)),:, domain_list]).unsqueeze(-1) * mask_embedding
        sum = torch.sum((P[torch.arange(P.size(0)),:, 0]) * segments_tensor * (1-zeros),dim=1).unsqueeze(-1)
        condition = sum == 0
        private_rep2 = torch.sum(embedded_inputs * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1),dim=1) / (sum + condition.float())

        pr = private_rep2.unsqueeze(1).repeat(1,MAX_LENGTH,1)
        h_private = hidden_embeddings * segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1) * pr
        #a = self.softmax(h_private)
        
        a = softmask_with_mask(self.sigmoid(h_private),segments_tensor,zeros)
        
        private_rep = hidden_embeddings *a* segments_tensor.unsqueeze(-1) * (1-zeros).unsqueeze(-1) #/ torch.sum(segments_tensor * (1-zeros),dim=1).unsqueeze(-1)

        y_pred2 = self.dcp(private_rep2)

        mask_perc = torch.sum((P[torch.arange(P.size(0)),:, 0]) * segments_tensor * (1-zeros),dim=1) / torch.sum(segments_tensor * (1-zeros),dim=1)
        self.i+=1
        #if self.training:
        #    h.remove()
        #hook_handle = shared_rep.register_hook(lambda grad: print("Gradient:", grad))
        
        return y_pred,summed_hs,y_pred2,private_rep,mask_perc,input_e#,P[torch.arange(P.size(0)),:, 0]

class BERTMasker_plus(nn.Module):
    def __init__(self,shared_domain_classifier,private_domain_classifier,shared_lcr, private_lcr,sentiment_classifier):
        super(BERTMasker_plus,self).__init__()
        self.shared_domain_classifier = shared_domain_classifier
        #self.private_domain_classifier = private_domain_classifier
        self.shared_lcr = shared_lcr
        self.private_lcr = private_lcr
        self.sentiment_classifier = sentiment_classifier
        self.descriptors = nn.Parameter(torch.rand(num_domains,descriptor_dimension)* 0.2-0.1 ,requires_grad=True)


    def forward(self, hidden_embeddings: torch.Tensor, input_embedding: torch.Tensor, mask_embedding: torch.Tensor, pad_embedding, segments_tensor: torch.Tensor, domain_list: list,target_ind):
        """
        :param hidden_embeddings: [batch_size x MAX_LENGTH x hidden_dim] containing all token embeddings
        :param input_embedding: [batch_size x MAX_LENGTH x hidden_dim] containing the input embeddings used by BERT to egenrate contextualized embeddings 
        :param mask_embedding: [hidden_dim] containing the input embedding of the [MASK] token
        :param segments_tensor: [batch_size x MAX_LENGTH] containing the attention masks used by BERT
        :param domain_list: [batch_size] list containing the domains of the respective samples
        
        :return shared_output: [batch_size x num_domains] predicted shared domain probabilities, private_output: [batch_size x num_domains] predicted private domain probabilities, sentiment_pred: [batch_size x num_polarities] with predicted sentiment
        """
        z = concatenate_tensors(representation=hidden_embeddings,descriptor=self.descriptors,domain_list=domain_list,att_masks = segments_tensor)
        #z = concatenate_tensors(representation=hidden_embeddings,descriptor=self.descriptors,domain_list=domain_list,att_masks = segments_tensor)
        #descriptor2 = self.descriptors.unsqueeze(0).unsqueeze(0).repeat(hidden_embeddings.size(0),MAX_LENGTH,1,1)
        #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).repeat(1,1,num_domains,1)
        #descriptor2 = self.descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
        #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
        #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
        #z = hidden_embeddings2 * descriptor2
        shared_output,shared_hidden,private_output,private_hidden,mask_perc,input_e = self.shared_domain_classifier(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list,z=z)
        #private_output,private_hidden = self.private_domain_classifier(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor,domain_list=domain_list,z=z)
        
        s_pad_target,s_att_target,s_pad_left,s_att_left,s_pad_right,s_att_right = get_contexts(shared_hidden,target_ind,pad_embedding,segments_tensor)
        p_pad_target,p_att_target,p_pad_left,p_att_left,p_pad_right,p_att_right = get_contexts(private_hidden,target_ind,pad_embedding,segments_tensor)
    

        shared_sentiment = self.shared_lcr(left = s_pad_left,target = s_pad_target,right = s_pad_right,att_left = s_att_left,att_target = s_att_target,att_right = s_att_right)
        private_sentiment = self.private_lcr(left = p_pad_left,target = p_pad_target,right = p_pad_right,att_left = p_att_left,att_target = p_att_target,att_right = p_att_right)
        
        total_lcr = torch.cat((shared_sentiment,private_sentiment),dim=-1)
        
        sentiment_pred = self.sentiment_classifier(total_lcr)
    
        return shared_output,private_output,sentiment_pred,mask_perc,input_e
        
