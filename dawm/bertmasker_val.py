import torch
from load_data import CustomDataset,CustomDataset2
from torch.utils.data import DataLoader
from config import *
from bertmasker import SentimentClassifier, SharedPart, PrivatePart, BERTMasker
import torch.nn as nn
from tqdm import tqdm
import torchmetrics
import optuna

def load_train(domain,target_values):
    out_path = 'Code/train_small/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset2(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right,target_tensor_index=8,target_values=target_values),mask_embedding,pad_embedding

def load_val(domain):
    out_path = 'Code/val/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right)

def load_train2(domain):
    out_path = 'Code/train_small/variables' + domain + '/'
    token_embeddings = torch.load(out_path + 'token_embeddings.pt')
    token_ids = torch.load(out_path + 'token_ids.pt')
    segment_ids = torch.load(out_path + 'segment_ids.pt')
    polarities = torch.load(out_path + 'polarities.pt')
    domain = torch.load(out_path + 'domain.pt')
    target_ind = torch.load(out_path + 'target_ind.pt')
    masking_constraints = torch.load(out_path + 'masking_constraints.pt')
    input_embeddings = torch.load(out_path + 'input_embeddings.pt')
    domain_list = torch.load(out_path + 'domain_list.pt')

    pad_target = torch.load(out_path + 'pad_target.pt')
    att_target = torch.load(out_path + 'att_target.pt')
    pad_left = torch.load(out_path + 'pad_left.pt')
    att_left = torch.load(out_path + 'att_left.pt')
    pad_right = torch.load(out_path + 'pad_right.pt')
    att_right = torch.load(out_path + 'att_right.pt')

    mask_embedding = torch.load(out_path + 'mask_embedding.pt')
    pad_embedding  = torch.load(out_path + 'pad_embedding.pt')

    return CustomDataset(token_embeddings,token_ids,segment_ids,polarities,domain,target_ind,masking_constraints,input_embeddings,domain_list,pad_target,att_target,pad_left,att_left,pad_right,att_right),mask_embedding,pad_embedding

def main(trial,validation_dataloader,data_loader1,source_loader,mask_embedding,pad_embedding):

    hidden_s = trial.suggest_categorical("hidden_s", [64,128,256])
    lr = trial.suggest_categorical("lr", [0.01,0.001,0.0001,0.0005,0.005,0.00005,0.00001])
    weight_decay = trial.suggest_categorical("weight_decay", [0.01,0.001,0.0001,0.0005,0.005])
    epochss = 7
    weight_shared = trial.suggest_categorical("weight_shared", [0.01,0.001,0.005])
    temp = trial.suggest_categorical("temp", [0.01,0.1,0.5,1])
    weight_private = weight_shared
    #weight_private = trial.suggest_categorical("weight_private", [1,0.1,0.001,0.05,0.005])
    weight_sent = trial.suggest_categorical("weight_sent", [1,2,3,4,5])
    alp = trial.suggest_categorical("alp", [0.5,1.0,1.5,2.0])
    #temp =0.001
    masking = 0.1

    #descriptors = nn.Parameter(torch.randn(num_domains,descriptor_dimension), requires_grad=True).to(device)
    private_part = PrivatePart(hidden_size=hidden_s,temp=temp).to(device)
    shared_part = SharedPart(hidden_size=hidden_s,temp=temp,alpha=alp,masking=masking).to(device)
    
    sentiment_classifier = SentimentClassifier(hidden_size=hidden_s).to(device)
    
    model = BERTMasker(shared_domain_classifier=shared_part,private_domain_classifier=private_part,sentiment_classifier=sentiment_classifier).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=weight_decay)
    shared_loss_fn = nn.CrossEntropyLoss()
    private_loss_fn = nn.CrossEntropyLoss()
    sentiment_loss_fn = nn.CrossEntropyLoss()


    '''
    vocab = {}

    # Loop through tokenizer's vocabulary and create embeddings
    for token, id in tokenizer.get_vocab().items():
        # Create tensor for current token ID
        token_id_tensor = torch.tensor(id)
        # Move token ID tensor to the device
        token_id_tensor = token_id_tensor.to(device)
        # Get embedding for current token ID
        embedding = model_bert.get_input_embeddings()(token_id_tensor)
        # Move embedding tensor to the device
        embedding = embedding.to(device)
        # Add token ID and embedding to vocab dictionary
        vocab[id] = embedding
    '''
    mask_embedding = mask_embedding.to(device)
    i = 0
    j = True

    train_acc_prev = torch.tensor(0.0,device=device)
    train_acc_prev2 = torch.tensor(0.0,device=device)
    train_acc_prev3 = torch.tensor(0.0,device=device)
    for epoch in range(epochss):
        model.train()
        total_loss = 0.0
        total_shared= 0.0
        total_private = 0.0 
        total_sentiment  = 0.0
        
        train_correct = 0
        train_total = 0.0
        
        if epoch < 1:
            data_loader = data_loader1
            for name, param in model.named_parameters():
                if 'bert' not in name:
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        param.requires_grad = False
                    
        else:
            data_loader = source_loader
            for name, param in model.named_parameters():
                if 'bert' not in name:
                    if 'shared_lcr' in name or 'private_lcr' in name or 'sentiment_classifier' in name:
                        
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            #for name, param in model.named_parameters():
             #   if 'descri' in name:
             #      param.requires_grad = False
        
        #else:
        #    for name, param in model.named_parameters():
        #        if 'bert' not in name:
        #            param.requires_grad = True
        # Use tqdm for progress bar
        with tqdm(total=len(data_loader), desc=f"Epoch {epoch+1}/{epochss}", unit="batch") as pbar:
            for batch_idx,(hidden_embeddings,_,segments_tensor,polarities,domain,_,_,input_embedding,domain_list,_,_,_,_,_,_) in enumerate(data_loader):
                
                if epoch > 0: 
                    domain_list = torch.zeros(domain_list.size(),device=device,dtype=torch.int64)

                # Zero the gradients
                optimizer.zero_grad()
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.to(device),segments_tensor.to(device),polarities.to(device),domain.to(device),domain_list.to(device),input_embedding.to(device)
                #descriptor2 = descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
                #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
                #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
                
                # Forward pass
                shared_output,private_output,sentiment_pred,p_embedded_inputs,s_embedded_inputs,p_P,s_P = model(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list)
                i+=1
                #print(torch.sum(s_P*segments_tensor,dim=-1))
                #print(torch.sum(p_P*ssegments_tensor,dim=-1))
        
                #print(p_embedded_inputs.size())
                
                if epoch <1:
                    shared_loss = shared_loss_fn(shared_output, domain_list)
                    private_loss = private_loss_fn(private_output, domain_list)
                    epoch_loss =  weight_shared*shared_loss + weight_private* private_loss
                  
                    total_loss += epoch_loss.item()
                    total_shared += shared_loss.item()
                    total_private += private_loss.item()

                elif epoch < 50:
                    sentiment_loss = sentiment_loss_fn(sentiment_pred,torch.argmax(polarities,dim=1))
                    epoch_loss = weight_sent* sentiment_loss

                    total_loss += epoch_loss.item() 
                
                else:
                    shared_loss = shared_loss_fn(shared_output, domain_list)
                    private_loss = private_loss_fn(private_output, domain_list)
                    sentiment_loss = sentiment_loss_fn(sentiment_pred,torch.argmax(polarities,dim=1))

                    epoch_loss = weight_shared*shared_loss + weight_private* private_loss + weight_sent* sentiment_loss

                    total_loss += epoch_loss.item()
                    total_shared += shared_loss.item()
                    total_private += private_loss.item()
                    total_sentiment += sentiment_loss.item()
                #print(epoch_loss)
                # Backward pass
                
                epoch_loss.backward(retain_graph=True)
                
                #train_acc += torchmetrics.functional.accuracy(torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1), torch.argmax(polarities,dim=1),task='multiclass',num_classes=num_polarities)
                train_correct += torch.sum((torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1) == torch.argmax(polarities,dim=1)).type(torch.int)).item()
                train_total += polarities.size(0)
                # Update weights
                optimizer.step()
                hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.cpu(),segments_tensor.cpu(),polarities.cpu(),domain.cpu(),domain_list.cpu(),input_embedding.cpu()
                
                
                # Update progress bar
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
                pbar.update(1)
                #print(descriptors)
                
                if i ==0:
                    
                    for name, param in model.named_parameters():
                        #if 'descr' in name:
                        #    print(param.grad)
                    
                        if param.grad == None:
                            print(name, param.grad)
                        ##else:
                        #if 0 in param.grad:
                        #    print(param.grad)
                    
                        
        i = 0
        
        
        print(f'Epoch [{epoch+1}/{epochss}], Total Loss: {total_loss:.4f}, shared loss {total_shared:.4f}, private loss: {total_private:.4f}, sentiment loss: {total_sentiment:.4f}')

        train_acc = torch.tensor(train_correct / train_total,device=device)
        print(f'acc: {train_acc}')
        train_acc_prev3 = train_acc_prev2
        train_acc_prev2 = train_acc_prev
        train_acc_prev = train_acc
        
        #if torch.max(train_acc_prev2,train_acc_prev) - train_acc_prev3 < eps:
        #    break
    
    model.eval()
    
    #f1_score = 0.0
    acc_shared = 0.0
    acc_private = 0.0
    f1 = 0.0

    test_correct = 0.0
    test_total = 0.0
    with torch.no_grad():
        for batch_idx,(hidden_embeddings,_,segments_tensor,polarities,domain,_,_,input_embedding,domain_list,_,_,_,_,_,_) in enumerate(validation_dataloader):
            hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.to(device),segments_tensor.to(device),polarities.to(device),domain.to(device),domain_list.to(device),input_embedding.to(device)
            #descriptor2 = descriptors.unsqueeze(0).unsqueeze(0).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,descriptor_dimension)
            #hidden_embeddings2 = hidden_embeddings.unsqueeze(2).expand(hidden_embeddings.size(0),MAX_LENGTH,num_domains,hidden_dim)
            #z = torch.cat((hidden_embeddings2,descriptor2),dim=-1)
            domain_list = torch.zeros(domain_list.size(),device=device,dtype=torch.int64)
            # Forward pass
            shared_output,private_output,sentiment_pred,p_embedded_inputs,s_embedded_inputs,p_P,s_P = model(hidden_embeddings=hidden_embeddings, input_embedding=input_embedding, mask_embedding=mask_embedding, segments_tensor=segments_tensor, domain_list=domain_list)
            
            pred = torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1)
            ds = torch.argmax(nn.functional.softmax(shared_output,dim=-1),dim=1)
            dp = torch.argmax(nn.functional.softmax(private_output,dim=-1),dim=1)
            polarities = torch.argmax(polarities,dim=1)
            
            test_correct += torch.sum((torch.argmax(nn.functional.softmax(sentiment_pred,dim=-1),dim=1) == polarities).type(torch.int)).item()
            test_total += polarities.size(0)
            
            acc_shared += torch.sum((torch.argmax(nn.functional.softmax(shared_output,dim=-1),dim=1) == domain_list).type(torch.int)).item()
            acc_private += torch.sum((torch.argmax(nn.functional.softmax(private_output,dim=-1),dim=1) == domain_list).type(torch.int)).item()
            
            #accuracy += torchmetrics.functional.accuracy(pred, polarities,task='multiclass',num_classes=num_polarities)
            f1 += torchmetrics.functional.f1_score(pred,polarities,task='multiclass',num_classes=num_polarities,average='macro')
            #f1_score += torchmetrics.functional.f1(pred, polarities, task='multiclass',num_classes=num_polarities)
            #domain = torch.argmax(domain,dim=1)
            #acc_shared += torchmetrics.functional.accuracy(ds, domain,task='multiclass',num_classes=num_domains)
            #acc_private += torchmetrics.functional.accuracy(dp, domain,task='multiclass',num_classes=num_domains)

            hidden_embeddings,segments_tensor,polarities,domain,domain_list,input_embedding = hidden_embeddings.cpu(),segments_tensor.cpu(),polarities.cpu(),domain.cpu(),domain_list.cpu(),input_embedding.cpu()

    print(f'Epoch [{epoch+1}/{epochss}], acc: {test_correct / test_total:.4f}, f1: {f1 / len(validation_dataloader):.4f}')
    print(f'Epoch [{epoch+1}/{epochss}], acc shared: {acc_shared / test_total:.4f},acc private {acc_private / test_total:.4f}')
    pbar.close()
    acc = test_correct / test_total
    return acc

if __name__ == '__main__':

    domain = ''
    targets = [2,1]
    torch.manual_seed(123)
    if torch.cuda.is_available():
        # Set seed for CUDA
        torch.cuda.manual_seed(123)

    dataset,mask_embedding,pad_embedding = load_train(domain,targets)
    domain = '_book'
    val_dataset = load_val(domain)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print('loaded data')
    
    

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    domain = '_book'
    dataset,mask_embedding,pad_embedding = load_train2(domain)
    source_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=123))
    study.optimize(lambda trial: main(trial, validation_dataloader,data_loader,source_loader,mask_embedding,pad_embedding), n_trials=15)

    # Get the best hyperparameters and results
    best_params = study.best_params
    best_loss = study.best_value
    print(best_params)
    print(best_loss)