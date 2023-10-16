def recall_k(y_true,y_pred,i2t_k,t2i_k):
    '''
    y_true: (n,)
    y_pred: (n,n)
    i2t_k: list of k
    t2i_k: list of k
    simplified for img-text retrieval(top-k accuracy)
    '''
    assert len(y_pred.shape)==2 and y_pred.shape[0] == y_pred.shape[1] == len(y_true), f"y_pred shape:{y_pred.shape}, len of y_true:{len(y_true)}"
    result = 0
    total_num = 0

    for k in i2t_k:
        y_pred_topk = y_pred.topk(k).indices
        all = 0
        correct = 0
        for true,pred in zip(y_true,y_pred_topk):
            all+=1
            if true in pred:
                correct+=1
        result+=correct/all
        total_num+=1

    for k in t2i_k:           
        y_pred_topk = y_pred.T.topk(k).indices
        all = 0
        correct = 0
        for true,pred in zip(y_true,y_pred_topk):
            all+=1
            if true in pred:
                correct+=1
        result+=correct/all
        total_num+=1
    
    return result/total_num
    
from typing import OrderedDict
def get_model_state_dict_requires_grad(model):
    model_state_dict=OrderedDict()
    for n,p in model.named_parameters():
        if p.requires_grad:
            model_state_dict[n] = p.data
    return model_state_dict