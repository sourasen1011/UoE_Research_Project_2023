from general_utils import *

class generic_Loss(torch.nn.Module):
    '''
    Generic Loss Function
    '''
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def nll_logistic_hazard(self , logits , targets , dur_idx):
        '''
        logits - parameterized inv.logit(hazards) from nn
        targets - survival matrix 
        '''
        logits = torch.Tensor(logits)
        targets = torch.Tensor(targets)
        dur_idx = torch.Tensor(dur_idx).to(torch.int64).view(-1 , 1)

        loss = F.binary_cross_entropy_with_logits(input = logits , target = targets , reduction = 'none')
        
        assert loss.shape == targets.shape , 'must match'

        # cumulative hazards
        loss = loss.cumsum(1)
        loss = loss.gather(1, dur_idx)
        
        return loss.view(-1).mean()

    def c_index_lbo_loss(self , logits , times , events):
        '''
        here logits are used to form the predicted survival times
        '''
        _haz = torch.sigmoid(logits)
        _survival = torch.cumprod(1 - _haz , dim = 1)

        # get last survival proba    
        last_pred_survival = _survival[: , -1]
            
        # get comparable mask
        comp_mask = self.get_comparable_mask(times , events)
            
        # get loss - in order to maximise the LBO of the C-index, its negative needs to be minimized
        loss = -self.cindex_lower_bound(comp_mask , last_pred_survival , times)

        return loss

    def cindex_lower_bound(self , comp_mask, pred_times, times):
        '''
        comp_mask - comparable mask (no need for times and events separately)
        pred_times - predicted survival times / (or survival probabilities)
        '''
        # Get order
        _, order = torch.sort(times)
        
        # Convert comp_mask and pred_times to PyTorch tensors - hen order accordingly
        pred_times_tensor = pred_times[order]

        # Lower Bound
        lb = torch.sum(comp_mask * (1 + torch.log(torch.sigmoid(pred_times_tensor - pred_times_tensor.view(-1, 1))) / torch.log(torch.tensor(2)))) / torch.sum(comp_mask)
            
        # Exact C index
        cindex = torch.sum(comp_mask * (pred_times_tensor - pred_times_tensor.view(-1, 1)) > 0) / torch.sum(comp_mask)
        # print(lb , cindex)
        assert lb <= cindex, 'not a lower bound'

        # add gradient tracking
        lb = lb.clone().detach().requires_grad_(True)
            
        return lb

    def get_comparable_mask(self , times, events):
        # Get order
        _, order = torch.sort(times)
        eve_ordered = events[order]

        # Build cross matrix
        cross_mat = torch.triu((torch.outer(eve_ordered, eve_ordered) + eve_ordered.view(-1, 1)).bool()).int()
        
        # Set all diagonal elements to zero
        cross_mat.fill_diagonal_(0)

        return cross_mat