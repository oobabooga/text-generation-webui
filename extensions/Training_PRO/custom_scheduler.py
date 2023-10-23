from functools import partial
import torch
import transformers
import math
from torch.optim.lr_scheduler import LambdaLR

from peft import (
    PeftModel,
)

RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

last_print_label = ''

custom_scheduler_params = {'trigger_loss': 0.0, 'ramp_down_ratio':1.0, 'current_loss': 0.0,'dynamic_scheduler_stop': False, 'calc_ramp_down_at_step': 0, 'calc_num_training_steps': 0}


def custom_scheduler_global_update(current_loss: float):
    custom_scheduler_params.update({'current_loss': current_loss})
  
def custom_scheduler_global_setup(trigger_loss: float, ramp_down_ratio: float):
    custom_scheduler_params.update({'trigger_loss': trigger_loss})
    custom_scheduler_params.update({'ramp_down_ratio': ramp_down_ratio})

    # calculates the total num steps after trigger
    custom_scheduler_params.update({'calc_num_training_steps': 0})
    #calculates steps when the ramp_down trigger occured
    custom_scheduler_params.update({'calc_ramp_down_at_step': 0})
    # triggers scheduler stopping after it reached calc_num_training_steps
    custom_scheduler_params.update({'dynamic_scheduler_stop': False})


# hold constant to the half of epochs then cosine down to 0
def _get_fp_half_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_firstepoch_steps: int):
    
    global last_print_label
    print_label = ''

    half_steps = num_training_steps//2
    
    num_warmup_steps = min(num_warmup_steps,half_steps)

    if current_step < num_warmup_steps:
        print_label = 'Scheduler: Warmup'
    elif current_step < half_steps:
        print_label = 'Scheduler: Hold'
    else:
        print_label = 'Scheduler: Annealing'
    
    if print_label != last_print_label:
        print(print_label)
    
    last_print_label = print_label

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    if current_step < half_steps:
        return 1.0 
    
    progress = float(current_step - half_steps) / float(max(1, num_training_steps - half_steps))
    num_cycles = 0.5
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))    
 

# raise up in cosine, then fall back in cosine
def _get_fp_cosine_raise_and_fall_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_firstepoch_steps: int):
    
    global last_print_label
    print_label = ''

    half_steps = num_training_steps//2
    
    #num_warmup_steps = min(num_warmup_steps,half_steps)

    if current_step < half_steps:
        print_label = 'Scheduler: Raise'
    else:
        print_label = 'Scheduler: Fall'
    
    if print_label != last_print_label:
        print(print_label)
    
    last_print_label = print_label

    
    # linear
    #    return float(current_step) / float(max(1, num_warmup_steps))
    
    progress = float(current_step - half_steps) / float(max(1, num_training_steps - half_steps))
    num_cycles = 0.5
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))    
 
# constant to the first epochs then cosine down to 0 over the rest epochs
def _get_fp_cosine_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_firstepoch_steps: int):
    
    global last_print_label
    print_label = ''
    
    num_warmup_steps = min(num_warmup_steps,num_firstepoch_steps)

    if current_step < num_warmup_steps:
        print_label = 'Scheduler: Warmup'
    elif current_step < num_firstepoch_steps:
        print_label = 'Scheduler: Hold'
    else:
        print_label = 'Scheduler: Annealing'
    
    if print_label != last_print_label:
        print(print_label)
    
    last_print_label = print_label

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    if current_step < num_firstepoch_steps:
        return 1.0 
    
    progress = float(current_step - num_firstepoch_steps) / float(max(1, num_training_steps - num_firstepoch_steps))
    num_cycles = 0.5
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))    
    
# halve lr each epoch   

def _get_fp_cdrop_rate_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int, num_firstepoch_steps: int):
    
    global last_print_label
    print_label = ''
    
    num_warmup_steps = min(num_warmup_steps, num_firstepoch_steps)

    current_epoch = (current_step // num_firstepoch_steps) + 1
    
    
    if current_step < num_warmup_steps:
        print_label = 'Scheduler: Warmup'
    elif current_step < num_firstepoch_steps:
        print_label = 'Scheduler: Hold'
    else:
        print_label = 'Scheduler: Drop Rate'
    
    if print_label != last_print_label:
        print(print_label)
    
    last_print_label = print_label

    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    
    if current_step < num_firstepoch_steps:
        return 1.0 

    # Compute the learning rate for the annealing phase
    
    learning_rate = 1.0 / float(2 ** (current_epoch - 1))
   
    return learning_rate

# epoch decay: 1/(1 + decay * epoch)

def custom_cosine_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_firstepoch_steps, last_epoch=-1):
    """
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    lr_lambda = partial(
        _get_fp_cosine_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_firstepoch_steps = num_firstepoch_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def custom_half_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_firstepoch_steps, last_epoch=-1):
    """
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    lr_lambda = partial(
        _get_fp_half_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_firstepoch_steps = num_firstepoch_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def custom_raise_fall_scheduler_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_firstepoch_steps, last_epoch=-1):
    """
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    
    lr_lambda = partial(
        _get_fp_cosine_raise_and_fall_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        num_firstepoch_steps = num_firstepoch_steps,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def neftune_forward(self, input: torch.Tensor):
    """
    Implements the NEFTune forward pass for the model. Note this works only for
    torch.nn.Embedding layers. This method is slightly adapted from the original source code
    that can be found here: https://github.com/neelsjain/NEFTune

    Args:
        input (`torch.Tensor`):
            The input tensor to the model.
        noise_alpha (`float`):
            The noise alpha value to use for the NEFTune forward pass.
    """
    embeddings = torch.nn.functional.embedding(
        input, self.weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse
    )

    if self.training:
        # Add noise to the embeddings
        dims = torch.tensor(embeddings.size(1) * embeddings.size(2))
        mag_norm = self.neftune_noise_alpha / torch.sqrt(dims)
        embeddings = embeddings + torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)

    return embeddings    


class FPNEFtuneTrainer(transformers.Trainer):
    def __init__(self,neftune_noise_alpha:float = 0.0, model = None, *args, **kwargs):
        self.neftune_noise_alpha = neftune_noise_alpha
        if self.neftune_noise_alpha > 0.0:
            model = self._activate_neftune(model)
        super().__init__(model = model, *args, **kwargs)

   
    def _activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        """
        print(f"Activating {RED}NEFtune{RESET} with scale: {self.neftune_noise_alpha}")
        if isinstance(model, transformers.PreTrainedModel):
            embeddings = model.get_input_embeddings()
        elif isinstance(model, PeftModel):
            embeddings = model.base_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        old_forward = embeddings.forward

        # This hack seems to be needed to properly use a custom forward pass
        # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
        bound_method = neftune_forward.__get__(embeddings, embeddings.__class__)
        setattr(embeddings, "forward", bound_method)

        # embeddings.forward = neftune_forward
        embeddings._trl_old_forward = old_forward

        return model
    
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer
        if self.neftune_noise_alpha is not None:

            if isinstance(self.model, transformers.PreTrainedModel):
                embeddings = self.model.get_input_embeddings()
            elif isinstance(self.model, PeftModel):
                embeddings = self.model.base_model.get_input_embeddings()

            if hasattr(embeddings, "_trl_old_forward"):
                embeddings.forward = embeddings._trl_old_forward
                del embeddings._trl_old_forward
                del embeddings.neftune_noise_alpha

        return output


class FPSchedulerTrainer(transformers.Trainer):
    def __init__(self,neftune_noise_alpha:float = 0.0, model = None, *args, **kwargs):
        self.neftune_noise_alpha = neftune_noise_alpha
        if self.neftune_noise_alpha > 0.0:
            model = self._activate_neftune(model)
        super().__init__(model = model, *args, **kwargs)

   
    def _activate_neftune(self, model):
        r"""
        Activates the neftune as presented in this code: https://github.com/neelsjain/NEFTune and paper: https://arxiv.org/abs/2310.05914
        """
        print(f"Activating {RED}NEFtune{RESET} with scale: {self.neftune_noise_alpha}")
        if isinstance(model, transformers.PreTrainedModel):
            embeddings = model.get_input_embeddings()
        elif isinstance(model, PeftModel):
            embeddings = model.base_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        old_forward = embeddings.forward

        # This hack seems to be needed to properly use a custom forward pass
        # all credits to: https://discuss.pytorch.org/t/how-can-i-replace-the-forward-method-of-a-predefined-torchvision-model-with-my-customized-forward-function/54224/11
        bound_method = neftune_forward.__get__(embeddings, embeddings.__class__)
        setattr(embeddings, "forward", bound_method)

        # embeddings.forward = neftune_forward
        embeddings._trl_old_forward = old_forward

        return model
    
    def train(self, *args, **kwargs):
        output = super().train(*args, **kwargs)

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer
        if self.neftune_noise_alpha is not None:

            if isinstance(self.model, transformers.PreTrainedModel):
                embeddings = self.model.get_input_embeddings()
            elif isinstance(self.model, PeftModel):
                embeddings = self.model.base_model.get_input_embeddings()

            if hasattr(embeddings, "_trl_old_forward"):
                embeddings.forward = embeddings._trl_old_forward
                del embeddings._trl_old_forward
                del embeddings.neftune_noise_alpha

        return output


    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        #Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or passed as an argument.
        
        num_train_epochs = self.args.num_train_epochs
        num_warmup_steps=self.args.get_warmup_steps(num_training_steps)
        num_firstepoch_steps = math.ceil(num_training_steps/num_train_epochs)
        num_warmup_acc = num_warmup_steps*self.args.gradient_accumulation_steps 
        num_firstepoch_steps_acc = num_firstepoch_steps*self.args.gradient_accumulation_steps
        num_training_steps_acc = num_training_steps*self.args.gradient_accumulation_steps

        custom_scheduler_params.update({'dynamic_scheduler_stop': False})
 
        print (f"Warm-up steps aligned to Gradient accumulation ({self.args.gradient_accumulation_steps}) = {num_warmup_acc} actual warmup steps")
        if self.args.lr_scheduler_type == 'cosine':
            
            num_warmup_acc_min = min(num_warmup_acc, num_firstepoch_steps_acc)

            if num_warmup_acc>num_firstepoch_steps_acc:
                print(f"\033[1;31;1mWARNING: The number of warmup steps is set too high! It will be clamped to 1 epoch, essentially going from warmup to annealing.\033[0;37;0m")
                print (f"FP Scheduler Warmup: 0-[{num_warmup_acc_min}], Hold [{num_warmup_acc_min}]-{num_firstepoch_steps_acc}, Annealing {num_firstepoch_steps_acc}-{num_training_steps_acc}")
            else:
                print (f"FP Scheduler Warmup: 0-{num_warmup_acc_min}, Hold {num_warmup_acc_min}-{num_firstepoch_steps_acc}, Annealing {num_firstepoch_steps_acc}-{num_training_steps_acc}")

            self.lr_scheduler = custom_cosine_scheduler_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps, 
                    num_firstepoch_steps = num_firstepoch_steps,
                )
            self._created_lr_scheduler = True
            return self.lr_scheduler
        elif self.args.lr_scheduler_type == 'constant':
           
            half_step_acc = num_training_steps_acc//2
            num_warmup_acc_min = min(num_warmup_acc, half_step_acc)

            if num_warmup_acc>half_step_acc:
                print(f"\033[1;31;1mWARNING: The number of warmup steps is set too high! It will be clamped to half of all epochs, essentially going from warmup to annealing in the middle.\033[0;37;0m")
                print (f"FP Scheduler Warmup: 0-[{num_warmup_acc_min}], Hold [{num_warmup_acc_min}]-{half_step_acc}, Annealing {half_step_acc}-{num_training_steps_acc}")
            else:
                print (f"FP Scheduler Warmup: 0-{num_warmup_acc_min}, Hold {num_warmup_acc_min}-{half_step_acc}, Annealing {half_step_acc}-{num_training_steps_acc}")

            self.lr_scheduler = custom_half_scheduler_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps, 
                    num_firstepoch_steps = num_firstepoch_steps,
                )
            self._created_lr_scheduler = True
            return self.lr_scheduler
        elif self.args.lr_scheduler_type == 'constant_with_warmup':
           
            half_step_acc = num_training_steps_acc//2
            
            if num_warmup_steps>0:
                print(f"Warmup doesn't apply to this scheduler [Raise-Fall]")

            print (f"Scheduler Raise: 0-{half_step_acc}, Fall {half_step_acc}-{num_training_steps_acc}")

            self.lr_scheduler = custom_raise_fall_scheduler_with_warmup(
                    optimizer=self.optimizer if optimizer is None else optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps, 
                    num_firstepoch_steps = num_firstepoch_steps,
                )
            self._created_lr_scheduler = True
            return self.lr_scheduler        
        else:
            return  super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)