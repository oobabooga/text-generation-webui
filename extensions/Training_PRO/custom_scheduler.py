from functools import partial
import torch
import transformers
import math
from torch.optim.lr_scheduler import LambdaLR


#FPHAM custom training scheduller block - should be extracted to separate file
last_print_label = ''

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

class FPSchedulerTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
        #Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or passed as an argument.
        
        num_train_epochs = self.args.num_train_epochs
        num_warmup_steps=self.args.get_warmup_steps(num_training_steps)
        num_firstepoch_steps = math.ceil(num_training_steps/num_train_epochs)
        num_warmup_acc = num_warmup_steps*self.args.gradient_accumulation_steps 
        num_firstepoch_steps_acc = num_firstepoch_steps*self.args.gradient_accumulation_steps
        num_training_steps_acc = num_training_steps*self.args.gradient_accumulation_steps
        
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
        else:
            return  super().create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)