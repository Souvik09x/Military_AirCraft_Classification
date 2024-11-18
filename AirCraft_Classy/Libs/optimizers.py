#import your nessessary libreries here


#define your Optimizer functions here
###DEMO
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def OptAdamax(model,**kwargs):
   return optim.Adamax(model.parameters(), lr=0.001)


class OptimizerWithScheduler:
   def __init__(self, optimizer, scheduler_type='StepLR', **kwargs):
            # Create Adamax optimizer
            self.optimizer = optimizer
            
            # Choose the scheduler
            if scheduler_type == 'StepLR':
                self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
            elif scheduler_type == 'ReduceLROnPlateau':
                self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
            else:
                raise ValueError(f"Unknown scheduler type: {scheduler_type}")

   def step(self, loss=None):
            """ Perform one step of optimization and adjust the learning rate using the scheduler. """
            self.optimizer.step()

            # If using ReduceLROnPlateau, pass the loss to adjust the learning rate
            if isinstance(self.scheduler, ReduceLROnPlateau) and loss is not None:
                self.scheduler.step(loss)  # This adjusts the LR based on validation loss
            elif isinstance(self.scheduler, StepLR):
                self.scheduler.step()  # This just steps based on epochs, so no need for loss

   def zero_grad(self):
            """ Zero out the gradients in the optimizer """
            self.optimizer.zero_grad()

   def get_lr(self):
            """ Get the current learning rate """
            return self.optimizer.param_groups[0]['lr']  # Assuming a single learning rate for all params
   def get_last_lr(self):
            """ Get the last learning rate from the scheduler """
            return self.scheduler.get_last_lr()[0] 
    

def OptAdamax_sc(model, **kwargs):
   optimizer = optim.Adamax(model.parameters(), lr=0.001, weight_decay=1e-5)
    # Create and return an instance of the OptimizerWithScheduler class
   return OptimizerWithScheduler(optimizer, "ReduceLROnPlateau", **kwargs)


# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# def OptAdamax(model, scheduler_type='StepLR', **kwargs):
    
#     class OptimizerWithScheduler:
#         def __init__(self, model, scheduler_type='StepLR', **kwargs):
#             # Create Adamax optimizer
#             self.optimizer = optim.Adamax(model.parameters(), lr=0.001)
            
#             # Choose the scheduler
#             if scheduler_type == 'StepLR':
#                 self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
#             elif scheduler_type == 'ReduceLROnPlateau':
#                 self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=3)
#             else:
#                 raise ValueError(f"Unknown scheduler type: {scheduler_type}")

#         def step(self, loss=None):
#             """ Perform one step of optimization and adjust the learning rate using the scheduler. """
#             self.optimizer.step()

#             # If using ReduceLROnPlateau, pass the loss to adjust the learning rate
#             if isinstance(self.scheduler, ReduceLROnPlateau) and loss is not None:
#                 self.scheduler.step(loss)  # This adjusts the LR based on validation loss
#             elif isinstance(self.scheduler, StepLR):
#                 self.scheduler.step()  # This just steps based on epochs, so no need for loss

#         def zero_grad(self):
#             """ Zero out the gradients in the optimizer """
#             self.optimizer.zero_grad()

#         def get_lr(self):
#             """ Get the current learning rate """
#             return self.optimizer.param_groups[0]['lr']  # Assuming a single learning rate for all params
        
#         def get_last_lr(self):
#             """ Get the last learning rate from the scheduler """
#             return self.scheduler.get_last_lr()[0]  # Get the last learning rate value (if using scheduler)

#     # Create and return an instance of the OptimizerWithScheduler class
#     return OptimizerWithScheduler(model, scheduler_type, **kwargs)
