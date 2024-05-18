import os
import time 
from validate import validate
from data import create_dataloader
# from earlystop import EarlyStopping 
from networks.trainer import Trainer
from options.train_options import TrainOptions
import numpy as np


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.val_acc = -10000.0

    def early_stop(self, curr_acc):
        if curr_acc > self.val_acc:
            self.val_acc = curr_acc
            self.counter = 0 
        elif curr_acc < (self.val_acc - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt



if __name__ == '__main__':
    opt = TrainOptions().parse()
    val_opt = get_val_opt()

    best_acc = 0.0
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt) 
    
    early_stopper = EarlyStopper(patience=opt.earlystop_epoch, min_delta=0.02)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    for epoch in range(opt.niter):
        for i, data in enumerate(data_loader):
            model.total_steps += 1
            model.set_input(data)
            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                print("Iter time: ", ((time.time()-start_time)/model.total_steps))

            if model.total_steps in [10,30,50,100,1000,5000,10000] and False: # save models at these iters 
                model.save_networks('model_iters_%s.pth' % model.total_steps)

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader) 
        print("(Val @ epoch {}) acc: {}; ap: {}".format(epoch, acc, ap))

        if early_stopper.early_stop(acc): 
            cont_train = model.adjust_learning_rate()
            if cont_train:
                print("Learning rate dropped by 10, continue training...") 
                early_stopper = EarlyStopper(patience=opt.earlystop_epoch, min_delta=0.01)
            else:
                print("Early stopping.")
                break

        if acc > best_acc:
            best_acc = acc 
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( f'model_acc_{acc:.5f}_epoch_{epoch}.pth' ) 

        model.train()
