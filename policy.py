import torch
import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class Policy(object):

    def __init__(self,pop_size):
        self.pop_size = pop_size    

    def update(self, all_rewards, population, pictures,index_list):
        rank = np.argsort(np.asarray(all_rewards))
        rank = index_list
        best_picture = pictures[rank[0]]
        # index_list = np.array([index_list[x] for x in rank])
        # child_index = [np.where(index_list == x)[0][0] for x in range (self.pop_size)]
        child_index = [rank[i] for i in range(self.pop_size)]
        pop = [population[child_index[i]] for i in range(self.pop_size)]
        fake = [pictures[child_index[i]] for i in range(self.pop_size)]
        fitness = [all_rewards[child_index[i]] for i in range(self.pop_size)]
        g_loss = sum(fitness)/self.pop_size
      
        return pop, fake, g_loss, best_picture

    def save_model(self, generator, discriminator):
        PATH = "Model.pt"
        
        torch.save({
            'discriminator': discriminator.state_dict(),
            'generator': generator[0].state_dict(),
            }, PATH)

    def get_parameters(self, net):
        net.cpu()
        parameters = parameters_to_vector(net.parameters())
        if torch.cuda.is_available():
            net.cuda()
        return parameters.detach().numpy()
        

    def set_parameters(self, child_param, netG, evaluate=False):
        perturb_parameters = torch.from_numpy(child_param)
        vector_to_parameters(perturb_parameters,netG.parameters())
        if torch.cuda.is_available():
            netG.cuda()
        return netG