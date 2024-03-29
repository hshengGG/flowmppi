import torch
import time

class MPPI:

    def __init__(self, cost, dx, du, horizon, num_samples, lambda_, sigma, control_constraints=None,
                 device='cuda:0', action_transform=None, flow=False, iterations=1):
        self.dx = dx
        self.du = du
        self.cost = cost
        self.H = horizon
        self.N = num_samples
        self.sigma = sigma
        self.lambda_ = lambda_
        self.device = device
        self.control_constraints = control_constraints
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)
        self.Z = self.U
        self.action_transform = action_transform
        self.flow = flow
        self.best_K_U = None
        self.K = 100
        self.iterations = iterations
        self.ref_z = torch.randn(self.H, self.du, device=self.device)
        
    def step(self, x):
        for iter in range(self.iterations):
            # Sample peturbations
            noise = torch.randn(self.N, self.H, self.du, device=self.device)
            noise[-1] *= 0.0
            # Get peturbation cost
            peturbed_actions = torch.zeros_like(noise)

            if self.flow:
                # Peturbed actions in action space
                start_time = time.time()
		        
                peturbed_actions[:self.N//2] = self.action_transform(x[0].unsqueeze(0), noise[:self.N//2])
                print("-----------x[0] input size-----------")
                print(x[0].unsqueeze(0).shape)
                print("-----------printing reverse flow input size-----------")
                print(noise[:self.N//2].shape)
                end_time = time.time()
                self.reverse_NF_time = end_time-start_time
                #To Do
                # should be reverse

                #appending times to be print out
                #self.forward_NF_times.append(forward_NF_time)
                peturbed_actions[self.N//2:] = self.U.unsqueeze(dim=0) + self.sigma * noise[self.N//2:]
                action_cost_Z = torch.sum(self.lambda_ * noise * (noise - self.Z), dim=[1, 2])
                action_cost_U = torch.sum(self.lambda_ * noise * self.U / self.sigma, dim=[1, 2])

                action_cost = torch.cat((action_cost_Z[:self.N//2], action_cost_U[self.N//2:]), dim=0) / self.du
                #print(action_cost_U.mean(), action_cost_U.min(), action_cost_U.max())


            else:
                peturbed_actions = self.U.unsqueeze(dim=0) + self.sigma * noise
                if self.control_constraints is not None:
                    peturbed_actions = torch.clamp(peturbed_actions, min=self.control_constraints[0],
                                                   max=self.control_constraints[1])
                action_cost = torch.sum(self.lambda_ * noise * self.U / self.sigma, dim=[1, 2]) / self.du
            # Get total cost
            start_time = time.time()
            total_cost = self.cost(x, peturbed_actions)
            end_time = time.time()
            self.cost_time = end_time-start_time


            total_cost += action_cost
            #total_cost -= torch.min(total_cost)
            #total_cost /= torch.max(total_cost)
            omega = torch.softmax(-total_cost / self.lambda_, dim=0)

            self.U = torch.sum((omega.reshape(-1, 1, 1) * peturbed_actions), dim=0)
            #self.U = peturbed_actions[0]
            #_, idx = torch.sort(total_cost, dim=0, descending=False)
            idx = torch.randperm(self.N).to(device=self.device)
            self.best_K_U = torch.gather(peturbed_actions, 0, idx[:self.K].reshape(-1, 1, 1).repeat(1, self.H, self.du))
            #self.best_K_U = peturbed_actions[0].reshape(1, self.H, self.du)
            
        #
        ## Shift U along by 1 timestep
        #if self.action_transform is not None:
        #    # We transform to true action space, do the shifting, then transform #back
        #    Z = self.action_transform(x[0].unsqueeze(0),
        #                              Z.unsqueeze(0))[0]
        #    self.U = torch.roll(Z, -1, dims=0)
        #    self.U[-1] = torch.randn(self.du, device=self.device)
        #
        #    # Save shifted U as the nominal trajectory for next time
        #    self.Z = self.action_transform(x[0].unsqueeze(0),
        #                                   self.U.unsqueeze(0),
        #                                   reverse=True)[0]
        #    # We return the unshifted version
        #    return Z

        out_U = self.U.clone()
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = torch.zeros(self.du, device=self.device)
        action_start = 0
        action_end = 0
        
        if self.action_transform is not None:
            action_start = time.time()
            self.Z = self.action_transform(x[0].unsqueeze(0),
                                           self.U.unsqueeze(0),
                                           reverse=True)
            print("-----------printing forward flow input size-----------")
            print(self.U.unsqueeze(0).shape)
            action_end = time.time()
        self.forward_NF_time = action_end - action_start
        #To Do
        #should be forward
        
        return out_U

    def reset(self):
        self.U = self.sigma * torch.randn(self.H, self.du, device=self.device)
        self.Z = torch.zeros_like(self.U)
