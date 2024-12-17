import copy
import os
import torch


class FoRLtoOnnxExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, actor, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = copy.deepcopy(actor.normalizer.to("cpu"))  # need to use this again in forward function
        self.actor = copy.deepcopy(actor.mu_net)  # only get the network for self.actor
        self.is_recurrent = False  # always False for FoRL so far

        # (test) add print statement to see if actors forward method is used or the method with normalization below
        # self.actor.forward = change_actor_forward_method.__get__(self.actor)

    def forward(self, x):
        return torch.tanh(self.actor(self.normalizer.normalize(x)))  # normalizes actions and observations

    def export(self, path, filename):
        self.to("cpu")
        if self.is_recurrent:
            obs = torch.zeros(1, self.rnn.input_size)
            h_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            c_in = torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size)
            actions, h_out, c_out = self(obs, h_in, c_in)
            torch.onnx.export(
                self,
                (obs, h_in, c_in),
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs", "h_in", "c_in"],
                output_names=["actions", "h_out", "c_out"],
                dynamic_axes={},
            )
        else:
            obs = torch.zeros(1, self.actor[0].in_features)
            torch.onnx.export(
                self,
                obs,
                os.path.join(path, filename),
                export_params=True,
                opset_version=11,
                verbose=self.verbose,
                input_names=["obs"],
                output_names=["actions"],
                dynamic_axes={},
            )


def change_actor_forward_method(self, x):
    # 'self' refers to the actor module instance
    print("[WARNING]: Using FoRL's actor.forward method might not include normalization")
    return self(x)
