from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from attacks.PreC_lib import rgb2lab_diff, ciede2000_diff
from loguru import logger

def quantization(x):
    """Quantize the continuous image tensors into 255 levels (8-bit encoding)."""
    return torch.round(x * 255) / 255


class PerC_CW:
    """
    PerC_CW: C&W with a new substitute penalty term concerning perceptual color differences.
    Modified from https://github.com/jeromerony/fast_adversarial/blob/master/fast_adv/adversarys/carlini.py

    Parameters
    ----------
    image_constraints : tuple
        Bounds of the images.
    num_classes : int
        Number of classes of the model to adversary.
    confidence : float, optional
        Confidence of the adversary for Carlini's loss, in terms of distance between logits.
    learning_rate : float
        Learning rate for the optimization.
    search_steps : int
        Number of search steps to find the best scale constant for Carlini's loss.
    max_iterations : int
        Maximum number of iterations during a single search step.
    initial_const : float
        Initial constant of the adversary.
    device : torch.device, optional
        Device to use for the adversary.
    """

    def __init__(self, model, config, image_constraints: Tuple[float, float] = (0, 1), num_classes: int = 10,
                 confidence: float = 0, learning_rate: float = 0.01, search_steps: int = 9,
                 max_iterations: int = 1000, abort_early: bool = True, initial_const: float = 10) -> None:
        self.model = model
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.binary_search_steps = search_steps
        self.max_iterations = max_iterations
        self.abort_early = abort_early
        self.initial_const = initial_const
        self.num_classes = num_classes
        self.device = config.device
        self.repeat = self.binary_search_steps >= 10
        self.boxmin, self.boxmax = image_constraints
        self.boxmul = (self.boxmax - self.boxmin) / 2
        self.boxplus = (self.boxmin + self.boxmax) / 2

    @staticmethod
    def _arctanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x *= (1. - eps)
        return 0.5 * torch.log((1 + x) / (1 - x))

    def _step(self, model: nn.Module, optimizer: optim.Optimizer, inputs: torch.Tensor, tinputs: torch.Tensor,
              modifier: torch.Tensor, labels: torch.Tensor, labels_infhot: torch.Tensor, targeted: bool,
              const: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size = inputs.shape[0]
        adv_input = torch.tanh(tinputs + modifier) * self.boxmul + self.boxplus
        l2 = torch.norm(ciede2000_diff(rgb2lab_diff(inputs, self.device), rgb2lab_diff(adv_input, self.device), self.device).view(batch_size, -1), dim=1)
        logits = model.predict((adv_input - 0.5) / 0.5)
        # labels_infhot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), float('inf'))
        real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = (logits - labels_infhot).max(1)[0]
        if targeted:
            logit_dists = torch.clamp(other - real + self.confidence, min=0)
        else:
            logit_dists = torch.clamp(real - other + self.confidence, min=0)

        logit_loss = (const * logit_dists).sum()
        l2_loss = l2.sum()
        loss = logit_loss + l2_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return quantization(adv_input).detach(), logits.detach(), l2.detach(), logit_dists.detach(), loss.detach()

    def attack(self, inputs: torch.Tensor, epsilon: float, labels: torch.Tensor, targeted: bool = False) -> torch.Tensor:
        inputs = inputs.clone().detach().to(self.device)
        batch_size = inputs.shape[0]
        tinputs = self._arctanh((inputs - self.boxplus) / self.boxmul)

        lower_bound = torch.zeros(batch_size, device=self.device)
        CONST = torch.full((batch_size,), self.initial_const, device=self.device, dtype=torch.float)
        upper_bound = torch.full((batch_size,), 1e10, device=self.device)

        o_best_l2 = torch.full((batch_size,), 1e10, device=self.device)
        o_best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)
        o_best_adversary = inputs.clone()

        labels_onehot = torch.zeros(labels.size(0), self.num_classes, device=self.device)
        # logger.debug(f'labels_onehot shape:{labels_onehot.shape}')
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        for outer_step in range(self.binary_search_steps):
            modifier = torch.zeros_like(inputs, requires_grad=True)
            optimizer = optim.Adam([modifier], lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-8)
            best_l2 = torch.full((batch_size,), 1e10, device=self.device)
            best_score = torch.full((batch_size,), -1, dtype=torch.long, device=self.device)

            if self.repeat and outer_step == (self.binary_search_steps - 1):
                CONST = upper_bound

            prev = float('inf')
            for iteration in range(epsilon):
                adv, logits, l2, logit_dists, loss = self._step(self.model, optimizer, inputs, tinputs, modifier,
                                                                labels, labels_infhot, targeted, CONST)

                if self.abort_early and iteration % (self.max_iterations // 10) == 0:
                    if loss > prev * 0.9999:
                        break
                    prev = loss

                predicted_classes = (self.model.predict((adv - 0.5) / 0.5) - labels_onehot * self.confidence).argmax(1) if targeted else \
                                   (self.model.predict((adv - 0.5) / 0.5) + labels_onehot * self.confidence).argmax(1)

                is_adv = (predicted_classes == labels) if targeted else (predicted_classes != labels)
                is_smaller = l2 < best_l2
                o_is_smaller = l2 < o_best_l2
                is_both = is_adv * is_smaller
                o_is_both = is_adv * o_is_smaller

                best_l2[is_both] = l2[is_both]
                best_score[is_both] = predicted_classes[is_both]
                o_best_l2[o_is_both] = l2[o_is_both]
                o_best_score[o_is_both] = predicted_classes[o_is_both]
                o_best_adversary[o_is_both] = adv[o_is_both]

            adv_found = (best_score == labels) if targeted else ((best_score != labels) * (best_score != -1))
            upper_bound[adv_found] = torch.min(upper_bound[adv_found], CONST[adv_found])
            adv_not_found = ~adv_found
            lower_bound[adv_not_found] = torch.max(lower_bound[adv_not_found], CONST[adv_not_found])
            is_smaller = upper_bound < 1e9
            CONST[is_smaller] = (lower_bound[is_smaller] + upper_bound[is_smaller]) / 2
            CONST[(~is_smaller) * adv_not_found] *= 10

        return o_best_adversary
