# Copyright 2022 The Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Taken from
https://github.com/v-wangg/OrthogonalPGD/blob/c92f11ee69723822f2179be1d6f50cd86d94bbff/attack.py#L15
"""
import torch
import numpy as np
import tqdm


class PGD:
  def __init__(self, classifier, detector, classifier_loss=None,
      detector_loss=None, steps=100, alpha=1 / 255, eps=8 / 255,
      use_projection=True, projection_norm='linf', target=None, lmbd=0, k=None,
      project_detector=False, project_classifier=False, img_min=0, img_max=1,
      verbose=True):
    '''
    :param classifier: model used for classification
    :param detector: model used for detection
    :param classifier_loss: loss used for classification model
    :param detector_loss: loss used for detection model. Need to have __call__
      method which outputs adversarial scores ranging from 0 to 1
      (0 if not afversarial and 1 if adversarial)
    :param steps: number of steps for which to perform gradient descent/ascent
    :param alpha: step size
    :param eps: constraint on noise that can be applied to images
    :param use_projection: True if gradients should be projected onto each other
    :param projection_norm: 'linf' or 'l2' for regularization of gradients
    :param target: target label to attack. if None, an untargeted attack is run
    :param lmbd: hyperparameter for 'f + lmbd * g' when 'use_projection' is False
    :param k: if not None, take gradients of g onto f every kth step
    :param project_detector: if True, take gradients of g onto f
    :param project_classifier: if True, take gradients of f onto g
    '''
    self.classifier = classifier
    self.detector = detector
    self.steps = steps
    self.alpha = alpha
    self.eps = eps
    self.classifier_loss = classifier_loss
    self.detector_loss = detector_loss
    self.use_projection = use_projection
    self.projection_norm = projection_norm
    self.project_classifier = project_classifier
    self.project_detector = project_detector
    self.target = target
    self.lmbd = lmbd
    self.k = k
    self.img_min = img_min
    self.img_max = img_max

    self.verbose = verbose

    # metrics to keep track of
    self.all_classifier_losses = []
    self.all_detector_losses = []

  def attack_batch(self, inputs, targets, device):
    adv_images = inputs.clone().detach()
    original_inputs_numpy = inputs.clone().detach().numpy()

    #         alarm_targets = torch.tensor(np.zeros(len(inputs)).reshape(-1, 1))

    # ideally no adversarial images should be detected
    alarm_targets = torch.tensor(np.zeros(len(inputs)))

    batch_size = inputs.shape[0]

    # targeted attack
    if self.target:
      targeted_targets = torch.tensor(
        torch.tensor(self.target * np.ones(len(inputs)), dtype=torch.int64)).to(
          device)

    advx_final = inputs.detach().numpy()
    loss_final = np.zeros(inputs.shape[0]) + np.inf

    if self.verbose:
      progress = tqdm.tqdm(range(self.steps))
    else:
      progress = range(self.steps)
    for i in progress:
      adv_images.requires_grad = True

      # calculating gradient of classifier w.r.t. images
      outputs = self.classifier(adv_images.to(device))

      if self.target is not None:
        loss_classifier = 1 * self.classifier_loss(outputs, targeted_targets)
      else:
        loss_classifier = self.classifier_loss(outputs, targets.to(device))

      loss_classifier.backward(retain_graph=True)
      grad_classifier = adv_images.grad.cpu().detach()

      # calculating gradient of detector w.r.t. images
      adv_images.grad = None
      adv_scores = self.detector(adv_images.to(device))

      if self.detector_loss:
        loss_detector = -self.detector_loss(adv_scores, alarm_targets.to(device))
      else:
        loss_detector = torch.mean(adv_scores)

      loss_detector.backward()
      grad_detector = adv_images.grad.cpu().detach()

      self.all_classifier_losses.append(loss_classifier.detach().data.item())
      self.all_detector_losses.append(loss_detector.detach().data.item())

      if self.target:
        has_attack_succeeded = (outputs.cpu().detach().numpy().argmax(
          1) == targeted_targets.cpu().numpy())
      else:
        has_attack_succeeded = (
              outputs.cpu().detach().numpy().argmax(1) != targets.numpy())

      adv_images_np = adv_images.cpu().detach().numpy()
      # print(torch.max(torch.abs(adv_images-inputs)))
      # print('b',torch.max(torch.abs(torch.tensor(advx_final)-inputs)))
      for i in range(len(advx_final)):
        if has_attack_succeeded[i] and loss_final[i] > adv_scores[i]:
          # print("assign", i, np.max(advx_final[i]-original_inputs_numpy[i]))
          advx_final[i] = adv_images_np[i]
          loss_final[i] = adv_scores[i]
          # print("Update", i, adv_scores[i])

      # using hyperparameter to combine gradient of classifier and gradient of detector
      if not self.use_projection:
        grad = grad_classifier + self.lmbd * grad_detector
      else:
        if self.project_detector:
          # using Orthogonal Projected Gradient Descent
          # projection of gradient of detector on gradient of classifier
          # then grad_d' = grad_d - (project grad_d onto grad_c)
          grad_detector_proj = grad_detector - torch.bmm((torch.bmm(
            grad_detector.view(batch_size, 1, -1),
            grad_classifier.view(batch_size, -1, 1))) / (1e-20 + torch.bmm(
            grad_classifier.view(batch_size, 1, -1),
            grad_classifier.view(batch_size, -1, 1))).view(-1, 1, 1),
                                                         grad_classifier.view(
                                                           batch_size, 1,
                                                           -1)).view(
            grad_detector.shape)
        else:
          grad_detector_proj = grad_detector

        if self.project_classifier:
          # using Orthogonal Projected Gradient Descent
          # projection of gradient of detector on gradient of classifier
          # then grad_c' = grad_c - (project grad_c onto grad_d)
          grad_classifier_proj = grad_classifier - torch.bmm((torch.bmm(
            grad_classifier.view(batch_size, 1, -1),
            grad_detector.view(batch_size, -1, 1))) / (1e-20 + torch.bmm(
            grad_detector.view(batch_size, 1, -1),
            grad_detector.view(batch_size, -1, 1))).view(-1, 1, 1),
                                                             grad_detector.view(
                                                               batch_size, 1,
                                                               -1)).view(
            grad_classifier.shape)
        else:
          grad_classifier_proj = grad_classifier

        # making sure adversarial images have crossed decision boundary
        outputs_perturbed = outputs.cpu().detach().numpy()
        if self.target:
          outputs_perturbed[
            np.arange(targeted_targets.shape[0]), targets] += .05
          has_attack_succeeded = np.array(
              (outputs_perturbed.argmax(1) == targeted_targets.cpu().numpy())[:,
              None, None, None], dtype=np.float32)
        else:
          outputs_perturbed[np.arange(targets.shape[0]), targets] += .05
          has_attack_succeeded = np.array(
              (outputs_perturbed.argmax(1) != targets.numpy())[:, None, None,
              None], dtype=np.float32)

        if self.verbose:
          progress.set_description(
              "Losses (%.3f/%.3f/%.3f/%.3f)" % (np.mean(self.all_classifier_losses[-10:]),
                                                np.mean(self.all_detector_losses[-10:]),
                                                np.mean(loss_final),
                                                has_attack_succeeded.mean()))

        # print('correct frac', has_attack_succeeded.mean())
        # print('really adv target reached', (outputs.argmax(1).cpu().detach().numpy() == self.target).mean())

        if self.k:
          # take gradients of g onto f every kth step
          if i % self.k == 0:
            grad = grad_detector_proj
          else:
            grad = grad_classifier_proj
        else:
          # print(outputs_perturbed, has_attack_succeeded, adv_scores)
          grad = grad_classifier_proj * (
                1 - has_attack_succeeded) + grad_detector_proj * has_attack_succeeded

        if np.any(np.isnan(grad.numpy())):
          print(np.mean(np.isnan(grad.numpy())))
          print("ABORT")
          break

      if self.target:
        grad = -grad

      # l2 regularization
      if self.projection_norm == 'l2':
        grad_norms = torch.norm(grad.view(batch_size, -1), p=2, dim=1) + 1e-20
        grad = grad / grad_norms.view(batch_size, 1, 1, 1)
      # linf regularization
      elif self.projection_norm == 'linf':
        grad = torch.sign(grad)
      else:
        raise Exception('Incorrect Projection Norm')

      adv_images = adv_images.detach() + self.alpha * grad
      delta = torch.clamp(adv_images - torch.tensor(original_inputs_numpy),
                          min=-self.eps, max=self.eps)
      adv_images = torch.clamp(torch.tensor(original_inputs_numpy) + delta,
                               min=self.img_min, max=self.img_max).detach()

    return torch.tensor(advx_final)

  def attack(self, inputs, targets, device):
    adv_images = []
    batch_adv_images = self.attack_batch(inputs, targets, device)
    adv_images.append(batch_adv_images)
    return torch.cat(adv_images)