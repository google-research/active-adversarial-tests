attack=${1:-pgd}
echo "Attack: $attack"

if [[ "$attack" == "thermometer-lspgd" ]]; then
  nsteps=7
  stepsize=0.01
  attackname="tlspgd"
else
  nsteps=10
  stepsize=0.005
  attackname="pgd"
fi

PYTHONPATH=$PYTHONPATH:$(pwd) python train_classifier.py \
  -bs=256 -lr=0.1 -op=sgd -sm=0.9 -wd=5e-4 -ne=200 --device="cuda" \
  --learning-rate-scheduler="multistep-0.1-60,120,160" \
  --classifier="networks.non_differentiable_16_thermometer_encoding_cifar_wideresnet344" \
  --output=checkpoints/thermometer_16_wrn304_linf_at_${attackname}_200_epochs.pth \
  -at="norm=linf epsilon=0.031372549 step_size=$stepsize n_steps=$nsteps attack=$attack"
