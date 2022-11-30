#!/bin/bash

./case_studies/vanilla_classifier/baseline.sh $1 $2
./case_studies/vanilla_classifier/gradient_masking.sh $1 $2
./case_studies/vanilla_classifier/inefficient_pgd.sh $1 $2
./case_studies/vanilla_classifier/noisy_pgd.sh $1 $2
./case_studies/vanilla_classifier/non_differentiable_input.sh $1 $2
