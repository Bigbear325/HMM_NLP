#!/bin/bash

# try different Mu with data/p1/test.txt concatenated with data/p2/unlabeled_20news.txt
for mu in  0.0 0.1 0.2
do
echo "predict $mu"
java -cp jars/jama.jar:bin/ HMM data/p1/train.txt data/p2/concatenated.txt results/p2/prediction_concatenated results/p2/training_log_concatenated $mu
done

# evaluate the results
for mu in 0.0 0.1 0.2
do
echo "evaluate $mu"
java -cp bin/ Evaluator data/p1/test.txt results/p2/prediction_${mu}.txt
done

