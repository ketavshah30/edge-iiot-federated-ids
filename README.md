# Federated Learning with XAI for Threat Detection in 6G Edge-IIoT Networks

A defensive research prototype for multiclass intrusion detection on Edge-IIoTset. It trains a federated neural network and uses SHAP and LIME to explain predictions.

## Saved evaluation

The archived run evaluated 443,841 test examples across 15 classes and reported 88.7% accuracy, 86.7% weighted F1, and 35.0% macro F1. The gap between weighted and macro F1 reflects uneven performance across classes. The JSON records the settings for that run. The script defaults are for exploratory runs and differ from the archived settings. The network latency chart in the script uses simulated data.

## Run the experiment

Install the dependencies with python -m pip install -r requirements.txt. Obtain Edge-IIoTset separately and place DNN-EdgeIIoT-dataset.csv next to experiment.py. Run python experiment.py from the repository root.
