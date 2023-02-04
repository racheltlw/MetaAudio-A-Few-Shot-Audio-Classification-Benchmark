from utils import *
import learn2learn as l2l
import yaml 
from model_selection import grab_model
from fit import validation_step_fixed, validation_step_variable

with open("maml_experiment_params.yaml") as stream:
    params = yaml.safe_load(stream)

# Setting of cuda device
device = torch.device('cuda:' + str(params['base']['cuda']) if \
    torch.cuda.is_available() else 'cpu')

# Loads in model params
with open("models/params/all_model_params.yaml") as stream:
    model_params = yaml.safe_load(stream)

name = 'Hybrid'

model = grab_model(name, model_params[name], out_dim=params['base']['n_way'])
model = model.to(device, dtype=torch.double)


maml = l2l.algorithms.MAML(model, lr=params['hyper']['inner_lr'], first_order=True)
learner = maml
path_to_model = "C:/Users/Rachel Tan/Documents/GitHub/MetaAudio-A-Few-Shot-Audio-Classification-Benchmark/Examples/MAML_ESC/results/MAML_ESC_TRIAL_4Jan23Hybrid_global_1_runs\MAML_ESC_TRIAL_4Jan23Hybrid_global_1_runs_2000_seed_922/best_val_model__05_01__10_47.pt"

best_model = load_model(path_to_model, learner)

print(f"model loaded from {path_to_model}")

#now just figure out how to use it 

# Chooses what type of validation step to take and sets according functions
if params['data']['variable']:
    validation_step = validation_step_variable
else:
    validation_step = validation_step_fixed






final_loss, final_pre, final_post, final_post_std = validation_step(evalLoader, learner,
                                                optimiser, val_batch,
                                                var_fit_function,
                                                loss_fn, params,
                                                **meta_func_kwargs)