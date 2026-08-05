#!/bin/bash
export PYTHONPATH="DiLA_for_vp2/vp2" # absolute path to the vp2 directory
echo PYTHONPATH=$PYTHONPATH

# for robodesk push_red, push_blue, push_green, upright_block_off_table, flat_block_off_table
accelerate launch --mixed_precision no scripts/run_control.py --multirun hydra.job.name=rd_case_study \
planning_modalities=[rgb] seed=1,2,3,4 agent.replan_interval=1 \
env=robodesk env.task=push_red env.a_dim=5 \
model=dila model_name=dila_model model.epoch=33 \
sweep=multi_task_epoch agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] \
agent.optimizer.log_every=5

# for robodesk open_slide, open_drawer
# accelerate launch --mixed_precision no scripts/run_control.py --multirun hydra.job.name=rd_case_study \
# planning_modalities=[rgb] seed=1,2,3,4 agent.replan_interval=1 \
# env=robodesk env.task=open_slide env.a_dim=5 \
# model=dila model_name=dila_model model.epoch=33 \
# sweep=multi_task_epoch agent.optimizer.init_std=[0.5,0.5,0.5,0.1,0.1] \
# agent.optimizer.log_every=5 agent.optimizer.num_samples=800

# for robosuite
# accelerate launch --mixed_precision no scripts/run_control.py --multirun hydra.job.name=rs_case_study \
# env=robosuite env.a_dim=4 \
# model=dila model_name=dila_model model.epoch=33 \
# agent/optimizer/objective=mse_rgb seed=1,2,3,4 agent.optimizer.log_every=5 sweep=single_task_epoch

