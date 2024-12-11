Code run on Ubuntu 20.04

To install the necessary dependencies, under /training_module:
`python3 setup.py install --user `
or 
`pip3 install -e . `

To define the environment containing the MPC task: training module/envs/MPC_task.py
Lines 216 through 230 defines the user command velocity. Lines 237 through 252 defines the disturbances.

To test the MPC (without RL policy): training module/mpc_test.py

To train the policy: training module/pytorch_train_mpc.py

To test the policy: training module/pytorch_test_mlp.py
Lines 31 through 33 selects the policy to test.
When running T6, exchange commenting between lines 113 and 114 to accomodate for the correct observation space.

To plot a comparison: training module/compare_plot.py.

Contact Muqun Hu for any questions: hu1033@purdue.edu

