# Run simulation

```
python main.py
```

# Analysis and plots


- **plot_cumulative_fallen_agents_time_lambda.py:** 
  plots Cumulative time series of fallen people for different initial number of pedestrians for different λ values
  alpha = fix


- **plot_cumulative_fallen_agents_time_alpha.py:** 
  plots Cumulative time series of fallen people for different initial number of pedestrians for different alpha values
  lambda = fix


- **heatmap.py:** 
  Plots many png figure of the survival heatmap.

- **plot_heatmap_once.py:** 
  Plots a pdf file of the survival heatmap. Once for a fixed lambda


# results 
## Run 5000 
  num_agents_list = [5000]
    lambda_decay_list = [0.2, 0.3]
    global_seed = 1234
    num_reps = 5
    gamma = 0.8
    alpha = [0.3, 0.7]  # 1.0 for physical shielding, 0.0 for targeted fire
