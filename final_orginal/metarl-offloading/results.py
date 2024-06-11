inner_lr = 5.5566373111494876e-05 ,outer_lr = 3.310582278465638e-05, num_inner_grad_steps = 10.0, inner_batch_size = 10.0, 
Teacher = 2356.216140477327, population_size = 15, bounds = [[1.e-20 5.e-04]
 [1.e-20 5.e-04]
 [1.e+01 2.e+01]
 [1.e+01 2.e+01]], iterations = 5
Logging to ./meta_evaluate_ppo_log/task_offloading
calculate baseline solution======
avg greedy solution:  -5.326296975575445
avg greedy solution:  802.4605928660801
avg greedy solution:  808.9166261023114
avg all remote solution:  1052.0136239681242
avg all local solution:  1478.0573242759704
Results for checkpoint meta_model_inner_step1/meta_model_final.ckpt:
  Average Return: -4.851698417822312
  Average PG Loss: -0.0002368597923145617
  Average VF Loss: 0.12884486107914536
  Average Latencies: 730.966460358892
Results for checkpoint meta_model_inner_step1/meta_model_0.ckpt:
  Average Return: -4.992803928902546
  Average PG Loss: 0.0019807949501239224
  Average VF Loss: 0.4061409117263041
  Average Latencies: 750.6904433634486
Results for checkpoint meta_model_inner_step1/meta_model_1.ckpt:
  Average Return: -4.945997264130764
  Average PG Loss: 0.0012112957321935229
  Average VF Loss: 0.2669190738672092
  Average Latencies: 743.692942399856
Results for checkpoint meta_model_inner_step1/meta_model_2.ckpt:
  Average Return: -4.8989037079276265
  Average PG Loss: 0.002616419267185308
  Average VF Loss: 0.37367334925098183
  Average Latencies: 736.6712465657098
Results for checkpoint meta_model_inner_step1/meta_model_3.ckpt:
  Average Return: -4.862355049655847
  Average PG Loss: 0.00024308134330275634
  Average VF Loss: 0.19823731003720083
  Average Latencies: 732.0288710041455
Results for checkpoint meta_model_inner_step1/meta_model_4.ckpt:
  Average Return: -4.849380632505264
  Average PG Loss: -0.0002754885519360318
  Average VF Loss: 0.13121393617288565
  Average Latencies: 729.7815359990665
#####################################################3
inner_lr = 0.00032397450148526324 ,outer_lr = 9.678026939744262e-05, num_inner_grad_steps = 96.97938114491716, inner_batch_size = 59.20168996809036, 
Teacher = 2312.656700198991, population_size = 15, bounds = [[1.e-20 5.e-04]
 [1.e-20 5.e-04]
 [1.e+01 1.e+03]
 [1.e+01 1.e+03]], iterations = 5
Logging to ./meta_evaluate_ppo_log/task_offloading
calculate baseline solution======
avg greedy solution:  -5.326296975575445
avg greedy solution:  802.4605928660801
avg greedy solution:  808.9166261023114
avg all remote solution:  1052.0136239681242
avg all local solution:  1478.0573242759704
2024-06-11 12:58:16.655952: E tensorflow/stream_executor/cuda/cuda_driver.cc:318] failed call to cuInit: UNKNOWN ERROR (303)
Results for checkpoint meta_model_inner_step1/meta_model_final.ckpt:
  Average Return: -4.751705895611848
  Average PG Loss: 0.03493789049946232
  Average VF Loss: 0.5289092288341052
  Average Latencies: 715.6292025431087
Results for checkpoint meta_model_inner_step1/meta_model_0.ckpt:
  Average Return: -5.2072377650803645
  Average PG Loss: 0.019009363274147484
  Average VF Loss: 0.7643318029097569
  Average Latencies: 783.1618547157833
Results for checkpoint meta_model_inner_step1/meta_model_1.ckpt:
  Average Return: -4.906199652107706
  Average PG Loss: 0.003946987455739514
  Average VF Loss: 0.41982930383564515
  Average Latencies: 738.0639030411721
Results for checkpoint meta_model_inner_step1/meta_model_2.ckpt:
  Average Return: -4.777512328039849
  Average PG Loss: 0.019272136461725573
  Average VF Loss: 0.5129580207077074
  Average Latencies: 719.3828720109123
Results for checkpoint meta_model_inner_step1/meta_model_3.ckpt:
  Average Return: -4.752522625180389
  Average PG Loss: 0.02553343614218412
  Average VF Loss: 0.6093240880671842
  Average Latencies: 716.0410014786312
Results for checkpoint meta_model_inner_step1/meta_model_4.ckpt:
  Average Return: -4.745961351245394
  Average PG Loss: 0.02858163919414819
  Average VF Loss: 0.5138616377924695
  Average Latencies: 714.3810721910205