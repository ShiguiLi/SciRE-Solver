
devices="0,1,2"
# steps="100"
# multi_steps=("100")
multi_steps=("20")
# multi_steps=("18" "20" "21" "24" "30" "36" "42" "48" "100")
multi_eps=("1e-4" "1e-3")
# skip="sigmoid"
skip="NSR"
# skip="time_quadratic"
# method="singlestep_fixed"
method="singlestep"
# order="3"
multi_orders=("3" "2")
dir="experiments/cifar10" 
# dir="experiments/celeba64" 
for eps in ${multi_eps[*]};
do
    for order in ${multi_orders[*]};
    do
        for step in ${multi_steps[*]};
        do
           CUDA_VISIBLE_DEVICES=$devices python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" --mode "eval" --workdir $dir --config.sampling.eps=$eps --config.sampling.method="scire_solver" --config.sampling.steps=$step --config.sampling.skip_type=$skip --config.sampling.scire_solver_order=$order --config.sampling.scire_solver_method=$method --config.eval.batch_size=300
       # sleep 30s
        done
    done
done
