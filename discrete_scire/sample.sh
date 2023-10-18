DEVICES='0'

# ##########################

# CIFAR-10 (DDPM checkpoint) example

data="cifar10"
# data="imagenet64"
# data="imagenet256_guided"
# data="imagenet512_guided"
# data="imagenet128_guided"
# data="bedroom_guided"
# data="celeba"
# scale="1"
sampleMethod="scire"
#sampleMethod="ei"
type="rde"
# type='dpm'
# type='fde'
STEPLIST=('50')
Trajectory="NSR"
#Trajectory="logSNR"
#Trajectory="time_uniform"
order="2"
endtime="1e-4"
method="singlestep_fixed"
# method="singlestep_agile"
# method="multistep"


for steps in ${STEPLIST[*]};
do
    workdir="experiments/"$data"/"$sampleMethod"_"$method"_order"$order"_"$steps"_"$Trajectory"_"$type"_"$endtime
    CUDA_VISIBLE_DEVICES=$DEVICES python main.py --config $data".yml" --exp=$workdir --sample --fid --timesteps $steps --eta 0 --ni --skip_type=$Trajectory --sample_type=$sampleMethod --scire_solver_order=$order --end_time=$endtime  --scire_solver_method=$method --scire_solver_type=$type --port 12355 #--scale=$scale
done

