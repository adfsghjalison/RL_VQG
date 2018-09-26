
update=$1
rank=$2
qg_update_epoch=10
total_training_data=20000
type_w_ini=500
qg_w_ini=100

plan=${update}_${rank}
dir=data/all/$plan
block=$[$total_training_data/$[$update+1]/$rank]
data_num=$[$block*$rank*($update+1)]

mkdir -p data/all/$plan
mkdir -p model/qg/$plan
mkdir -p model/final/$plan
mkdir -p log/$plan
mkdir -p output/$plan

echo -e "\nunits per block = $block\n"
echo -e "\nFor $plan\n"

for run in $(seq 0 $update)
do
	start=$[$run*$block]
	echo -e " \n\n ----- Generate Questions for (plan, run) = ($plan, $run) ----- \n\n"
	if (( run == 0 )); then
		python code/qg_gen.py $type_w_ini $qg_w_ini $dir/q-$run $start $block $rank  2>&1 | tee log/$plan/gen_q-$run.txt
	else
		python code/qg_gen.py $type_w_ini $plan/${run}_$[$run*$qg_update_epoch] $dir/q-$run $start $block $rank  2>&1 | tee log/$plan/gen_q-$run.txt
	fi
	
	echo -e " \n\n ----- Generating Answers for (plan, run) = ($plan, $run) ----- \n\n"
	python code/expert_gen.py $dir/q-$run $dir/qa-$run $dir/update-$run $start $block $rank  2>&1 | tee log/$plan/gen_a-$run.txt

	if (( run < update )); then
		echo -e " \n\n ----- Update QG for (plan, run) = ($plan, $run) ----- \n\n"
		if (( run == 0 )); then
			python code/qg_train.py $dir/update-$run $qg_update_epoch $qg_w_ini $plan/1_$qg_update_epoch  2>&1 | tee log/$plan/update_q-$run.txt
		else
			python code/qg_train.py $dir/update-$run $qg_update_epoch $plan/${run}_$[$run*$qg_update_epoch] $plan/$[$run+1]_$[($run+1)*$qg_update_epoch]  2>&1 | tee log/$plan/update_q-$run.txt
		fi
	fi
	echo -e "\n---------------------------------------------------------------------------\n"
done


echo -e " \n\n ----- Generate word table for (plan) = ($plan) ----- \n\n"
rm -f $dir/qa
for i in $(seq 0 $run)
do
	cat $dir/qa-$i >> $dir/qa
done
python code/gen_word.py $dir/qa model/final/$plan/ans.txt

echo -e "\n---------------------------------------------------------------------------\n"

echo -e " \n\n ----- Train final QA for (plan) = ($plan) ----- \n\n"
python code/final_train_spe.py $dir/qa $data_num  $plan   2>&1 | tee log/$plan/train_final.txt


