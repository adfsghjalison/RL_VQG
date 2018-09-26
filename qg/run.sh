
python train_type.py 10

for i in $(seq 200 50 500)
do
    echo -e "\n------------  $i epoch  ---------------\n"
    python train_type.py 50 $i
done

#python predict_type.py 300 200 > model/out/300_200.test
python predict_type.py 500 600 > model/out/500_600.test

python qg_train.py 15 10


for i in $(seq 100 100 600)
do
    echo -e "\n------------  $i  ---------------\n"
    python qg_test_fre.py $i > model/2_1_lstm/output_test_f.$i
done

