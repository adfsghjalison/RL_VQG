
#generate question
python code/qg_gen.py [load_type] [load_qg] [output-f] [start] [num] [rank]

#generate answer
python code/expert_gen.py [input-q] [output-a] [output-up] [start] [num] [rank]

#update qg
python code/qg_train.py [input] [EPOCH] [load_weights] [write_weights]

#generate word table
python code/gen_word.py [in] [out]


#train final QA
python code/final_train.py [input] [plan] [EPOCH] ([load_weights])

#test final QA
python code/final_test.py [plan] [load_weights]

