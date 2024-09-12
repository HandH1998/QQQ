# we recommend to try this method first for its model accuracy and inference speed
# rotation + gptq 
# activation per-channel quant + weight per-channel quant
model_path=your_model_path
tokenizer_path=your_tokenizer_path
save_path=your_save_path
log_name=your_log_name

python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--dtype float16 \
--smooth false \
--rotation true \
--w_quantizer FixedQuantize \
--w_group_size -1 \
--gptq_mse true \
--gptq_groupsize -1 \
--save_path ${save_path} \
&> ${log_name}.log


# rotation + gptq
# activation per-channel quant + weight per-group quant + groupsize 128
model_path=your_model_path
tokenizer_path=your_tokenizer_path
save_path=your_save_path
log_name=your_log_name

python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--dtype float16 \
--smooth false \
--rotation true \
--w_quantizer GroupFixedQuantize \
--w_group_size 128 \
--gptq_mse true \
--gptq_groupsize 128 \
--save_path ${save_path} \
&> ${log_name}.log


# smooth + gptq
# activation per-channel quant + weight per-channel quant
model_path=your_model_path
tokenizer_path=your_tokenizer_path
save_path=your_save_path
log_name=your_log_name

python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--dtype float16 \
--smooth true \
--rotation false \
--w_quantizer FixedQuantize \
--w_group_size -1 \
--gptq_mse false \
--gptq_groupsize -1 \
--save_path ${save_path} \
&> ${log_name}.log


# smooth + gptq
# activation per-channel quant + weight per-group quant + groupsize 128
model_path=your_model_path
tokenizer_path=your_tokenizer_path
save_path=your_save_path
log_name=your_log_name

python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--dtype float16 \
--smooth true \
--rotation false \
--w_quantizer GroupFixedQuantize \
--w_group_size 128 \
--gptq_mse false \
--gptq_groupsize 128 \
--save_path ${save_path} \
&> ${log_name}.log




