model_path=your_quantized_model_path
tokenizer_path=your_tokenizer_path
log_name=your_log_name

python3 examples/eval_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--eval_ppl \
--tasks piqa,winogrande,hellaswag,arc_challenge,arc_easy \
--batch_size 8 \
--max_length 2048 \
&> ${log_name}.log