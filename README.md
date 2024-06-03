# QQQ: Quality Quatuor bit Quantization for Large Language Models
The paper will published on arXiv soon.

QQQ is an innovative and hardware-optimized W4A8 quantization solution. QQQ incorporates adaptive smoothing and Hessian-based compensation, significantly boosting the model's performance without the need for extensive training.
Moreover, we meticulously crafted W4A8 GEMM kernels to expedite the inference speed. Our comprehensive experiments demonstrate that QQQ not only matches the performance of the leading W4A8, W8A8, and W4A16 quantization methods but also significantly accelerates inference—achieving up to **2.24x**, **2.10x**, and **1.25x** speed boosting compared to FP16, W8A8, and W4A16, respectively.
Additionally, our specialized per-channel W4A8 GEMM and per-group W4A8 GEMM kernels have attained remarkable speedups of **3.67x** and **3.29x** over FP16 GEMM.

## Install
### Prerequisites
- Your GPU(s) must be of Compute Capability 8.0 or higher. Amphere and later architectures are supported.
- Your CUDA version must be CUDA 11.4 or later.
- Python 3.9+
- Transformers 4.36.2
- lm_eval 0.4.2
### Build from source
Currently this repo only support build form source. We will release package soon.

```
git clone https://github.com/HandH1998/QQQ.git
cd QQQ
pip install -v -e .
```

## Supported models
Model support list:

| Models   | Sizes                       |
| ---------| ----------------------------|
| LLaMA-1  | 7B/13B/30B/65B              |
| LLaMA-2  | 7B/13B/70B                  |
| LLaMA-3  | 8B/70B                      |

## Usage
### Quantize model
Here is an example for quantizing a LLaMA model with per-channel weight quantization.
```
python3 examples/quant_model.py \
--model_path ${model_path} \
--tokenizer_path ${tokenizer_path} \
--batch_size 8 \
--dtype float16 \
--quant_config quant_config/llama/w4a8.yaml \ # uses quant_config/llama/w4a8-pergroup.yaml for per-group weigth quantization
--save_path ${save_path}
```
### Evaluate Model
Here is an example for evaluating perplexity on WikiText2 and accuracy on some zero-shot tasks.
```
python3 examples/eval_model.py \
--model_path ${quantized_model_path} \
--tokenizer_path ${tokenizer_path} \
--tasks piqa,winogrande,hellaswag,arc_challenge,arc_easy \ # lm_eval tasks
--eval_ppl \ # whether evaluate perplexity on WikiText2
--batch_size 8 \
--max_length 2048 
```
### inference
- inference with vLLM 

  We recommand to infer with vllm for a faster speed. Refer to this [PR](https://github.com/vllm-project/vllm/pull/5218). Here is an offline inference example.
  ```
  from vllm import LLM, SamplingParams

  # Sample prompts.
  prompts = [
      "Hello, my name is",
      "The president of the United States is",
      "The capital of France is a",
      "A pig",
  ]
  # Create a sampling params object.
  sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
  model = your_quantized_model_path
  tokenizer = your_tokenizer_path

  # Create an LLM.
  llm = LLM(
      model=model,
      tokenizer=tokenizer,
      quantization="qqq",
  )
  # Generate texts from the prompts. The output is a list of RequestOutput objects
  # that contain the prompt, generated text, and other information.
  outputs = llm.generate(prompts, sampling_params)
  # Print the outputs.
  for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

  ```
  

- inference in this repo
  ```
  python3 examples/test_model.py \
  --model_path ${quantized_model_path} \
  --tokenizer_path ${tokenizer_path} \
  --prompt "Are you a pig?" \
  --max_new_tokens 128
  ```

## Key results
### Model performance
We evaluated the model performance on WikiText2 and five zero-shot tasks.
![model_performance](assets/figures/model_performance.png)
### Throughput
We conducted the same-batch throughput comparison of quantized LLaMA-2 models under various batch sizes. The input sequence length is 1024 and the output sequence length is 128.
![speedup](assets/figures/speedup.png)
### W4A8 GEMM performance
Here is the speedup over PyTorch FP16 GEMM (Calling CUTLASS) of all GEMMs under different numbers of input tokens. The weight matrix size is (N=8192, K=21760).
![gemm_performance](assets/figures/gemm_performance.png)

## Acknowledgement
- Special thanks the **GPTQ Team** for proposing **GPTQ** algorithm and open source the [code](https://github.com/IST-DASLab/gptq), and for releasing [Marlin kernel](https://github.com/IST-DASLab/marlin) which our W4A8 GEMM refers to.
- Special thanks the **Outlier Suppression Plus Team** for proposing **Outlier Suppression Plus** algorithm and open source the [code](https://github.com/ModelTC/Outlier_Suppression_Plus/tree/main).

## Reference
If you find QQQ useful or relevant to your research, please cite our paper:

```bibtex
It will come soon!

```