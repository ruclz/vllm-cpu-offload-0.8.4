#!/usr/bin/env python3
import json
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_MLA"
# os.environ["LD_PRELOAD"]= "/home/liuzhuan/moe-cpu-engine/ext/xft/libiomp5.so"
# os.environ["OMP_NUM_THREADS"] = "120"

prompts = [ "Tellme the recipe for tea",
            # "write a story about bird",
            # "What is Black Myth Wukong",
            # "how many Rs in strawberry",
            # "The strawberry is a swollen receptacle, covered with many small achenes, the botanical fruits. In culinary terms, a strawberry is an edible fruit. From a botanical point of view, it is not a berry but an aggregate accessory fruit, because the fleshy part is derived from the receptacle"
]

def main():
    # Set the model ID. Here we use a distilled variant for demonstration.
    # You can change this to "deepseek-ai/DeepSeek-R1" if you have sufficient hardware.
    model_id = "/mnt/nvme2p1/model/deepseek-ai/DeepSeek-R1"

    print(f"Loading model: {model_id}")

    compilation_config = CompilationConfig(
                use_cudagraph=True,
                cudagraph_capture_sizes=[256],
            )
    # Initialize the model offline.
    # Adjust 'tensor_parallel_size' based on your GPU availability (or set to 1 for CPU mode).
    llm = LLM(
        model_id,
        block_size=16,
        tensor_parallel_size=1,
        max_model_len=4096,
        trust_remote_code=True,
        #compilation_config=compilation_config,
        enforce_eager=True
    )

    # Define sampling parameters.
    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=1.0,
        top_k=3,
        max_tokens=4096,
        min_tokens=32
    )

    for prompt in prompts:
        # Generate output offline by passing a list of prompts.
        outputs = llm.generate([prompt], sampling_params)

        # Process and print the output.
        for output in outputs:
            print("\n--- Generated Response ---")
            # The prompt used (should match our input)
            print("Prompt:", output.prompt)
            # The generated text (final answer)
            print("Output:", output.outputs[0].text)
            # If available, also print any reasoning steps
            # reasoning = output.outputs[0].get("reasoning_content", "")
            # if reasoning:
            #     print("\n--- Reasoning Steps ---")
            #     print(reasoning)

if __name__ == "__main__":
    main()
