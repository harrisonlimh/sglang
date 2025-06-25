# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import multiprocessing as mp
import os
import random
import unittest
from typing import List

from utils import (
    ALL_OTHER_MULTI_LORA_MODELS,
    CI_MULTI_LORA_MODELS,
    TORCH_DTYPES,
    LoRAModelCase,
)

from sglang.test.runners import HFRunner, SRTRunner
from sglang.test.test_utils import CustomTestCase, calculate_rouge_l, is_in_ci

TEST_MULTIPLE_BATCH_PROMPTS = [
    "AI is a field of computer science focused on",
]


class TestLoRA(CustomTestCase):
    def _run_lora_multiple_batch_on_model_cases(self, model_cases: List[LoRAModelCase]):
        for model_case in model_cases:
            for torch_dtype in TORCH_DTYPES:
                max_new_tokens = 32
                backend = "triton"
                base_path = model_case.base
                lora_adapter_paths = [a.name for a in model_case.adaptors]
                assert len(lora_adapter_paths) >= 2

                batches = [
                    # (
                    #     [
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #     ],
                    #     [
                    #         None,
                    #         lora_adapter_paths[0],
                    #         lora_adapter_paths[1],
                    #     ],
                    # ),
                    # (
                    #     [
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #     ],
                    #     [
                    #         lora_adapter_paths[0],
                    #         None,
                    #         lora_adapter_paths[1],
                    #     ],
                    # ),
                    # (
                    #     [
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #     ],
                    #     [lora_adapter_paths[0], lora_adapter_paths[1], None],
                    # ),
                    # (
                    #     [
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #         random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                    #     ],
                    #     [None, lora_adapter_paths[1], None],
                    # ),
                    (
                        [
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                            random.choice(TEST_MULTIPLE_BATCH_PROMPTS),
                        ],
                        [None, None, None],
                    ),
                ]

                print(
                    f"\n========== Testing multiple batches on base '{base_path}' with backend={backend}, dtype={torch_dtype} ---"
                )

                # Initialize runners
                srt_runner = SRTRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    lora_paths=[lora_adapter_paths[0], lora_adapter_paths[1]],
                    max_loras_per_batch=len(lora_adapter_paths) + 1,
                    lora_backend=backend,
                    disable_radix_cache=True,
                    disable_chunked_prefix_cache=True,
                    sleep_on_idle=True,  # Eliminate non-determinism by forcing all requests to be processed in one batch.
                )
                hf_runner = HFRunner(
                    base_path,
                    torch_dtype=torch_dtype,
                    model_type="generation",
                    batch_generation=True,
                    output_str_only=True,
                )

                with srt_runner, hf_runner:
                    for i, (prompts, lora_paths) in enumerate(batches):
                        print(
                            f"\n--- Running Batch {i + 1} --- prompts: {prompts}, lora_paths: {lora_paths}"
                        )

                        srt_outputs = srt_runner.batch_forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,
                        )

                        hf_outputs = hf_runner.forward(
                            prompts,
                            max_new_tokens=max_new_tokens,
                            lora_paths=lora_paths,

                        )

                        print("SRT outputs:", [s for s in srt_outputs.output_strs])
                        print("HF outputs:", [s for s in hf_outputs.output_strs])

                        for srt_out, hf_out in zip(
                            srt_outputs.output_strs, hf_outputs.output_strs
                        ):
                            srt_str = srt_out.strip()
                            hf_str = hf_out.strip()
                            rouge_tol = model_case.rouge_l_tolerance
                            rouge_score = calculate_rouge_l([srt_str], [hf_str])[0]
                            if rouge_score < rouge_tol:
                                raise AssertionError(
                                    f"ROUGE-L score {rouge_score} below tolerance {rouge_tol} "
                                    f"for base '{base_path}', adaptor '{lora_paths}', backend '{backend}', prompt: '{prompts}...'"
                                )

                        print(f"--- Batch {i + 1} Comparison Passed --- ")

    def test_ci_lora_models(self):
        mp.set_start_method("spawn")
        self._run_lora_multiple_batch_on_model_cases(CI_MULTI_LORA_MODELS)

    def test_all_lora_models(self):
        if is_in_ci():
            return

        filtered_models = []
        for model_case in ALL_OTHER_MULTI_LORA_MODELS:
            if "ONLY_RUN" in os.environ and os.environ["ONLY_RUN"] != model_case.base:
                continue
            filtered_models.append(model_case)

        self._run_lora_multiple_batch_on_model_cases(filtered_models)


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    unittest.main(warnings="ignore")
