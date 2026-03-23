import importlib.util
import pathlib
import unittest
from types import SimpleNamespace

import torch

from exllamav2.cache import ExLlamaV2Cache
from exllamav2.gated_delta_net import GDNRecurrentState
from exllamav2.generator.base import ExLlamaV2BaseGenerator
from exllamav2.generator.dynamic import ExLlamaV2DynamicGenerator
from exllamav2.generator.sampler import ExLlamaV2Sampler
from exllamav2.generator.streaming import ExLlamaV2StreamingGenerator

ARCHITECTURE_PATH = pathlib.Path(__file__).resolve().parents[1] / "exllamav2" / "architecture.py"
ARCHITECTURE_SPEC = importlib.util.spec_from_file_location("exllamav2_architecture_only", ARCHITECTURE_PATH)
ARCHITECTURE_MODULE = importlib.util.module_from_spec(ARCHITECTURE_SPEC)
assert ARCHITECTURE_SPEC.loader is not None
ARCHITECTURE_SPEC.loader.exec_module(ARCHITECTURE_MODULE)
ExLlamaV2ArchParams = ARCHITECTURE_MODULE.ExLlamaV2ArchParams


class DummyModel:

    def __init__(self, recurrent = True):
        self.config = SimpleNamespace(
            max_seq_len = 32,
            max_input_len = 32,
            max_batch_size = 1,
            vocab_size = 128,
            no_flash_attn = True,
            num_key_value_heads = 2,
            num_hidden_layers = 4,
            head_dim = 8,
            layer_types = ["linear_attention", "full_attention", "linear_attention", "full_attention"] if recurrent else None,
        )
        self.cache_map = {}

    def forward(self, *args, **kwargs):
        raise AssertionError("DummyModel.forward should not be reached in these guard tests")


class DummyTokenizer:

    eos_token_id = 0
    pad_token_id = 0

    def encode(self, prompt, encode_special_tokens = False, return_offsets = False, add_bos = False):
        batch = 1 if isinstance(prompt, str) else len(prompt)
        ids = torch.tensor([[1, 2, 3]], dtype = torch.long).repeat(batch, 1)
        if return_offsets:
            return ids, None
        return ids

    def padding_mask(self, ids):
        return None

    def decode(self, ids, decode_special_tokens = False):
        batch = ids.shape[0] if isinstance(ids, torch.Tensor) and ids.ndim == 2 else 1
        return [""] * batch

    def get_id_to_piece_list(self, decode_special_tokens = False):
        return [""] * 256


class Qwen35ConfigTests(unittest.TestCase):

    def _prepare_arch(self, arch_string, config_data):
        return ExLlamaV2ArchParams(arch_string, config_data)

    def test_qwen35_text_architecture_is_registered(self):
        config = self._prepare_arch("Qwen3_5ForCausalLM", {
            "model_type": "qwen3_5_text",
        })
        self.assertEqual(config.lm_prefix, "")
        self.assertEqual(config.keymap, [("$model.language_model.", "model.")])
        self.assertEqual(config.lm.norm_constant_bias, 1)
        self.assertEqual(config.lm.keys["lm_head"], "model.embed_tokens")
        self.assertTrue(config.lm.default_use_qk_norm)

    def test_qwen35_multimodal_architecture_still_detects_text_stack(self):
        config = self._prepare_arch("Qwen3_5MoeForConditionalGeneration", {
            "model_type": "qwen3_5_moe",
            "text_config": {
                "model_type": "qwen3_5_moe_text",
            },
            "vision_config": {
                "model_type": "qwen3_5_moe",
            },
        })
        self.assertEqual(config.lm_prefix, "")
        self.assertEqual(config.keymap, [("$model.language_model.", "model.")])

    def test_other_unknown_architectures_still_fall_back_to_llama(self):
        config = self._prepare_arch("CustomUnknownForCausalLM", {})
        self.assertEqual(config.arch_string, "LlamaForCausalLM")

    def test_recurrent_cache_clones_current_state_but_rejects_rewind(self):
        cache = ExLlamaV2Cache(DummyModel(), batch_size = 1, max_seq_len = 8)
        self.assertEqual(cache.recurrent_layer_indices, [0, 2])

        state = GDNRecurrentState()
        state.position = 3
        state.last_conv_state = torch.randn(1, 4, 4)
        state.last_recurrent_state = torch.randn(1, 2, 8, 8)
        cache.recurrent_states[0] = state
        cache.current_seq_len = 3

        clone = cache.clone()
        self.assertEqual(clone.current_seq_len, 3)
        self.assertEqual(clone.recurrent_states[0].position, 3)
        self.assertIsNot(clone.recurrent_states[0].last_conv_state, state.last_conv_state)
        self.assertTrue(torch.equal(clone.recurrent_states[0].last_conv_state, state.last_conv_state))

        with self.assertRaisesRegex(RuntimeError, "do not support rewinding"):
            cache.current_seq_len = 2

        cache.reset()
        self.assertEqual(cache.current_seq_len, 0)
        self.assertEqual(cache.recurrent_states, {})

    def test_base_generator_rejects_token_healing_for_recurrent_cache(self):
        generator = ExLlamaV2BaseGenerator(DummyModel(), ExLlamaV2Cache(DummyModel()), DummyTokenizer())
        settings = ExLlamaV2Sampler.Settings()

        with self.assertRaisesRegex(ValueError, "token_healing"):
            generator.generate_simple("Hello", settings, 1, token_healing = True)

    def test_streaming_generator_rejects_rewind_features_for_recurrent_cache(self):
        model = DummyModel()
        cache = ExLlamaV2Cache(model)
        tokenizer = DummyTokenizer()
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
        settings = ExLlamaV2Sampler.Settings()
        input_ids = torch.tensor([[1, 2, 3]], dtype = torch.long)

        with self.assertRaisesRegex(ValueError, "token_healing"):
            generator.begin_stream_ex(input_ids, settings, token_healing = True)

        with self.assertRaisesRegex(ValueError, "banned_strings"):
            generator.begin_stream_ex(input_ids, settings, banned_strings = ["bad"])

        generator.speculative_ngram = True
        with self.assertRaisesRegex(ValueError, "speculative_ngram"):
            generator.begin_stream_ex(input_ids, settings)

    def test_dynamic_generator_rejects_recurrent_cache(self):
        model = DummyModel()
        cache = ExLlamaV2Cache(model)
        tokenizer = DummyTokenizer()

        with self.assertRaisesRegex(ValueError, "DynamicGenerator does not support recurrent layers"):
            ExLlamaV2DynamicGenerator(model, cache, tokenizer)


if __name__ == "__main__":
    unittest.main()
