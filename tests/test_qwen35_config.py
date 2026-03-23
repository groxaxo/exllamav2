import importlib.util
import pathlib
import unittest

ARCHITECTURE_PATH = pathlib.Path(__file__).resolve().parents[1] / "exllamav2" / "architecture.py"
ARCHITECTURE_SPEC = importlib.util.spec_from_file_location("exllamav2_architecture_only", ARCHITECTURE_PATH)
ARCHITECTURE_MODULE = importlib.util.module_from_spec(ARCHITECTURE_SPEC)
assert ARCHITECTURE_SPEC.loader is not None
ARCHITECTURE_SPEC.loader.exec_module(ARCHITECTURE_MODULE)
ExLlamaV2ArchParams = ARCHITECTURE_MODULE.ExLlamaV2ArchParams


class Qwen35ConfigTests(unittest.TestCase):

    def _prepare_arch(self, arch_string, config_data):
        return ExLlamaV2ArchParams(arch_string, config_data)

    def test_qwen35_text_architecture_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Qwen 3.5 / Qwen3 Next family"):
            self._prepare_arch("Qwen3_5ForCausalLM", {
                "model_type": "qwen3_5_text",
            })

    def test_qwen35_multimodal_architecture_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Qwen 3.5 / Qwen3 Next family"):
            self._prepare_arch("Qwen3_5MoeForConditionalGeneration", {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                },
                "vision_config": {
                    "model_type": "qwen3_5_moe",
                },
            })

    def test_other_unknown_architectures_still_fall_back_to_llama(self):
        config = self._prepare_arch("CustomUnknownForCausalLM", {})
        self.assertEqual(config.arch_string, "LlamaForCausalLM")


if __name__ == "__main__":
    unittest.main()
