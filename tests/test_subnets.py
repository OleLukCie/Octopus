# tests/test_subnets.py

import unittest
from src.factory import OctopusTranslatorFactory


class TestMedicalSubnets(unittest.TestCase):
    """Test subnets with medical domain configuration."""

    def setUp(self):
        self.translator = OctopusTranslatorFactory.create_from_config(
            "configs/zh2en_medical.yaml"
        )
        self.src_text = "心梗患者需要紧急处理"
        self.expected_expansion = "心肌梗死患者需要紧急处理"  # "心梗" → "心肌梗死"

    def test_domain_subnet_abbreviation_expansion(self):
        domain_subnet = self.translator.subnets[3]  # Assuming 4th subnet is domain
        output, _ = domain_subnet.forward(self.src_text)
        self.assertIn("心肌梗死", output)  # Verify abbreviation expansion

    def test_lexical_subnet_term_translation(self):
        lexical_subnet = self.translator.subnets[0]
        output, _ = lexical_subnet.forward(self.expected_expansion)
        self.assertIn("myocardial infarction", output)
        self.assertIn("emergency treatment", output)


if __name__ == "__main__":
    unittest.main()