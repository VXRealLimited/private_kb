import os
import unittest
import numpy as np
from aspe import ASPEEngine, ASPEKey


class TestASPEEngine(unittest.TestCase):

    def setUp(self):
        """Set up test environment and engine instance before each test."""
        self.dimension = 4
        self.engine = ASPEEngine(dimension=self.dimension)
        self.key = self.engine.generate_keys()
        self.temp_key_file = "test_aspe_key.json"

    def tearDown(self):
        """Clean up temporary test files after each test."""
        if os.path.exists(self.temp_key_file):
            os.remove(self.temp_key_file)

    def test_key_generation(self):
        """Test that generated key matrices have correct dimensions."""
        self.assertEqual(self.key.M.shape, (self.dimension * 2, self.dimension * 2))
        self.assertEqual(self.key.M_inv.shape, (self.dimension * 2, self.dimension * 2))
        self.assertEqual(self.key.S.shape, (self.dimension * 2,))
        self.assertEqual(len(self.key.aes_key), 32)  # 256 bits = 32 bytes

    def test_key_save_and_load(self):
        """Test key serialization to disk and deserialization back."""
        self.key.save_keys(self.temp_key_file)
        self.assertTrue(os.path.exists(self.temp_key_file))

        loaded_key = ASPEKey.load_keys(self.temp_key_file)
        
        np.testing.assert_array_almost_equal(self.key.M, loaded_key.M)
        np.testing.assert_array_almost_equal(self.key.M_inv, loaded_key.M_inv)
        np.testing.assert_array_equal(self.key.S, loaded_key.S)
        self.assertEqual(self.key.aes_key, loaded_key.aes_key)

    def test_vector_normalization(self):
        """Test internal helper normalizes non-unit vectors to length 1."""
        raw_vec = np.array([3.0, 4.0, 0.0, 0.0])
        norm_vec = self.engine._normalize(raw_vec)
        self.assertAlmostEqual(np.linalg.norm(norm_vec), 1.0)

    def test_invalid_dimension_raises_error(self):
        """Test that passing a vector of incorrect dimension raises a ValueError."""
        invalid_vec = [1.0, 2.0]  # Expected dimension is 4
        with self.assertRaises(ValueError):
            self.engine.encrypt_record(self.key, invalid_vec, "sample text")

        with self.assertRaises(ValueError):
            self.engine.transform_query(self.key, invalid_vec)

    def test_payload_encryption_and_decryption(self):
        """Test text roundtrip encryption and local decryption."""
        doc_vector = [0.5, -0.5, 0.5, -0.5]
        original_text = "Top Secret Financial Report Q4"

        _, encrypted_payload = self.engine.encrypt_record(
            self.key, doc_vector, original_text
        )
        decrypted_text = self.engine.decrypt_payload(self.key, encrypted_payload)

        self.assertEqual(original_text, decrypted_text)

    def test_invalid_key_decryption_fails(self):
        """Test that decrypting with a different AES key raises ValueError."""
        doc_vector = [0.5, -0.5, 0.5, -0.5]
        _, encrypted_payload = self.engine.encrypt_record(
            self.key, doc_vector, "Sensitive Data"
        )
        
        wrong_key = self.engine.generate_keys()
        with self.assertRaises(ValueError):
            self.engine.decrypt_payload(wrong_key, encrypted_payload)

    def test_cosine_similarity_dot_product(self):
        """
        Test that dot product of obfuscated vectors reflects cosine similarity 
        of normalized inputs.
        """
        # Identical direction vectors -> Cosine similarity should be ~1.0
        vec_a = [0.5, 0.5, 0.5, 0.5]
        vec_b = [1.0, 1.0, 1.0, 1.0]

        obf_doc_vec, _ = self.engine.encrypt_record(self.key, vec_a, "Doc A")
        obf_query_vec = self.engine.transform_query(self.key, vec_b)

        similarity = np.dot(obf_query_vec, obf_doc_vec)
        self.assertAlmostEqual(similarity, 1.0, places=2)


if __name__ == "__main__":
    unittest.main()