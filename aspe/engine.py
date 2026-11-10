import os
import json
import base64
import numpy as np
from typing import Union, Tuple, List
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class ASPEKey:
    """Container for ASPE transformation matrices, splitting vector, and AES key."""

    def __init__(self, M: np.ndarray, M_inv: np.ndarray, S: np.ndarray, aes_key: bytes):
        self.M = M                # (2d, 2d) Secret Transformation Matrix
        self.M_inv = M_inv        # (2d, 2d) Inverse Transformation Matrix
        self.S = S                # (2d,) Binary Splitting Vector (0 or 1)
        self.aes_key = aes_key    # 256-bit AES-GCM Key for text encryption

    def save_keys(self, filepath: str) -> None:
        """Saves key configuration to a secure local JSON file."""
        data = {
            "M": base64.b64encode(self.M.tobytes()).decode('utf-8'),
            "M_inv": base64.b64encode(self.M_inv.tobytes()).decode('utf-8'),
            "M_shape": list(self.M.shape),
            "M_dtype": str(self.M.dtype),
            "S": base64.b64encode(self.S.tobytes()).decode('utf-8'),
            "S_shape": list(self.S.shape),
            "S_dtype": str(self.S.dtype),
            "aes_key": base64.b64encode(self.aes_key).decode('utf-8')
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_keys(cls, filepath: str) -> "ASPEKey":
        """Loads key configuration from a local JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Key file not found at: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        M_bytes = base64.b64decode(data["M"])
        M = np.frombuffer(M_bytes, dtype=np.dtype(data["M_dtype"])).reshape(data["M_shape"])

        M_inv_bytes = base64.b64decode(data["M_inv"])
        M_inv = np.frombuffer(M_inv_bytes, dtype=np.dtype(data["M_dtype"])).reshape(data["M_shape"])

        S_bytes = base64.b64decode(data["S"])
        S = np.frombuffer(S_bytes, dtype=np.dtype(data["S_dtype"])).reshape(data["S_shape"])

        aes_key = base64.b64decode(data["aes_key"])

        return cls(M=M, M_inv=M_inv, S=S, aes_key=aes_key)


class ASPEEngine:
    """ASPE Obfuscation Engine optimized for Cosine Similarity (Unit Vectors)."""

    def __init__(self, dimension: int):
        """
        :param dimension: Length of original input embedding vectors (e.g., 1536)
        """
        self.dim = dimension
        self.ext_dim = dimension * 2

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Helper to convert input vector to a unit vector."""
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("Cannot normalize a zero vector.")
        return vector / norm

    def generate_keys(self) -> ASPEKey:
        """Generates secret matrices (M, M^-1), splitting vector (S), and AES key."""
        while True:
            M = np.random.randn(self.ext_dim, self.ext_dim)
            det = np.linalg.det(M)
            if abs(det) > 1e-2:
                break

        M_inv = np.linalg.inv(M)
        S = np.random.randint(0, 2, size=self.ext_dim)
        aes_key = AESGCM.generate_key(bit_length=256)

        return ASPEKey(M=M, M_inv=M_inv, S=S, aes_key=aes_key)

    def encrypt_record(
        self, key: ASPEKey, vector: Union[List[float], np.ndarray], text: str
    ) -> Tuple[List[float], str]:
        """
        Normalizes document vector to unit length, obfuscates it, and encrypts text payload via AES-256-GCM.
        
        :param key: Active ASPEKey instance
        :param vector: Original embedding vector (will be auto-normalized to unit vector)
        :param text: Text content to encrypt
        :return: (obfuscated_vector_2d, base64_encrypted_text)
        """
        v = np.array(vector, dtype=np.float64)
        if len(v) != self.dim:
            raise ValueError(f"Expected vector dimension {self.dim}, got {len(v)}")

        # 1. Normalize to unit vector for cosine similarity
        v = self._normalize(v)

        # 2. ASPE Dimension Splitting
        v_ext = np.zeros(self.ext_dim, dtype=np.float64)
        for i in range(self.dim):
            if key.S[i] == 1:
                rand_val = np.random.uniform(-0.01, 0.01)
                v_ext[i] = v[i] - rand_val
                v_ext[self.dim + i] = rand_val
            else:
                v_ext[i] = v[i]
                v_ext[self.dim + i] = v[i]

        # Matrix transformation: p' = M^T * v_ext
        protected_vector = np.dot(key.M.T, v_ext)

        # 3. AES-256-GCM Payload Encryption
        aesgcm = AESGCM(key.aes_key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, text.encode('utf-8'), None)

        payload = {
            "nonce": base64.b64encode(nonce).decode('utf-8'),
            "ciphertext": base64.b64encode(ciphertext).decode('utf-8')
        }
        protected_msg = base64.b64encode(json.dumps(payload).encode('utf-8')).decode('utf-8')

        return protected_vector.tolist(), protected_msg

    def transform_query(
        self, key: ASPEKey, vector: Union[List[float], np.ndarray]
    ) -> List[float]:
        """
        Normalizes query vector to unit length and transforms it via M^-1.
        
        :param key: Active ASPEKey instance
        :param vector: Search query vector (will be auto-normalized to unit vector)
        :return: obfuscated_query_vector_2d
        """
        q = np.array(vector, dtype=np.float64)
        if len(q) != self.dim:
            raise ValueError(f"Expected vector dimension {self.dim}, got {len(q)}")

        # 1. Normalize to unit vector for cosine similarity
        q = self._normalize(q)

        # 2. ASPE Dimension Splitting
        q_ext = np.zeros(self.ext_dim, dtype=np.float64)
        for i in range(self.dim):
            if key.S[i] == 0:
                rand_val = np.random.uniform(-0.01, 0.01)
                q_ext[i] = q[i] - rand_val
                q_ext[self.dim + i] = rand_val
            else:
                q_ext[i] = q[i]
                q_ext[self.dim + i] = q[i]

        # Matrix transformation: q' = M^-1 * q_ext
        protected_query = np.dot(key.M_inv, q_ext)

        return protected_query.tolist()

    def decrypt_payload(self, key: ASPEKey, encrypted_msg: str) -> str:
        """Decrypts base64 AES-GCM ciphertext payload back to plaintext string."""
        try:
            raw_json = base64.b64decode(encrypted_msg.encode('utf-8')).decode('utf-8')
            payload = json.loads(raw_json)

            nonce = base64.b64decode(payload["nonce"])
            ciphertext = base64.b64decode(payload["ciphertext"])

            aesgcm = AESGCM(key.aes_key)
            decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, None)
            return decrypted_bytes.decode('utf-8')
        except Exception as e:
            raise ValueError("Decryption failed. Invalid key or corrupted payload.") from e