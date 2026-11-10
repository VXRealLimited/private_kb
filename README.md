Our solution features a Client-Side Vector Obfuscation Layer (ASPE). Before any knowledge data reaches the cloud host, raw text is encrypted using standard AES-256, and vector embeddings undergo a private matrix transformation locally on your network. The cloud host executes similarity matches over obfuscated vector space without ever holding the decryption key or seeing original vector structures. This enables full cloud scalability across any provider while preventing direct data disclosure.

# ASPE Engine (`aspe-engine`)

A lightweight Python library for **Asymmetric Scalar Product Preserving Encryption (ASPE)** combined with **AES-256-GCM** payload encryption.

This package allows client applications to obfuscate high-dimensional vector embeddings before sending them to any third-party cloud provider. The cloud provider can calculate direct inner products (Cosine Similarity over unit vectors) without ever holding the decryption key or viewing raw embedding structures.

---

## Features

- **Universal Cloud Portability:** Works over standard floating-point vector indexes (`pgvector`, FAISS, Qdrant, Milvus) across any cloud provider.
- **Client-Managed Security Boundary:** Key generation, vector transformation, and payload encryption happen exclusively on your client environment.
- **Cosine Similarity Optimized:** Automatic unit vector normalization ensures that cloud inner-product calculations accurately compute Cosine Similarity.
- **Key Serialization:** Built-in methods to save and reload transformation matrices ($M, M^{-1}$) and AES payload keys locally.

---

## Installation

```bash
pip install -r requirements.txt
```
