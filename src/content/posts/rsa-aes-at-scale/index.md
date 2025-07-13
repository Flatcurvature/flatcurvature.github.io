---
title: "Cryptography for Data Security: How to Secure Your Big Data with RSA and AES"
published: 2025-01-04
image: "./cover.jpeg"
description: >
  This post introduces hybrid cryptography for securing large datasets by combining RSA (for key encryption) and AES (for data encryption). Ideal for anyone looking to secure big data at rest or in transit using industry-standard cryptographic techniques.
tags:
  - Data Security
  - Data
  - Security
  - Cryptography
category: Security
draft: false
---

> Reading time: 30 minutes
> Cover image source: [Source](https://i.pximg.net/img-master/img/2025/06/09/02/00/54/131346494_p0_master1200.jpg)

Storing sensitive data — from customer records to scientific datasets — comes with serious responsibility. You must ensure that the data remains confidential, even in the event of a breach. That’s where cryptography step in.

For example, consider a healthcare analytics company that processes large volumes of patient records, lab results, and diagnostic images. These files often need to be stored securely in the cloud and shared across systems or departments. Using a hybrid cryptographic system, where AES efficiently encrypts the bulk data and RSA secures the encryption keys to ensures that sensitive information remains unreadable to attackers, even if storage systems are compromised.

In this post, we’ll explore how to secure large data sets using a hybrid cryptographic system that combines the strength of RSA and AES.

## Why Not Just Use RSA or AES Alone?

- RSA is excellent for key exchange, but inefficient for encrypting large files.
- AES is fast and efficient for encrypting data, but requires secure key sharing.

So, we combine them:
1. Use **AES** to encrypt the large data.
2. Use **RSA** to encrypt the AES key.

This is how most real-world systems (including HTTPS) works.

## Step-by-Step Encryption Process

### Step 1: Generate AES Key

AES (Advanced Encryption Standard) works with a symmetric key. We'll generate a random 256-bit key.

### Step 2: Encrypt Data with AES

The bulk of the data is encrypted using this AES key.

### Step 3: Encrypt AES Key with RSA

Use the recipient's RSA public key to encrypt the AES key, enabling secure key exchange.

### Step 4: Save Encrypted Data + Encrypted Key

Store both in a secure format, ready for transfer or storage.

## Mathematical Recap

### AES (Symmetric Encryption)

For data block \( M \) and key \( K \), the ciphertext \( C \) is:

\[
C = \text{AES}_K(M)
\]

Decryption:

\[
M = \text{AES}^{-1}_K(C)
\]

### RSA (Asymmetric Key Encryption)

Given a key pair:
- Public key: \( (e, n) \)
- Private key: \( (d, n) \)

To encrypt the AES key \( K \):

\[
K_{\text{enc}} = K^e \mod n
\]

To decrypt:

\[
K = K_{\text{enc}}^d \mod n
\]

## Example Case

### Real-World Use Case: Securing Cloud Storage for Patient Data

Imagine you're working with a healthcare dataset like the [Disease Symptoms and Patient Profile dataset](https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset), which contains sensitive information such as patient age, gender, medical conditions, and reported symptoms. This type of data is not only subject to strict privacy regulations (e.g., HIPAA, GDPR) but is also a prime target for misuse if exposed.

Suppose you need to store this data securely on private cloud, a shared institutional server, or a cloud-based data lake. To protect patient confidentiality even in the event of a storage breach, you can use a hybrid encryption strategy:

* Use AES to encrypt the dataset efficiently before uploading. AES is ideal for encrypting large tabular files or JSON records.
* Use RSA to encrypt the AES key with each authorized user's public key, ensuring that only intended recipients can decrypt the data.
* Manage access securely by distributing private RSA keys only to verified and authorized personnel.

This way, even if the encrypted file is accessed by an unauthorized party, the data remains unreadable and secure, reinforcing both technical and regulatory safeguards for handling medical records.

Enough talking, lets try the secure data transport. You will need to install `pycryptodome` for this.

```bash
pip install pycryptodome
```

### Encrypting Data

To protect sensitive patient records before storage or transfer, we apply hybrid encryption using the AES and RSA algorithms. In this example, we load a real-world healthcare dataset containing patient profiles and symptoms, and encrypt it in a secure, efficient way. AES is used to encrypt the actual dataset due to its speed and suitability for large files, while RSA encrypts the AES key to ensure that only authorized users can decrypt the data. The result is a JSON package containing all encrypted components, which can safely be stored or shared without exposing the original records.

```python
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes
import base64
import json
import pandas as pd

# Load patient dataset from CSV
df = pd.read_csv("/content/Disease_symptom_and_patient_profile_dataset.csv")

# Convert dataset to JSON string (compact)
data_string = df.to_json(orient='records')  # a list of patient dicts

# Save raw JSON to plaintext file (optional)
with open("plaintext_data.txt", "w") as f:
    f.write(data_string)

# Convert to bytes for encryption
data = data_string.encode('utf-8')

# Generate RSA key pair
key = RSA.generate(2048)
private_key = key
public_key = key.publickey()

# Generate AES key
aes_key = get_random_bytes(32)  # AES-256

# Encrypt data with AES
cipher_aes = AES.new(aes_key, AES.MODE_EAX)
ciphertext, tag = cipher_aes.encrypt_and_digest(data)

# Encrypt AES key with RSA
cipher_rsa = PKCS1_OAEP.new(public_key)
enc_aes_key = cipher_rsa.encrypt(aes_key)

# Store encrypted package
secure_package = {
    'enc_key': base64.b64encode(enc_aes_key).decode(),
    'nonce': base64.b64encode(cipher_aes.nonce).decode(),
    'tag': base64.b64encode(tag).decode(),
    'ciphertext': base64.b64encode(ciphertext).decode()
}

# Write encrypted output
with open("encrypted_package.json", "w") as f:
    json.dump(secure_package, f)

# Save private key
with open("private_key.pem", "wb") as f:
    f.write(private_key.export_key())

# Optionally save public key too
with open("public_key.pem", "wb") as f:
    f.write(public_key.export_key())

print("Patient dataset encrypted and saved.")

```

### Decrypting Data

To access the original patient records, an authorized user with the correct RSA private key can decrypt the AES key and subsequently decrypt the dataset. This process make sure that only individuals with the proper credentials can recover the original information. In the following code, we reverse the encryption process by loading the encrypted package, decrypting the AES key using RSA and then decrypting the patient data using AES. Finally, we verify data integrity by comparing SHA-256 hashes of the decrypted output with the original file.

```python
import json
import base64
import hashlib
from Crypto.PublicKey import RSA
from Crypto.Cipher import AES, PKCS1_OAEP

# Load encrypted package
with open("encrypted_package.json", "r") as f:
    secure_package = json.load(f)

# Load private RSA key
with open("private_key.pem", "rb") as f:
    private_key = RSA.import_key(f.read())

# Step 1: Decrypt AES key
cipher_rsa = PKCS1_OAEP.new(private_key)
aes_key = cipher_rsa.decrypt(base64.b64decode(secure_package['enc_key']))

# Step 2: Decrypt the data using AES
cipher_aes = AES.new(aes_key, AES.MODE_EAX, nonce=base64.b64decode(secure_package['nonce']))
plaintext = cipher_aes.decrypt_and_verify(
    base64.b64decode(secure_package['ciphertext']),
    base64.b64decode(secure_package['tag'])
)

# Step 3: Write decrypted output to file
with open("decrypted_data.txt", "wb") as f:
    f.write(plaintext)

# Step 4: Define SHA-256 hash function
def sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

# Step 5: Compare original and decrypted hashes
original_hash = sha256("plaintext_data.txt")
decrypted_hash = sha256("decrypted_data.txt")

print("Original Hash:  ", original_hash)
print("Decrypted Hash: ", decrypted_hash)

if original_hash == decrypted_hash:
    print("Success: The decrypted file matches the original.")
else:
    print("Error: The decrypted file does not match the original.")
```

If the data matched, you will find result like this

```
Original Hash:   6b552ead432c7d3c2dee6cf61283a49f4cdd779da158eac7e7d74b5a83646f7f
Decrypted Hash:  6b552ead432c7d3c2dee6cf61283a49f4cdd779da158eac7e7d74b5a83646f7f
Success: The decrypted file matches the original.
```

We’ve built a secure and scalable encryption system suitable for protecting sensitive healthcare datasets — such as patient profiles, symptoms, and medical conditions. This hybrid approach ensures that even if the storage medium is compromised, the data remains protected through strong encryption and controlled key access.

This workflow is valuable for data-heavy domains like healthcare, finance, and research, where confidentiality and compliance are critical. Whether you’re storing data in the cloud or sharing it across organizational boundaries, hybrid encryption provides a robust, standards-based solution to safeguard sensitive information at rest and in transit.

### Final Thoughts

In this era where data breaches are increasingly common and regulations around data privacy are stricter than ever, building cryptographic safeguards into your data pipeline is no longer optional — well, it's essential. This post demonstrated how hybrid encryption using RSA and AES can be applied to real-world healthcare datasets to enforce confidentiality, integrity, and access control.

By encrypting sensitive records before they leave your laptop and securing encryption keys with asymmetric cryptography, you reduce the risk of exposure even in untrusted environments like public cloud storage or shared infrastructure. Whether you're handling patient profiles, financial reports, or proprietary models, this approach scales well and aligns with best practices for modern data security.

Cryptography doesn't need to be abstract or overly academic — well, don't worry about that — with the right tools and understanding, it's a practical and powerful friend in keeping your data safe.

---
