import base64
import os

def xor_encrypt_decrypt(data, key=0x55):  # Choose a simple key
    """Encrypt or decrypt data using XOR."""
    return bytes([b ^ key for b in data])

def custom_encode(data):
    """Custom encoding that mimics Base32 but with a custom character set."""
    base32_encoded = base64.b32encode(data).decode('utf-8').rstrip('=')
    custom_chars = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567', 'ZXYWVUTSRQPONMLKJIHGFEDCBA576432')
    return base32_encoded.translate(custom_chars)

def custom_decode(data):
    """Reverse the custom encoding."""
    custom_chars = str.maketrans('ZXYWVUTSRQPONMLKJIHGFEDCBA576432', 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567')
    data = data.translate(custom_chars)
    padded_data = data + '=' * (-len(data) % 8)
    return base64.b32decode(padded_data)

def encode_id(youtube_id):
    """Encode a YouTube ID with a salt."""
    salt = os.urandom(5)
    data = salt + youtube_id.encode('utf-8')
    encrypted_bytes = xor_encrypt_decrypt(data)
    encoded_id = custom_encode(encrypted_bytes)
    return encoded_id

def decode_id(encoded_id):
    """Decode a previously encoded YouTube ID, remove the salt."""
    encrypted_bytes = custom_decode(encoded_id)
    decrypted_bytes = xor_encrypt_decrypt(encrypted_bytes)
    return decrypted_bytes[5:].decode('utf-8')  # Skip the salt