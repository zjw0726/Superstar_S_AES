from flask import Flask, render_template,request, jsonify
from S_AES import Encryption

app = Flask(__name__)

encryption = Encryption()

@app.route('/')
def index():
    return render_template( 'index.html')

@app.route('/encrypt', methods=['POST'])
def encrypt():
    data = request.json
    output = encryption.encryptionAPI(data['input'], data['key'])
    return jsonify({'output': output})

@app.route('/decrypt', methods=['POST'])
def decrypt():
    data = request.json
    output = encryption.decryptionAPI(data['input'], data['key'])
    return jsonify({'output': output})

@app.route('/double_encrypt', methods=['POST'])
def double_encrypt():
    data = request.json
    output = encryption.doubleEncryptionAPI(data['input'], data['key1'], data['key2'])
    return jsonify({'output': output})

@app.route('/double_decrypt', methods=['POST'])
def double_decrypt():
    data = request.json
    output = encryption.doubleDecryptionAPI(data['input'], data['key1'], data['key2'])
    return jsonify({'output': output})

@app.route('/third_encrypt', methods=['POST'])
def third_encrypt():
    data = request.json
    output = encryption.thirdEncryptionAPI(data['input'], data['key1'], data['key2'])
    return jsonify({'output': output})

@app.route('/third_decrypt', methods=['POST'])
def third_decrypt():
    data = request.json
    output = encryption.thirdDecryptionAPI(data['input'], data['key1'], data['key2'])
    return jsonify({'output': output})

@app.route('/cbc_encrypt', methods=['POST'])
def cbc_encrypt_route():
    data = request.get_json()
    output = encryption.CBC_encrypt(data['plaintext'], data['key'], data['iv'])
    return jsonify({'output': output})

@app.route('/cbc_decrypt', methods=['POST'])
def cbc_decrypt_route():
    data = request.get_json()
    output =encryption.CBC_decrypt(data['ciphertext'], data['key'], data['iv'])
    return jsonify({'output': output})

if __name__ == '__main__':
    app.run(debug=True, port=5555)
