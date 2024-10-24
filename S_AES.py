class Encryption:
    def __init__(self):
        self.globalKey = []
        self.hexMap = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        self.hex2IntMap = list(range(16))
        self.sBox = [
            [9, 4, 10, 11],
            [13, 1, 8, 5],
            [6, 2, 0, 3],
            [12, 14, 15, 7]
        ]
        self.sBox_reverse = [
            [10, 5, 9, 11],
            [1, 7, 8, 15],
            [6, 0, 2, 3],
            [12, 4, 13, 14]
        ]
        self.mulitiKernel = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13],
    [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2],
    [0, 4, 8, 12, 3, 7, 11, 15, 6, 2, 14, 10, 5, 1, 13, 9],
    [0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1, 9, 12, 3, 6],
    [0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4],
    [0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11],
    [0, 8, 3, 11, 6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1],
    [0, 9, 1, 8, 2, 11, 3, 10, 4, 13, 5, 12, 6, 15, 7, 14],
    [0, 10, 7, 13, 14, 4, 9, 3, 15, 5, 8, 2, 1, 11, 6, 12],
    [0, 11, 5, 14, 10, 1, 15, 4, 7, 12, 2, 9, 13, 6, 8, 3],
    [0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1, 13, 15, 3, 4, 8],
    [0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3, 14, 10, 7],
    [0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10, 11, 5],
    [0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10]
]

    def swap(self, a, b):
        return b, a

    def xorStrings(self,str1, str2):
        result = ''
        for i in range(min(len(str1), len(str2))):
            result += chr(ord(str1[i]) ^ ord(str2[i]))  # 异或操作
        return result

    def display(self,input):
        res = ""
        for i in range(2):
            left = input[i] >> 4
            right = input[i] & 0xF
            res += self.hexMap[left] + self.hexMap[right]
        return res

    def A_k(self, plianText, key):
        output = [0, 0]
        for i in range(2):
            output[i] = plianText[i] ^ key[i]
        return output

    def NS(self,input):
        output = [0, 0]
        for i in range(len(input)):
            left = input[i] >> 4
            right = input[i] & 0xF
            left_i = left >> 2
            left_j = left & 0x3
            right_i = right >> 2
            right_j = right & 0x3
            sum_value = (self.sBox[left_i][left_j] << 4) + self.sBox[right_i][right_j]
            output[i] = sum_value
        return output

    def NS_reverse(self, input):
        output = [0, 0]
        for i in range(len(input)):
            left = input[i] >> 4
            right = input[i] & 0xF
            left_i = left >> 2
            left_j = left & 0x3
            right_i = right >> 2
            right_j = right & 0x3
            sum_value = (self.sBox_reverse[left_i][left_j] << 4) + self.sBox_reverse[right_i][right_j]
            output[i] = sum_value
        return output

    def SR(self,input):
        output = [0, 0]
        output[0] = input[0]
        left = input[1] >> 4
        right = input[1] & 0xF
        output[1] = (right << 4) + left
        return output

    def MC(self,input):
        output = [0, 0]
        kernel = [[1, 4], [4, 1]]
        grid = [[0, 0], [0, 0]]
        # 将输入的每一个 int 拆分为高 4 位和低 4 位
        for i in range(2):
            left = input[i] >> 4
            right = input[i] % 16
            grid[i][0] = left
            grid[i][1] = right

        temp = [[0, 0], [0, 0]]
        # 进行矩阵运算
        for i in range(2):
            for j in range(2):
                temp[i][j] = (self.mulitiKernel[kernel[i][0]][grid[0][j]] ^self.mulitiKernel[kernel[i][1]][grid[1][j]])
        # 将结果重新组合为输出
        for i in range(2):
            output[i] = (temp[i][0] << 4) + temp[i][1]

        return output
    def MC_reverse(self, input):
        output = [0, 0]
        kernel = [[9, 2], [2, 9]]
        grid = [[0, 0], [0, 0]]
        # 将输入的每一个 int 拆分为高 4 位和低 4 位
        for i in range(2):
            left = input[i] >> 4
            right = input[i] % 16
            grid[i][0] = left
            grid[i][1] = right

        temp = [[0, 0], [0, 0]]
        # 进行矩阵运算
        for i in range(2):
            for j in range(2):
                temp[i][j] = (self.mulitiKernel[kernel[i][0]][grid[0][j]] ^self.mulitiKernel[kernel[i][1]][grid[1][j]])
        # 将结果重新组合为输出
        for i in range(2):
            output[i] = (temp[i][0] << 4) + temp[i][1]

        return output

    def int2Binary(self,input):
        output = []
        if input <= 3:
            output = [0, 0]
            index = 1
        elif input <= 255:
            output = [0] * 8
            index = 7
        else:
            output = [0] * 16
            index = 15

        while index >= 0 and input != 0:
            output[index] = input % 2
            input //= 2
            index -= 1

        return output

    def g(self,word, turn):
        left = word >> 4
        right = word & 0xF
        RCON = 128 if turn == 1 else 48
        # Swap left and right
        left, right = right, left
        left = self.sBox[left >> 2][left & 0x3]
        right = self.sBox[right >> 2][right & 0x3]
        res = ((left << 4) + right) ^ RCON
        output = self.int2Binary(res)
        return output

    def binary2Int(self,input):
        base = 1
        sum_value = 0
        for i in range(len(input) - 1, -1, -1):
            sum_value += base * input[i]
            base *= 2
        return sum_value

    def binary2Str(self,input):
        res = ""
        c1 = chr(input[0])
        c2 = chr(input[1])
        res += c1
        res += c2
        return res

    def subKey(self,originKey):
        output = [originKey.copy()]  # Initialize the output with the original key
        for i in range(1, 3):
            word = self.binary2Int(output[i - 1])  # Convert the previous subkey to an integer
            word_left = word >> 8  # Get the left 8 bits
            word_right = word & 0xFF  # Get the right 8 bits
            wordRightAfterG = self.binary2Int(self.g(word_right, i))  # Apply function g to the right part
            newWordLeft = word_left ^ wordRightAfterG  # XOR with the result of g
            newWordRight = newWordLeft ^ word_right  # XOR to get the new right part
            newWord = (newWordLeft << 8) + newWordRight  # Combine the two parts
            output.append(self.int2Binary(newWord))  # Convert back to binary and append to output
        return output

    def char2Binary(self,input):
        asciiCode = ord(input)  # Get the ASCII code of the character
        output = [0] * 8  # Initialize binary output with 8 bits
        index = 7
        while index >= 0 and asciiCode != 0:
            output[index] = asciiCode % 2  # Set the last bit
            asciiCode //= 2  # Right shift the ASCII code
            index -= 1
        return output

    def strToBinary(self,plainText):
        length = len(plainText)
        output = []
        for i in range(length):
            output.append(self.char2Binary(plainText[i]))
        return output

    def getKey(self,input):
        temp = self.strToBinary(input)
        output = [0] * 16  # Initialize with 16 bits of 0
        m = len(temp)
        n = len(temp[0]) if m > 0 else 0
        index = 0
        for i in range(m):
            for j in range(n):
                output[index] = temp[i][j]
                index += 1
                if index == 16:
                    break
            if index == 16:
                break
        return output

    def encryptionAPI(self, s, key):
        res = ""
        self.globalKey = self.subKey(self.getKey(key))
        plainText=self.strToBinary(s)
        len_ = len(plainText)
        if len_ % 2 != 0:
            plainText.append(self.char2Binary('$'))
            len_ += 1
        for i in range(0, len_, 2):
            temp = [self.binary2Int(plainText[i]), self.binary2Int(plainText[i + 1])]
            # print(temp)
            temp = self.A_k(temp, self.globalKey[0])
            # print(temp)
            temp = self.NS(temp)
            # print(temp)
            temp = self.SR(temp)
            temp = self.MC(temp)
            temp = self.A_k(temp, self.globalKey[1])
            temp = self.NS(temp)
            temp = self.SR(temp)
            temp = self.A_k(temp, self.globalKey[2])
            res += self.display(temp)
        return res

    def hexTobinary(self,s):
        hexStr = s.strip()  # 去除首尾空白字符
        for char in s:
            if not (char.isdigit() or 'A' <= char <= 'F' or 'a' <= char <= 'f'):
                raise ValueError(f"Invalid hex character: {char}")
        length = len(s)
        output = []  # Initialize the output list
        count = 0

        for i in range(0, length, 2):  # Iterate in steps of 2
            hexStr = s[i:i + 2]  # Get 2 characters (1 byte)
            base1 = self.hex2IntMap[ord(hexStr[0]) - ord('A') + 10] if hexStr[0] >= 'A' else self.hex2IntMap[
                ord(hexStr[0]) - ord('0')]
            base2 = self.hex2IntMap[ord(hexStr[1]) - ord('A') + 10] if hexStr[1] >= 'A' else self.hex2IntMap[
                ord(hexStr[1]) - ord('0')]
            num = base1 * 16 + base2  # Convert hex to decimal
            output.append(self.int2Binary(num))  # Convert decimal to binary and add to output
            count += 1

        return output

    def decryptionAPI(self,s, key):
        res = ""
        globalKey = self.subKey(self.getKey(key))  # Generate the global key using the provided key
        plainText = self.hexTobinary(s)  # Convert hex input to binary representation
        length = len(plainText)

        for i in range(0, length, 2):  # Process pairs of binary inputs
            temp = [self.binary2Int(plainText[i]), self.binary2Int(plainText[i + 1])]  # Get two bytes at a time
            temp = self.A_k(temp, globalKey[2])  # Apply the A_k transformation with the third round key
            temp = self.SR(temp)  # Apply the Shift Rows transformation
            temp = self.NS_reverse(temp)  # Apply the inverse of Namespace transformation
            temp = self.A_k(temp, globalKey[1])  # Apply the A_k transformation with the second round key
            temp = self.MC_reverse(temp)  # Apply the inverse of Mix Columns transformation
            temp = self.SR(temp)  # Apply the Shift Rows transformation again
            temp = self.NS_reverse(temp)  # Apply the inverse Namespace transformation again
            temp = self.A_k(temp, globalKey[0])  # Apply the A_k transformation with the first round key
            res += self.binary2Str(temp)  # Convert the result back to a string and append to result

        return res

    def doubleEncryptionAPI(self, plaintext, key1, key2):
        midText = self.encryptionAPI(plaintext, key1)
        res = self.encryptionAPI(midText, key2)
        return res

    def doubleDecryptionAPI(self, text, key1, key2):
        midText = self.decryptionAPI(text, key1)
        res = self.decryptionAPI(midText, key2)
        return res

    def thirdEncryptionAPI(self, plaintext, key1, key2):
        midText = self.encryptionAPI(plaintext, key1)
        midText = self.encryptionAPI(midText, key2)
        res = self.encryptionAPI(midText, key1)
        return res

    def thirdDecryptionAPI(self, text, key1, key2):
        midText = self.decryptionAPI(text, key1)
        midText = self.decryptionAPI(midText, key2)
        res = self.decryptionAPI(midText, key1)
        return res

    def hexToStr(self,s):
        length = len(s)
        res = ""
        for i in range(0, length, 2):
            hexStr = s[i:i + 2]  # Get 2 characters (1 byte)
            base1 = self.hex2IntMap[ord(hexStr[0]) - ord('A') + 10] if hexStr[0] >= 'A' else self.hex2IntMap[
                ord(hexStr[0]) - ord('0')]
            base2 =self.hex2IntMap[ord(hexStr[1]) - ord('A') + 10] if hexStr[1] >= 'A' else self.hex2IntMap[
                ord(hexStr[1]) - ord('0')]
            num = base1 * 16 + base2  # Convert hex to decimal
            c = chr(num)  # Convert decimal to character
            res += c  # Append character to result string
        return res

    def str2Int(self,s):
        c1 = ord(s[0])  # Convert the first character to its ASCII/Unicode value
        c2 = ord(s[1])  # Convert the second character to its ASCII/Unicode value
        num = c1 * 256 + c2  # Combine the two values into a single integer
        return num

    def CBC_encrypt(self,plaintext, key, iv):
        ciphertext = ""
        previous_block = iv  # Initialize IV
        len_=len(plaintext)
        if len_ % 2 != 0:
            plaintext+=('$')
            len_ += 1
        for i in range(0, len(plaintext), 2):
            block = plaintext[i:i + 2]  # Every 2 characters as a block
            xored_block = self.xorStrings(block, previous_block)  # XOR with previous block
            encrypted_block = self.encryptionAPI(xored_block, key)  # Encrypt
            ciphertext += encrypted_block  # Append to ciphertext
            previous_block = encrypted_block  # Update previous block

        return ciphertext

    def CBC_decrypt(self,ciphertext, key, iv):
        plaintext = ""
        previous_block = iv  # Initialize IV

        for i in range(0, len(ciphertext), 4):
            block = ciphertext[
                    i:i + 4]  # Every 4 characters as a block (since encryptionAPI outputs 4 characters per input block)
            decrypted_block = self.decryptionAPI(block, key)  # Decrypt
            xored_block = self.xorStrings(decrypted_block, previous_block)  # XOR with previous block
            plaintext += xored_block  # Append to plaintext
            previous_block = block  # Update previous block

        return plaintext
    """
    def test_third_decryption():
    # 创建 Encryption 类的实例
    encryption = Encryption()

    # 定义测试用例
    ciphertext = "1234"
    key1 = "abcd"
    key2 = "zjw1"

    # 执行三重解密
    decrypted_text = encryption.thirdDecryptionAPI(ciphertext, key1, key2)

    # 打印结果
    print(f"Decrypted text: {decrypted_text}")

# 运行测试用例
test_third_decryption()
    """


