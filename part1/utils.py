
import hashlib
import random
import re

TYPO_PROBABILITY = 0.50
PROTECTED_WORDS = {"never", "neither", "nor", "not", "no", "without"}

def custom_transform(example):
    text = example["text"]

    # 每条文本使用独立且稳定的随机种子，不依赖标签或调用顺序
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))

    def change_word(match):
        word = match.group(0)

        if len(word) < 5 or word.lower() in PROTECTED_WORDS:
            return word

        if rng.random() >= TYPO_PROBABILITY:
            return word

        # 只交换内部相邻且不同的字母，首尾字母不动
        candidates = [
            i for i in range(1, len(word) - 2)
            if word[i].lower() != word[i + 1].lower()
        ]
        if not candidates:
            return word

        i = rng.choice(candidates)
        letters = list(word)
        letters[i], letters[i + 1] = letters[i + 1], letters[i]
        return "".join(letters)

    transformed = re.sub(r"\b[A-Za-z]+\b", change_word, text)
    return {**example, "text": transformed}
