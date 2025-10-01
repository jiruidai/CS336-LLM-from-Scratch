from dataclasses import dataclass
from collections import defaultdict
import regex


# -----------------------------------------------------------------------------
# Tokenizer params and base class
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class BPETokenizerParams:
    """All you need to specify a BPETokenizer."""
    vocab: dict[int, bytes]     # index -> bytes
    merges: dict[tuple[int, int], int]  # index1,index2 -> new_index


class Tokenizer:
    pass


# -----------------------------------------------------------------------------
# Character and Byte tokenizers
# -----------------------------------------------------------------------------

class CharacterTokenizer(Tokenizer):
    """Represent a string as a sequence of Unicode code points."""
    def encode(self, string: str) -> list[int]:
        return list(map(ord, string))

    def decode(self, indices: list[int]) -> str:
        return "".join(map(chr, indices))


class ByteTokenizer(Tokenizer):
    """Represent a string as a sequence of bytes."""
    def encode(self, string: str) -> list[int]:
        string_bytes = string.encode("utf-8")  # @inspect string_bytes
        indices = list(map(int, string_bytes))  # @inspect indices
        return indices

    def decode(self, indices: list[int]) -> str:
        string_bytes = bytes(indices)  # @inspect string_bytes
        string = string_bytes.decode("utf-8")  # @inspect string
        return string


# -----------------------------------------------------------------------------
# BPE utilities
# -----------------------------------------------------------------------------

def merge(indices: list[int], pair: tuple[int, int], new_index: int) -> list[int]:  # @inspect indices, @inspect pair, @inspect new_index
    """Return `indices`, but with all instances of `pair` replaced with `new_index`."""
    new_indices = []  # @inspect new_indices
    i = 0  # @inspect i
    while i < len(indices):
        if i + 1 < len(indices) and indices[i] == pair[0] and indices[i + 1] == pair[1]:
            new_indices.append(new_index)
            i += 2
        else:
            new_indices.append(indices[i])
            i += 1
    return new_indices


class BPETokenizer(Tokenizer):
    """BPE tokenizer given a set of merges and a vocabulary."""
    def __init__(self, params: BPETokenizerParams):
        self.params = params

    def encode(self, string: str) -> list[int]:
        indices = list(map(int, string.encode("utf-8")))  # @inspect indices
        # Note: this is a very slow implementation
        for pair, new_index in self.params.merges.items():  # @inspect pair, @inspect new_index
            indices = merge(indices, pair, new_index)
        return indices

    def decode(self, indices: list[int]) -> str:
        bytes_list = list(map(self.params.vocab.get, indices))  # @inspect bytes_list
        string = b"".join(bytes_list).decode("utf-8")  # @inspect string
        return string


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def get_compression_ratio(string: str, indices: list[int]) -> float:
    """Given `string` that has been tokenized into `indices`, ."""
    num_bytes = len(bytes(string, encoding="utf-8"))  # @inspect num_bytes
    num_tokens = len(indices)                       # @inspect num_tokens
    return num_bytes / num_tokens


# -----------------------------------------------------------------------------
# Demonstrations / notes
# -----------------------------------------------------------------------------

def character_tokenizer():
    """
    Character-based tokenization
    A Unicode string is a sequence of Unicode characters.
    Each character can be converted into a code point (integer) via ord.
    assert ord("a") == 97
    assert ord("🌍") == 127757
    It can be converted back via chr.
    assert chr(97) == "a"
    assert chr(127757) == "🌍"
    Now let's build a Tokenizer and make sure it round-trips:
    """
    tokenizer = CharacterTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    # There are approximately 150K Unicode characters.  [Wikipedia]
    vocabulary_size = max(indices) + 1  # This is a lower bound @inspect vocabulary_size
    # Problem 1: this is a very large vocabulary.
    # Problem 2: many characters are quite rare (e.g., 🌍), which is inefficient use of the vocabulary.
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio


def byte_tokenizer():
    """
    Byte-based tokenization
    Unicode strings can be represented as a sequence of bytes, which can be represented by integers between 0 and 255.
    The most common Unicode encoding is  UTF-8
    Some Unicode characters are represented by one byte:
    assert bytes("a", encoding="utf-8") == b"a"
    Others take multiple bytes:
    assert bytes("🌍", encoding="utf-8") == b"\xf0\x9f\x8c\x8d"
    Now let's build a Tokenizer and make sure it round-trips:
    """
    tokenizer = ByteTokenizer()
    string = "Hello, 🌍! 你好!"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string
    # The vocabulary is nice and small: a byte can represent 256 values.
    vocabulary_size = 256  # @inspect vocabulary_size
    # What about the compression rate?
    compression_ratio = get_compression_ratio(string, indices)  # @inspect compression_ratio
    assert compression_ratio == 1
    # The compression ratio is terrible, which means the sequences will be too long.
    # Given that the context length of a Transformer is limited (since attention is quadratic), this is not looking great...


def word_tokenizer():
    """
    Word-based tokenization
    Another approach (closer to what was done classically in NLP) is to split strings into words.
    """
    string = "I'll say supercalifragilisticexpialidocious!"
    segments = regex.findall(r"\w+|.", string)  # @inspect segments
    # This regular expression keeps all alphanumeric characters together (words).
    # Here is a fancier version:
    pattern = GPT2_TOKENIZER_REGEX  # @inspect pattern
    segments = regex.findall(pattern, string)  # @inspect segments
    # To turn this into a Tokenizer, we need to map these segments into integers.
    # Then, we can build a mapping from each segment into an integer.
    # But there are problems:
    # 
    # The number of words is huge (like for Unicode characters).
    # 
    # Many words are rare and the model won't learn much about them.
    # 
    # This doesn't obviously provide a fixed vocabulary size.
    # 
    # New words we haven't seen during training get a special UNK token, which is ugly and can mess up perplexity calculations.
    vocabulary_size = "Number of distinct segments in the training data"
    compression_ratio = get_compression_ratio(string, segments)  # @inspect compression_ratio


def bpe_tokenizer():
    """
    Byte Pair Encoding (BPE)

    Basic idea: train the tokenizer on raw text to automatically determine the vocabulary.
        Intuition: common sequences of characters are represented by a single token, rare sequences are represented by many tokens.
        The GPT-2 paper used word-based tokenization to break up the text into inital segments and run the original BPE algorithm on each segment.
        Sketch: start with each byte as a token, and successively merge the most common pair of adjacent tokens.
    Training the tokenizer
    """
    string = "the cat in the hat"  # @inspect string
    params = train_bpe(string, num_merges=3)
    # Using the tokenizer
    # Now, given a new text, we can encode it.
    tokenizer = BPETokenizer(params)
    string = "the quick brown fox"  # @inspect string
    indices = tokenizer.encode(string)  # @inspect indices
    reconstructed_string = tokenizer.decode(indices)  # @inspect reconstructed_string
    assert string == reconstructed_string


# -----------------------------------------------------------------------------
# BPE training
# -----------------------------------------------------------------------------

def train_bpe(string: str, num_merges: int) -> BPETokenizerParams:  # @inspect string, @inspect num_merges
    """Start with the list of bytes of string."""
    indices = list(map(int, string.encode("utf-8")))  # @inspect indices
    merges: dict[tuple[int, int], int] = {}  # index1, index2 => merged index
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}  # index -> bytes
    
    for i in range(num_merges):

        # Count the number of occurrences of each pair of tokens
        counts = defaultdict(int)
        for index1, index2 in zip(indices, indices[1:]):  # For each adjacent pair
            counts[(index1, index2)] += 1  # @inspect counts
       
        # Find the most common pair.
        pair = max(counts, key=counts.get)  # @inspect pair
        index1, index2 = pair
        
        # Merge that pair.
        new_index = 256 + i  # @inspect new_index
        merges[pair] = new_index  # @inspect merges
        vocab[new_index] = vocab[index1] + vocab[index2]  # @inspect vocab
        indices = merge(indices, pair, new_index)  # @inspect indices
   
    return BPETokenizerParams(vocab=vocab, merges=merges)


if __name__ == "__main__":
    main()
