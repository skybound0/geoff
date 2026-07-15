import os
import urllib.request

# trie node implementation for word hunt solver
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True

class WordGameSolver:
    def __init__(self, dictionary_path="dictionary.txt"):
        self.dictionary_path = dictionary_path
        self.words = set()
        self.trie = Trie()
        self.load_dictionary()

    def load_dictionary(self):
        # try to download or find a dictionary file on the system
        if not os.path.exists(self.dictionary_path):
            # check common linux dictionary path first
            linux_dict = "/usr/share/dict/words"
            if os.path.exists(linux_dict):
                print(f"copying dictionary from {linux_dict}")
                try:
                    with open(linux_dict, "r") as f:
                        words = f.read().splitlines()
                    # filter words for gameplay (3-8 letters, alphabetic only)
                    valid_words = [w.lower() for w in words if w.isalpha() and 3 <= len(w) <= 8]
                    with open(self.dictionary_path, "w") as f:
                        f.write("\n".join(valid_words))
                except Exception as e:
                    print(f"failed to read system dictionary: {e}")

        # if still not found, download a scrabble word list
        if not os.path.exists(self.dictionary_path):
            print("downloading scrabble word list...")
            url = "https://raw.githubusercontent.com/raun/Scrabble/master/words.txt"
            try:
                urllib.request.urlretrieve(url, self.dictionary_path)
            except Exception as e:
                print(f"failed to download word list: {e}")
                # fallback minimal wordlist
                fallback_words = ["cat", "dog", "hunt", "word", "game", "bird", "nest", "play", "anagram"]
                with open(self.dictionary_path, "w") as f:
                    f.write("\n".join(fallback_words))

        # read the words from dictionary file
        if os.path.exists(self.dictionary_path):
            with open(self.dictionary_path, "r") as f:
                for line in f:
                    word = line.strip().lower()
                    if word.isalpha() and 3 <= len(word) <= 8:
                        self.words.add(word)
                        self.trie.insert(word)
            print(f"loaded {len(self.words)} words into dictionary")

    def solve_word_hunt(self, grid):
        # grid is a 2d list of size 4x4 containing lowercase letters
        rows = 4
        cols = 4
        found_words = {} # maps word to coordinate path

        # dfs over the 4x4 grid
        def dfs(r, c, node, path, visited):
            letter = grid[r][c]
            if letter not in node.children:
                return
            
            next_node = node.children[letter]
            new_path = path + [(r, c)]
            new_visited = visited | {(r, c)}
            current_word = "".join(grid[row][col] for row, col in new_path)

            if next_node.is_word:
                # first path found wins (a word's path length is fixed by the word itself)
                if current_word not in found_words:
                    found_words[current_word] = new_path

            # explore 8 directions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in new_visited:
                        dfs(nr, nc, next_node, new_path, new_visited)

        # start search from every cell in the grid
        for r in range(rows):
            for c in range(cols):
                dfs(r, c, self.trie.root, [], set())

        # longest words first (scores higher)
        sorted_words = sorted(found_words.items(), key=lambda item: len(item[0]), reverse=True)
        return sorted_words

    def solve_anagrams(self, letters):
        # letters is a list or string of lowercase letters
        letters = sorted(l.lower() for l in letters)
        found_words = []

        # check word is a submultiset of letters
        def can_form(word, available_letters):
            word_list = sorted(list(word))
            i = 0
            j = 0
            while i < len(word_list) and j < len(available_letters):
                if word_list[i] == available_letters[j]:
                    i += 1
                j += 1
            return i == len(word_list)

        # filter dictionary for words of length 3 to 6
        for word in self.words:
            if 3 <= len(word) <= 6:
                if can_form(word, letters):
                    found_words.append(word)

        # sort anagrams by length descending
        sorted_words = sorted(found_words, key=len, reverse=True)
        return sorted_words
