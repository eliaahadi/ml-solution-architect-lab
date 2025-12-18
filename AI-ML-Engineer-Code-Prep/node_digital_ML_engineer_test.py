"""
=============================================================================
PROBLEM STATEMENT
=============================================================================

Implement a text similarity calculator using TF-IDF (Term Frequency–Inverse
Document Frequency) scoring and cosine similarity. Given a corpus of documents,
calculate how similar any two documents are.

Your system must:
1. Calculate TF (term frequency) for each word in each document
2. Calculate IDF (inverse document frequency) across the corpus
3. Compute TF-IDF vectors for documents
4. Calculate cosine similarity between document vectors

=============================================================================
INTERFACE (Let's implement in here.)
=============================================================================

import math
from collections import Counter, defaultdict

class TextSimilarity:
    def __init__(self):
        pass

    def add_document(self, doc_id: str, text: str) -> None:
        pass

    def similarity(self, doc_id1: str, doc_id2: str) -> float:
        '''
        Calculate cosine similarity between two documents.
        Returns: float between 0.0 and 1.0 (1.0 = identical, 0.0 = no overlap)
        '''
        pass

=============================================================================
EXAMPLE
=============================================================================
"""

import math
from collections import Counter, defaultdict

class TextSimilarity:
    def __init__(self):
        self.docs = {}                  # doc_id -> list[str] tokens
        self.tf = {}                    # doc_id -> dict(term -> tf)
        self.df = defaultdict(int)      # term -> number of docs containing term
        self.N = 0

    def _tokenize(self, text: str):
        return text.lower().split()

    def add_document(self, doc_id: str, text: str) -> None:
        # if re-adding same doc_id, remove old stats first
        if doc_id in self.docs:
            raise ValueError(f"doc_id already exists: {doc_id}")

        # increment tokens
        tokens = self._tokenize(text)
        print('tokens ', tokens)
        self.docs[doc_id] = tokens
        self.N += 1

        counts = Counter(tokens)
        total = len(tokens) if tokens else 1

        # TF (Term Frequency normalized)
        tf_map = {term: cnt / total for term, cnt in counts.items()}
        self.tf[doc_id] = tf_map
        # print("add doc ", self.tf)

        # DF (document frequency) update (unique terms per doc)
        for term in counts.keys():
            self.df[term] += 1
        
    # determine inverse doc frequency
    def _idf(self, term: str) -> float:
        # smooth IDF so it's well-behaved for small docs
        df = self.df.get(term, 0)
        return math.log((self.N + 1) / (df + 1)) + 1.0
    
    def _tfidf(self, doc_id: str) -> dict:
        test_dict = {'doc1': {'i': 0.3333333333333333}}
        # print('self.tf[doc_id] ', self.tf, doc_id)
        print('test dict access ', test_dict['doc1'])
        # print('tfmap ', self.tf, 'doc_id is: ', doc_id)
        tf_map = self.tf[doc_id]
        return {t: tfv * self._idf(t) for t, tfv in tf_map.items()}
    
    def similarity(self, doc_id1: str, doc_id2: str) -> float:
        if doc_id1 not in self.docs or doc_id2 not in self.docs:
            raise KeyError("Unknown doc_id")
        v1 = self._tfidf(doc_id1)
        print('\n similarity function \n')
        v2 = self._tfidf(doc_id2)

        print('v1 tfidf ')
        if not v1 or not v2:
            return 0.0
        
        # dot over intersection
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        dot = sum(w * v2.get(t, 0.0) for t, w in v1.items())

        n1 = math.sqrt(sum(w * w for w in v1.values()))
        n2 = math.sqrt(sum(w * w for w in v2.values()))

        if n1 == 0.0 or n2 == 0.0:
            return 0.0
        
        return dot / (n1 * n2)
    

if __name__ == "__main__":
    ts = TextSimilarity()
    ts.add_document('doc1', "I love tech")
    ts.add_document('doc2', "I love tech and beaches")
    ts.add_document('doc3', "Japan has tech and beaches")

    print("doc1 vs doc2: ", ts.similarity("doc1", "doc2"))
    print("doc1 vs doc3: ", ts.similarity("doc1", "doc3"))


'''
import math
import re
from collections import Counter, defaultdict

_WORD_RE = re.compile(r"[a-zA-Z0-9']+")
print('wordre ', _WORD_RE)

class TextSimilarity:
    def __init__(self):
        self.docs = {}                 # doc_id -> list[str] tokens
        self.tf = {}                   # doc_id -> dict(term -> tf)
        self.df = defaultdict(int)     # term -> number of docs containing term
        self.N = 0

    def _tokenize(self, text: str):
        return [w.lower() for w in _WORD_RE.findall(text)]

    def add_document(self, doc_id: str, text: str) -> None:
        # If re-adding same doc_id, remove old stats first (optional; simplest: disallow)
        if doc_id in self.docs:
            raise ValueError(f"doc_id already exists: {doc_id}")

        tokens = self._tokenize(text)
        print('tokens ', tokens)
        self.docs[doc_id] = tokens
        self.N += 1

        counts = Counter(tokens)
        total = len(tokens) if tokens else 1

        # TF (normalized)
        tf_map = {term: cnt / total for term, cnt in counts.items()}
        self.tf[doc_id] = tf_map

        # DF update (unique terms per doc)
        for term in counts.keys():
            self.df[term] += 1

    def _idf(self, term: str) -> float:
        # Smooth IDF so it's well-behaved for small corpora
        df = self.df.get(term, 0)
        return math.log((self.N + 1) / (df + 1)) + 1.0

    def _tfidf(self, doc_id: str) -> dict:
        tf_map = self.tf[doc_id]
        return {t: tfv * self._idf(t) for t, tfv in tf_map.items()}

    def similarity(self, doc_id1: str, doc_id2: str) -> float:
        if doc_id1 not in self.docs or doc_id2 not in self.docs:
            raise KeyError("Unknown doc_id")

        v1 = self._tfidf(doc_id1)
        v2 = self._tfidf(doc_id2)

        if not v1 or not v2:
            return 0.0

        # dot over intersection
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        dot = sum(w * v2.get(t, 0.0) for t, w in v1.items())

        n1 = math.sqrt(sum(w * w for w in v1.values()))
        n2 = math.sqrt(sum(w * w for w in v2.values()))
        if n1 == 0.0 or n2 == 0.0:
            return 0.0

        return dot / (n1 * n2)


if __name__ == "__main__":
    ts = TextSimilarity()
    ts.add_document("d1", "The cat sat on the mat.")
    ts.add_document("d2", "The cat sat.")
    ts.add_document("d3", "Quantum mechanics is hard and cat like.")

    # print("d1 vs d2:", ts.similarity("d1", "d2"))  # should be relatively high
    # print("d1 vs d3:", ts.similarity("d1", "d3"))  # should be near 0

'''