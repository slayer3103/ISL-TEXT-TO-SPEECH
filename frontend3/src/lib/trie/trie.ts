import WORDLIST from './wordlist.json';

export type TrieNode = {
  children: { [ch: string]: TrieNode } | null;
  isWord: boolean;
  freq?: number;
};

export class Trie {
  root: TrieNode;
  wordFreqMap: Map<string, number>;

  constructor() {
    this.root = { children: {}, isWord: false };
    this.wordFreqMap = new Map();
    this.buildFromWordlist();
  }

  private buildFromWordlist() {
    // Attempt to load bundled JSON. Using static import ensures Vite includes the file in the bundle.
    const wordlist: Array<{word:string;freq:number}> = (WORDLIST as any) || [];
    for (const entry of wordlist as Array<{ word: string; freq: number }>) {
      const w = entry.word;
      const f = entry.freq || 1;
      this.insert(w, f);
      this.wordFreqMap.set(w, f);
    }
  }

  insert(word: string, freq: number = 1) {
    let node = this.root;
    for (let ch of word) {
      if (!node.children) node.children = {};
      if (!node.children[ch]) {
        node.children[ch] = { children: {}, isWord: false };
      }
      node = node.children[ch];
    }
    node.isWord = true;
    node.freq = freq;
  }

  getCompletions(prefix: string, limit: number = 500): Array<{ word: string; freq: number }> {
    let node = this.root;
    for (const ch of prefix) {
      if (!node.children || !node.children[ch]) {
        return [];
      }
      node = node.children[ch];
    }

    const out: Array<{ word: string; freq: number }> = [];
    const stack: Array<{ node: TrieNode; word: string }> = [{ node, word: prefix }];

    while (stack.length && out.length < limit) {
      const { node: n, word } = stack.pop()!;
      if (n.isWord) {
        out.push({ word, freq: n.freq ?? (this.wordFreqMap.get(word) ?? 1) });
      }
      if (n.children) {
        const keys = Object.keys(n.children).sort().reverse();
        for (const k of keys) {
          stack.push({ node: n.children[k], word: word + k });
        }
      }
    }

    return out;
  }
}
