// File: src/lib/trie/trie.ts

import { TrieNode, Completion } from "./trieTypes";
import wordlist from "./wordlist.json";  // assumed: list of words + freq

export class Trie {
  private nodes: TrieNode[];      // compact array of nodes
  private words: string[];        // parallel array of words (optional)
  
  constructor() {
    this.nodes = [];
    this.words = [];
    this.loadWordlist();
  }
  
  private loadWordlist() {
    // Example: wordlist.json format: [{ word: "this", freq: 43210 }, ...]
    // Build nodes array with root at index 0
    this.nodes.push({ children: {} });
    for (let i = 0; i < wordlist.length; i++) {
      const { word, freq } = wordlist[i];
      this.insert(word, freq);
    }
  }
  
  private insert(word: string, freq: number) {
    let nodeIdx = 0;
    for (let i = 0; i < word.length; i++) {
      const ch = word[i];
      const node = this.nodes[nodeIdx];
      if (!node.children) node.children = {};
      if (!(ch in node.children)) {
        const newIndex = this.nodes.length;
        node.children[ch] = newIndex;
        this.nodes.push({ children: {} });
      }
      nodeIdx = node.children[ch]!;
    }
    this.nodes[nodeIdx].isWord = true;
    this.nodes[nodeIdx].freq = freq;
    this.words.push(word);
    this.nodes[nodeIdx].wordIndex = this.words.length - 1;
  }
  
  public getCompletions(prefix: string, limit: number = 10): string[] {
    let nodeIdx = 0;
    for (let ch of prefix) {
      const node = this.nodes[nodeIdx];
      if (!node.children || !(ch in node.children)) {
        return [];
      }
      nodeIdx = node.children[ch]!;
    }
    // Now we’re at the node matching the prefix.
    // Do DFS/BFS to collect up to `limit` words beneath this node.
    const results: string[] = [];
    const stack: Array<{ idx: number; wordSoFar: string }> = [{ idx: nodeIdx, wordSoFar: prefix }];
    while (stack.length > 0 && results.length < limit) {
      const { idx, wordSoFar } = stack.pop()!;
      const n = this.nodes[idx];
      if (n.isWord && typeof n.wordIndex === 'number') {
        results.push(this.words[n.wordIndex]);
      }
      if (n.children) {
        for (const ch in n.children) {
          stack.push({ idx: n.children[ch], wordSoFar: wordSoFar + ch });
        }
      }
    }
    return results;
  }
}
