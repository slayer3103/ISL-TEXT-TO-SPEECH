// src/lib/trie/trie.ts
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
    try {
      // Attempt to load bundled JSON. Using static import ensures Vite includes the file in the bundle.
      const wordlist: Array<{ word: string; freq: number }> = (WORDLIST as any) || [];
      if (!Array.isArray(wordlist) || wordlist.length === 0) {
        console.warn('[Trie] wordlist.json loaded but is empty or not an array: src/lib/trie/wordlist.json');
      } else {
        console.info(`[Trie] loaded wordlist.json (${wordlist.length} entries)`);
      }
      for (const entry of wordlist) {
        const w = String(entry.word || '').trim().toLowerCase();
        if (!w) continue;
        const f = typeof entry.freq === 'number' ? entry.freq : 1;
        this.insert(w, f);
        this.wordFreqMap.set(w, f);
      }
    } catch (err) {
      console.error('[Trie] failed to load wordlist.json — check path & tsconfig.resolveJsonModule', err);
    }
  }

  insert(word: string, freq: number = 1) {
    const w = String(word || '').toLowerCase();
    if (!w) return;
    let node = this.root;
    for (let ch of w) {
      if (!node.children) node.children = {};
      if (!node.children[ch]) {
        node.children[ch] = { children: {}, isWord: false };
      }
      node = node.children[ch];
    }
    node.isWord = true;
    node.freq = freq;
  }

  /**
   * Raw DFS completions from a prefix. Returned words are lowercase.
   */
  getCompletions(prefix: string, limit: number = 500): Array<{ word: string; freq: number }> {
    const p = String(prefix || '').toLowerCase();
    let node = this.root;
    for (const ch of p) {
      if (!node.children || !node.children[ch]) {
        return [];
      }
      node = node.children[ch];
    }

    const out: Array<{ word: string; freq: number }> = [];
    const stack: Array<{ node: TrieNode; word: string }> = [{ node, word: p }];

    while (stack.length && out.length < limit) {
      const { node: n, word } = stack.pop()!;
      if (n.isWord) {
        out.push({ word, freq: n.freq ?? (this.wordFreqMap.get(word) ?? 1) });
      }
      if (n.children) {
        // deterministic traversal: alphabetical order reversed so smallest letter visited last
        const keys = Object.keys(n.children).sort().reverse();
        for (const k of keys) {
          stack.push({ node: n.children[k], word: word + k });
        }
      }
    }

    return out;
  }

  /**
   * Preferred API for UI: returns top completions sorted by a light-weight combined score.
   *
   * options:
   *  - limit: how many results to return
   *  - minLen: exclude words shorter than this (default 2)
   *  - maxLen: exclude words longer than this (default 12)
   *  - requireVowel: if true, exclude words without vowels
   *
   * The scoring is: score = freq * (1 + englishScore), where englishScore is a small boost for vowel presence
   * and penalties for non-alpha characters and extreme lengths.
   */
  getTopCompletions(
    prefix: string,
    limit = 5,
    opts?: { minLen?: number; maxLen?: number; requireVowel?: boolean }
  ): Array<{ word: string; freq: number }> {
    const minLen = typeof opts?.minLen === 'number' ? opts!.minLen : 2; // exclude single-letter by default
    const maxLen = typeof opts?.maxLen === 'number' ? opts!.maxLen : 12;
    const requireVowel = !!opts?.requireVowel;

    const raw = this.getCompletions(prefix, 200); // get a reasonable set then filter & sort
    if (!raw || raw.length === 0) return [];

    // scoring helper: simple english-likeness
    const englishScore = (w: string) => {
      const vowels = (w.match(/[aeiou]/g) || []).length;
      const letters = (w.match(/[a-z]/g) || []).length;
      const nonAlpha = w.length - letters;
      const vf = letters > 0 ? vowels / letters : 0;
      const nonAlphaPenalty = Math.max(0, nonAlpha) * 0.1;
      // small penalty for extreme lengths: prefer around length ~5
      const lengthPenalty = Math.abs(5 - w.length) / 10; // 0..n
      return Math.max(0, vf) - nonAlphaPenalty - lengthPenalty * 0.05;
    };

    const filtered = raw
      .map(r => ({ word: r.word, freq: r.freq ?? 1 }))
      .filter(r => r.word.length >= minLen && r.word.length <= maxLen)
      .filter(r => !requireVowel || /[aeiou]/i.test(r.word));

    const scored = filtered.map(r => {
      const s = englishScore(r.word);
      const score = (r.freq || 1) * (1 + Math.max(0, s));
      return { ...r, score };
    });

    scored.sort((a, b) => (b.score - a.score) || (b.freq - a.freq) || a.word.localeCompare(b.word));

    return scored.slice(0, limit).map(s => ({ word: s.word, freq: s.freq ?? 1 }));
  }
}

export default Trie;
