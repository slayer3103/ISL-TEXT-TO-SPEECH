import { Trie } from "../trie/trie";

export type Completion = {
  word: string;
  score: number;
  logPrior: number;
  logEmission: number;
  freq: number;
};

const SMALL = 1e-9;

export class SuggestionEngine {
  trie: Trie;
  constructor(trie: Trie) {
    this.trie = trie;
  }

  getTopSuggestions(committedPrefix: string, letterDist: { [char: string]: number } = {}, K: number = 5): Completion[] {
    if (!committedPrefix || committedPrefix.length < 2) return [];

    const vals = Object.values(letterDist);
    if (vals.length === 0) return [];

    const candidates = this.trie.getCompletions(committedPrefix, 1000).filter(c => c.word.length >= 3);
    const topProb = Math.max(...vals, 0);
    const alpha = topProb < 0.5 ? 0.8 : (topProb < 0.75 ? 0.5 : 0.2);

    const results: Completion[] = [];
    for (const cand of candidates) {
      const word = cand.word;
      const freq = cand.freq || 1;
      const logPrior = Math.log((freq || 1) + SMALL);

      const nextPos = committedPrefix.length;
      let logEmission = Math.log(SMALL);
      if (nextPos < word.length) {
        const nextChar = word.charAt(nextPos);
        const p = (letterDist[nextChar] ?? 0);
        logEmission = Math.log((p > 0 ? p : SMALL));
      } else {
        logEmission = Math.log(0.99);
      }

      const score = alpha * logPrior + (1 - alpha) * logEmission;
      results.push({ word, score, logPrior, logEmission, freq });
    }

    results.sort((a, b) => b.score - a.score);
    return results.slice(0, K);
  }
}
