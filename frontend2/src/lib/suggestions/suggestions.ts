// File: src/lib/suggestions/suggestions.ts

import { Trie } from "../trie/trie";
import { Completion } from "../trie/trieTypes";

const SMALL_PROB = 1e-6;

export class SuggestionEngine {
  private trie: Trie;
  
  constructor(trie: Trie) {
    this.trie = trie;
  }
  
  /**
   * Get top suggestions given committed prefix + current letter distribution.
   * @param committedPrefix current accepted letters (without space)
   * @param letterDist map of char -> probability (from letter model)
   * @param K number of top suggestions desired
   */
  public getTopSuggestions(
    committedPrefix: string,
    letterDist: { [char: string]: number },
    K: number = 5
  ): Completion[] {
    const candidates = this.trie.getCompletions(committedPrefix, 500);
    const completions: Completion[] = [];
    
    for (const word of candidates) {
      const nextChar = word.charAt(committedPrefix.length);
      const emissionProb = letterDist[nextChar] ?? SMALL_PROB;
      const logEmission = Math.log(emissionProb);
      // Assume freq stored in trie node; for simplicity get via wordlist or separate map
      // For MVP let's assume freq = 1 if missing
      const freq = 1;
      const logPrior = Math.log(freq);
      const score = logPrior + logEmission;
      
      completions.push({ word, score, prior: logPrior, emission: logEmission });
    }
    
    completions.sort((a, b) => b.score - a.score);
    return completions.slice(0, K);
  }
}
