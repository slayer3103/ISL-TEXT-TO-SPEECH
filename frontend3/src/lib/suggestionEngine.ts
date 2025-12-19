import { Trie } from './trie';

export interface Distribution {
  [letter: string]: number;
}

export class SuggestionEngine {
  private trie: Trie;

  constructor(wordList: string[]) {
    this.trie = new Trie();
    wordList.forEach((word, index) => {
      // Use inverse index as frequency proxy
      const frequency = wordList.length - index;
      this.trie.insert(word, frequency);
    });
  }

  getSuggestions(
    prefix: string,
    modelDistribution?: Distribution
  ): string[] {
    if (prefix.length < 2) {
      return [];
    }

    const candidates = this.trie.search(prefix);

    if (!modelDistribution) {
      return candidates;
    }

    // Rank by combining frequency with next-letter likelihood
    return this.rankBySuggestions(prefix, candidates, modelDistribution);
  }

  private rankBySuggestions(
    prefix: string,
    candidates: string[],
    distribution: Distribution
  ): string[] {
    const scored = candidates.map(word => {
      const nextChar = word[prefix.length];
      const likelihood = distribution[nextChar?.toUpperCase()] || 0;
      return { word, score: likelihood };
    });

    return scored
      .sort((a, b) => b.score - a.score)
      .map(item => item.word);
  }
}
