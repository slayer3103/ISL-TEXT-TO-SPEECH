// File: src/lib/trie/trieTypes.ts

export type TrieNode = {
  children?: { [char: string]: number };  // index to node list (compact representation)
  isWord?: boolean;
  freq?: number;                           // word-frequency prior (for scoring)
  wordIndex?: number;                      // index into words array if needed
};

export type Completion = {
  word: string;
  score: number;       // combined prior + emission score (log-space)
  prior: number;       // log prior
  emission: number;    // log emission from letter model
};
