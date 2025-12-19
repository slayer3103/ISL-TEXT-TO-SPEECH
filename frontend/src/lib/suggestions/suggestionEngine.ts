// src/lib/suggestions/suggestionEngine.ts
import type { Trie } from "../trie/trie";

export type DistMap = { [token: string]: number };

export class SuggestionEngine {
  trie: Trie;

  constructor(trieInstance: Trie) {
    this.trie = trieInstance as any;
  }

  getTopCompletions(
    prefix: string,
    limit = 5,
    opts?: { minLen?: number; maxLen?: number; requireVowel?: boolean }
  ): Array<{ word: string; freq: number }> {
    const p = String(prefix || "").toLowerCase();
    const defaultMin = typeof opts?.minLen === "number" ? opts.minLen! : Math.max(1, p.length || 1);
    const safeOpts = { minLen: defaultMin, maxLen: opts?.maxLen ?? 12, requireVowel: opts?.requireVowel ?? false };
    return (this.trie.getTopCompletions(p, Math.max(limit, 5), safeOpts)) || [];
  }

  getTopSuggestions(prefix: string, distMap?: DistMap, limit = 5): string[] {
    const p = String(prefix || "").toLowerCase().trim();

    // DEBUG: log inputs
    try {
      console.debug("[Suggest] getTopSuggestions called", { prefix: p, distMap: distMap || {}, limit });
    } catch (e) { /* ignore console errors */ }

    const results: string[] = [];
    const seen = new Set<string>();

    const pushWords = (arr: Array<{ word: string; freq?: number }>) => {
      for (const r of arr) {
        if (!r || !r.word) continue;
        const w = String(r.word).toLowerCase();
        if (seen.has(w)) continue;
        seen.add(w);
        results.push(w);
        if (results.length >= limit) break;
      }
    };

    if (p.length > 0) {
      const tops = this.getTopCompletions(p, Math.max(limit, 6), { minLen: Math.max(1, p.length) });
      pushWords(tops);
      // DEBUG: log intermediate completions
      console.debug("[Suggest] completions for prefix", p, tops.map(t => t.word));
      return results.slice(0, limit);
    }

    if (distMap && Object.keys(distMap).length > 0) {
      const tokens = Object.keys(distMap)
        .map(k => ({ k: String(k || "").toLowerCase(), v: Number(distMap[k] || 0) }))
        .filter(x => x.k.length > 0)
        .sort((a, b) => b.v - a.v);

      const TOP_TOKENS = Math.min(4, tokens.length);
      const perTokenLimit = Math.max(3, Math.ceil(limit / Math.max(1, TOP_TOKENS)));

      for (let i = 0; i < TOP_TOKENS && results.length < limit; ++i) {
        const tok = tokens[i].k;
        const seed = tok.length === 1 ? tok : tok.slice(0, Math.min(3, tok.length));
        const comps = this.getTopCompletions(seed, perTokenLimit, { minLen: 2 });
        pushWords(comps);
        console.debug("[Suggest] seed", seed, "->", comps.map(c => c.word));
      }

      if (results.length === 0) {
        const fallback = this.getTopCompletions("", limit, { minLen: 3 });
        pushWords(fallback);
        console.debug("[Suggest] fallback ->", fallback.map(c => c.word));
      }

      console.debug("[Suggest] final results (distMap path)", results.slice(0, limit));
      return results.slice(0, limit);
    }

    const fallback = this.getTopCompletions("", limit, { minLen: 3 });
    pushWords(fallback);
    console.debug("[Suggest] final fallback results", results.slice(0, limit));
    return results.slice(0, limit);
  }
}

export default SuggestionEngine;
