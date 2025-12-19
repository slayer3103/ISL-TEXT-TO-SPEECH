export class TrieNode {
  children: Map<string, TrieNode> = new Map();
  isEndOfWord: boolean = false;
  frequency: number = 1;
}

export class Trie {
  private root: TrieNode = new TrieNode();

  insert(word: string, frequency: number = 1): void {
    let node = this.root;
    for (const char of word.toLowerCase()) {
      if (!node.children.has(char)) {
        node.children.set(char, new TrieNode());
      }
      node = node.children.get(char)!;
    }
    node.isEndOfWord = true;
    node.frequency = frequency;
  }

  search(prefix: string): string[] {
    const results: Array<{ word: string; frequency: number }> = [];
    let node = this.root;

    // Navigate to the prefix
    for (const char of prefix.toLowerCase()) {
      if (!node.children.has(char)) {
        return [];
      }
      node = node.children.get(char)!;
    }

    // DFS to collect all words with this prefix
    this.dfs(node, prefix.toLowerCase(), results);

    // Sort by frequency (descending) and return top results
    return results
      .sort((a, b) => b.frequency - a.frequency)
      .slice(0, 10)
      .map(item => item.word);
  }

  private dfs(
    node: TrieNode,
    prefix: string,
    results: Array<{ word: string; frequency: number }>
  ): void {
    if (node.isEndOfWord) {
      results.push({ word: prefix, frequency: node.frequency });
    }

    for (const [char, childNode] of node.children) {
      this.dfs(childNode, prefix + char, results);
    }
  }
}
