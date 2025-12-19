import React from 'react';
import WORDLIST from './lib/trie/wordlist.json';

export default function TestWordlist() {
  console.log('WORDLIST import ->', WORDLIST);
  return (
    <div style={{ padding: 20 }}>
      <h3>Wordlist test</h3>
      <div>{Array.isArray(WORDLIST) ? `Loaded ${WORDLIST.length} items` : 'wordlist not an array'}</div>
      <pre style={{ maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(WORDLIST.slice(0, 20), null, 2)}</pre>
    </div>
  );
}
